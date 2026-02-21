"""Gemini Cloud extraction service for comparative analysis.

Calls cloud Gemini models (gemini-2.5-flash, gemini-3-flash-preview) via
google.generativeai SDK and returns KVT4 extraction results alongside
timing/privacy metadata for side-by-side comparison with local MedGemma.

When no API key is available or an API call fails, the service transparently
falls back to pre-cached results stored in data/cached_cloud_results/.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from kvt_utils import extract_kvt_fact_lines, normalize_readmission_kvt4_lines

CACHED_DIR = Path(__file__).resolve().parents[1] / "data" / "cached_cloud_results"

# ---------------------------------------------------------------------------
# Extraction prompt – reuse the project's optimized, battle-tested prompt
# ---------------------------------------------------------------------------

from prompts.optimized_prompt import READMISSION_MVP_PROMPT_OPTIMIZED

_CLOUD_EXTRACTION_PROMPT = READMISSION_MVP_PROMPT_OPTIMIZED

AVAILABLE_MODELS = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CloudExtractionResult:
    model_id: str
    raw_response: str
    kvt4_lines: List[str]
    normalized_lines: List[str]
    normalization_stats: Dict[str, Any]
    fact_count: int
    cluster_coverage: List[str]
    format_validity_rate: float
    duration_sec: float
    is_cached: bool
    error: Optional[str] = None
    privacy_label: str = "☁️ Cloud API"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_kvt4_lines(text: str) -> List[str]:
    """Parse KVT4 fact lines from raw model response."""
    lines: List[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        # Skip markdown fences, headers, comments
        if stripped.startswith("```") or stripped.startswith("#") or stripped.startswith("//"):
            continue
        parts = stripped.split("|")
        if len(parts) == 4:
            lines.append(stripped)
    return lines


def _cluster_list(lines: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 1:
            c = parts[0].strip().upper()
            if c and c not in seen:
                seen.add(c)
                out.append(c)
    return out


def _format_validity(raw_lines: List[str], total_non_empty: int) -> float:
    if total_non_empty == 0:
        return 0.0
    return len(raw_lines) / total_non_empty if total_non_empty else 0.0


def _count_non_empty_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("```") and not line.strip().startswith("#"))


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

import hashlib

def _param_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]

def _cache_key(model_id: str, case_id: str, prompt: str, note_text: str) -> str:
    safe_model = model_id.replace("/", "_").replace(".", "_")
    p_hash = _param_hash(prompt)
    n_hash = _param_hash(note_text)
    return f"{safe_model}__{case_id}__{p_hash}_{n_hash}"


def _load_cached(model_id: str, case_id: str, prompt: str, note_text: str) -> Optional[CloudExtractionResult]:
    key = _cache_key(model_id, case_id, prompt, note_text)
    path = CACHED_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CloudExtractionResult(
            model_id=data["model_id"],
            raw_response=data["raw_response"],
            kvt4_lines=data["kvt4_lines"],
            normalized_lines=data["normalized_lines"],
            normalization_stats=data.get("normalization_stats", {}),
            fact_count=data["fact_count"],
            cluster_coverage=data["cluster_coverage"],
            format_validity_rate=data.get("format_validity_rate", 0.0),
            duration_sec=data.get("duration_sec", 0.0),
            is_cached=True,
            error=None,
            privacy_label="☁️ Cloud API (cached)",
        )
    except Exception:
        return None


def _save_cache(model_id: str, case_id: str, prompt: str, note_text: str, result: CloudExtractionResult) -> None:
    CACHED_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(model_id, case_id, prompt, note_text)
    path = CACHED_DIR / f"{key}.json"
    data = asdict(result)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Gemini API call
# ---------------------------------------------------------------------------

def _get_api_key() -> Optional[str]:
    """Read API key from env or .env file."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    # Try .env at repo root
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return None


def _call_gemini(model_id: str, note_text: str) -> tuple[str, float]:
    """Call Gemini API and return (response_text, duration_sec)."""
    from google import genai
    from google.genai import types

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment or .env file")

    client = genai.Client(api_key=api_key)

    prompt = _CLOUD_EXTRACTION_PROMPT.replace("{EHR_TEXT}", note_text)

    start = time.perf_counter()
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8192,
        ),
    )
    duration = round(time.perf_counter() - start, 3)

    if response.candidates:
        try:
            finish_reason = response.candidates[0].finish_reason
            if finish_reason != "STOP":
                print(f"[{model_id}] Generated with finish_reason: {finish_reason}")
        except Exception:
            pass

    text = response.text if response.text else ""
    return text, duration


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_with_cloud(
    model_id: str,
    note_text: str,
    case_id: str = "custom",
) -> CloudExtractionResult:
    """Run KVT4 extraction with a cloud Gemini model.

    Falls back to cached results if the API is unavailable.
    """
    # Try cache first for known synthetic cases
    if case_id != "custom":
        cached = _load_cached(model_id, case_id, _CLOUD_EXTRACTION_PROMPT, note_text)
        if cached is not None:
            return cached

    # Try live API
    try:
        raw_text, duration = _call_gemini(model_id, note_text)
    except Exception as exc:
        # Build a clean error message
        exc_str = str(exc)
        if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
            clean_error = "⏳ API quota exhausted (429). Try again later or use a different API key."
        elif "403" in exc_str or "PERMISSION_DENIED" in exc_str:
            clean_error = "🔑 API key invalid or lacks permissions (403)."
        elif "404" in exc_str or "NOT_FOUND" in exc_str:
            clean_error = f"❌ Model '{model_id}' not found (404). Check model name."
        elif "timeout" in exc_str.lower():
            clean_error = "⏱️ API request timed out. Try again."
        else:
            clean_error = f"API error: {exc_str[:200]}"

        # If API fails, try cache even for custom
        cached = _load_cached(model_id, case_id, _CLOUD_EXTRACTION_PROMPT, note_text)
        if cached is not None:
            cached.error = f"{clean_error} (showing cached results)"
            return cached
        return CloudExtractionResult(
            model_id=model_id,
            raw_response="",
            kvt4_lines=[],
            normalized_lines=[],
            normalization_stats={},
            fact_count=0,
            cluster_coverage=[],
            format_validity_rate=0.0,
            duration_sec=0.0,
            is_cached=False,
            error=clean_error,
        )

    # Parse results
    kvt4_lines = _parse_kvt4_lines(raw_text)
    non_empty = _count_non_empty_lines(raw_text)
    validity = _format_validity(kvt4_lines, non_empty)
    normalized_lines, stats = normalize_readmission_kvt4_lines(kvt4_lines)
    clusters = _cluster_list(normalized_lines)

    result = CloudExtractionResult(
        model_id=model_id,
        raw_response=raw_text,
        kvt4_lines=kvt4_lines,
        normalized_lines=normalized_lines,
        normalization_stats=stats,
        fact_count=len(normalized_lines),
        cluster_coverage=clusters,
        format_validity_rate=round(validity, 4),
        duration_sec=duration,
        is_cached=False,
    )

    # Cache for future use
    if case_id != "custom":
        try:
            _save_cache(model_id, case_id, _CLOUD_EXTRACTION_PROMPT, note_text, result)
        except Exception:
            pass

    return result


def list_available_models() -> List[Dict[str, str]]:
    """Return available cloud models for the UI selector."""
    has_key = _get_api_key() is not None
    out = []
    for display, model_id in AVAILABLE_MODELS.items():
        status = "ready" if has_key else "no API key (cached only)"
        out.append({"display": display, "model_id": model_id, "status": status})
    return out
