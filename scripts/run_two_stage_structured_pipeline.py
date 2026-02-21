#!/usr/bin/env python3
"""
Two-stage structured pipeline (low-memory friendly):

Stage 1 (BASE): Structured Output JSON (schema enforced when supported) -> Markdown
Stage 2 (FT):   Consume ONLY Markdown -> output KVT4 facts

Designed for OpenAI-compatible inference backends (LM Studio, llama.cpp server, vLLM, etc.)
via a minimal DSPy-free OpenAI-compatible client.

Typical low-memory workflow:
  1) Start backend with base model only.
  2) Run:  stage1
  3) Restart backend with LoRA/merged FT model.
  4) Run:  stage2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai_compat import OpenAICompatibleChatClient  # noqa: E402
from json_extract import extract_first_json_object  # noqa: E402
from kvt_utils import extract_kvt_fact_lines, normalize_readmission_kvt4_lines, strip_medgemma_internal_tokens  # noqa: E402
from prompts.optimized_prompt import (  # noqa: E402
    READMISSION_DOMAIN_JSON_SYSTEM_PROMPT,
    READMISSION_DOMAIN_JSON_SYSTEM_PROMPT_SGR_V1,
    READMISSION_DOMAIN_JSON_SYSTEM_PROMPT_SGR_V2,
    READMISSION_DOMAIN_JSON_SYSTEM_PROMPT_SGR_V2_STRICT,
    READMISSION_DOMAIN_JSON_SYSTEM_PROMPT_SGR_V2_STRICT_CASCADE,
    READMISSION_DOMAIN_JSON_SYSTEM_PROMPT_SGR_V3,
    READMISSION_DOMAIN_JSON_SYSTEM_PROMPT_SGR_V4,
    READMISSION_DOMAIN_JSON_SYSTEM_PROMPT_SGR_V2_COMPACT,
    READMISSION_STAGE2_FROM_DOMAIN_MARKDOWN_PROMPT_FULL,
    READMISSION_STAGE2_FROM_DOMAIN_MARKDOWN_PROMPT_STRICT_LINES,
    READMISSION_STAGE2_OBJECTIVE_FROM_DOMAIN_MARKDOWN_PROMPT_LINES,
    READMISSION_STAGE2_ALL_FROM_DOMAIN_MARKDOWN_PROMPT_LINES,
    READMISSION_STAGE2_ALL_TRAINING_MATCH_PROMPT,
)
from readmission_metrics import (  # noqa: E402
    DEFAULT_DOWNSTREAM_CONFIG,
    DownstreamMetricConfig,
    compute_downstream_score,
    compute_metrics,
)


def _project_gt_to_kvt4_lines(gt_obj: Any) -> List[str]:
    """
    Best-effort conversion of a ground-truth JSON object into KVT4 lines:

        CLUSTER|Keyword|Value|Timestamp

    Supported shapes:
    - list[str] (already lines)
    - list[dict] with keys: (cluster|C), (keyword|K), (value|V), (timestamp|T)
    - dict with a list under one of: facts / toon_facts / kvt_facts / fact_lines / lines
    """

    def _as_str(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, bool):
            return "true" if x else "false"
        return str(x)

    def _from_fact_dict(d: Dict[str, Any]) -> Optional[str]:
        cluster = _as_str(d.get("cluster") or d.get("CLUSTER") or d.get("C")).strip()
        keyword = _as_str(d.get("keyword") or d.get("KEYWORD") or d.get("K")).strip()
        value = _as_str(d.get("value") or d.get("VALUE") or d.get("V")).strip()
        timestamp = _as_str(d.get("timestamp") or d.get("TIMESTAMP") or d.get("T")).strip()
        if not cluster or not keyword:
            return None
        if not timestamp:
            timestamp = "Unknown"
        return f"{cluster}|{keyword}|{value}|{timestamp}"

    if isinstance(gt_obj, list):
        out: List[str] = []
        for item in gt_obj:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
                continue
            if isinstance(item, dict):
                line = _from_fact_dict(item)
                if line:
                    out.append(line)
        return out

    if isinstance(gt_obj, dict):
        for key in ["facts", "toon_facts", "kvt_facts", "fact_lines", "lines"]:
            v = gt_obj.get(key)
            if isinstance(v, list):
                return _project_gt_to_kvt4_lines(v)
        # Fallback: some GT formats store a single string blob.
        blob = gt_obj.get("text") or gt_obj.get("raw") or ""
        if isinstance(blob, str) and "|" in blob:
            return [ln.strip() for ln in blob.splitlines() if ln.strip()]

    return []


DOMAIN_KEYS = [
    "DEMOGRAPHICS",
    "VITALS",
    "LABS",
    "PROBLEMS",
    "SYMPTOMS",
    "MEDICATIONS",
    "PROCEDURES",
    "UTILIZATION",
    "DISPOSITION",
]

_STAR_SUFFIX_RE = re.compile(r"(?P<num>-?\d+(?:\.\d+)?)\*")
_SPO2_RE = re.compile(
    r"\bSpO2\s*=\s*(?P<num>-?\d+(?:\.\d+)?)\s*%?(?:\s*RA)?(?=[;,\s]|$)",
    re.IGNORECASE,
)
_BP_PAIR_RE = re.compile(
    r"\b(?P<label>Systolic BP|BP)\s*=\s*(?P<sbp>-?\d+(?:\.\d+)?)\s*/\s*(?P<dbp>-?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_BP_RE = re.compile(
    r"\b(?P<key>Systolic BP|Diastolic BP)\s*=\s*(?P<sbp>-?\d+(?:\.\d+)?)\s*/\s*(?P<dbp>-?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)

_STAGE2_FACT_OBJ_RE = re.compile(
    r'\{\s*"(?:cluster|CLUSTER)"\s*:\s*"(?P<cluster>[^"]+)"\s*,\s*'
    r'"(?:keyword|KEYWORD)"\s*:\s*"(?P<keyword>[^"]+)"\s*,\s*'
    r'"(?:value|VALUE)"\s*:\s*(?P<value>"[^"]*"|-?\d+(?:\.\d+)?|true|false|null)\s*,\s*'
    r'"(?:timestamp|TIMESTAMP)"\s*:\s*"(?P<timestamp>[^"]+)"\s*\}',
    re.MULTILINE,
)

_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_FIRST_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

_BINARY_FLAG_KEYWORDS = {
    "Anticoagulation",
    "Insulin Therapy",
    "Opioid Therapy",
    "Diuretic Therapy",
    "Any Procedure",
    "Surgery",
    "Dialysis",
}
_BINARY_TRUE_VALUES = {"1", "1.0", "true", "yes", "y"}
_BINARY_FALSE_VALUES = {"0", "0.0", "false", "no", "n"}
_UTILIZATION_PLACEHOLDERS = {"", "not stated", "none", "unknown", "n/a", "___", "..."}

_CANON_VITALS_ORDER = [
    "Heart Rate",
    "Systolic BP",
    "Diastolic BP",
    "Respiratory Rate",
    "Temperature",
    "SpO2",
    "Weight",
]
_CANON_LABS_ORDER = [
    "Hemoglobin",
    "Hematocrit",
    "WBC",
    "Platelet",
    "Sodium",
    "Potassium",
    "Creatinine",
    "BUN",
    "Glucose",
    "Bicarbonate",
]

_LAB_KEY_ALIAS: Dict[str, str] = {
    "wbc": "WBC",
    "white blood cell": "WBC",
    "hemoglobin": "Hemoglobin",
    "hgb": "Hemoglobin",
    "hematocrit": "Hematocrit",
    "hct": "Hematocrit",
    "platelet": "Platelet",
    "plt": "Platelet",
    "sodium": "Sodium",
    "na": "Sodium",
    "potassium": "Potassium",
    "k": "Potassium",
    "creatinine": "Creatinine",
    "creat": "Creatinine",
    "bun": "BUN",
    "urea n": "BUN",
    "urean": "BUN",
    "blood urea nitrogen": "BUN",
    "glucose": "Glucose",
    "bicarbonate": "Bicarbonate",
    "hco3": "Bicarbonate",
    "bicarb": "Bicarbonate",
    "co2": "Bicarbonate",
    "total co2": "Bicarbonate",
}

_TEXT_PLACEHOLDERS = {
    "",
    "not stated",
    "none",
    "none.",
    "unknown",
    "n/a",
    "na",
    "null",
    "...",
    "___",
}
_SURGERY_RE = re.compile(
    r"(surgery|surgical|ectomy|otomy|plasty|arthro|laparotomy|repair|resection|bypass|stent|amputation|transplant)",
    re.IGNORECASE,
)
_NEGATED_SURGERY_RE = re.compile(r"\b(without surgery|no surgery|denies surgery|declined surgery)\b", re.IGNORECASE)
_SURGERY_KV_NEG_RE = re.compile(r"^\s*(surgery|any procedure)\s*=\s*(no|0|0\.0|false|not stated)\s*$", re.IGNORECASE)
_SURGERY_KV_POS_RE = re.compile(r"^\s*surgery\s*=\s*(yes|1|1\.0|true)\s*$", re.IGNORECASE)
_GENERIC_SURGICAL_HEADER_RE = re.compile(r"^\s*major surgical or invasive procedure\b", re.IGNORECASE)
_HISTORICAL_PROC_RE = re.compile(r"\b(history of|hx of|h/o|s/p|status post|prior|previous)\b", re.IGNORECASE)
_DIALYSIS_RE = re.compile(r"(dialysis|hemodialysis|cvvh)", re.IGNORECASE)
_VENT_RE = re.compile(r"(ventilation|ventilator|intubat)", re.IGNORECASE)
_PROCEDURE_MENTION_RE = re.compile(
    r"(surgery|surgical|ectomy|otomy|plasty|arthro|laparotomy|repair|resection|bypass|stent|amputation|transplant|"
    r"dialysis|hemodialysis|cvvh|ventilation|ventilator|intubat|extubat|tracheostom)",
    re.IGNORECASE,
)
_SYMPTOM_FRAGMENT_STOP = {
    "and arm",
    "and leg",
    "and hand",
    "and face",
    "arm",
    "leg",
    "hand",
    "face",
}
_ANATOMIC_TOKENS = {
    "arm",
    "arms",
    "leg",
    "legs",
    "hand",
    "hands",
    "foot",
    "feet",
    "face",
    "ear",
    "ears",
    "eye",
    "eyes",
    "tongue",
}
_DIR_OR_JOIN_TOKENS = {"left", "right", "bilateral", "and", "or"}


def _parse_stop_list(stop_args: Optional[List[str]]) -> Optional[List[str]]:
    if not stop_args:
        return None
    out: List[str] = []
    for item in stop_args:
        for s in str(item).split(","):
            t = s.strip()
            if t:
                out.append(t)
    return out or None


def _kvt4_is_valid_line(line: str) -> bool:
    ln = (line or "").strip()
    if not ln:
        return False
    if ln.count("|") != 3:
        return False
    parts = [p.strip() for p in ln.split("|")]
    return len(parts) == 4 and all(parts)


def _compute_kvt4_format_stats(
    *,
    raw_text: str,
    extracted_lines: List[str],
    output_mode: str,
    did_retry: bool,
    facts_after_sanitize_count: int,
) -> Dict[str, Any]:
    mode = (output_mode or "").strip().casefold()
    recovered_cluster_prefix = 0
    end_lines = 0
    kvt3_not_stated_lines = 0
    kvt3_not_stated_examples: List[str] = []
    if mode == "json":
        # In JSON mode the raw output is not line-oriented; evaluate stability on extracted KVT4 candidates.
        basis = "extracted_fact_lines"
        candidates = [ln.strip() for ln in (extracted_lines or []) if ln.strip()]
    else:
        basis = "raw_output_lines"
        raw_lines = [ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()]
        end_lines = sum(1 for ln in raw_lines if ln.casefold() == "end")
        # Exclude the mandated terminator line from format stability scoring.
        tmp = [ln for ln in raw_lines if ln.casefold() != "end"]
        # Some models emit explicit "not stated" KVT3 placeholders like:
        #   SYMPTOMS|not stated|Admission
        # These are not facts (they are banned elsewhere); track them transparently but
        # exclude from KVT4 format stability scoring.
        candidates = []
        for ln in tmp:
            s = ln.strip()
            if s.count("|") == 2:
                parts = [p.strip() for p in s.split("|")]
                if len(parts) == 3 and parts[1].casefold() == "not stated":
                    kvt3_not_stated_lines += 1
                    if len(kvt3_not_stated_examples) < 5:
                        kvt3_not_stated_examples.append(s)
                    continue
            candidates.append(ln)
        for ln in candidates:
            s = (ln or "").strip()
            if s.count("|") == 4:
                parts = [p.strip() for p in s.split("|")]
                if len(parts) == 5 and parts[0].strip().upper() in {"CLUSTER", "CLUSTERS"}:
                    recovered = "|".join(parts[1:])
                    if _kvt4_is_valid_line(recovered):
                        recovered_cluster_prefix += 1

    valid = [ln for ln in candidates if _kvt4_is_valid_line(ln)]
    invalid = [ln for ln in candidates if ln and not _kvt4_is_valid_line(ln)]

    uniq = set(valid)
    duplicates_exact = len(valid) - len(uniq)

    effective_valid = len(valid) + int(recovered_cluster_prefix)
    effective_rate = (effective_valid / len(candidates)) if candidates else 0.0

    return {
        "format_stats_basis": basis,
        "raw_total_lines": len(candidates),
        "raw_valid_kvt4_lines": len(valid),
        "raw_invalid_lines": len(invalid),
        "raw_invalid_examples": invalid[:5],
        "raw_duplicates_exact": int(duplicates_exact),
        "raw_recovered_cluster_prefix_lines": int(recovered_cluster_prefix),
        "raw_effective_valid_kvt4_lines": int(effective_valid),
        "raw_effective_valid_rate": float(round(effective_rate, 6)),
        "raw_end_lines": int(end_lines),
        "raw_kvt3_not_stated_lines": int(kvt3_not_stated_lines),
        "raw_kvt3_not_stated_examples": kvt3_not_stated_examples,
        "facts_after_sanitize_count": int(facts_after_sanitize_count),
        "did_retry": bool(did_retry),
    }


def _raw_kvt4_validity(raw_text: str, extracted_lines: List[str]) -> Tuple[int, int, float]:
    """Return (raw_valid, raw_total, raw_valid_rate) for stage2 retry heuristics."""
    raw_lines = [ln for ln in (raw_text or "").splitlines() if ln.strip()]
    raw_total = len(raw_lines)
    raw_valid = len(extracted_lines or [])
    raw_rate = (raw_valid / raw_total) if raw_total else 0.0
    return raw_valid, raw_total, raw_rate


def _drop_stage2_prompt_leakage_lines(lines: List[str]) -> List[str]:
    """Drop obvious prompt/instruction leakage lines from Stage2 extracted candidates."""
    if not lines:
        return lines

    leak_substrings = [
        "output limits",
        "input limits",
        "hard cap",
        "canonical keywords",
        "must match exactly",
        "begin extraction",
        "one fact per line",
        "cluster|keyword|value|timestamp",
    ]
    leak_prefixes = ("##", "<h1", "<h2", "<h3", "<p", "<ul", "<li")
    out: List[str] = []
    dropped = 0

    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue
        sl = s.casefold()

        if sl.startswith(leak_prefixes):
            dropped += 1
            continue
        if any(tok in sl for tok in leak_substrings):
            dropped += 1
            continue
        out.append(s)

    if dropped:
        print(f"    [post-filter] dropped {dropped} prompt-leakage line(s)", flush=True)
    return out


def _sanitize_stage2_lines(lines: List[str], *, scope: str) -> List[str]:
    """
    Deterministic hygiene for Stage2 outputs before normalization:
    - strip common numeric decorations (e.g. leading '$')
    - enforce numeric-only for VITALS/LABS/UTILIZATION
    - drop objective facts with value == 'not stated' (avoid downstream FP noise)

    Scope behavior:
    - scope='objective': dedupe by (CLUSTER, Keyword), prefer Discharge, normalize timestamps.
    - scope='all': dedupe objective clusters only; semantic clusters keep model timestamps.
      Objective timestamps can still be canonically normalized via
      MEDGEMMA_OBJECTIVE_TS_CANONICAL_ALL.
    """

    scope_l = (scope or "").strip().casefold()
    if scope_l not in {"objective", "all"}:
        raise ValueError("scope must be 'objective' or 'all'")

    objective_clusters = {"DEMOGRAPHICS", "VITALS", "LABS", "UTILIZATION", "DISPOSITION"}
    numeric_clusters = {"VITALS", "LABS", "UTILIZATION"}
    semantic_clusters = {"PROBLEMS", "SYMPTOMS", "MEDICATIONS", "PROCEDURES"}
    recover_3part = _env_truthy_stage2("MEDGEMMA_STAGE2_RECOVER_3PART_LINES", validated_default="0", experimental_default="1")
    reclassify_non_numeric = _env_truthy_stage2(
        "MEDGEMMA_STAGE2_RECLASSIFY_NONNUMERIC_CLUSTERS",
        validated_default="0",
        experimental_default="1",
    )

    # Dedup objective clusters by (CLUSTER, Keyword), prefer Discharge over Admission.
    best_objective: Dict[Tuple[str, str], Tuple[str, str, str, str, int]] = {}
    other_lines: List[str] = []

    def ts_rank(ts: str) -> int:
        t = (ts or "").strip().casefold()
        if t in {"discharge", "dc"}:
            return 2
        if t in {"admission", "adm"}:
            return 1
        return 0

    def _normalize_semantic_keyword(keyword: str) -> str:
        k = " ".join((keyword or "").strip().split())
        return k.rstrip(" :;,.")

    def _split_semantic_items(value: str) -> List[str]:
        raw = (value or "").strip()
        if not raw:
            return []
        parts: List[str] = []
        # First split by semicolon/newline, then comma.
        level1 = [x.strip() for x in re.split(r"[;\n]+", raw) if x.strip()]
        for seg in level1:
            for item in [x.strip() for x in seg.split(",") if x.strip()]:
                parts.append(item)

        out: List[str] = []
        seen: set[str] = set()
        for item in parts:
            v = " ".join(item.split()).strip(" -")
            if not v:
                continue
            if v.casefold() in _TEXT_PLACEHOLDERS:
                continue
            if v.casefold() in {"none", "nil"}:
                continue
            if v not in seen:
                out.append(v)
                seen.add(v)
        return out

    def _normalize_problem_value(value: str, ts_raw: str) -> Optional[str]:
        vv = re.sub(r"\s+", " ", (value or "").strip().casefold())
        if not vv or vv in _TEXT_PLACEHOLDERS:
            return None
        if vv in {"chronic", "acute", "exist", "not exist"}:
            return vv
        if vv in {"past", "history", "historical", "pmh", "chronic condition", "chronic disease"}:
            return "chronic"
        if vv in {"discharge", "discharged", "active", "current"}:
            return "acute"
        if vv in {"present", "yes", "true", "1", "positive", "confirmed", "exists"}:
            return "exist"
        if vv in {"no", "none", "false", "0", "absent", "negative", "not present", "ruled out"}:
            return "not exist"
        ts_cf = (ts_raw or "").strip().casefold()
        if ts_cf in {"discharge", "dc"} and "discharg" in vv:
            return "acute"
        if ts_cf == "past" and ("hist" in vv or "past" in vv):
            return "chronic"
        return None

    def _normalize_symptom_value(value: str) -> Optional[str]:
        vv = re.sub(r"\s+", " ", (value or "").strip().casefold())
        if not vv or vv in _TEXT_PLACEHOLDERS:
            return None
        if vv in {"yes", "no", "severe"}:
            return vv
        if vv in {"present", "positive", "true", "1", "y", "symptomatic"}:
            return "yes"
        if vv in {"none", "absent", "negative", "false", "0", "n", "denied", "denies"}:
            return "no"
        if "severe" in vv or vv in {"marked", "significant"}:
            return "severe"
        return None

    def _expand_semantic_line(cluster_u: str, keyword: str, value: str, ts_raw: str) -> List[str]:
        # Keep MEDICATIONS/PROCEDURES as-is for post-filters; only expand PROBLEMS/SYMPTOMS.
        if cluster_u not in {"PROBLEMS", "SYMPTOMS"}:
            return [f"{cluster_u}|{keyword}|{value}|{ts_raw}"]

        kw = _normalize_semantic_keyword(keyword)
        kw_cf = kw.casefold()
        items = _split_semantic_items(value)

        if cluster_u == "PROBLEMS":
            acute_keys = {"discharge dx", "working dx", "complication", "complications"}
            chronic_keys = {"pmh/comorbidities", "pmh", "comorbidities", "past medical history"}
            if kw_cf in acute_keys and items:
                return [f"PROBLEMS|{it}|acute|Discharge" for it in items]
            if kw_cf in chronic_keys and items:
                return [f"PROBLEMS|{it}|chronic|Past" for it in items]
            norm_v = _normalize_problem_value(value, ts_raw)
            if norm_v is None:
                return []
            ts_out = ts_raw
            if (ts_out or "").strip().casefold() == "unknown":
                if norm_v == "acute":
                    ts_out = "Discharge"
                elif norm_v == "chronic":
                    ts_out = "Past"
                else:
                    ts_out = "Admission"
            return [f"PROBLEMS|{kw}|{norm_v}|{ts_out}"]

        if cluster_u == "SYMPTOMS":
            adm_keys = {"adm symptoms", "admission symptoms", "admission sx"}
            dc_keys = {"dc symptoms", "discharge symptoms", "discharge sx"}
            if kw_cf in adm_keys and items:
                return [f"SYMPTOMS|{it}|yes|Admission" for it in items]
            if kw_cf in dc_keys and items:
                return [f"SYMPTOMS|{it}|yes|Discharge" for it in items]
            norm_v = _normalize_symptom_value(value)
            if norm_v is None:
                return []
            return [f"SYMPTOMS|{kw}|{norm_v}|{ts_raw}"]

        return [f"{cluster_u}|{kw}|{value}|{ts_raw}"]

    # Known keyword → cluster mapping for recovering 3-part lines (missing cluster prefix).
    _KW_TO_CLUSTER: Dict[str, str] = {}
    for _kw in ("Heart Rate", "Systolic BP", "Diastolic BP", "Respiratory Rate", "Temperature", "SpO2", "Weight"):
        _KW_TO_CLUSTER[_kw] = "VITALS"
    for _kw in ("Hemoglobin", "Hematocrit", "WBC", "Platelet", "Sodium", "Potassium", "Creatinine", "BUN", "Glucose", "Bicarbonate"):
        _KW_TO_CLUSTER[_kw] = "LABS"
    for _kw in ("Sex", "Age"):
        _KW_TO_CLUSTER[_kw] = "DEMOGRAPHICS"
    for _kw in ("Prior Admissions 12mo", "ED Visits 6mo", "Days Since Last Admission", "Current Length of Stay"):
        _KW_TO_CLUSTER[_kw] = "UTILIZATION"
    for _kw in ("Discharge Disposition", "Mental Status"):
        _KW_TO_CLUSTER[_kw] = "DISPOSITION"
    for _kw in ("Any Procedure", "Surgery", "Dialysis", "Mechanical Ventilation"):
        _KW_TO_CLUSTER[_kw] = "PROCEDURES"
    for _kw in ("Medication Count", "New Medications Count", "Polypharmacy", "Anticoagulation", "Insulin Therapy", "Opioid Therapy", "Diuretic Therapy"):
        _KW_TO_CLUSTER[_kw] = "MEDICATIONS"

    for ln in lines:
        parts = [p.strip() for p in (ln or "").split("|")]
        # Recover 3-part lines (missing cluster prefix) using keyword→cluster mapping.
        if recover_3part and len(parts) == 3:
            a, b, c = parts

            # Case A: model emitted "<CLUSTER>|<Keyword>|<Value>" (missing Timestamp).
            cluster_guess = a.strip().strip("*<>").strip().upper()
            if cluster_guess in (objective_clusters | semantic_clusters) and b and c:
                # Canonical default timestamps for recovery.
                if cluster_guess == "DISPOSITION":
                    ts = "Discharge"
                elif cluster_guess == "UTILIZATION":
                    ts = "Past"
                else:
                    ts = "Admission"
                parts = [cluster_guess, b, c, ts]
            else:
                # Case B: model emitted "<Keyword>|<Value>|<Timestamp>" (missing Cluster).
                kw, val, ts = a, b, c
                inferred = _KW_TO_CLUSTER.get(kw, "")
                if inferred and kw and val and ts:
                    parts = [inferred, kw, val, ts]
        if len(parts) != 4:
            continue
        cluster, keyword, value, timestamp = parts
        if not (cluster and keyword and value and timestamp):
            continue

        value = value.strip()
        if value.startswith("$"):
            value = value.lstrip("$").strip()

        # Strip markdown bold/italic markers and angle-bracket wrappers
        # that the model sometimes emits (e.g. *DEMOGRAPHICS**, <LABS>).
        cluster_u = cluster.strip().strip("*<>").strip().upper()
        value_cf = value.casefold()
        ts = timestamp.strip()

        # "not stated" is a placeholder, not a fact. Drop it for all clusters.
        if value_cf == "not stated":
            continue

        # Normalize common timestamp shorthands.
        if ts.casefold() == "adm":
            ts = "Admission"
        elif ts.casefold() == "dc":
            ts = "Discharge"

        if cluster_u in numeric_clusters:
            if not _NUM_RE.match(value):
                continue

        # Fix cluster confusion: reclassify semantic-cluster lines that have
        # canonical non-numeric objective keywords (MEDICATIONS, PROCEDURES, DISPOSITION,
        # DEMOGRAPHICS). Do NOT reclassify to VITALS/LABS — the model often echoes
        # wrong numeric values under PROBLEMS, and reclassifying would overwrite correct
        # entries via timestamp-priority dedup.
        if reclassify_non_numeric and cluster_u in semantic_clusters:
            correct_cluster = _KW_TO_CLUSTER.get(keyword, "")
            if correct_cluster and correct_cluster not in numeric_clusters and correct_cluster in objective_clusters:
                cluster_u = correct_cluster

        if cluster_u in objective_clusters:
            key = (cluster_u, keyword)
            r = ts_rank(ts)
            prev = best_objective.get(key)
            if prev is None or r > prev[4]:
                best_objective[key] = (cluster_u, keyword, value, ts, r)
        elif cluster_u in semantic_clusters:
            if scope_l == "all":
                if _env_truthy_stage2("MEDGEMMA_EXPAND_SEMANTIC_LINES", validated_default="0", experimental_default="1"):
                    other_lines.extend(_expand_semantic_line(cluster_u, keyword, value, ts))
                else:
                    other_lines.append(f"{cluster_u}|{keyword}|{value}|{ts}")
        else:
            # Unknown cluster: keep only in "all" mode, unchanged.
            if scope_l == "all":
                other_lines.append(f"{cluster_u}|{keyword}|{value}|{ts}")

    force_objective_ts_canonical_in_all = _env_truthy_stage2(
        "MEDGEMMA_OBJECTIVE_TS_CANONICAL_ALL",
        validated_default="0",
        experimental_default="1",
    )

    out: List[str] = []
    for (_cluster_u, _keyword), (c, k, v, ts_raw, r) in best_objective.items():
        if scope_l == "objective" or force_objective_ts_canonical_in_all:
            # Canonical objective timestamps (policy default).
            if c == "DISPOSITION":
                ts = "Discharge"
            elif c == "UTILIZATION":
                ts = "Past"
            else:
                ts = "Admission"
        else:
            # Keep what the model produced (after ADM/DC normalization).
            ts = ts_raw
        out.append(f"{c}|{k}|{v}|{ts}")

    if scope_l == "all":
        # Dedup semantic lines by exact string only (keep multiple problems/symptoms).
        seen = set(out)
        for ln in other_lines:
            if ln not in seen:
                out.append(ln)
                seen.add(ln)

    # Stable order: cluster then keyword.
    out.sort(key=lambda s: (s.split("|", 1)[0], s.split("|", 2)[1]))
    return out


def _sanitize_demographics_text(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return "Sex=not stated\nAge=not stated"

    # Common degenerate outputs: "F" / "M"
    if t.upper() in {"F", "M"}:
        sex = "female" if t.upper() == "F" else "male"
        return f"Sex={sex}\nAge=not stated"

    t = t.replace("Sex=F", "Sex=female").replace("Sex=M", "Sex=male")

    # Ensure both Sex and Age exist (Stage2 hard-lock needs Sex explicit).
    has_sex = "sex=" in t.casefold()
    has_age = "age=" in t.casefold()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not has_sex:
        lines.insert(0, "Sex=not stated")
    if not has_age:
        lines.append("Age=not stated")
    return "\n".join(lines[:2])


def _sanitize_demographics_obj(v: Any) -> Dict[str, Any]:
    """
    SGR-v4: DEMOGRAPHICS is a structured object (keys: sex/age).
    Normalize to stable values but keep the object shape.
    """
    if not isinstance(v, dict):
        return {"sex": "not stated", "age": "not stated"}
    sex_raw = _as_text(v.get("sex") or v.get("Sex") or "").strip().casefold()
    if sex_raw in {"m", "male"}:
        sex = "male"
    elif sex_raw in {"f", "female"}:
        sex = "female"
    else:
        sex = "not stated"
    age_raw = _as_text(v.get("age") or v.get("Age") or "").strip()
    age_num = _extract_numeric_value(age_raw) if age_raw and age_raw.casefold() != "not stated" else None
    age = age_num if age_num is not None else ("not stated" if not age_raw or age_raw.casefold() == "not stated" else age_raw)
    return {"sex": sex, "age": age}


def _sanitize_obj_numeric(value: str, *, prefer_second: bool = False, prefer_last_if_kg: bool = False) -> str:
    """
    Best-effort numeric-only sanitizer for SGR-v4 object fields.
    Returns numeric string or "not stated".
    """
    s = (value or "").strip()
    if not s or s.casefold() == "not stated":
        return "not stated"
    if "/" in s:
        parts = [p for p in s.split("/") if p.strip()]
        if len(parts) >= 2:
            pick = parts[1] if prefer_second else parts[0]
            num = _extract_numeric_value(pick)
            return num if num is not None else "not stated"
    if prefer_last_if_kg and "kg" in s.casefold():
        nums = _FIRST_NUM_RE.findall(s)
        return nums[-1] if nums else "not stated"
    num = _extract_numeric_value(s.replace("%", "").replace("RA", "").replace("ra", ""))
    return num if num is not None else "not stated"


def _sanitize_objective_obj(v: Any, *, kind: str) -> Dict[str, Any]:
    """
    SGR-v4: VITALS/LABS are structured objects with admission/discharge dicts.
    Keep the object shape and normalize values to numeric-only strings or "not stated".
    """
    if not isinstance(v, dict):
        return {"admission": {}, "discharge": {}}
    out: Dict[str, Any] = {"admission": {}, "discharge": {}}
    for part in ("admission", "discharge"):
        src = v.get(part) or {}
        if not isinstance(src, dict):
            src = {}
        dst: Dict[str, str] = {}
        for k, raw in src.items():
            key = str(k)
            val = _as_text(raw)
            if kind == "vitals":
                if key in {"diastolic_bp", "Diastolic BP"}:
                    dst[key] = _sanitize_obj_numeric(val, prefer_second=True)
                elif key in {"weight", "Weight"}:
                    dst[key] = _sanitize_obj_numeric(val, prefer_last_if_kg=True)
                else:
                    dst[key] = _sanitize_obj_numeric(val)
            else:
                dst[key] = _sanitize_obj_numeric(val)
        out[part] = dst
    return out


def _sanitize_vitals_or_labs_text(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return "not stated"

    # Normalize common numeric decorations.
    t = _STAR_SUFFIX_RE.sub(lambda m: m.group("num"), t)
    t = _SPO2_RE.sub(lambda m: f"SpO2={m.group('num')}", t)

    # Expand paired BP into two fields when emitted as "Systolic BP=120/80" or "BP=120/80".
    def bp_pair_fix(match: re.Match[str]) -> str:
        sbp = match.group("sbp")
        dbp = match.group("dbp")
        return f"Systolic BP={sbp}; Diastolic BP={dbp}"

    t = _BP_PAIR_RE.sub(bp_pair_fix, t)

    # Fix BP fields if model emits "169/99".
    def bp_fix(match: re.Match[str]) -> str:
        key = match.group("key").strip().lower()
        sbp = match.group("sbp")
        dbp = match.group("dbp")
        if key.startswith("systolic"):
            return f"Systolic BP={sbp}"
        return f"Diastolic BP={dbp}"

    t = _BP_RE.sub(bp_fix, t)
    return t


def _sanitize_disposition_text(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return "not stated"
    allowed_dispo = {
        "home": "Home",
        "home with services": "Home with Services",
        "home with service": "Home with Services",
        "home w services": "Home with Services",
        "home w service": "Home with Services",
        "home + services": "Home with Services",
        "snf": "SNF",
        "skilled nursing facility": "SNF",
        "extended care": "SNF",
        "extended": "SNF",
        "rehab": "Rehab",
        "rehabilitation": "Rehab",
        "ltac": "LTAC",
        "hospice": "Hospice",
        "ama": "AMA",
        "left ama": "AMA",
    }
    fields: Dict[str, str] = {}

    def set_if_empty(key: str, val: str) -> None:
        v = (val or "").strip()
        if not v:
            return
        if key not in fields or fields[key].strip().casefold() in {"", "not stated"}:
            fields[key] = v

    for ln in [x.strip() for x in t.splitlines() if x.strip()]:
        # Normalize common separators.
        if ":" in ln and "=" not in ln:
            k0, v0 = ln.split(":", 1)
            ln = f"{k0.strip()}={v0.strip()}"

        if "=" in ln:
            k, v = ln.split("=", 1)
            key = k.strip()
            val = v.strip()
            if key.casefold() in {"discharge disposition", "disposition"}:
                raw = val.casefold().strip()
                mapped = allowed_dispo.get(raw, val)
                set_if_empty("Discharge Disposition", mapped)
            elif key.casefold() in {"mental status", "mental"}:
                set_if_empty("Mental Status", _sanitize_mental_status_value(val))
            elif key.casefold() in {"support needs", "support"}:
                set_if_empty("Support Needs", val)
            else:
                # Preserve unknown key=value lines but keep them deterministic.
                set_if_empty(key, val)
            continue

        # Bare token(s) like "Home" are common on small models; map them deterministically.
        raw = ln.strip()
        raw_cf = raw.casefold()
        if raw_cf in allowed_dispo:
            set_if_empty("Discharge Disposition", allowed_dispo[raw_cf])
            continue
        ms = _sanitize_mental_status_value(raw)
        if ms != "not stated":
            set_if_empty("Mental Status", ms)
            continue

    # Ensure Stage2 objective parser always sees canonical keys in stable order.
    dispo = fields.get("Discharge Disposition", "").strip() or "not stated"
    mental = fields.get("Mental Status", "").strip() or "not stated"
    support = fields.get("Support Needs", "").strip() or "not stated"
    return "\n".join([f"Discharge Disposition={dispo}", f"Mental Status={mental}", f"Support Needs={support}"])


def _normalize_binary_text_value(v: str) -> str:
    lv = (v or "").strip().casefold()
    if lv in _TEXT_PLACEHOLDERS:
        return "not stated"
    if lv in {"1", "1.0", "true", "yes", "y"}:
        return "yes"
    if lv in {"0", "0.0", "false", "no", "n"}:
        return "no"
    return v.strip()


def _sanitize_medications_text(s: str) -> str:
    """
    Normalize Stage1 MEDICATIONS block into stable integral keys/values.
    Keeps only canonical keys expected by Stage2, normalizes binary flags to yes/no.
    """
    t = (s or "").strip()
    if not t:
        return "not stated"

    canon_keys = [
        "Medication Count",
        "New Medications Count",
        "Polypharmacy",
        "Anticoagulation",
        "Insulin Therapy",
        "Opioid Therapy",
        "Diuretic Therapy",
    ]
    vals: Dict[str, str] = {k: "not stated" for k in canon_keys}

    for ln in [x.strip() for x in t.splitlines() if x.strip()]:
        if "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        key = k.strip()
        val = v.strip()
        if key not in vals:
            continue
        if key in {"Medication Count", "New Medications Count"}:
            num = _extract_numeric_value(val)
            vals[key] = num if num is not None else "not stated"
        else:
            vals[key] = _normalize_binary_text_value(val)

    return "\n".join([f"{k}={vals[k]}" for k in canon_keys])


def _note_has_procedure_mention(note_text: str) -> bool:
    t = (note_text or "").strip()
    if not t:
        return False
    return bool(_PROCEDURE_MENTION_RE.search(t))


def _sanitize_procedures_text(s: str, *, note_text: str = "") -> str:
    """
    Normalize Stage1 PROCEDURES into canonical integral keys for Stage2/RiskEngine:
    Any Procedure, Surgery, Dialysis, Mechanical Ventilation.
    """
    t = (s or "").strip()
    if not t:
        return "Any Procedure=not stated\nSurgery=not stated\nDialysis=not stated\nMechanical Ventilation=not stated"

    any_proc = "no"
    surgery = "no"
    dialysis = "no"
    ventilation = "no"

    # Prefer explicit canonical keys if already present.
    explicit: Dict[str, str] = {}
    for ln in [x.strip() for x in t.splitlines() if x.strip()]:
        if "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        key = k.strip()
        val = v.strip()
        if key in {"Any Procedure", "Surgery", "Dialysis", "Mechanical Ventilation"}:
            explicit[key] = val

    if explicit:
        any_proc = _normalize_binary_text_value(explicit.get("Any Procedure", any_proc))
        surgery = _normalize_binary_text_value(explicit.get("Surgery", surgery))
        dial_raw = (explicit.get("Dialysis", dialysis) or "").strip().casefold()
        if dial_raw in {"0", "0.0", "false", "n", "no"}:
            dial_raw = "no"
        elif dial_raw in {"1", "1.0", "true", "y", "yes"}:
            dial_raw = "done"
        if dial_raw in {"started", "done", "decided", "cancelled", "no"}:
            dialysis = dial_raw
        elif dial_raw in _TEXT_PLACEHOLDERS:
            dialysis = "not stated"
        else:
            dialysis = "done" if dial_raw else "not stated"
        vent_raw = (explicit.get("Mechanical Ventilation", ventilation) or "").strip()
        if vent_raw.casefold() in _TEXT_PLACEHOLDERS:
            ventilation = "not stated"
        elif vent_raw.casefold() in {"0", "0.0", "false", "n", "no"}:
            ventilation = "no"
        elif vent_raw.casefold() in {"1", "1.0", "true", "y", "yes"}:
            ventilation = "1"
        else:
            num = _extract_numeric_value(vent_raw)
            ventilation = num if num is not None else _normalize_binary_text_value(vent_raw)
    else:
        for ln in [x.strip() for x in t.splitlines() if x.strip()]:
            raw = ln
            if "=" in ln:
                k, v = ln.split("=", 1)
                key = k.strip()
                val = v.strip()
            else:
                key = ""
                val = ln.strip()
            merged = f"{key} {val}".strip()
            lv = val.casefold().strip()
            bin_lv = _normalize_binary_text_value(val).casefold()
            is_historical = bool(_HISTORICAL_PROC_RE.search(merged))
            is_present = bin_lv == "yes"
            if not is_present and bin_lv not in {"yes", "no", "not stated"}:
                is_present = lv not in _TEXT_PLACEHOLDERS and lv != "no"
            if is_present and not is_historical:
                any_proc = "yes"
            if _SURGERY_RE.search(merged):
                if is_present:
                    surgery = "yes"
            if _DIALYSIS_RE.search(merged):
                if bin_lv == "no":
                    dialysis = "no"
                elif bin_lv == "not stated" or lv in _TEXT_PLACEHOLDERS:
                    pass
                else:
                    dialysis = "done"
            if _VENT_RE.search(merged):
                if bin_lv == "no":
                    ventilation = "no"
                elif bin_lv == "not stated" or lv in _TEXT_PLACEHOLDERS:
                    pass
                else:
                    num = _extract_numeric_value(val)
                    ventilation = num if num is not None else "yes"

    # Harmonize "yes" to minimum valid values for typed fields.
    if dialysis == "yes":
        dialysis = "done"
    if ventilation == "yes":
        ventilation = "1"

    # Optional strict anti-hallucination mode for Stage1 QA:
    # treat typed PROCEDURES negatives as unknown to reduce unsupported defaults.
    if _env_truthy("MEDGEMMA_STAGE1_PROCEDURES_TYPED_NEG_TO_NOT_STATED", "0"):
        if surgery == "no":
            surgery = "not stated"
        if dialysis == "no":
            dialysis = "not stated"
        if ventilation == "no":
            ventilation = "not stated"
        if any_proc == "no" and surgery == "not stated" and dialysis == "not stated" and ventilation == "not stated":
            any_proc = "not stated"

    # Hallucination guard: Stage1 often emits all-zeros defaults for PROCEDURES with no note evidence.
    # Convert this pattern to "not stated" to avoid unsupported negatives in downstream scoring.
    if _env_truthy("MEDGEMMA_STAGE1_PROCEDURES_REQUIRE_EVIDENCE", "1"):
        has_pos_dialysis = dialysis in {"started", "done", "decided", "cancelled"}
        has_pos_vent = bool(ventilation) and ventilation not in {"no", "not stated", "0", "0.0", "false"}
        has_positive = surgery == "yes" or any_proc == "yes" or has_pos_dialysis or has_pos_vent
        has_mention = _note_has_procedure_mention(note_text)
        if not has_positive and not has_mention:
            any_proc = "not stated"
            surgery = "not stated"
            dialysis = "not stated"
            ventilation = "not stated"

    return (
        f"Any Procedure={any_proc}\n"
        f"Surgery={surgery}\n"
        f"Dialysis={dialysis}\n"
        f"Mechanical Ventilation={ventilation}"
    )


def _extract_numeric_value(value: str) -> Optional[str]:
    m = _FIRST_NUM_RE.search(value or "")
    return m.group(0) if m else None


def _sanitize_vitals_text(s: str) -> str:
    """
    Normalize Stage1 VITALS block into canonical numeric-only Key=Value pairs.
    Preserves ADM/DC line prefixes when present.
    """
    t = _sanitize_vitals_or_labs_text(s)
    if not t:
        return "not stated"

    out_lines: List[str] = []
    for ln in [x.strip() for x in t.splitlines() if x.strip()]:
        prefix = ""
        rest = ln
        if ":" in ln and ln.split(":", 1)[0].strip().casefold() in {"adm", "dc"}:
            prefix, rest = ln.split(":", 1)
            prefix = prefix.strip().upper()
            rest = rest.strip()

        kv: Dict[str, str] = {}
        parts = [p.strip() for p in rest.split(";") if p.strip()]
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            key = k.strip()
            val = v.strip()
            if key not in _CANON_VITALS_ORDER:
                continue
            if val.casefold() == "not stated":
                continue

            if key in {"Systolic BP", "Diastolic BP"}:
                # Already expanded earlier if paired.
                num = _extract_numeric_value(val)
            elif key == "SpO2":
                num = _extract_numeric_value(val.replace("%", "").replace("RA", "").replace("ra", ""))
            elif key == "Temperature":
                num = _extract_numeric_value(val)
            elif key == "Weight":
                # Prefer kg if present.
                nums = _FIRST_NUM_RE.findall(val)
                if "kg" in val.casefold() and nums:
                    num = nums[-1]
                else:
                    num = nums[0] if nums else None
            else:
                num = _extract_numeric_value(val)
            if num is None:
                continue
            kv[key] = num

        if not kv:
            continue

        rebuilt = "; ".join([f"{k}={kv[k]}" for k in _CANON_VITALS_ORDER if k in kv])
        if prefix:
            out_lines.append(f"{prefix}: {rebuilt}")
        else:
            out_lines.append(rebuilt)

    return "\n".join(out_lines) if out_lines else "not stated"


def _sanitize_labs_text(s: str) -> str:
    """
    Normalize Stage1 LABS block into canonical numeric-only Key=Value pairs.
    Filters non-canonical keys and non-numeric values (e.g., ASA=NEG).
    """
    t = _sanitize_vitals_or_labs_text(s)
    if not t:
        return "not stated"

    out_lines: List[str] = []
    canon = set(_CANON_LABS_ORDER)
    for ln in [x.strip() for x in t.splitlines() if x.strip()]:
        prefix = ""
        rest = ln
        if ":" in ln and ln.split(":", 1)[0].strip().casefold() in {"adm", "dc"}:
            prefix, rest = ln.split(":", 1)
            prefix = prefix.strip().upper()
            rest = rest.strip()

        kv: Dict[str, str] = {}
        parts = [p.strip() for p in rest.split(";") if p.strip()]
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            key_raw = k.strip()
            key = _LAB_KEY_ALIAS.get(_normalize_sparse_key(key_raw), key_raw)
            val = v.strip()
            if key not in canon:
                continue
            if val.casefold() == "not stated":
                continue
            num = _extract_numeric_value(val)
            if num is None:
                continue
            kv[key] = num

        if not kv:
            continue

        rebuilt = "; ".join([f"{k}={kv[k]}" for k in _CANON_LABS_ORDER if k in kv])
        if prefix:
            out_lines.append(f"{prefix}: {rebuilt}")
        else:
            out_lines.append(rebuilt)

    return "\n".join(out_lines) if out_lines else "not stated"


def _stage1_objective_to_kvt4_lines(stage1_normalized: Dict[str, Any]) -> List[str]:
    """Build deterministic objective KVT4 lines from Stage1 normalized payload.

    Stage2 may drift into partial KVT3 lines (missing cluster or timestamp).
    Stage1 already contains stabilized objective evidence, so we export it as
    KVT4 for robust downstream merging and strict normalization.
    """

    def _emit(cluster: str, keyword: str, value: str, timestamp: str) -> None:
        c = (cluster or "").strip().upper()
        k = (keyword or "").strip()
        v = (value or "").strip()
        t = (timestamp or "").strip() or "Unknown"
        if not (c and k and v):
            return
        if v.casefold() in {"not stated", "unknown", "n/a", "na", "null", "none", "___"}:
            return
        out.append(f"{c}|{k}|{v}|{t}")

    def _parse_kv_pairs(text: str) -> Dict[str, str]:
        kv: Dict[str, str] = {}
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            chunks = [c.strip() for c in line.split(";") if c.strip()] if ";" in line else [line]
            for ch in chunks:
                if "=" not in ch:
                    continue
                k, v = ch.split("=", 1)
                key = k.strip()
                val = v.strip()
                if key and val and key not in kv:
                    kv[key] = val
        return kv

    out: List[str] = []

    # DEMOGRAPHICS
    demo = stage1_normalized.get("DEMOGRAPHICS")
    if isinstance(demo, dict):
        sex = _as_text(demo.get("sex") or demo.get("Sex") or "").strip().casefold()
        age = _as_text(demo.get("age") or demo.get("Age") or "").strip()
        if sex in {"male", "female"}:
            _emit("DEMOGRAPHICS", "Sex", sex, "Admission")
        age_num = _extract_numeric_value(age) if age and age.casefold() != "not stated" else None
        if age_num:
            _emit("DEMOGRAPHICS", "Age", age_num, "Admission")
    else:
        kv = _parse_kv_pairs(_as_text(demo))
        sex = kv.get("Sex") or kv.get("sex") or ""
        age = kv.get("Age") or kv.get("age") or ""
        sex_cf = sex.strip().casefold()
        if sex_cf in {"male", "female"}:
            _emit("DEMOGRAPHICS", "Sex", sex_cf, "Admission")
        age_num = _extract_numeric_value(age) if age and age.strip().casefold() != "not stated" else None
        if age_num:
            _emit("DEMOGRAPHICS", "Age", age_num, "Admission")

    def _emit_objective_block(cluster: str, block: Any, allowed_keys: List[str]) -> None:
        if isinstance(block, dict):
            canon_norm_map = {_normalize_sparse_key(k): k for k in allowed_keys}
            for part, ts in (("admission", "Admission"), ("discharge", "Discharge")):
                src = block.get(part) if isinstance(block.get(part), dict) else {}
                for k, v in src.items():
                    key_raw = str(k).strip()
                    key_normed = _normalize_sparse_key(key_raw)
                    key_norm = canon_norm_map.get(key_normed) or _LAB_KEY_ALIAS.get(key_normed) or key_raw
                    if key_norm not in allowed_keys:
                        continue
                    num = _extract_numeric_value(_as_text(v))
                    if num is None:
                        continue
                    _emit(cluster, key_norm, num, ts)
            return

        for raw in _as_text(block).splitlines():
            line = raw.strip()
            if not line or line.casefold() == "not stated":
                continue
            prefix = ""
            rest = line
            if ":" in line and line.split(":", 1)[0].strip().casefold() in {"adm", "dc"}:
                prefix, rest = line.split(":", 1)
                prefix = prefix.strip().casefold()
                rest = rest.strip()
            ts = "Admission" if prefix == "adm" else ("Discharge" if prefix == "dc" else "Admission")
            kv = _parse_kv_pairs(rest)
            for key in allowed_keys:
                if key not in kv:
                    continue
                num = _extract_numeric_value(kv[key])
                if num is None:
                    continue
                _emit(cluster, key, num, ts)

    _emit_objective_block("VITALS", stage1_normalized.get("VITALS"), _CANON_VITALS_ORDER)
    _emit_objective_block("LABS", stage1_normalized.get("LABS"), _CANON_LABS_ORDER)

    util_keys = {"Prior Admissions 12mo", "ED Visits 6mo", "Days Since Last Admission", "Current Length of Stay"}
    util = stage1_normalized.get("UTILIZATION")
    if util:
        kv = _parse_kv_pairs(_as_text(util))
        for k, v in kv.items():
            if k not in util_keys:
                continue
            num = _extract_numeric_value(v)
            if num is None:
                continue
            _emit("UTILIZATION", k, num, "Past")

    dispo_keys = {"Discharge Disposition", "Mental Status"}
    dispo = stage1_normalized.get("DISPOSITION")
    if dispo:
        kv = _parse_kv_pairs(_as_text(dispo))
        for k, v in kv.items():
            if k not in dispo_keys:
                continue
            _emit("DISPOSITION", k, v, "Discharge")

    return out


def _sanitize_mental_status_value(v: str) -> str:
    vv = (v or "").strip().casefold()
    if not vv or vv in {"...", "…"}:
        return "not stated"
    if "clear" in vv or "coherent" in vv:
        return "alert"
    if "intact" in vv or "oriented" in vv:
        return "oriented"
    if "letharg" in vv:
        return "lethargic"
    if "confus" in vv:
        return "confused"
    if vv in {"alert", "confused", "oriented", "lethargic"}:
        return vv
    return "not stated"


def _discover_hadm_ids(cohort_root: Path, n: int, require_ground_truth: bool = True) -> List[int]:
    ids: List[int] = []
    for p in sorted(cohort_root.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.isdigit():
            continue
        hadm = int(p.name)
        ehr = p / f"ehr_{hadm}.txt"
        gt = p / f"ground_truth_{hadm}.json"
        if not ehr.exists():
            continue
        if require_ground_truth and not gt.exists():
            continue
        if ehr.exists():
            ids.append(hadm)
        if len(ids) >= n:
            break
    return ids


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def _stage2_profile_name() -> str:
    return (os.getenv("MEDGEMMA_STAGE2_PROFILE", "v41_validated") or "").strip().casefold()


def _env_truthy_stage2(name: str, *, validated_default: str, experimental_default: str) -> bool:
    """Stage2 env gate with profile-aware defaults.

    - Explicit env var always wins.
    - If MEDGEMMA_STAGE2_PROFILE is in {"experimental", "tuning", "curated10_tuning"},
      use experimental_default.
    - Otherwise use validated_default ("v41_validated" baseline behavior).
    """
    if os.getenv(name) is not None:
        return _env_truthy(name, validated_default)

    profile = _stage2_profile_name()
    if profile in {"experimental", "tuning", "curated10_tuning"}:
        return _env_truthy(name, experimental_default)
    return _env_truthy(name, validated_default)


def _sha256_hex_utf8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stage2_prompt_template_for_run(*, mode_l: str, scope_l: str, use_training_match_prompt: bool) -> Tuple[str, str]:
    if mode_l == "json":
        if scope_l != "objective":
            raise SystemExit("--output-mode json currently supports only --scope objective")
        return ("READMISSION_STAGE2_FROM_DOMAIN_MARKDOWN_PROMPT_STRICT_LINES", READMISSION_STAGE2_FROM_DOMAIN_MARKDOWN_PROMPT_STRICT_LINES)

    if mode_l != "lines":
        raise SystemExit("--output-mode must be one of: lines, json")

    if scope_l == "objective":
        return ("READMISSION_STAGE2_OBJECTIVE_FROM_DOMAIN_MARKDOWN_PROMPT_LINES", READMISSION_STAGE2_OBJECTIVE_FROM_DOMAIN_MARKDOWN_PROMPT_LINES)
    if scope_l != "all":
        raise SystemExit("--scope must be one of: objective, all")

    if use_training_match_prompt:
        return ("READMISSION_STAGE2_ALL_TRAINING_MATCH_PROMPT", READMISSION_STAGE2_ALL_TRAINING_MATCH_PROMPT)
    return ("READMISSION_STAGE2_ALL_FROM_DOMAIN_MARKDOWN_PROMPT_LINES", READMISSION_STAGE2_ALL_FROM_DOMAIN_MARKDOWN_PROMPT_LINES)


def _stage2_prompt_prefix(template: str) -> str:
    """Compute a byte-stable Stage2 prefix used for llama.cpp prompt-cache hits.

    llama.cpp prompt cache requires the prompt prefix to be byte-for-byte stable.
    We hash the template prefix (everything before `{EHR_TEXT}`) as a regression guard.
    """
    t = (template or "").strip()
    if "{EHR_TEXT}" not in t:
        return t
    return t.split("{EHR_TEXT}", 1)[0]


def _trim_text(raw: str, *, max_chars: int, strategy: str) -> str:
    """Deterministically trim long clinical text to avoid backend 400s on small ctx sizes.

    Note: This is intentionally simple and DSPy-free. It mirrors the intent of
    `dspy_integration.py` trimming but is scoped to this runner.
    """

    s = str(raw or "")
    if max_chars <= 0 or len(s) <= max_chars:
        return s

    strat = (strategy or "").strip().casefold()
    if strat in {"middle", "center"}:
        start = max(0, (len(s) - max_chars) // 2)
        return s[start : start + max_chars]

    def _kw_to_regex(kw: str) -> str:
        # Avoid accidental full-document coverage from short substrings (e.g., "k", "na", "cr")
        # by requiring word boundaries for simple alphanumeric tokens.
        if re.fullmatch(r"[a-z0-9]+", kw or ""):
            return r"\b" + re.escape(kw) + r"\b"
        return re.escape(kw or "")

    if strat in {"keyword_window_objective_last", "keyword_window_obj_last", "keyword_window_strict"}:
        # Variant of keyword_window that keeps objective evidence (VITALS/LABS windows) close to the
        # end of the prompt. This helps SWA-style models that attend mostly to recent tokens.
        default_terms = [t.casefold() for t in (_CANON_VITALS_ORDER + _CANON_LABS_ORDER)]
        objective_terms = [
            "vitals:",
            "vital signs",
            "labs:",
            "pertinent labs",
            "bp",
            "heart rate",
            "respiratory rate",
            "spo2",
            "wbc",
            "hgb",
            "hct",
            "plt",
            "bun",
            "urea n",
            "creat",
            "total co2",
            "hco3",
            "glucose",
            "bicarb",
        ]
        terms = [t.strip().casefold() for t in os.getenv("MEDGEMMA_KEYWORD_WINDOW_TERMS", "").split(",") if t.strip()]
        terms = terms or (default_terms + objective_terms)
        objective_set = set(t.casefold() for t in (default_terms + objective_terms))

        window = int(os.getenv("MEDGEMMA_KEYWORD_WINDOW_CHARS", "900"))
        head_keep = int(os.getenv("MEDGEMMA_KEYWORD_WINDOW_HEAD_CHARS", "900"))
        try:
            obj_min = int(os.getenv("MEDGEMMA_KEYWORD_WINDOW_OBJ_MIN_CHARS", "2000"))
        except Exception:
            obj_min = 2000

        lower = s.casefold()
        obj_spans: List[Tuple[int, int]] = []
        other_spans: List[Tuple[int, int]] = []
        for kw in terms:
            if not kw:
                continue
            pat = _kw_to_regex(kw)
            if not pat:
                continue
            for m in re.finditer(pat, lower):
                span = (max(0, m.start() - window), min(len(s), m.end() + window))
                (obj_spans if kw in objective_set else other_spans).append(span)

        if not obj_spans and not other_spans:
            return _trim_text(s, max_chars=max_chars, strategy="middle")

        def _merge(spans: List[Tuple[int, int]]) -> List[List[int]]:
            if not spans:
                return []
            spans.sort()
            merged: List[List[int]] = []
            for st, en in spans:
                if not merged or st > merged[-1][1]:
                    merged.append([st, en])
                else:
                    merged[-1][1] = max(merged[-1][1], en)
            return merged

        obj_merged = _merge(obj_spans)
        other_merged = _merge(other_spans)

        head = s[: min(len(s), head_keep)]
        budget = max_chars - len(head)
        if budget <= 0:
            return head

        # This strategy is explicitly for keeping objective evidence close to the end.
        # Prefer using most/all remaining budget for objective windows; keep head_keep
        # as the only "global" context.
        obj_budget = budget
        other_budget = 0

        def _collect(merged: List[List[int]], budget_chars: int) -> str:
            if budget_chars <= 0 or not merged:
                return ""
            pieces: List[str] = []
            remain = budget_chars
            for st, en in merged:
                if remain <= 0:
                    break
                chunk = s[st:en]
                if len(chunk) <= remain:
                    pieces.append("\n\n" + chunk)
                    remain -= len(chunk)
                else:
                    pieces.append("\n\n" + chunk[:remain])
                    remain = 0
            return "".join(pieces)

        other_txt = _collect(other_merged, other_budget)
        obj_txt = _collect(obj_merged, obj_budget)
        return head + other_txt + obj_txt

    if strat in {"keyword_window", "kw", "window"}:
        # Anchor windows around objective terms so VITALS/LABS remain visible even if the note is long.
        default_terms = [t.casefold() for t in (_CANON_VITALS_ORDER + _CANON_LABS_ORDER)]
        extra = [
            "vitals:",
            "vital signs",
            "labs:",
            "pertinent labs",
            "bp",
            "heart rate",
            "respiratory rate",
            "spo2",
            "wbc",
            "hgb",
            "hct",
            "plt",
            "bun",
            "urea n",
            "creat",
            "total co2",
            "hco3",
            "glucose",
            "bicarb",
        ]
        terms = [t.strip().casefold() for t in os.getenv("MEDGEMMA_KEYWORD_WINDOW_TERMS", "").split(",") if t.strip()]
        terms = terms or (default_terms + extra)
        window = int(os.getenv("MEDGEMMA_KEYWORD_WINDOW_CHARS", "900"))
        head_keep = int(os.getenv("MEDGEMMA_KEYWORD_WINDOW_HEAD_CHARS", "900"))

        lower = s.casefold()
        spans: List[Tuple[int, int]] = []
        for kw in terms:
            if not kw:
                continue
            pat = _kw_to_regex(kw)
            if not pat:
                continue
            for m in re.finditer(pat, lower):
                spans.append((max(0, m.start() - window), min(len(s), m.end() + window)))

        if not spans:
            # No anchors found; fall back to middle slice.
            return _trim_text(s, max_chars=max_chars, strategy="middle")

        spans.sort()
        merged: List[List[int]] = []
        for st, en in spans:
            if not merged or st > merged[-1][1]:
                merged.append([st, en])
            else:
                merged[-1][1] = max(merged[-1][1], en)

        pieces: List[str] = [s[: min(len(s), head_keep)]]
        budget = max_chars - len(pieces[0])
        if budget <= 0:
            return pieces[0]

        for st, en in merged:
            if budget <= 0:
                break
            chunk = s[st:en]
            if len(chunk) <= budget:
                pieces.append("\n\n" + chunk)
                budget -= len(chunk)
            else:
                pieces.append("\n\n" + chunk[:budget])
                budget = 0
        return "".join(pieces)

    # Default: head+tail
    head_chars = int(os.getenv("MEDGEMMA_TRIM_HEAD_CHARS", str(max_chars // 2)))
    tail_chars = int(os.getenv("MEDGEMMA_TRIM_TAIL_CHARS", str(max_chars - head_chars)))
    head = s[: max(0, head_chars)]
    tail = s[-max(0, tail_chars) :] if tail_chars > 0 else ""
    return head + "\n\n" + tail


def _load_schema_response_format(schema_path: Path) -> Dict[str, Any]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    # OpenAI-compatible JSON schema wrapper (LM Studio supports this).
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "readmission_domain_summary",
            "strict": True,
            "schema": schema,
        },
    }


def _sanitize_objective_evidence_line(s: str) -> str:
    # Keep one-line evidence; remove obvious placeholder prefixes while keeping the numeric payload.
    t = re.sub(r"\s+", " ", (s or "").replace("|", " ")).strip()
    t = re.sub(r"^_+\s*", "", t)  # de-id placeholder prefix (if present)
    t = t.replace("___", "not stated")
    if len(t) > 240:
        t = t[:240].rstrip()
    return t or "not stated"


def _extract_objective_lines(note_text: str) -> tuple[List[str], List[str]]:
    """Extract compact objective evidence lines (vitals + labs) from the note."""

    lines = [ln.strip() for ln in (note_text or "").splitlines() if ln.strip()]
    if not lines:
        return [], []

    vitals: List[str] = []
    labs: List[str] = []

    for idx, ln in enumerate(lines):
        up = ln.upper()
        if ("VITALS:" not in up) and ("VITAL SIGNS" not in up):
            continue
        t0 = _sanitize_objective_evidence_line(ln)
        if t0 and t0 not in vitals:
            vitals.append(t0)
        # If it's a header-only line (or metadata-only line), include the next numeric line.
        has_payload = bool(
            re.search(
                r"(?i)(\\b[0-9]{2,3}/[0-9]{2,3}\\b|"
                r"\\bTemp\\s*[:=]?\\s*[0-9]|\\bT\\s*[:=]?\\s*[0-9]|"
                r"\\bBP\\s*[:=]?\\s*[0-9]|\\bHR\\s*[:=]?\\s*[0-9]|\\bRR\\s*[:=]?\\s*[0-9]|"
                r"\\b(?:SaO2|SpO2)\\s*[:=]?\\s*[0-9]|\\bO2\\s*sat\\s*[:=]?\\s*[0-9]|\\bWt\\s*[:=]?\\s*[0-9])",
                ln,
            )
        )
        if not has_payload:
            # Pull up to 2 continuation lines (vitals frequently wrap across lines).
            for j in range(idx + 1, min(len(lines), idx + 6)):
                if len(vitals) >= 3:
                    break
                ln2 = lines[j].strip()
                if not ln2:
                    continue
                if not re.search(r"\d", ln2):
                    continue
                t1 = _sanitize_objective_evidence_line(ln2)
                if t1 and t1 not in vitals:
                    vitals.append(t1)
        if len(vitals) >= 3:
            break

    # Fallback: unlabeled vitals line (common in exam sections): "169/68 55 17 99 99% RA"
    if not vitals:
        for ln in lines:
            if len(vitals) >= 3:
                break
            if not re.search(r"\b[0-9]{2,3}/[0-9]{2,3}\b", ln):
                continue
            if not re.search(r"(?i)(%|\bRA\b|SpO2|SaO2)", ln):
                continue
            t = _sanitize_objective_evidence_line(ln)
            if t and t not in vitals:
                vitals.append(t)

    # Labs often show up as compact hyphenated key-value sequences on 1-2 lines.
    lab_tokens = [
        "WBC",
        "HGB",
        "HCT",
        "PLT",
        "HEMOGLOBIN",
        "HEMATOCRIT",
        "PLATELET",
        "SODIUM",
        "POTASSIUM",
        "GLUCOSE",
        "BUN",
        "UREA N",
        "UREAN",
        "CREAT",
        "CREATININE",
        "TOTAL CO2",
        "CO2",
        "HCO3",
        "BICARB",
        "BICARBONATE",
    ]
    for ln in lines:
        up = ln.upper()
        if not any(tok in up for tok in lab_tokens):
            continue
        # Require digits on the line to reduce false positives (e.g., a section header).
        if not re.search(r"\d", ln):
            continue
        # Avoid obvious meds lines like "Losartan Potassium 100 mg ..." which can look lab-ish.
        if any(re.search(pat, ln, flags=re.IGNORECASE) for pat in [r"\bPO\b", r"\bIV\b", r"\bBID\b", r"\bTID\b", r"\bDAILY\b", r"\bQHS\b"]):
            continue
        # Require at least one lab token to appear in a key-value-ish form to reduce false positives.
        if not re.search(
            r"(?i)(WBC\s*[-:]|HGB\s*[-:]|HCT\s*[-:]|PLT(?:\s*(?:COUNT|CT))?\s*[-:]|"
            r"GLUCOSE\s*[-:]|UREA\s*N\s*[-:]|UREAN\s*[-:]|CREAT\s*[-:]|CREATININE\s*[-:]|"
            r"SODIUM\s*[-:]|POTASSIUM\s*[-:]|TOTAL\s*CO2\s*[-:]|HCO3\s*[-:]|BICARB\s*[-:]|CO2\s*[-:]|"
            r"\bNa\s*[-:]|\bK\s*[-:])",
            ln,
        ):
            continue
        ln2 = _sanitize_objective_evidence_line(ln)
        if ln2 and ln2 not in labs:
            labs.append(ln2)
        if len(labs) >= 6:
            break

    return vitals, labs


def _objective_lines_to_appendix(vitals: List[str], labs: List[str]) -> str:
    if not vitals and not labs:
        return ""
    out: List[str] = []
    out.append("OBJECTIVE EVIDENCE EXCERPT (verbatim):")
    if vitals:
        out.append("VITALS:")
        out.extend(vitals)
    if labs:
        out.append("LABS:")
        out.extend(labs)
    return "\n".join(out).strip()


def _extract_objective_appendix(note_text: str) -> str:
    """Extract a compact objective evidence excerpt to append near the end of the user prompt.

    Motivation: some SWA-style models (Gemma3/MedGemma) attend more strongly to recent tokens,
    causing objective evidence (vitals/labs) earlier in long notes to be ignored.
    """

    vitals, labs = _extract_objective_lines(note_text)
    return _objective_lines_to_appendix(vitals, labs)


_VITALS_INLINE_RE = re.compile(
    r"(?i)\bvitals:\s*"
    r"(?P<temp>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<hr>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<sbp>[0-9]+(?:\.[0-9]+)?)/(?P<dbp>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<rr>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<spo2>[0-9]+(?:\.[0-9]+)?)",
)


def _parse_vitals_from_lines(vitals_lines: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ln in vitals_lines:
        m = _VITALS_INLINE_RE.search(ln)
        if not m:
            continue
        out["temperature"] = m.group("temp")
        out["heart_rate"] = m.group("hr")
        out["systolic_bp"] = m.group("sbp")
        out["diastolic_bp"] = m.group("dbp")
        out["respiratory_rate"] = m.group("rr")
        out["spo2"] = m.group("spo2")
        return out

    text = " ".join(vitals_lines)
    if not text.strip():
        return out

    # Labeled/compact format: "Vitals: T: 98.4  P: 54 R: 16  BP: 141/40  SaO2: 94% on RA"
    # Also supports compact variants like "T98.7, HR90, RR18, BP116/77, SaO2 96".
    m_bp = re.search(r"(?i)\bBP\s*[:=]?\s*(?P<sbp>[0-9]{2,3})/(?P<dbp>[0-9]{2,3})\b", text)
    if not m_bp:
        # Unlabeled BP pair (common in exam sections).
        m_bp = re.search(r"\b(?P<sbp>[0-9]{2,3})/(?P<dbp>[0-9]{2,3})\b", text)
    if m_bp:
        out["systolic_bp"] = m_bp.group("sbp")
        out["diastolic_bp"] = m_bp.group("dbp")

    m_temp = re.search(r"(?i)\bT(?:emp(?:erature)?)?\s*[:=]?\s*(?P<v>[0-9]+(?:\.[0-9]+)?)\b", text)
    if m_temp:
        out["temperature"] = m_temp.group("v")

    m_hr = re.search(r"(?i)\b(?:HR|P|Pulse|Heart Rate)\s*[:=]?\s*(?P<v>[0-9]+(?:\.[0-9]+)?)\b", text)
    if m_hr:
        out["heart_rate"] = m_hr.group("v")

    m_rr = re.search(r"(?i)\b(?:RR|Resp(?:iratory)?\s*Rate|R)\s*[:=]?\s*(?P<v>[0-9]+(?:\.[0-9]+)?)\b", text)
    if m_rr:
        out["respiratory_rate"] = m_rr.group("v")

    m_spo2 = re.search(r"(?i)\b(?:SaO2|SpO2|O2\s*sat)\s*[:=]?\s*(?P<v>[0-9]+(?:\.[0-9]+)?)\b", text)
    if m_spo2:
        out["spo2"] = m_spo2.group("v")

    m_wt = re.search(r"(?i)\b(?P<v>[0-9]+(?:\.[0-9]+)?)\s*kg\b", text)
    if m_wt:
        out["weight"] = m_wt.group("v")

    # Fallback: unlabeled compact admission exam format like "169/68 55 17 99 99% RA".
    if out.get("systolic_bp") and out.get("diastolic_bp") and ("heart_rate" not in out or "respiratory_rate" not in out or "temperature" not in out or "spo2" not in out):
        for ln in vitals_lines:
            m = re.search(r"\b(?P<sbp>[0-9]{2,3})/(?P<dbp>[0-9]{2,3})\b", ln)
            if not m:
                continue
            rest = re.sub(r"\b[0-9]{2,3}/[0-9]{2,3}\b", " ", ln)
            # Avoid capturing the "2" in tokens like "SaO2" / "O2".
            nums = re.findall(r"(?<![A-Za-z])[0-9]+(?:\.[0-9]+)?(?![A-Za-z])", rest)
            if len(nums) < 4:
                continue
            out.setdefault("heart_rate", nums[0])
            out.setdefault("respiratory_rate", nums[1])
            out.setdefault("temperature", nums[2])
            out.setdefault("spo2", nums[-1])
            break

    return out


def _parse_labs_from_lines(labs_lines: List[str]) -> Dict[str, str]:
    # Avoid mis-parsing urine UA entries (e.g., "URINE WBC-1") as blood labs.
    filtered = [ln for ln in labs_lines if "URINE" not in (ln or "").upper()]
    text = " ".join(filtered)
    if not text.strip():
        return {}

    def first(patterns: List[str]) -> Optional[str]:
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                return m.group("v")
        return None

    pats: Dict[str, List[str]] = {
        "wbc": [
            r"\bWBC-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bWBC\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
        ],
        "hemoglobin": [
            r"\bHGB-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bHEMOGLOBIN-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bHGB\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bHgb\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
        ],
        "hematocrit": [
            r"\bHCT-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bHEMATOCRIT-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bHCT\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bHct\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
        ],
        "platelet": [
            r"\bPLT-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bPLT\s*COUNT-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bPLATELET-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bPlt(?:\s*Ct|\s*Count)?\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
        ],
        "sodium": [
            r"\bSODIUM-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bNa-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bSODIUM\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bNa\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
        ],
        "potassium": [
            r"\bPOTASSIUM-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bK-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bPOTASSIUM\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bK\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
        ],
        "creatinine": [
            r"\bCREAT(?:ININE)?-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bCreat(?:inine)?\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
        ],
        "bun": [
            r"\bBUN-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bUREA\s*N-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bUreaN-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bUREAN-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bBUN\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bUREA\s*N\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bUreaN\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
        ],
        "glucose": [
            r"\bGLUCOSE-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bGlucose\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
        ],
        "bicarbonate": [
            r"\bTOTAL\s*CO2-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bHCO3-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bBICARB(?:ONATE)?-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bCO2-(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bTOTAL\s*CO2\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bHCO3\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
            r"\bCO2\s*[:=]\s*(?P<v>[0-9]+(?:\.[0-9]+)?)(?:[*#])?\b",
        ],
    }

    out: Dict[str, str] = {}
    for k, ps in pats.items():
        v = first(ps)
        if v is not None:
            out[k] = v
    return out


def _fill_stage1_strict_cascade_objective_inplace(
    obj: Dict[str, Any],
    *,
    vitals_lines: List[str],
    labs_lines: List[str],
) -> None:
    """Deterministic objective fill for sgr_v2_strict_cascade.

    This is a pragmatic fallback: if the model fails to populate objective keys even when
    evidence is present, we fill missing numeric values from the extracted evidence lines.
    """

    def is_not_stated(v: Any) -> bool:
        return str(v or "").strip().casefold() == "not stated"

    vit = obj.get("VITALS")
    if isinstance(vit, dict):
        # Evidence slots: fill only if missing.
        for i in range(1, 4):
            k = f"evidence_line{i}"
            if is_not_stated(vit.get(k)) and (i - 1) < len(vitals_lines):
                vit[k] = vitals_lines[i - 1]

        vals = _parse_vitals_from_lines(vitals_lines)
        adm = vit.get("admission")
        if isinstance(adm, dict):
            for k, v in vals.items():
                if k in adm and is_not_stated(adm.get(k)):
                    adm[k] = v

    lab = obj.get("LABS")
    if isinstance(lab, dict):
        for i in range(1, 7):
            k = f"evidence_line{i}"
            if is_not_stated(lab.get(k)) and (i - 1) < len(labs_lines):
                lab[k] = labs_lines[i - 1]

        vals = _parse_labs_from_lines(labs_lines)
        adm = lab.get("admission")
        if isinstance(adm, dict):
            for k, v in vals.items():
                if k in adm and is_not_stated(adm.get(k)):
                    adm[k] = v


def _as_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _normalize_sparse_key(key: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (key or "").casefold())).strip()


def _semantic_item_clean(v: Any) -> str:
    """Normalize PROBLEMS/SYMPTOMS item text and strip placeholder leakage."""
    s = _as_text(v).strip()
    if not s:
        return ""
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.replace("|", " ").replace("___", "not stated")
    s = re.sub(r"\bnot stated\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bn/?a\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bunknown\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" ,;:-()[]{}")
    if not s or s.casefold() in _TEXT_PLACEHOLDERS:
        return ""
    return s


def _sanitize_stage1_value(v: Any) -> Any:
    """
    Best-effort sanitation for Stage1 JSON payloads.
    Keeps structured values (SGR-v1) intact while removing common drift tokens.
    """
    if isinstance(v, str):
        return v.replace("|", " ").replace("___", "not stated")
    if isinstance(v, list):
        out: List[str] = []
        for it in v:
            s = _as_text(it).strip()
            if not s:
                continue
            s = s.replace("\n", " ").replace("\r", " ")
            s = s.replace("|", " ").replace("___", "not stated")
            out.append(s)
        return out
    if isinstance(v, dict):
        return {str(k): _sanitize_stage1_value(val) for k, val in v.items()}
    return _as_text(v).replace("|", " ").replace("___", "not stated")


def _domain_json_to_markdown(obj: Dict[str, Any]) -> str:
    def norm_val(v: Any) -> str:
        s = _as_text(v).strip()
        s = s.replace("|", " ")
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
        s = s.replace("Sex=F", "Sex=female").replace("Sex=M", "Sex=male")
        return s

    def render_problems(v: Any) -> List[str]:
        if not isinstance(v, dict):
            t = norm_val(v)
            return [ln.rstrip() for ln in t.split("\n") if ln.strip()] if t else []
        # Keep markdown compatible with Stage2 expectations (key=value lines).
        def join_items(items: Any) -> str:
            if not isinstance(items, list):
                return "not stated"
            cleaned: List[str] = []
            seen: set[str] = set()
            for it in items:
                s = _semantic_item_clean(it)
                if not s:
                    continue
                nk = _normalize_sparse_key(s)
                if not nk or nk in seen:
                    continue
                seen.add(nk)
                cleaned.append(s)
            return ", ".join(cleaned) if cleaned else "not stated"

        pmh = join_items(v.get("pmh_comorbidities") or [])
        discharge = join_items(v.get("discharge_dx") or [])
        complications = join_items(v.get("complications") or [])
        working = join_items(v.get("working_dx") or [])
        return [
            f"PMH/Comorbidities={pmh}",
            f"Discharge Dx={discharge}",
            f"Complications={complications}",
            f"Working Dx={working}",
        ]

    def render_symptoms(v: Any) -> List[str]:
        if not isinstance(v, dict):
            t = norm_val(v)
            return [ln.rstrip() for ln in t.split("\n") if ln.strip()] if t else []
        # Keep markdown compatible with Stage2 expectations (key=value lines).
        def join_items(items: Any, *, max_items: int) -> str:
            if not isinstance(items, list):
                return "not stated"
            cleaned: List[str] = []
            seen: set[str] = set()
            for it in items:
                s = _semantic_item_clean(it)
                if not s:
                    continue
                nk = _normalize_sparse_key(s)
                if not nk or nk in seen:
                    continue
                seen.add(nk)
                cleaned.append(s)
                if len(cleaned) >= max_items:
                    break
            return ", ".join(cleaned) if cleaned else "not stated"

        # Keep SYMPTOMS conservative for small-model stability.
        adm = join_items(v.get("admission") or [], max_items=3)
        dc = join_items(v.get("discharge") or [], max_items=1)
        return [f"ADM symptoms={adm}", f"DC symptoms={dc}"]

    def render_demographics(v: Any) -> List[str]:
        if not isinstance(v, dict):
            t = norm_val(v)
            return [ln.rstrip() for ln in t.split("\n") if ln.strip()] if t else []
        # Support both legacy ("Sex"/"Age") and sgr_v4 safe keys ("sex"/"age").
        sex = _as_text(v.get("Sex") or v.get("sex") or "").strip() or "not stated"
        age = _as_text(v.get("Age") or v.get("age") or "").strip() or "not stated"
        return [f"Sex={sex}", f"Age={age}"]

    def render_objective_kv(v: Any, *, fields: List[Tuple[str, List[str]]]) -> List[str]:
        if not isinstance(v, dict):
            t = norm_val(v)
            return [ln.rstrip() for ln in t.split("\n") if ln.strip()] if t else []
        adm = v.get("admission") or {}
        dc = v.get("discharge") or {}
        if not isinstance(adm, dict):
            adm = {}
        if not isinstance(dc, dict):
            dc = {}

        def fmt_line(src: Dict[str, Any]) -> str:
            # Fixed-shape line: always include all canonical keys in order.
            # This makes Stage2 parsing significantly more stable on small models.
            parts: List[str] = []
            for display, candidates in fields:
                raw = ""
                for ck in candidates:
                    raw = _as_text(src.get(ck, "")).strip()
                    if raw:
                        break
                val = raw if raw and raw.casefold() != "not stated" else "not stated"
                parts.append(f"{display}={val}")
            return "; ".join(parts)

        adm_line = fmt_line(adm)
        dc_line = fmt_line(dc)
        return [f"ADM: {adm_line}", f"DC: {dc_line}"]

    out: List[str] = []
    for key in DOMAIN_KEYS:
        out.append(f"## {key}")
        raw_val = obj.get(key, "")
        if key == "PROBLEMS":
            lines = render_problems(raw_val)
            out.extend(lines if lines else ["not stated"])
        elif key == "SYMPTOMS":
            lines = render_symptoms(raw_val)
            out.extend(lines if lines else ["not stated"])
        elif key == "DEMOGRAPHICS":
            lines = render_demographics(raw_val)
            out.extend(lines if lines else ["not stated"])
        elif key == "VITALS":
            lines = render_objective_kv(
                raw_val,
                fields=[
                    ("Heart Rate", ["Heart Rate", "heart_rate"]),
                    ("Systolic BP", ["Systolic BP", "systolic_bp"]),
                    ("Diastolic BP", ["Diastolic BP", "diastolic_bp"]),
                    ("Respiratory Rate", ["Respiratory Rate", "respiratory_rate"]),
                    ("Temperature", ["Temperature", "temperature"]),
                    ("SpO2", ["SpO2", "spo2"]),
                    ("Weight", ["Weight", "weight"]),
                ],
            )
            out.extend(lines if lines else ["not stated"])
        elif key == "LABS":
            lines = render_objective_kv(
                raw_val,
                fields=[
                    ("Hemoglobin", ["Hemoglobin", "hemoglobin"]),
                    ("Hematocrit", ["Hematocrit", "hematocrit"]),
                    ("WBC", ["WBC", "wbc"]),
                    ("Platelet", ["Platelet", "platelet"]),
                    ("Sodium", ["Sodium", "sodium"]),
                    ("Potassium", ["Potassium", "potassium"]),
                    ("Creatinine", ["Creatinine", "creatinine"]),
                    ("BUN", ["BUN", "bun"]),
                    ("Glucose", ["Glucose", "glucose"]),
                    ("Bicarbonate", ["Bicarbonate", "bicarbonate"]),
                ],
            )
            out.extend(lines if lines else ["not stated"])
        else:
            val = norm_val(raw_val)
            if not val:
                out.append("not stated")
            else:
                out.extend([ln.rstrip() for ln in val.split("\n") if ln.strip()])
        out.append("")
    return "\n".join(out).strip() + "\n"


def _extract_domain_json(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    obj, json_text = extract_first_json_object(text or "")
    if isinstance(obj, dict):
        # Return the raw object (key presence is used as a stability gate).
        return obj, (json_text or "").strip()
    return None, (json_text or text or "").strip()


def _json_has_placeholders(obj: Any) -> bool:
    try:
        s = json.dumps(obj, ensure_ascii=False).casefold()
    except Exception:
        s = str(obj).casefold()
    return "___" in s


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


@dataclass
class Stage2Row:
    hadm_id: int
    stage1_json_ok: str
    stage2_lines: int
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    downstream_score_nogate: float


def _markdown_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    out = [header, sep]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def _filter_stage1_markdown_for_stage2(md: str, allowed_clusters: List[str]) -> str:
    """
    Keep only selected clusters from the Stage1 Markdown summary.
    Stage2 must never see the raw EHR note; this is still derived-only content.
    """
    want = {c.strip().upper() for c in allowed_clusters if c.strip()}
    out: List[str] = []
    cur: str | None = None
    for ln in (md or "").splitlines():
        if ln.startswith("## "):
            cur = ln[3:].strip().upper()
        if cur is not None and cur in want:
            out.append(ln.rstrip())
    # Ensure trailing newline.
    return "\n".join(out).strip() + "\n"


def _strip_not_stated_lines(md: str) -> str:
    """
    Stage2 must consume ONLY Stage1-derived Markdown, but we can remove non-informative
    filler lines that tend to cause format drift (e.g., models echoing 'not stated'
    as if it were a fact).
    """
    out: List[str] = []
    for ln in (md or "").splitlines():
        if ln.strip().casefold() == "not stated":
            continue
        out.append(ln.rstrip())
    return "\n".join(out).strip() + "\n"


def _compact_stage1_markdown(md: str) -> str:
    """Remove 'not stated' noise from Stage1 Markdown before feeding to Stage2.

    Goes beyond _strip_not_stated_lines by also stripping:
    - key=not stated pairs from semicolon-separated lines (VITALS/LABS)
    - single KV lines where value is "not stated"
    - empty cluster section headers left after stripping

    This reduces token waste and prevents truncation of late clusters
    (DISPOSITION, MEDICATIONS, PROCEDURES).
    """
    out: List[str] = []
    for ln in (md or "").splitlines():
        stripped = ln.strip()

        # 1) Bare "not stated" → drop
        if stripped.casefold() == "not stated":
            continue

        # 2) Section headers → keep (pruned later if empty)
        if stripped.startswith("## "):
            out.append(ln.rstrip())
            continue

        # 3) Semicolon-separated KV lines (VITALS/LABS: "ADM: key=val; key=val")
        if ";" in stripped and "=" in stripped:
            prefix = ""
            body = stripped
            # Extract ADM:/DC: prefix if present
            for pfx in ("ADM:", "DC:"):
                if body.upper().startswith(pfx):
                    prefix = body[:len(pfx)] + " "
                    body = body[len(pfx):].strip()
                    break
            pairs = [p.strip() for p in body.split(";")]
            kept = [p for p in pairs if not p.strip().casefold().endswith("=not stated")]
            if kept:
                out.append(prefix + "; ".join(kept))
            continue  # drop line entirely if all pairs were "not stated"

        # 4) Single KV line "Key=not stated" → drop
        if "=" in stripped and stripped.split("=", 1)[1].strip().casefold() == "not stated":
            continue

        # 5) Keep everything else
        out.append(ln.rstrip())

    # 6) Remove empty section headers (## CLUSTER with no content below)
    final: List[str] = []
    for i, ln in enumerate(out):
        if ln.strip().startswith("## "):
            # Look ahead: next non-empty line
            has_content = False
            for j in range(i + 1, len(out)):
                if out[j].strip().startswith("## "):
                    break
                if out[j].strip():
                    has_content = True
                    break
            if not has_content:
                continue  # drop empty section header
        final.append(ln)

    return "\n".join(final).strip() + "\n"


def _drop_hallucinated_negatives(lines: List[str], md_in: str) -> List[str]:
    """Drop MEDICATIONS/PROCEDURES facts with Value=no when that cluster is absent from Stage1 input.

    The LoRA model sometimes fabricates negative facts (e.g. Surgery|no, Dialysis|no)
    even when the PROCEDURES section was entirely stripped by compaction.
    This post-filter enforces the 'absence is NOT evidence' rule deterministically.
    """
    guarded_clusters = {"MEDICATIONS", "PROCEDURES"}
    # Determine which guarded clusters have a non-empty section in the compacted markdown.
    present: set[str] = set()
    for ln in (md_in or "").splitlines():
        stripped = ln.strip()
        if stripped.startswith("## "):
            cluster_name = stripped[3:].strip().upper()
            if cluster_name in guarded_clusters:
                present.add(cluster_name)

    out: List[str] = []
    dropped = 0
    for ln in lines:
        parts = ln.split("|")
        if len(parts) == 4:
            cluster_u = parts[0].strip().upper()
            keyword = parts[1].strip()
            value_cf = parts[2].strip().casefold()
            if cluster_u in guarded_clusters and cluster_u not in present and value_cf == "no":
                # Keep PROCEDURES|Any Procedure|no as a weak fallback signal.
                if not (cluster_u == "PROCEDURES" and keyword == "Any Procedure"):
                    dropped += 1
                    continue
        out.append(ln)

    if dropped:
        print(f"    [post-filter] dropped {dropped} hallucinated negative(s) from absent cluster(s)", flush=True)
    return out


def _parse_stage1_procedures_from_markdown(md_in: str) -> Dict[str, str]:
    in_section = False
    out: Dict[str, str] = {}
    for raw in (md_in or "").splitlines():
        ln = raw.strip()
        if ln.startswith("## "):
            in_section = (ln[3:].strip().upper() == "PROCEDURES")
            continue
        if not in_section or not ln or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        key = k.strip()
        if key in {"Any Procedure", "Surgery", "Dialysis", "Mechanical Ventilation"}:
            out[key] = v.strip()
    return out


def _has_surgery_text_evidence(md_in: str) -> bool:
    text = (md_in or "")
    if not text.strip():
        return False
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        # Explicit structured negatives are not evidence.
        if _SURGERY_KV_NEG_RE.match(ln):
            continue
        # Generic headers are not evidence by themselves.
        if _GENERIC_SURGICAL_HEADER_RE.match(ln):
            continue
        if _NEGATED_SURGERY_RE.search(ln):
            continue
        if _SURGERY_KV_POS_RE.match(ln):
            return True
        if re.search(r"\bs/p\b", ln, flags=re.IGNORECASE):
            return True
        if re.search(
            r"\b(cholecystectomy|appendectomy|hysterectomy|salpingectomy|arthroplasty|cabg|craniotomy|laparoscopic|hernia repair|resection)\b",
            ln,
            flags=re.IGNORECASE,
        ):
            return True
        if _SURGERY_RE.search(ln):
            return True
    return False


def _has_procedure_text_evidence(md_in: str) -> bool:
    """Any positive procedure evidence in Stage1 markdown (surgery/dialysis/ventilation)."""
    if _has_surgery_text_evidence(md_in):
        return True
    for ln in _markdown_section_lines(md_in, "PROCEDURES"):
        s = (ln or "").strip()
        if not s:
            continue
        if "=" in s:
            _k, _v = s.split("=", 1)
            vv = _v.strip().casefold()
            if vv in _TEXT_PLACEHOLDERS or vv in {"no", "0", "0.0", "false"}:
                continue
        if _DIALYSIS_RE.search(s) or _VENT_RE.search(s):
            return True
    return False


def _inject_stage1_procedure_fallback(lines: List[str], md_in: str) -> List[str]:
    """Inject conservative PROCEDURES fallback facts from Stage1 markdown evidence."""
    stage1_proc = _parse_stage1_procedures_from_markdown(md_in)
    has_surgery_evidence = _has_surgery_text_evidence(md_in)
    has_procedure_evidence = _has_procedure_text_evidence(md_in)
    has_procedure_section = "## PROCEDURES" in (md_in or "").upper()

    parsed: List[Tuple[str, str, str, str]] = []
    other: List[Tuple[str, str, str, str]] = []
    for ln in lines:
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) != 4:
            continue
        c, k, v, t = parts
        if c.upper() == "PROCEDURES":
            parsed.append((c, k, v, t))
        else:
            other.append((c, k, v, t))

    if not parsed and not stage1_proc and not has_procedure_evidence:
        # If Stage1 kept the PROCEDURES section but no typed keys survived compaction,
        # emit a conservative fallback signal for Any Procedure.
        if has_procedure_section and _env_truthy("MEDGEMMA_STAGE1_ANY_PROCEDURE_DEFAULT_NO", "1"):
            return [*lines, "PROCEDURES|Any Procedure|no|Admission"]
        return lines

    by_kw: Dict[str, Tuple[str, str, str, str]] = {k: item for item in parsed for k in [item[1]]}

    any_v = _normalize_binary_text_value(stage1_proc.get("Any Procedure", "")).casefold()
    surg_v = _normalize_binary_text_value(stage1_proc.get("Surgery", "")).casefold()
    dial_raw = (stage1_proc.get("Dialysis", "") or "").strip().casefold()
    vent_raw = (stage1_proc.get("Mechanical Ventilation", "") or "").strip().casefold()
    dial_pos = dial_raw in {"started", "done", "decided", "cancelled", "yes", "1", "1.0", "true", "y"}
    vent_pos = bool(vent_raw) and vent_raw not in _TEXT_PLACEHOLDERS and vent_raw not in {"no", "0", "0.0", "false", "n"}
    has_typed_positive = surg_v == "yes" or dial_pos or vent_pos or has_procedure_evidence

    # Upcast only when Surgery is unknown; do not override explicit no.
    if has_surgery_evidence and surg_v not in {"yes", "no"}:
        surg_v = "yes"
        has_typed_positive = True

    # Keep explicit Any Procedure only when Stage1 explicitly provided it.
    if any_v in {"yes", "no"}:
        # Guard inconsistent Any Procedure=yes when Stage1 has no positive evidence.
        if any_v == "yes" and not has_typed_positive:
            any_v = "no"
        by_kw["Any Procedure"] = ("PROCEDURES", "Any Procedure", any_v, "Admission")
    else:
        if not has_typed_positive and _env_truthy("MEDGEMMA_STAGE1_ANY_PROCEDURE_DEFAULT_NO", "1"):
            by_kw["Any Procedure"] = ("PROCEDURES", "Any Procedure", "no", "Admission")
        elif by_kw.get("Any Procedure", ("", "", "", ""))[2].strip().casefold() == "yes" and not has_typed_positive:
            # Unsupported generic positive from Stage2 (without Stage1 evidence)
            # tends to be a high-FP pattern on curated cohorts.
            del by_kw["Any Procedure"]

    if surg_v == "yes":
        by_kw["Surgery"] = ("PROCEDURES", "Surgery", "yes", "Past")
    elif surg_v == "no" and not has_surgery_evidence and by_kw.get("Surgery", ("", "", "", ""))[2].strip().casefold() == "yes":
        # Drop Stage2 Surgery=yes if Stage1 explicitly says no and no evidence was found.
        del by_kw["Surgery"]

    # Rebuild lines in stable order.
    proc_order = ["Any Procedure", "Surgery", "Dialysis", "Mechanical Ventilation"]
    out: List[str] = [f"{c}|{k}|{v}|{t}" for (c, k, v, t) in other]
    injected = 0
    for k in proc_order:
        item = by_kw.get(k)
        if item is None:
            continue
        out.append(f"{item[0]}|{item[1]}|{item[2]}|{item[3]}")
        if not any(p[1] == k for p in parsed):
            injected += 1

    if injected:
        print(f"    [post-filter] injected {injected} PROCEDURES fallback fact(s) from Stage1 evidence", flush=True)
    return out


def _drop_low_information_negatives(lines: List[str]) -> List[str]:
    """Drop low-information negative facts that tend to inflate FP without adding risk signal.

    Rationale:
    - Binary negative flags (e.g. MEDICATIONS|Opioid Therapy|no) are usually neutral in scoring.
    - PROCEDURES specific negatives (Surgery/Dialysis/Mechanical Ventilation=no) are often emitted
      as defaults and create GT mismatches.
    - New Medications Count=0 is similarly low-information for downstream risk scoring.
    """

    med_binary = {
        "Polypharmacy",
        "Anticoagulation",
        "Insulin Therapy",
        "Opioid Therapy",
        "Diuretic Therapy",
    }
    proc_specific = {"Surgery", "Dialysis", "Mechanical Ventilation"}

    out: List[str] = []
    dropped = 0
    for ln in lines:
        parts = ln.split("|")
        if len(parts) != 4:
            out.append(ln)
            continue
        cluster, keyword, value, timestamp = [p.strip() for p in parts]
        value_cf = value.casefold()

        if cluster.upper() == "MEDICATIONS":
            if keyword in med_binary and value_cf == "no":
                dropped += 1
                continue
            if keyword == "New Medications Count":
                n = _extract_numeric_value(value)
                if n is not None:
                    try:
                        if float(n) == 0.0:
                            dropped += 1
                            continue
                    except Exception:
                        pass

        if cluster.upper() == "PROCEDURES" and keyword in proc_specific and value_cf == "no":
            dropped += 1
            continue

        out.append(f"{cluster}|{keyword}|{value}|{timestamp}")

    if dropped:
        print(f"    [post-filter] dropped {dropped} low-information negative fact(s)", flush=True)
    return out


def _semantic_key_norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (s or "").casefold())).strip()


def _is_fragmented_symptom_keyword(keyword: str) -> bool:
    nk = _semantic_key_norm(keyword)
    if not nk:
        return True
    if nk in _SYMPTOM_FRAGMENT_STOP:
        return True
    if nk.startswith("and ") or nk.startswith("or "):
        return True
    toks = nk.split()
    if len(toks) <= 2 and all(t in _ANATOMIC_TOKENS or t in _DIR_OR_JOIN_TOKENS for t in toks):
        return True
    return False


def _semantic_postprocess_gate(lines: List[str]) -> List[str]:
    """Reduce fragmented semantic lines and duplicate semantic keywords.

    Rules:
    - Drop fragmented SYMPTOMS keywords like "and arm"/"face".
    - Dedupe SYMPTOMS by normalized keyword, prefer Admission.
    - Dedupe PROBLEMS by normalized keyword, prefer chronic/Past.
    """

    non_semantic: List[str] = []
    problems: Dict[str, List[Tuple[str, str, str, str]]] = {}
    symptoms: Dict[str, List[Tuple[str, str, str, str]]] = {}
    passthrough_semantic: List[str] = []
    dropped_fragments = 0
    dedup_problems = 0
    dedup_symptoms = 0

    for ln in lines:
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) != 4:
            continue
        cluster, keyword, value, timestamp = parts
        cu = cluster.upper()
        if cu == "PROBLEMS":
            problems.setdefault(_semantic_key_norm(keyword), []).append((cluster, keyword, value, timestamp))
            continue
        if cu == "SYMPTOMS":
            if _is_fragmented_symptom_keyword(keyword):
                dropped_fragments += 1
                continue
            symptoms.setdefault(_semantic_key_norm(keyword), []).append((cluster, keyword, value, timestamp))
            continue
        if cu in {"MEDICATIONS", "PROCEDURES"}:
            passthrough_semantic.append(f"{cluster}|{keyword}|{value}|{timestamp}")
            continue
        non_semantic.append(f"{cluster}|{keyword}|{value}|{timestamp}")

    out: List[str] = list(non_semantic)

    for _nk, items in problems.items():
        if len(items) == 1:
            c, k, v, t = items[0]
            out.append(f"{c}|{k}|{v}|{t}")
            continue
        dedup_problems += len(items) - 1
        pick = None
        for it in items:
            _c, _k, v, t = it
            if v.strip().casefold() == "chronic" or t.strip() == "Past":
                pick = it
                break
        if pick is None:
            pick = items[0]
        c, k, v, t = pick
        out.append(f"{c}|{k}|{v}|{t}")

    for _nk, items in symptoms.items():
        if len(items) == 1:
            c, k, v, t = items[0]
            out.append(f"{c}|{k}|{v}|{t}")
            continue
        dedup_symptoms += len(items) - 1
        pick = None
        for it in items:
            _c, _k, _v, t = it
            if t.strip() == "Admission":
                pick = it
                break
        if pick is None:
            pick = items[0]
        c, k, v, t = pick
        out.append(f"{c}|{k}|{v}|{t}")

    out.extend(passthrough_semantic)
    out.sort(key=lambda s: (s.split("|", 1)[0], s.split("|", 2)[1]))
    if dropped_fragments or dedup_problems or dedup_symptoms:
        print(
            "    [post-filter] semantic gate "
            f"(dropped_symptom_fragments={dropped_fragments}, "
            f"dedup_problems={dedup_problems}, dedup_symptoms={dedup_symptoms})",
            flush=True,
        )
    return out


def _normalize_binary_flag_values(lines: List[str]) -> List[str]:
    """Normalize binary categorical flags to canonical yes/no values.

    Stage2 sometimes emits 0/1 for categorical flags. Risk engine expects yes/no.
    """
    out: List[str] = []
    changed = 0
    for ln in lines:
        parts = ln.split("|")
        if len(parts) != 4:
            out.append(ln)
            continue
        cluster, keyword, value, timestamp = [p.strip() for p in parts]
        if keyword in _BINARY_FLAG_KEYWORDS:
            value_cf = value.casefold()
            if keyword == "Dialysis":
                if value_cf in _BINARY_TRUE_VALUES:
                    if value_cf != "done":
                        changed += 1
                    value = "done"
                elif value_cf in _BINARY_FALSE_VALUES:
                    if value_cf != "no":
                        changed += 1
                    value = "no"
            else:
                if value_cf in _BINARY_TRUE_VALUES:
                    if value_cf != "yes":
                        changed += 1
                    value = "yes"
                elif value_cf in _BINARY_FALSE_VALUES:
                    if value_cf != "no":
                        changed += 1
                    value = "no"
        out.append(f"{cluster}|{keyword}|{value}|{timestamp}")

    if changed:
        print(f"    [post-filter] normalized {changed} binary flag value(s) to yes/no", flush=True)
    return out


def _markdown_section_lines(md_in: str, section_name: str) -> List[str]:
    target = (section_name or "").strip().upper()
    current = ""
    lines: List[str] = []
    for raw in (md_in or "").splitlines():
        stripped = raw.strip()
        if stripped.startswith("## "):
            current = stripped[3:].strip().upper()
            continue
        if current == target and stripped:
            lines.append(stripped)
    return lines


def _utilization_has_stage1_evidence(md_in: str) -> bool:
    section = _markdown_section_lines(md_in, "UTILIZATION")
    if not section:
        return False

    for ln in section:
        for token in ln.split(";"):
            t = token.strip()
            if not t:
                continue
            value = t.split("=", 1)[1].strip() if "=" in t else t
            if value.casefold() not in _UTILIZATION_PLACEHOLDERS:
                return True
    return False


def _drop_utilization_without_stage1_evidence(lines: List[str], md_in: str) -> List[str]:
    """Gate UTILIZATION facts when Stage1 provides no utilization evidence."""
    if _utilization_has_stage1_evidence(md_in):
        return lines

    out: List[str] = []
    dropped = 0
    for ln in lines:
        parts = ln.split("|")
        if len(parts) == 4 and parts[0].strip().upper() == "UTILIZATION":
            dropped += 1
            continue
        out.append(ln)
    if dropped:
        print(f"    [post-filter] dropped {dropped} UTILIZATION fact(s) without Stage1 evidence", flush=True)
    return out


def run_stage1(
    *,
    cohort_root: Path,
    out_dir: Path,
    hadm_ids: List[int],
    url: str,
    model: str,
    schema_path: Path,
    stage1_profile: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    overwrite_stage1: bool,
    debug: bool,
) -> None:
    client = OpenAICompatibleChatClient(url=url, model=model, debug=debug)
    client.assert_model_available()
    response_format = _load_schema_response_format(schema_path)
    baseline_fallback_enabled = (stage1_profile or "").strip().casefold().startswith("sgr_") and _env_truthy(
        "STAGE1_SGR_BASELINE_FALLBACK",
        "1",
    )
    baseline_response_format: Optional[Dict[str, Any]] = None
    if baseline_fallback_enabled:
        baseline_response_format = _load_schema_response_format(Path("schemas/readmission_domain_summary.schema.json"))

    def _has_required_top_keys(o: Any) -> bool:
        return isinstance(o, dict) and all(k in o for k in DOMAIN_KEYS)

    # NOTE: This script is sometimes invoked in a per-document loop with a single HADM id
    # (e.g. by sequential orchestration). Avoid shrinking an existing run scope by
    # overwriting <out_dir>/hadm_ids.json with a 1-element list.
    hadm_ids_path = out_dir / "hadm_ids.json"
    existing_hadm_ids: List[int] = []
    if hadm_ids_path.exists():
        try:
            loaded = json.loads(hadm_ids_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list) and all(isinstance(x, int) for x in loaded):
                existing_hadm_ids = loaded
        except Exception:
            existing_hadm_ids = []

    if not (
        existing_hadm_ids
        and len(existing_hadm_ids) > len(hadm_ids)
        and set(hadm_ids).issubset(set(existing_hadm_ids))
    ):
        hadm_ids_path.write_text(json.dumps(hadm_ids, indent=2), encoding="utf-8")
    meta = {
        "stage": "stage1",
        "cohort_root": str(cohort_root),
        "url": url,
        "model": model,
        "stage1_profile": str(stage1_profile),
        "schema_path": str(schema_path),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "meta_stage1.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for i, hadm in enumerate(hadm_ids, 1):
        doc_dir = cohort_root / str(hadm)
        ehr_path = doc_dir / f"ehr_{hadm}.txt"
        ehr_text = ehr_path.read_text(encoding="utf-8", errors="ignore")
        # Avoid leaking de-identification placeholders into Stage1 JSON (common drift signal).
        # This keeps the pipeline stable without changing clinical evidence.
        ehr_text = ehr_text.replace("___", "not stated")
        ehr_chars_original = len(ehr_text)

        per_dir = out_dir / str(hadm)
        raw_path = per_dir / "stage1_raw.txt"
        raw_try0_path = per_dir / "stage1_raw_try0.txt"
        raw_model_path = per_dir / "stage1_raw_model.txt"
        json_path = per_dir / "stage1.json"
        norm_json_path = per_dir / "stage1_normalized.json"
        md_path = per_dir / "stage1.md"

        if not overwrite_stage1 and json_path.exists() and norm_json_path.exists() and md_path.exists():
            print(f"[{i}/{len(hadm_ids)}] HADM {hadm} | skip (exists)", flush=True)
            continue

        trim_enabled = _env_truthy("MEDGEMMA_TRIM_INPUT", "0")
        try:
            max_chars = int(os.getenv("MEDGEMMA_MAX_TEXT_CHARS", "6000"))
        except Exception:
            max_chars = 6000
        trim_strategy = os.getenv("MEDGEMMA_TRIM_STRATEGY", "middle").strip() or "middle"
        fallback_order = [s.strip() for s in os.getenv("MEDGEMMA_TRIM_FALLBACK_ORDER", "middle,keyword_window,head_tail").split(",") if s.strip()]

        objective_appendix_enabled = _env_truthy("MEDGEMMA_STAGE1_OBJECTIVE_APPENDIX", "0") or _env_truthy(
            "OBJECTIVE_APPENDIX",
            "0",
        )
        objective_vitals_lines: List[str] = []
        objective_labs_lines: List[str] = []
        if objective_appendix_enabled:
            objective_vitals_lines, objective_labs_lines = _extract_objective_lines(ehr_text)
            objective_appendix = _objective_lines_to_appendix(objective_vitals_lines, objective_labs_lines)
        else:
            objective_appendix = ""

        def _mk_prompt(note_text: str) -> str:
            base = "EHR NOTE:\n" + (note_text or "").strip()
            if objective_appendix:
                base = base + "\n\n" + objective_appendix.strip()
            return base + "\n\nBegin Stage 1 now."

        _RF_OMIT = object()

        def _call_stage1(base_prompt: str, extra_suffix: str = "", rf_override: Any = None) -> tuple[str, Dict[str, Any]]:
            up = base_prompt + (extra_suffix or "")
            if rf_override is _RF_OMIT:
                rf = None
            elif rf_override is None:
                rf = response_format
            else:
                rf = rf_override
            rr = client.chat(
                user_prompt=up,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=rf,
            )
            return strip_medgemma_internal_tokens(rr.text), dict(rr.meta or {})

        def _call_stage1_baseline(base_prompt: str, extra_suffix: str = "") -> tuple[str, Dict[str, Any]]:
            up = base_prompt + (extra_suffix or "")
            rr = client.chat(
                user_prompt=up,
                system_prompt=READMISSION_DOMAIN_JSON_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=baseline_response_format,
            )
            return strip_medgemma_internal_tokens(rr.text), dict(rr.meta or {})

        strategies: List[str] = []
        if ehr_chars_original <= max_chars:
            strategies = ["full"]
        else:
            # If trimming is enabled, prefer trimmed-first; otherwise try full-first.
            first = [trim_strategy] if trim_enabled else ["full", trim_strategy]
            for s in first + fallback_order:
                if s and s not in strategies:
                    strategies.append(s)
            if not trim_enabled and "full" not in strategies:
                strategies.insert(0, "full")

        base_prompt = ""
        raw_str = ""
        raw_client_meta: Dict[str, Any] = {}
        prompt_variant = ""
        prompt_errors: List[Dict[str, str]] = []

        for strat in strategies:
            note_used = ehr_text if strat == "full" else _trim_text(ehr_text, max_chars=max_chars, strategy=strat)
            base_prompt = _mk_prompt(note_used)
            prompt_variant = strat
            try:
                raw_str, raw_client_meta = _call_stage1(base_prompt)
                break
            except Exception as e:  # noqa: BLE001
                prompt_errors.append({"variant": strat, "error": str(e)})
                raw_str = ""
                raw_client_meta = {}
                base_prompt = ""

        if not raw_str:
            per_dir.mkdir(parents=True, exist_ok=True)
            err_obj = {
                "hadm_id": int(hadm),
                "stage": "stage1",
                "error": "all_stage1_prompt_variants_failed",
                "prompt_variants_tried": prompt_errors,
                "trim": {
                    "enabled": bool(trim_enabled),
                    "max_chars": int(max_chars),
                    "strategy": str(trim_strategy),
                    "fallback_order": fallback_order,
                    "ehr_chars_original": int(ehr_chars_original),
                },
            }
            (per_dir / "stage1_error.json").write_text(json.dumps(err_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{i}/{len(hadm_ids)}] HADM {hadm} | stage1_error=all_variants_failed", flush=True)
            continue

        obj, _extracted_json_text = _extract_domain_json(raw_str)
        parse_ok = _has_required_top_keys(obj)

        # Hygiene retry: even if JSON parses, some models emit placeholders like "___".
        # This breaks strict gates and should be fixed at the source (Stage1 output).
        did_hygiene_retry = False
        if parse_ok and "___" in raw_str:
            did_hygiene_retry = True
            _write_text(raw_try0_path, raw_str)
            hygiene_suffix = (
                "\n\nHYGIENE FIX (MANDATORY):\n"
                "- Output ONLY the JSON object.\n"
                "- NEVER output placeholders like ___ or redaction tokens like [** ... **].\n"
                "- For any unknown value, write exactly: not stated.\n"
                "- Keep the same 9 top-level keys.\n"
                "- Ensure JSON strings do not contain raw newlines (use \\n inside strings if needed).\n"
            )
            try:
                raw_h_str, meta_h = _call_stage1(base_prompt, hygiene_suffix)
                obj_h, _ = _extract_domain_json(raw_h_str)
                if _has_required_top_keys(obj_h) and "___" not in raw_h_str:
                    raw_str = raw_h_str
                    obj = obj_h
                    parse_ok = True
                    raw_client_meta = meta_h
            except Exception:
                pass

        did_retry = False
        if not parse_ok:
            did_retry = True
            # When Stage1 schema is complex, some OpenAI-compatible backends accept the schema
            # but yield empty/truncated outputs. Try one pass with lighter `json_object` and one
            # pass with no response_format at all to maximize portability.
            rf_chain: List[Any] = [None, {"type": "json_object"}, _RF_OMIT]

            retry_suffix = (
                "\n\nCOMPACT MODE (MANDATORY):\n"
                "- Output ONLY the JSON object.\n"
                "- For VITALS: ONLY these keys: Heart Rate, Systolic BP, Diastolic BP, Respiratory Rate, Temperature, SpO2, Weight.\n"
                "- For LABS: ONLY these keys: WBC, Hemoglobin, Hematocrit, Platelet, Sodium, Potassium, Creatinine, BUN, Glucose, Bicarbonate.\n"
                "- For any missing value, write exactly: not stated.\n"
                "- Do NOT include any other tests (no urine studies, tox, CMP extras, etc.).\n"
                "- Do NOT use placeholders like ___.\n"
                "- Ensure the JSON is complete and valid (close quotes/braces).\n"
            )
            raw2_str: str | None = None
            meta2: Dict[str, Any] | None = None
            for rf in rf_chain:
                try:
                    raw2_str, meta2 = _call_stage1(base_prompt, retry_suffix, rf_override=rf)
                    _write_text(per_dir / "stage1_raw_retry1.txt", raw2_str)
                    obj, _extracted_json_text = _extract_domain_json(raw2_str)
                    parse_ok = _has_required_top_keys(obj)
                    if parse_ok:
                        raw_str = raw2_str
                        raw_client_meta = meta2 or {}
                        break
                except Exception:
                    continue

            # If the note is long and we still failed to parse, retry compact mode on a trimmed slice.
            if not parse_ok and ehr_chars_original > max_chars:
                alt_text = _trim_text(ehr_text, max_chars=max_chars, strategy=trim_strategy)
                alt_prompt = _mk_prompt(alt_text)
                if alt_prompt and alt_prompt != base_prompt:
                    for rf in rf_chain:
                        try:
                            raw2_str, meta2 = _call_stage1(alt_prompt, retry_suffix, rf_override=rf)
                            _write_text(per_dir / "stage1_raw_retry1_trimmed.txt", raw2_str)
                            obj, _extracted_json_text = _extract_domain_json(raw2_str)
                            parse_ok = _has_required_top_keys(obj)
                            if parse_ok:
                                raw_str = raw2_str
                                raw_client_meta = meta2 or {}
                                base_prompt = alt_prompt
                                prompt_variant = f"retry_trim:{trim_strategy}"
                                break
                        except Exception:
                            continue

        did_placeholder_retry = False
        if "___" in raw_str:
            did_placeholder_retry = True
            placeholder_suffix = (
                "\n\nPLACEHOLDER BAN (MANDATORY):\n"
                "- Output ONLY the JSON object.\n"
                '- Replace every "___" with exactly: not stated.\n'
                "- Do NOT invent values.\n"
                "- Ensure the JSON is complete and valid.\n"
            )
            try:
                raw_p_str, meta_p = _call_stage1(base_prompt, placeholder_suffix)
                _write_text(per_dir / "stage1_raw_retry_placeholders.txt", raw_p_str)
                obj_p, _ = _extract_domain_json(raw_p_str)
                if _has_required_top_keys(obj_p) and "___" not in raw_p_str:
                    raw_str = raw_p_str
                    obj = obj_p
                    parse_ok = True
                    raw_client_meta = meta_p
            except Exception:
                pass

        fallback_used = False
        fallback_error: str | None = None
        if not parse_ok and baseline_fallback_enabled and base_prompt:
            try:
                raw_fb_str, meta_fb = _call_stage1_baseline(base_prompt)
                obj_fb, _ = _extract_domain_json(raw_fb_str)
                if _has_required_top_keys(obj_fb):
                    raw_str = raw_fb_str
                    obj = obj_fb
                    parse_ok = True
                    raw_client_meta = meta_fb
                    fallback_used = True
                    prompt_variant = "fallback:strings_v1"
            except Exception as e:  # noqa: BLE001
                fallback_error = str(e)

        raw_model_str = raw_str
        raw_sanitized_str = raw_model_str.replace("___", "not stated")
        _write_text(raw_model_path, raw_model_str)
        _write_text(raw_path, raw_sanitized_str)
        extracted_obj: Dict[str, Any]
        if obj is None:
            extracted_obj = {k: "" for k in DOMAIN_KEYS}
        else:
            extracted_obj = {k: obj.get(k, "") for k in DOMAIN_KEYS}

        # sgr_v2_strict_cascade: fill missing objective values from extracted evidence lines (deterministic fallback).
        if stage1_profile == "sgr_v2_strict_cascade" and (objective_vitals_lines or objective_labs_lines):
            _fill_stage1_strict_cascade_objective_inplace(
                extracted_obj,
                vitals_lines=objective_vitals_lines,
                labs_lines=objective_labs_lines,
            )

        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(extracted_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        normalized_obj: Dict[str, Any] = {k: _sanitize_stage1_value(extracted_obj.get(k, "")) for k in DOMAIN_KEYS}
        if isinstance(normalized_obj.get("DEMOGRAPHICS"), dict):
            normalized_obj["DEMOGRAPHICS"] = _sanitize_demographics_obj(normalized_obj.get("DEMOGRAPHICS"))
        else:
            normalized_obj["DEMOGRAPHICS"] = _sanitize_demographics_text(_as_text(normalized_obj.get("DEMOGRAPHICS", "")))
        if isinstance(normalized_obj.get("VITALS"), dict):
            normalized_obj["VITALS"] = _sanitize_objective_obj(normalized_obj.get("VITALS"), kind="vitals")
        else:
            normalized_obj["VITALS"] = _sanitize_vitals_text(_as_text(normalized_obj.get("VITALS", "")))
        if isinstance(normalized_obj.get("LABS"), dict):
            normalized_obj["LABS"] = _sanitize_objective_obj(normalized_obj.get("LABS"), kind="labs")
        else:
            normalized_obj["LABS"] = _sanitize_labs_text(_as_text(normalized_obj.get("LABS", "")))
        normalized_obj["MEDICATIONS"] = _sanitize_medications_text(_as_text(normalized_obj.get("MEDICATIONS", "")))
        normalized_obj["PROCEDURES"] = _sanitize_procedures_text(
            _as_text(normalized_obj.get("PROCEDURES", "")),
            note_text=ehr_text,
        )
        normalized_obj["DISPOSITION"] = _sanitize_disposition_text(_as_text(normalized_obj.get("DISPOSITION", "")))

        hygiene_stats = {
            "raw_model_had_placeholders": bool("___" in raw_model_str),
            "raw_sanitized_had_placeholders": bool("___" in raw_sanitized_str),
            "json_had_placeholders": bool(_json_has_placeholders(extracted_obj)),
            "normalized_had_placeholders": bool(_json_has_placeholders(normalized_obj)),
        }
        norm_json_path.write_text(
            json.dumps(
                {"normalized": normalized_obj, "hygiene_stats": hygiene_stats, "openai_compat": raw_client_meta},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        md = _domain_json_to_markdown(normalized_obj)
        _write_text(md_path, md)
        stage1_facts_path = per_dir / "stage1_facts.txt"
        stage1_facts_lines = _stage1_objective_to_kvt4_lines(normalized_obj)
        _write_text(stage1_facts_path, "\n".join(stage1_facts_lines) + ("\n" if stage1_facts_lines else ""))

        stage1_meta = {
            "hadm_id": hadm,
            "json_parse_ok": bool(parse_ok),
            "missing_keys": [k for k in DOMAIN_KEYS if not (isinstance(obj, dict) and k in obj)],
            "md_chars": len(md),
            "did_retry": bool(did_retry),
            "did_hygiene_retry": bool(did_hygiene_retry),
            "did_placeholder_retry": bool(did_placeholder_retry),
            "prompt_variant": str(prompt_variant),
            "ehr_chars_original": int(ehr_chars_original),
            "ehr_max_chars": int(max_chars),
            "trim_enabled": bool(trim_enabled),
            "trim_strategy": str(trim_strategy),
            "trim_variants_tried": [d.get("variant") for d in prompt_errors if isinstance(d, dict)],
            "fallback_to_baseline_used": bool(fallback_used),
            "fallback_to_baseline_error": fallback_error,
            "raw_model_chars": len(raw_model_str),
            "raw_sanitized_chars": len(raw_sanitized_str),
            "had_placeholders_model": bool("___" in raw_model_str),
            "had_placeholders_sanitized": bool("___" in raw_sanitized_str),
            "openai_compat": raw_client_meta,
        }
        (per_dir / "stage1_meta.json").write_text(json.dumps(stage1_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        ok = "yes" if stage1_meta["json_parse_ok"] and not stage1_meta["missing_keys"] else "no"
        print(f"[{i}/{len(hadm_ids)}] HADM {hadm} | json_ok={ok} | md_chars={len(md)}", flush=True)


def run_stage2(
    *,
    cohort_root: Path,
    out_dir: Path,
    hadm_ids: List[int],
    url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    repetition_penalty: Optional[float],
    top_p: Optional[float],
    min_p: Optional[float],
    typical_p: Optional[float],
    stop: Optional[List[str]],
    require_timestamp_match: bool,
    semantic_keyword_only_match: bool,
    output_mode: str,
    scope: str,
    overwrite_stage2: bool,
    debug: bool,
) -> None:
    client = OpenAICompatibleChatClient(url=url, model=model, debug=debug)
    client.assert_model_available()

    scope_l = (scope or "").strip().casefold()
    mode_l = (output_mode or "").strip().casefold()

    # Keep baseline objective-mode behavior unchanged by default.
    effective_rep = repetition_penalty
    if effective_rep is None and scope_l == "all":
        effective_rep = 1.10
    # Allow env override for repetition penalty (useful for raw completion mode).
    rep_env = (os.getenv("MEDGEMMA_STAGE2_REPETITION_PENALTY") or "").strip()
    if rep_env:
        try:
            effective_rep = float(rep_env)
        except ValueError:
            pass

    effective_stop = _parse_stop_list(stop)
    if effective_stop is None and mode_l == "lines":
        # Stop on a *line break* before END to avoid truncating immediately if the model begins with "END".
        effective_stop = ["\nEND"]
    # Note: triple-newline stop was tested but caused premature termination
    # when the model naturally uses blank lines between cluster sections.

    stage2_profile = _stage2_profile_name()
    stage2_behavior = {
        "stage2_profile": stage2_profile,
        "training_match_prompt": _env_truthy_stage2(
            "MEDGEMMA_STAGE2_TRAINING_MATCH_PROMPT",
            validated_default="0",
            experimental_default="1",
        ),
        "recover_3part_lines": _env_truthy_stage2(
            "MEDGEMMA_STAGE2_RECOVER_3PART_LINES",
            validated_default="0",
            experimental_default="1",
        ),
        "reclassify_nonnumeric_clusters": _env_truthy_stage2(
            "MEDGEMMA_STAGE2_RECLASSIFY_NONNUMERIC_CLUSTERS",
            validated_default="0",
            experimental_default="1",
        ),
        "expand_semantic_lines": _env_truthy_stage2(
            "MEDGEMMA_EXPAND_SEMANTIC_LINES",
            validated_default="0",
            experimental_default="1",
        ),
        "objective_ts_canonical_all": _env_truthy_stage2(
            "MEDGEMMA_OBJECTIVE_TS_CANONICAL_ALL",
            validated_default="0",
            experimental_default="1",
        ),
        "consecutive_dedup": _env_truthy_stage2(
            "MEDGEMMA_STAGE2_CONSECUTIVE_DEDUP",
            validated_default="0",
            experimental_default="1",
        ),
        "stage1_procedure_fallback": _env_truthy_stage2(
            "MEDGEMMA_STAGE1_PROCEDURE_FALLBACK",
            validated_default="0",
            experimental_default="1",
        ),
        "drop_low_info_negatives": _env_truthy_stage2(
            "MEDGEMMA_DROP_LOW_INFO_NEGATIVES",
            validated_default="0",
            experimental_default="1",
        ),
        "stage2_semantic_gate": _env_truthy_stage2(
            "MEDGEMMA_STAGE2_SEMANTIC_GATE",
            validated_default="0",
            experimental_default="1",
        ),
        "utilization_evidence_gate": _env_truthy_stage2(
            "MEDGEMMA_UTILIZATION_EVIDENCE_GATE",
            validated_default="0",
            experimental_default="1",
        ),
    }
    use_training_match_prompt = bool(stage2_behavior["training_match_prompt"])
    use_consecutive_dedup = bool(stage2_behavior["consecutive_dedup"])
    use_stage1_procedure_fallback = bool(stage2_behavior["stage1_procedure_fallback"])
    use_drop_low_info_negatives = bool(stage2_behavior["drop_low_info_negatives"])
    use_semantic_gate = bool(stage2_behavior["stage2_semantic_gate"])
    use_utilization_evidence_gate = bool(stage2_behavior["utilization_evidence_gate"])

    prompt_template_id, prompt_template = _stage2_prompt_template_for_run(
        mode_l=mode_l,
        scope_l=scope_l,
        use_training_match_prompt=use_training_match_prompt,
    )
    prompt_prefix = _stage2_prompt_prefix(prompt_template)
    prompt_prefix_sha256 = _sha256_hex_utf8(prompt_prefix)
    print(
        f"[stage2:cag] prompt_template={prompt_template_id} | prefix_sha256={prompt_prefix_sha256} | prefix_chars={len(prompt_prefix)}",
        flush=True,
    )

    generation_params: Dict[str, Any] = {
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "repetition_penalty": float(effective_rep) if effective_rep is not None else None,
        "top_p": float(top_p) if top_p is not None else None,
        "min_p": float(min_p) if min_p is not None else None,
        "typical_p": float(typical_p) if typical_p is not None else None,
        "stop": effective_stop,
        "output_mode": str(output_mode),
        "scope": str(scope),
        "stage2_behavior": stage2_behavior,
    }
    generation_params = {k: v for k, v in generation_params.items() if v is not None}

    meta = {
        "stage": "stage2",
        "cohort_root": str(cohort_root),
        "url": url,
        "model": model,
        "prompt_template_id": prompt_template_id,
        "prompt_prefix_sha256": prompt_prefix_sha256,
        "prompt_prefix_chars": int(len(prompt_prefix)),
        "generation_params": generation_params,
        "require_timestamp_match": require_timestamp_match,
        "semantic_keyword_only_match": semantic_keyword_only_match,
        "output_mode": str(output_mode),
        "scope": str(scope),
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "meta_stage2.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    downstream_cfg_obj = asdict(DEFAULT_DOWNSTREAM_CONFIG)
    downstream_cfg_nogate = DownstreamMetricConfig(**{**downstream_cfg_obj, "gates": {}})

    rows: List[Stage2Row] = []

    for i, hadm in enumerate(hadm_ids, 1):
        per_dir = out_dir / str(hadm)
        json_path = per_dir / "stage1.json"
        md_path = per_dir / "stage1.md"
        meta_path = per_dir / "stage1_meta.json"

        if not json_path.exists() or not md_path.exists():
            print(f"[{i}/{len(hadm_ids)}] HADM {hadm} | missing stage1 artifacts", flush=True)
            continue

        stage2_raw_path = per_dir / "stage2_raw.txt"
        stage2_lines_path = per_dir / "stage2_facts.txt"
        stage2_norm_path = per_dir / "stage2_normalized.json"
        stage2_metrics_path = per_dir / "stage2_metrics.json"

        if stage2_metrics_path.exists() and not overwrite_stage2:
            print(f"[{i}/{len(hadm_ids)}] HADM {hadm} | skip (exists)", flush=True)
            continue
        if overwrite_stage2 and stage2_metrics_path.exists():
            # Avoid stale retry artifacts when re-running in-place.
            try:
                (per_dir / "stage2_raw_retry1.txt").unlink(missing_ok=True)  # py>=3.8
            except TypeError:
                p = per_dir / "stage2_raw_retry1.txt"
                if p.exists():
                    p.unlink()

        md = md_path.read_text(encoding="utf-8", errors="ignore")
        if scope_l == "objective":
            md_in = _filter_stage1_markdown_for_stage2(md, ["DEMOGRAPHICS", "VITALS", "LABS", "UTILIZATION", "DISPOSITION"])
        elif scope_l == "all":
            md_in = _compact_stage1_markdown(md)
        else:
            raise SystemExit("--scope must be one of: objective, all")
        system_prompt = prompt_template.replace("{EHR_TEXT}", md_in).strip()

        response_format = None
        if mode_l == "json":
            # Ask for a strict JSON object to reduce truncation-induced parse failures.
            facts_schema = {
                "type": "object",
                "additionalProperties": False,
                "required": ["facts"],
                "properties": {
                    "facts": {
                        "type": "array",
                        "maxItems": 25,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["cluster", "keyword", "value", "timestamp"],
                            "properties": {
                                "cluster": {"type": "string", "enum": ["DEMOGRAPHICS", "VITALS", "LABS", "UTILIZATION", "DISPOSITION"]},
                                "keyword": {
                                    "type": "string",
                                    "enum": [
                                        "Sex",
                                        "Age",
                                        "Heart Rate",
                                        "Systolic BP",
                                        "Diastolic BP",
                                        "Respiratory Rate",
                                        "Temperature",
                                        "SpO2",
                                        "Weight",
                                        "Hemoglobin",
                                        "Hematocrit",
                                        "WBC",
                                        "Platelet",
                                        "Sodium",
                                        "Potassium",
                                        "Creatinine",
                                        "BUN",
                                        "Glucose",
                                        "Bicarbonate",
                                        "Prior Admissions 12mo",
                                        "ED Visits 6mo",
                                        "Days Since Last Admission",
                                        "Current Length of Stay",
                                        "Discharge Disposition",
                                        "Mental Status",
                                    ],
                                },
                                "value": {"type": "string"},
                                "timestamp": {"type": "string", "enum": ["Past", "Admission", "Discharge", "Unknown", "ADM", "DC"]},
                            },
                        },
                    }
                },
            }
            response_format = {"type": "json_schema", "json_schema": {"name": "kvt4_facts", "strict": True, "schema": facts_schema}}

        use_raw_completion = _env_truthy("MEDGEMMA_STAGE2_RAW_COMPLETION", "0")
        if use_raw_completion and mode_l == "lines":
            # Use /completion endpoint (no chat template) for LoRA trained on raw text
            completion = client.complete(
                prompt=system_prompt + "\n",
                max_tokens=max_tokens,
                temperature=temperature,
                repetition_penalty=effective_rep,
                top_p=top_p,
                min_p=min_p,
                typical_p=typical_p,
                stop=effective_stop,
            )
        else:
            completion = client.chat(
                user_prompt="BEGIN",
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                repetition_penalty=effective_rep,
                top_p=top_p,
                min_p=min_p,
                typical_p=typical_p,
                stop=effective_stop,
                response_format=response_format,
            )
        comp_str = strip_medgemma_internal_tokens(completion.text)
        completion_client_meta = dict(completion.meta or {})
        _write_text(stage2_raw_path, comp_str)

        lines = extract_kvt_fact_lines(comp_str)
        if mode_l == "json" and not lines:
            # Salvage: llama.cpp may return truncated JSON despite schema hints.
            # Extract fact objects from partial JSON text.
            salvaged: List[str] = []
            for m in _STAGE2_FACT_OBJ_RE.finditer(comp_str):
                c = m.group("cluster").strip()
                k = m.group("keyword").strip()
                v_tok = m.group("value").strip()
                if v_tok.startswith('"') and v_tok.endswith('"') and len(v_tok) >= 2:
                    v = v_tok[1:-1]
                else:
                    v = v_tok
                v = v.strip().lstrip("$")
                t = m.group("timestamp").strip()
                salvaged.append(f"{c}|{k}|{v}|{t}")
            if salvaged:
                lines = salvaged
        if _env_truthy("MEDGEMMA_STAGE2_DROP_PROMPT_LEAKAGE", "1"):
            lines = _drop_stage2_prompt_leakage_lines(lines)
        extracted_before_sanitize = list(lines)
        raw_valid0, raw_total0, raw_rate0 = _raw_kvt4_validity(comp_str, extracted_before_sanitize)
        retry_on_low_valid = _env_truthy("MEDGEMMA_STAGE2_RETRY_ON_LOW_VALID_RATE", "0")
        try:
            retry_low_valid_threshold = float(os.getenv("MEDGEMMA_STAGE2_RETRY_VALID_RATE_THRESHOLD", "0.45"))
        except Exception:
            retry_low_valid_threshold = 0.45
        try:
            retry_low_valid_min_total = int(os.getenv("MEDGEMMA_STAGE2_RETRY_MIN_RAW_LINES", "20"))
        except Exception:
            retry_low_valid_min_total = 20
        low_valid_quality = (
            retry_on_low_valid
            and mode_l == "lines"
            and raw_total0 >= retry_low_valid_min_total
            and raw_rate0 < retry_low_valid_threshold
        )

        did_retry = False
        if not lines or low_valid_quality:
            # Retry once: some small models produce checklists unless forced.
            retry_lock = (
                "\n\nFAILSAFE:\n"
                "- Return ONLY KVT4 lines now.\n"
                "- One fact per line: CLUSTER|Keyword|Value|Timestamp\n"
                "- No headers, no extra text.\n"
                "- DO NOT repeat prompt text, sections, markdown, or keyword lists.\n"
                if mode_l == "lines"
                else "\n\nFAILSAFE: Return ONLY a valid JSON object: {\"facts\": [...]} and nothing else."
            )
            completion2 = client.chat(
                user_prompt="BEGIN" + retry_lock,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                repetition_penalty=effective_rep,
                top_p=top_p,
                min_p=min_p,
                typical_p=typical_p,
                stop=effective_stop,
                response_format=response_format,
            )
            comp_str2 = strip_medgemma_internal_tokens(completion2.text)
            completion_client_meta = dict(completion2.meta or completion_client_meta)
            _write_text(per_dir / "stage2_raw_retry1.txt", comp_str2)
            lines = extract_kvt_fact_lines(comp_str2)
            if mode_l == "json" and not lines:
                salvaged2: List[str] = []
                for m in _STAGE2_FACT_OBJ_RE.finditer(comp_str2):
                    c = m.group("cluster").strip()
                    k = m.group("keyword").strip()
                    v_tok = m.group("value").strip()
                    if v_tok.startswith('"') and v_tok.endswith('"') and len(v_tok) >= 2:
                        v = v_tok[1:-1]
                    else:
                        v = v_tok
                    v = v.strip().lstrip("$")
                    t = m.group("timestamp").strip()
                    salvaged2.append(f"{c}|{k}|{v}|{t}")
                if salvaged2:
                    lines = salvaged2
            if lines:
                did_retry = True
                comp_str = comp_str2
                _write_text(stage2_raw_path, comp_str)
                extracted_before_sanitize = list(lines)
        # Optional dedup of consecutive identical lines (repetition loop suppression).
        if use_consecutive_dedup and len(lines) > 1:
            deduped: List[str] = [lines[0]]
            for ln in lines[1:]:
                if ln != deduped[-1]:
                    deduped.append(ln)
            lines = deduped
        lines = _sanitize_stage2_lines(lines, scope=scope)
        if _env_truthy("MEDGEMMA_POSTPROCESS_BINARY_FLAGS", "1"):
            lines = _normalize_binary_flag_values(lines)
        if scope_l == "all":
            lines = _drop_hallucinated_negatives(lines, md_in)
            if use_stage1_procedure_fallback:
                # Use un-compacted Stage1 markdown for fallback extraction to avoid
                # changing Stage2 prompt context with empty PROCEDURES sections.
                lines = _inject_stage1_procedure_fallback(lines, md)
            if use_drop_low_info_negatives:
                lines = _drop_low_information_negatives(lines)
            if use_semantic_gate:
                lines = _semantic_postprocess_gate(lines)
        if use_utilization_evidence_gate:
            lines = _drop_utilization_without_stage1_evidence(lines, md_in)
        _write_text(stage2_lines_path, "\n".join(lines) + ("\n" if lines else ""))
        norm, norm_stats = normalize_readmission_kvt4_lines(lines)
        format_stats = _compute_kvt4_format_stats(
            raw_text=comp_str,
            extracted_lines=extracted_before_sanitize,
            output_mode=output_mode,
            did_retry=did_retry,
            facts_after_sanitize_count=len(lines),
        )
        stage2_norm_path.write_text(
            json.dumps(
                {
                    "normalized": norm,
                    "normalization_stats": norm_stats,
                    "generation_params": generation_params,
                    "prompt_template_id": prompt_template_id,
                    "prompt_prefix_sha256": prompt_prefix_sha256,
                    "format_stats": format_stats,
                    "openai_compat": completion_client_meta,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        raw_total = int(format_stats.get("raw_total_lines") or 0)
        raw_valid = int(format_stats.get("raw_valid_kvt4_lines") or 0)
        recovered = int(format_stats.get("raw_recovered_cluster_prefix_lines") or 0)
        eff_valid = int(format_stats.get("raw_effective_valid_kvt4_lines") or 0)
        valid_rate = (raw_valid / raw_total) if raw_total else 0.0
        eff_rate = (eff_valid / raw_total) if raw_total else 0.0
        fmt_summary = (
            f"valid_rate={valid_rate:.3f} ({raw_valid}/{raw_total})"
            + (f" | effective_rate={eff_rate:.3f} ({eff_valid}/{raw_total}, recovered={recovered})" if recovered else "")
            + f" | invalid={int(format_stats.get('raw_invalid_lines') or 0)}"
            + f" | dup={int(format_stats.get('raw_duplicates_exact') or 0)}"
            + f" | facts={int(format_stats.get('facts_after_sanitize_count') or 0)}"
        )

        # Evaluate vs GT if available (curated cohorts).
        doc_dir = cohort_root / str(hadm)
        gt_path = doc_dir / f"ground_truth_{hadm}.json"
        if gt_path.exists():
            gt_obj = json.loads(gt_path.read_text(encoding="utf-8"))
            gt_lines_raw = _project_gt_to_kvt4_lines(gt_obj)
            gt_norm, _gt_norm_stats = normalize_readmission_kvt4_lines(gt_lines_raw)
            m, details = compute_metrics(
                norm,
                gt_norm,
                require_timestamp_match=require_timestamp_match,
                semantic_keyword_only_match=semantic_keyword_only_match,
            )
            score_ng, score_report_ng = compute_downstream_score(details, cfg=downstream_cfg_nogate)
            metrics_payload = {
                "metrics": {"precision": m.precision, "recall": m.recall, "f1": m.f1, "tp": m.tp, "fp": m.fp, "fn": m.fn},
                "downstream_score_nogate": score_report_ng,
                "stage1_json_ok": None,
            }
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    metrics_payload["stage1_json_ok"] = "yes" if (meta.get("json_parse_ok") and not meta.get("missing_keys")) else "no"
                except Exception:
                    metrics_payload["stage1_json_ok"] = None
            stage2_metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            rows.append(
                Stage2Row(
                    hadm_id=hadm,
                    stage1_json_ok=str(metrics_payload.get("stage1_json_ok") or "unknown"),
                    stage2_lines=len(norm),
                    precision=m.precision,
                    recall=m.recall,
                    f1=m.f1,
                    tp=m.tp,
                    fp=m.fp,
                    fn=m.fn,
                    downstream_score_nogate=float(score_ng),
                )
            )
            print(f"[{i}/{len(hadm_ids)}] HADM {hadm} | {fmt_summary} | F1={m.f1:.3f}", flush=True)
        else:
            stage2_metrics_path.write_text(json.dumps({"metrics": None, "note": "GT missing"}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{i}/{len(hadm_ids)}] HADM {hadm} | {fmt_summary} | GT missing", flush=True)

    if rows:
        # Write summary
        summary_csv = out_dir / "summary_stage2.csv"
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))

        cols = ["hadm_id", "stage1_json_ok", "stage2_lines", "precision", "recall", "f1", "tp", "fp", "fn", "downstream_score_nogate"]
        md = _markdown_table([asdict(r) for r in rows], cols)
        (out_dir / "summary_stage2.md").write_text(md, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort-root", type=str, default="Curated_EHR_Test_Sets/cohort10_initial/EHR_test_data")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--hadm-ids", type=str, default="", help="Comma-separated hadm_ids. If empty: discover by --num-docs.")
    p.add_argument("--num-docs", type=int, default=10)
    p.add_argument(
        "--allow-missing-gt",
        action="store_true",
        help="When --hadm-ids is empty: discover docs that have ehr_*.txt even if ground_truth_*.json is missing (useful for proxy cohorts).",
    )
    p.add_argument("--debug", action="store_true")

    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("stage1")
    p1.add_argument("--url", type=str, default=os.getenv("OPENAI_COMPAT_URL", os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234")))
    p1.add_argument("--model", type=str, default=os.getenv("OPENAI_COMPAT_MODEL_STAGE1", os.getenv("LMSTUDIO_MODEL", "medgemma-base-q5_k_m")))
    p1.add_argument(
        "--profile",
        type=str,
        default=os.getenv("STAGE1_PROFILE", "sgr_v2"),
        choices=[
            "strings_v1",
            "sgr_v1",
            "sgr_v2",
            "sgr_v2_strict",
            "sgr_v2_strict_cascade",
            "sgr_v2_compact",
            "sgr_v3",
            "sgr_v4",
        ],
    )
    p1.add_argument("--schema-path", type=str, default="schemas/readmission_domain_summary.schema.json")
    p1.add_argument("--max-tokens", type=int, default=int(os.getenv("STAGE1_MAX_TOKENS", "1536")))
    p1.add_argument("--temperature", type=float, default=0.0)
    p1.add_argument("--overwrite-stage1", action="store_true", help="Overwrite existing stage1_* outputs if present.")

    p2 = sub.add_parser("stage2")
    p2.add_argument("--url", type=str, default=os.getenv("OPENAI_COMPAT_URL", os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234")))
    p2.add_argument("--model", type=str, default=os.getenv("OPENAI_COMPAT_MODEL_STAGE2", "medgemma-ft-lora-adapters-q5_k_m"))
    p2.add_argument("--max-tokens", type=int, default=768)
    p2.add_argument("--temperature", type=float, default=0.0)
    p2.add_argument("--repetition-penalty", type=float, default=None, help="If omitted: default is 1.10 for --scope all, unchanged for objective.")
    p2.add_argument("--top-p", type=float, default=None)
    p2.add_argument("--min-p", type=float, default=None)
    p2.add_argument("--typical-p", type=float, default=None)
    p2.add_argument("--stop", action="append", default=None, help="Stop sequences (repeatable or comma-separated).")
    p2.add_argument("--overwrite-stage2", action="store_true", help="Overwrite existing stage2_* outputs if present.")
    p2.add_argument("--require-timestamp-match", action="store_true")
    p2.add_argument("--semantic-keyword-only-match", action="store_true")
    p2.add_argument("--output-mode", type=str, default=os.getenv("STAGE2_OUTPUT_MODE", "lines"), choices=["lines", "json"])
    p2.add_argument("--scope", type=str, default=os.getenv("STAGE2_SCOPE", "objective"), choices=["objective", "all"])

    args = p.parse_args()

    cohort_root = Path(args.cohort_root)
    if args.hadm_ids.strip():
        hadm_ids = [int(x.strip()) for x in args.hadm_ids.split(",") if x.strip()]
    else:
        hadm_ids = _discover_hadm_ids(cohort_root, int(args.num_docs), require_ground_truth=not bool(args.allow_missing_gt))

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (Path("results") / f"two_stage_structured_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "stage1":
        stage1_profile = str(getattr(args, "profile", "sgr_v2")).strip() or "sgr_v2"
        allow_sgr_v4 = _env_truthy("MEDGEMMA_ALLOW_SGR_V4_STAGE1", "0")
        if stage1_profile == "sgr_v4" and not allow_sgr_v4:
            print("[guardrail] Stage1 profile sgr_v4 is disabled for production; forcing sgr_v2", flush=True)
            stage1_profile = "sgr_v2"

        schema_path = Path(args.schema_path)
        if (
            not allow_sgr_v4
            and stage1_profile == "sgr_v2"
            and str(schema_path).strip().endswith("readmission_domain_summary_sgr_v4.schema.json")
        ):
            print("[guardrail] Stage1 schema sgr_v4 is disabled for production; forcing sgr_v2 schema", flush=True)
            schema_path = Path("schemas/readmission_domain_summary_sgr_v2.schema.json")

        if str(args.schema_path).strip() == "schemas/readmission_domain_summary.schema.json":
            if stage1_profile == "sgr_v1":
                schema_path = Path("schemas/readmission_domain_summary_sgr_v1.schema.json")
            elif stage1_profile in ("sgr_v2", "sgr_v2_compact"):
                schema_path = Path("schemas/readmission_domain_summary_sgr_v2.schema.json")
            elif stage1_profile == "sgr_v2_strict":
                schema_path = Path("schemas/readmission_domain_summary_sgr_v2_strict.schema.json")
            elif stage1_profile == "sgr_v2_strict_cascade":
                schema_path = Path("schemas/readmission_domain_summary_sgr_v2_strict_cascade.schema.json")
            elif stage1_profile == "sgr_v3":
                schema_path = Path("schemas/readmission_domain_summary_sgr_v3.schema.json")
            elif stage1_profile == "sgr_v4":
                schema_path = Path("schemas/readmission_domain_summary_sgr_v4.schema.json")

        if stage1_profile == "sgr_v1":
            system_prompt = READMISSION_DOMAIN_JSON_SYSTEM_PROMPT_SGR_V1
        elif stage1_profile == "sgr_v2":
            system_prompt = READMISSION_DOMAIN_JSON_SYSTEM_PROMPT_SGR_V2
        elif stage1_profile == "sgr_v2_strict":
            system_prompt = READMISSION_DOMAIN_JSON_SYSTEM_PROMPT_SGR_V2_STRICT
        elif stage1_profile == "sgr_v2_strict_cascade":
            system_prompt = READMISSION_DOMAIN_JSON_SYSTEM_PROMPT_SGR_V2_STRICT_CASCADE
        elif stage1_profile == "sgr_v2_compact":
            system_prompt = READMISSION_DOMAIN_JSON_SYSTEM_PROMPT_SGR_V2_COMPACT
        elif stage1_profile == "sgr_v3":
            system_prompt = READMISSION_DOMAIN_JSON_SYSTEM_PROMPT_SGR_V3
        elif stage1_profile == "sgr_v4":
            system_prompt = READMISSION_DOMAIN_JSON_SYSTEM_PROMPT_SGR_V4
        else:
            system_prompt = READMISSION_DOMAIN_JSON_SYSTEM_PROMPT
        run_stage1(
            cohort_root=cohort_root,
            out_dir=out_dir,
            hadm_ids=hadm_ids,
            url=args.url,
            model=args.model,
            schema_path=schema_path,
            stage1_profile=stage1_profile,
            system_prompt=system_prompt,
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            overwrite_stage1=bool(getattr(args, "overwrite_stage1", False)),
            debug=bool(args.debug),
        )
    elif args.cmd == "stage2":
        run_stage2(
            cohort_root=cohort_root,
            out_dir=out_dir,
            hadm_ids=hadm_ids,
            url=args.url,
            model=args.model,
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            repetition_penalty=args.repetition_penalty,
            top_p=args.top_p,
            min_p=args.min_p,
            typical_p=args.typical_p,
            stop=args.stop,
            require_timestamp_match=bool(args.require_timestamp_match),
            semantic_keyword_only_match=bool(args.semantic_keyword_only_match),
            output_mode=str(args.output_mode),
            scope=str(args.scope),
            overwrite_stage2=bool(args.overwrite_stage2),
            debug=bool(args.debug),
        )
    else:
        raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
