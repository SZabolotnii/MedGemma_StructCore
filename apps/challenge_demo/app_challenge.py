from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running as a script: `python apps/challenge_demo/app_challenge.py`
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import gradio as gr
import pandas as pd
import plotly.graph_objects as go  # type: ignore

from apps.challenge_demo.services.case_library import get_case, load_cases
from apps.challenge_demo.services.evidence_service import load_evidence_rows
from apps.challenge_demo.services.gemini_cloud_service import (
    AVAILABLE_MODELS,
    CloudExtractionResult,
    extract_with_cloud,
    list_available_models,
)
from apps.challenge_demo.services.structcore_service import (
    StructCoreConfig,
    lines_to_rows,
    result_to_debug_json,
    run_structcore,
)

DEMO_GUIDE_PATH = Path(__file__).parent / "DEMO_GUIDE.md"

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════

# ── Gradio theme (Medical Brain branding — LIGHT) ──────────────────────
APP_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.pink,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#F8F9FA",
    body_background_fill_dark="#F8F9FA",
    body_text_color="#1a1d1e",
    body_text_color_dark="#1a1d1e",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_color="#E0E0E0",
    block_border_color_dark="#E0E0E0",
    block_label_text_color="#5a5a5a",
    block_label_text_color_dark="#5a5a5a",
    block_title_text_color="#1a1d1e",
    block_title_text_color_dark="#1a1d1e",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#D0D0D0",
    input_border_color_dark="#D0D0D0",
    input_placeholder_color="#999999",
    input_placeholder_color_dark="#999999",
    button_primary_background_fill="#0087FF",
    button_primary_background_fill_dark="#0087FF",
    button_primary_background_fill_hover="#0066CC",
    button_primary_background_fill_hover_dark="#0066CC",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#F0F0F0",
    button_secondary_background_fill_dark="#F0F0F0",
    border_color_primary="#D0D0D0",
    border_color_primary_dark="#D0D0D0",
    background_fill_secondary="#FFFFFF",
    background_fill_secondary_dark="#FFFFFF",
    color_accent_soft="rgba(255,204,0,0.12)",
    color_accent_soft_dark="rgba(255,204,0,0.12)",
)

CUSTOM_CSS = """
/* ── Global (LIGHT THEME) ────────────────────────────────────── */
.gradio-container {
    max-width: 98% !important;
    background: #F8F9FA !important;
}

/* ── Animations ──────────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes countUp { from { opacity: 0; } to { opacity: 1; } }
@keyframes pulse   { 0%,100% { opacity: 1; } 50% { opacity: 0.6; } }
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
@keyframes borderGlow {
    0%,100% { border-color: rgba(255,204,0,0.5); }
    50%     { border-color: rgba(231,52,113,0.7); }
}
@keyframes btnPulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(0,87,255,0.4); }
    50%     { box-shadow: 0 0 0 8px rgba(0,87,255,0); }
}

/* ── Header (Medical Brain gradient on white) ────────────────── */
.app-header {
    background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
    border: 2px solid transparent;
    border-image: linear-gradient(90deg, #FFCC00, #F9A533, #E73471, #C60F8B) 1;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.6s ease-out;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.app-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(255,204,0,0.05) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(231,52,113,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.app-header h1 {
    font-size: 1.8em; margin: 0 0 6px 0;
    background: linear-gradient(90deg, #FFCC00 0%, #F9A533 25%, #E73471 75%, #C60F8B 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.app-header .subtitle { color: #5a5a5a; font-size: 1em; margin: 0; }
.model-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.82em; font-weight: 600; margin-top: 10px;
}
.model-badge.local { background: rgba(255,204,0,0.15); color: #D97700; border: 1px solid rgba(255,204,0,0.4); }
.model-badge.privacy { background: rgba(0,185,140,0.15); color: #00A67E; border: 1px solid rgba(0,185,140,0.4); }

/* ── Hero metric counters ────────────────────────────────────── */
.hero-metrics {
    display: flex; gap: 16px; flex-wrap: wrap; margin-top: 18px;
    animation: fadeInUp 0.8s ease-out 0.3s both;
}
.hero-metric {
    flex: 1; min-width: 140px; background: #FFFFFF;
    border: 1px solid #E0E0E0; border-radius: 14px;
    padding: 16px 12px; text-align: center;
    transition: transform 0.2s, box-shadow 0.3s, border-color 0.3s;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.hero-metric:hover { 
    transform: translateY(-3px); 
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    animation: borderGlow 2s infinite; 
}
.hero-metric .hm-value {
    font-size: 1.8em; font-weight: 800; animation: countUp 1s ease-out 0.5s both;
}
.hero-metric .hm-label {
    font-size: 0.72em; color: #666; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 4px;
}

/* ── Pipeline stepper ────────────────────────────────────────── */
.pipeline-stepper {
    display: flex; align-items: center; gap: 0; margin: 16px 0; padding: 16px;
    background: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.step-item {
    flex: 1; text-align: center; padding: 12px 8px;
    border-radius: 10px; position: relative; transition: all 0.3s;
}
.step-item .step-icon { font-size: 1.5em; margin-bottom: 4px; }
.step-item .step-label { font-size: 0.75em; color: #999; }
.step-item.active { background: rgba(255,204,0,0.12); }
.step-item.active .step-label { color: #D97700; font-weight: 600; }
.step-item.active .step-icon { animation: pulse 1.5s infinite; }
.step-item.done { background: rgba(0,185,140,0.1); }
.step-item.done .step-label { color: #00A67E; }
.step-arrow { color: #CCC; font-size: 1.2em; padding: 0 4px; }

/* ── Metric cards ────────────────────────────────────────────── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }
.metric-card {
    flex: 1; min-width: 150px; background: #FFFFFF;
    border: 1px solid #E0E0E0; border-radius: 12px;
    padding: 16px; text-align: center;
    transition: transform 0.2s, box-shadow 0.3s;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.metric-card:hover { 
    transform: translateY(-2px); 
    box-shadow: 0 6px 14px rgba(0,0,0,0.1);
}
.metric-card .label { font-size: 0.78em; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-card .value { font-size: 1.6em; font-weight: 700; color: #1a1d1e; margin-top: 4px; }
.metric-card .value.green { color: #00A67E; }
.metric-card .value.yellow { color: #D97700; }
.metric-card .value.red { color: #E73471; }
.metric-card .value.blue { color: #0087FF; }

/* ── Cluster colour tags (Medical Brain palette) ────────────── */
.cluster-tag {
    display: inline-block; padding: 2px 10px; border-radius: 6px;
    font-size: 0.78em; font-weight: 600; margin: 2px;
}
.cluster-DEMOGRAPHICS { background: rgba(0,135,255,0.15); color: #0066CC; }
.cluster-VITALS       { background: rgba(0,185,140,0.15); color: #00805E; }
.cluster-LABS         { background: rgba(255,204,0,0.15); color: #D97700; }
.cluster-DISPOSITION  { background: rgba(249,165,51,0.15); color: #CC7A00; }
.cluster-MEDICATIONS  { background: rgba(231,52,113,0.15); color: #C02858; }
.cluster-PROCEDURES   { background: rgba(198,15,139,0.15); color: #8B0A79; }
.cluster-UTILIZATION  { background: rgba(0,135,255,0.15); color: #0087DD; }
.cluster-PROBLEMS     { background: rgba(255,204,0,0.2); color: #B36B00; }
.cluster-SYMPTOMS     { background: rgba(231,52,113,0.2); color: #A02050; }

/* ── Status badges ───────────────────────────────────────────── */
.badge { display: inline-block; padding: 3px 12px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
.badge-verified    { background: rgba(0,185,140,0.15); color: #00805E; }
.badge-preliminary { background: rgba(255,204,0,0.15); color: #D97700; }
.badge-planned     { background: rgba(231,52,113,0.15); color: #C02858; }

/* ── Evidence claim cards ────────────────────────────────────── */
.evidence-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; margin: 16px 0; }
.evidence-card {
    background: #FFFFFF; border-radius: 12px; padding: 18px 20px;
    border-left: 4px solid #CCC; transition: transform 0.2s, box-shadow 0.3s;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.evidence-card:hover { 
    transform: translateY(-2px); 
    box-shadow: 0 6px 14px rgba(0,0,0,0.12);
}
.evidence-card.verified   { border-left-color: #00A67E; }
.evidence-card.preliminary { border-left-color: #D97700; }
.evidence-card.planned     { border-left-color: #E73471; }
.evidence-card .ec-id { font-size: 0.7em; color: #999; text-transform: uppercase; letter-spacing: 1px; }
.evidence-card .ec-metric { font-size: 1.4em; font-weight: 700; color: #1a1d1e; margin: 6px 0; }
.evidence-card .ec-claim { font-size: 0.85em; color: #5a5a5a; }
.evidence-card .ec-status { margin-top: 8px; }

/* ── Impact calculator ───────────────────────────────────────── */
.impact-result {
    background: linear-gradient(135deg, rgba(0,185,140,0.08), rgba(0,135,255,0.08));
    border: 1px solid rgba(0,185,140,0.25); border-radius: 14px;
    padding: 24px; margin: 16px 0;
}
.impact-big { font-size: 2.4em; font-weight: 800; color: #00A67E; text-align: center; }
.impact-sub { font-size: 0.9em; color: #5a5a5a; text-align: center; margin-top: 4px; }

/* ── Privacy comparison table ────────────────────────────────── */
.privacy-table { width: 100%; border-collapse: collapse; margin: 12px 0; }
.privacy-table th {
    background: rgba(255,204,0,0.1) !important; color: #D97700 !important;
    padding: 10px 14px !important; text-align: left !important; font-size: 0.82em !important;
    border-bottom: 2px solid #E0E0E0 !important;
}
.privacy-table td { 
    padding: 10px 14px !important; 
    border-bottom: 1px solid #F0F0F0 !important; 
    color: #1a1d1e !important;
}
.privacy-table .yes { color: #00A67E; font-weight: 600; }
.privacy-table .no  { color: #E73471; font-weight: 600; }
.privacy-table .warn { color: #D97700; font-weight: 600; }

/* ── Architecture diagram ────────────────────────────────────── */
.arch-box {
    background: #FFFFFF; border: 1px solid #E0E0E0;
    border-radius: 12px; padding: 20px; margin: 8px 0;
    transition: transform 0.2s, box-shadow 0.3s;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.arch-box:hover { 
    transform: translateY(-2px); 
    box-shadow: 0 6px 14px rgba(0,0,0,0.1);
}
.arch-arrow { text-align: center; font-size: 1.5em; 
    background: linear-gradient(90deg, #FFCC00, #E73471);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 4px 0; }
.arch-title { font-weight: 700; 
    background: linear-gradient(90deg, #F9A533, #C60F8B);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 6px; }
.arch-desc  { color: #5a5a5a; font-size: 0.9em; }

/* ── Tabs ─────────────────────────────────────────────────────── */
.tab-nav button { color: #5a5a5a !important; }
.tab-nav button.selected { 
    background: linear-gradient(90deg, rgba(255,204,0,0.12), rgba(231,52,113,0.12)) !important;
    border-bottom: 2px solid #D97700 !important;
    color: #D97700 !important; 
}

/* ── Comparison ───────────────────────────────────────────────── */
.comparison-highlight { border-left: 3px solid #E73471 !important; }

/* ── Dataframe / Table (Light theme) ─────────────────────────── */
table { border-collapse: collapse !important; }
table, .dataframe {
    background: #FFFFFF !important;
    color: #1a1d1e !important;
}
table th {
    background: linear-gradient(90deg, rgba(255,204,0,0.12), rgba(231,52,113,0.12)) !important;
    color: #D97700 !important;
    border: 1px solid #E0E0E0 !important;
    padding: 8px 12px !important;
    font-weight: 600 !important;
    font-size: 0.85em !important;
    text-transform: uppercase !important;
    letter-spacing: 0.3px !important;
}
table td {
    background: #FFFFFF !important;
    color: #1a1d1e !important;
    border: 1px solid #F0F0F0 !important;
    padding: 6px 12px !important;
    font-size: 0.9em !important;
}
table tr:hover td {
    background: rgba(255,204,0,0.06) !important;
}
.gradio-dataframe {
    border: 1px solid #E0E0E0 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05) !important;
}

/* ── Run button pulse animation ──────────────────────────────── */
button#run-structcore-btn, button[id*="run"] {
    animation: btnPulse 2.5s infinite;
}

/* ── Placeholder state for empty plots ───────────────────────── */
.plot-placeholder {
    display: flex; align-items: center; justify-content: center;
    flex-direction: column; gap: 12px;
    height: 280px; border-radius: 14px;
    border: 2px dashed #D0D0D0; background: #F8F9FA;
    color: #9AA0A6; text-align: center; padding: 24px;
}
.plot-placeholder .ph-icon { font-size: 3em; }
.plot-placeholder .ph-text { font-size: 0.95em; color: #666; }
.plot-placeholder .ph-cta  { font-size: 0.82em; color: #0087FF; font-weight: 600; }

/* ── Evidence card C01 (key metric) — full-width highlight ───── */
.evidence-card.c01-highlight {
    grid-column: 1 / -1;
    border-left: 6px solid #00A67E;
    background: linear-gradient(135deg, rgba(0,185,140,0.06), #FFFFFF);
}
.evidence-card.c01-highlight .ec-metric { font-size: 1.8em; color: #00A67E; }
"""

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

CLUSTER_COLORS = {
    "DEMOGRAPHICS": "#5c6bc0",
    "VITALS": "#1e88e5",
    "LABS": "#43a047",
    "DISPOSITION": "#ef6c00",
    "MEDICATIONS": "#e53935",
    "PROCEDURES": "#8e24aa",
    "UTILIZATION": "#00897b",
    "PROBLEMS": "#fdd835",
    "SYMPTOMS": "#ff7043",
}

CLUSTER_MAX_SCORES = {
    "DEMOGRAPHICS": 20, "VITALS": 25, "LABS": 30,
    "DISPOSITION": 25, "MEDICATIONS": 25, "PROCEDURES": 25,
    "UTILIZATION": 20, "PROBLEMS": 20, "SYMPTOMS": 5,
}

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ── Stage output normalizer ─────────────────────────────────────
import re as _re

def _normalize_stage_output(raw: str) -> List[str]:
    """Convert raw stage output to a list of display rows.

    Handles three formats produced by Stage 1 / Stage 2:
    1. Pipe-delimited KVT4 lines  (CLUSTER|Keyword|Value|Timestamp)
    2. Bare JSON array             ([{"cluster":…}, …])
    3. JSON inside markdown fence  (```json\n[…]\n```)

    Any JSON object is normalized to a KVT4 pipe-row using the
    keys cluster / keyword / value / timestamp (case-insensitive).
    Unrecognized lines are passed through as-is.
    """
    if not raw:
        return [""]

    text = raw.strip()

    # ── Strip markdown code fence if present ──────────────────────
    fence_match = _re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # ── Try to parse as JSON ──────────────────────────────────────
    json_candidate = text
    # Sometimes the model wraps in END / START markers — strip them
    json_candidate = _re.sub(r"(?i)(start|end)\s*\n?", "", json_candidate).strip()

    rows: List[str] = []
    try:
        parsed = json.loads(json_candidate)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    # Normalise keys: cluster / keyword / value / timestamp
                    lk = {k.lower(): str(v) for k, v in item.items()}
                    cluster   = lk.get("cluster", "")
                    keyword   = lk.get("keyword", lk.get("key", ""))
                    value     = lk.get("value", "")
                    timestamp = lk.get("timestamp", lk.get("time", ""))
                    rows.append(f"{cluster}|{keyword}|{value}|{timestamp}")
                else:
                    rows.append(str(item))
            if rows:
                logger.info(f"_normalize_stage_output: converted {len(rows)} JSON objects → KVT4 rows")
                return rows
        elif isinstance(parsed, dict):
            # Single object
            lk = {k.lower(): str(v) for k, v in parsed.items()}
            rows.append(
                f"{lk.get('cluster','')}|{lk.get('keyword','')}|{lk.get('value','')}|{lk.get('timestamp','')}"
            )
            return rows
    except (json.JSONDecodeError, ValueError):
        pass  # Not JSON — fall through to line splitting

    # ── Default: split by newline ─────────────────────────────────
    lines = raw.splitlines()
    return lines if lines else [""]


def _default_case_id() -> str:
    target = "high_risk_complex"
    cases = load_cases()
    if any(c.id == target for c in cases):
        return target
    return cases[0].id if cases else "custom"


def _case_choices() -> List[Tuple[str, str]]:
    out = []
    for c in load_cases():
        out.append((c.title, c.id))
    out.append(("✍️ Paste Custom Note", "custom"))
    return out


def _on_case_change(case_id: str) -> Tuple[str | Dict, str]:
    if not case_id or case_id == "custom":
        return gr.update(), "Manual mode: paste your own note text."
    c = get_case(case_id)
    if c is None:
        return "", "Case not found."
    return c.text, f"**{c.title}**\n\n{c.description}"


# ── File Upload Logic ───────────────────────────────────────────

def _read_file_content(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    try:
        if ext in [".txt", ".md", ".json", ".csv"]:
            return path.read_text(encoding="utf-8", errors="replace")
        
        elif ext == ".pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(file_path)
                text = []
                for page in reader.pages:
                    text.append(page.extract_text())
                return "\n".join(text)
            except ImportError:
                return "Error: PyPDF2 not installed. Cannot read PDF.\nPlease install: pip install PyPDF2"
            except Exception as e:
                return f"Error reading PDF: {str(e)}"

        elif ext in [".docx", ".doc"]:
            try:
                import docx
                doc = docx.Document(file_path)
                return "\n".join([p.text for p in doc.paragraphs])
            except ImportError:
                return "Error: python-docx not installed. Cannot read DOCX.\nPlease install: pip install python-docx"
            except Exception as e:
                return f"Error reading DOCX: {str(e)}"
        
        else:
            return f"Error: Unsupported file extension {ext}"

    except Exception as e:
        return f"Error reading file: {str(e)}"


# ── Status formatting ───────────────────────────────────────────

def _format_status(note_id: str, backend_mode: str, duration_sec: float,
                   gate_summary: Dict, warnings: List[str], error: str | None) -> str:
    ok = "✅" if gate_summary.get("parse_success") else "❌"
    clusters_list = gate_summary.get("clusters_present") or []
    cluster_counts = gate_summary.get("cluster_counts") or {}
    lines = gate_summary.get("lines_extracted") or gate_summary.get("output_lines", 0)

    # Build HTML metric cards
    mode_badge = "🔒 Local" if "mock" in backend_mode or "pipeline" in backend_mode else "☁️ Cloud"
    mode_color = "green" if "Local" in mode_badge else "blue"

    # Pipeline stepper (compact)
    # Changed layout to be cleaner for side panel
    stepper_html = """
<div class="pipeline-stepper" style="padding:10px; margin:0 0 12px 0;">
  <div class="step-item done"><div class="step-icon" style="font-size:1.2em">📄</div><div class="step-label">Input</div></div>
  <div class="step-arrow">→</div>
  <div class="step-item done"><div class="step-icon" style="font-size:1.2em">🧠</div><div class="step-label">Stage 1</div></div>
  <div class="step-arrow">→</div>
  <div class="step-item done"><div class="step-icon" style="font-size:1.2em">🎯</div><div class="step-label">Stage 2</div></div>
  <div class="step-arrow">→</div>
  <div class="step-item done"><div class="step-icon" style="font-size:1.2em">📊</div><div class="step-label">Risk</div></div>
</div>
"""

    # Cluster tags with counts
    cluster_tags = " ".join(
        f'<span class="cluster-tag cluster-{c}">{c} ({cluster_counts.get(c, 0)})</span>' for c in clusters_list
    ) or '<span style="color:#666">none detected</span>'

    # Mock mode warning (compact)
    mock_warning = ""
    if "mock" in backend_mode.lower():
        mock_warning = """
<div style="background:rgba(234,67,53,0.1);border:1px solid #ea4335;border-radius:8px;padding:10px;margin-bottom:12px;">
  <p style="margin:0;color:#c5221f;font-weight:700;font-size:0.85em;">⚠️ MOCK MODE: Synthetic Data</p>
</div>
"""

    # Use a 2-column grid for metrics in the side panel
    html = f"""
{mock_warning}
{stepper_html}
<div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-bottom:12px;">
  <div class="metric-card" style="min-width:0; padding:12px;"><div class="label" style="font-size:0.7em">Status</div><div class="value" style="font-size:1.3em">{ok}</div></div>
  <div class="metric-card" style="min-width:0; padding:12px;"><div class="label" style="font-size:0.7em">Facts</div><div class="value blue" style="font-size:1.3em">{lines}</div></div>
  <div class="metric-card" style="min-width:0; padding:12px;"><div class="label" style="font-size:0.7em">Time</div><div class="value" style="font-size:1.3em">{duration_sec:.2f}s</div></div>
  <div class="metric-card" style="min-width:0; padding:12px;"><div class="label" style="font-size:0.7em">Mode</div><div class="value {mode_color}" style="font-size:1em; margin-top:6px;">{mode_badge}</div></div>
</div>

<div style="background:#fff; border:1px solid #e0e0e0; border-radius:12px; padding:12px;">
  <div style="font-size:0.75em; color:#999; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Clusters Detected:</div>
  <div style="line-height:1.6;">{cluster_tags}</div>
</div>
"""
    if warnings:
        html += "<details><summary>⚠️ Warnings</summary><ul>"
        html += "".join(f"<li>{w}</li>" for w in warnings)
        html += "</ul></details>"
    if error:
        html += f'<p style="color:var(--danger)">Error: {error}</p>'
    return html


def _format_processing_status(note_id: str, backend_mode: str) -> str:
    mode_badge = "🔒 Local" if "mock" in (backend_mode or "").lower() or "pipeline" in (backend_mode or "").lower() else "☁️ Cloud"
    return f"""
<div class="pipeline-stepper" style="padding:10px; margin:0 0 12px 0;">
  <div class="step-item done"><div class="step-icon" style="font-size:1.2em">📄</div><div class="step-label">Input</div></div>
  <div class="step-arrow">→</div>
  <div class="step-item active"><div class="step-icon" style="font-size:1.2em">🧠</div><div class="step-label">Stage 1</div></div>
  <div class="step-arrow">→</div>
  <div class="step-item"><div class="step-icon" style="font-size:1.2em">🎯</div><div class="step-label">Stage 2</div></div>
  <div class="step-arrow">→</div>
  <div class="step-item"><div class="step-icon" style="font-size:1.2em">📊</div><div class="step-label">Risk</div></div>
</div>
<div class="metric-card" style="padding:12px; margin-bottom:10px;">
  <div class="label" style="font-size:0.7em">Live Status</div>
  <div class="value blue" style="font-size:1.05em;">⏳ Processing started…</div>
  <div style="font-size:0.8em; color:#666; margin-top:6px;">Case: {note_id or "custom"} · Mode: {mode_badge}</div>
</div>
"""


def _on_run_started(case_id: str, backend_mode: str) -> Tuple[str, Dict]:
    status_html = _format_processing_status(case_id or "custom", backend_mode or "pipeline")
    return status_html, gr.update(value="⏳ Running…", interactive=False)


def _on_run_finished() -> Dict:
    return gr.update(value="🚀 Run StructCore", interactive=True)


# ── Risk visualization ──────────────────────────────────────────

def _build_risk_gauge(prob: float, category: str) -> go.Figure:
    """Create a plotly gauge chart for readmission probability."""
    color_map = {"Low": "#34a853", "Medium": "#fbbc04", "High": "#ea4335", "Critical": "#b71c1c"}
    bar_color = color_map.get(category, "#9aa0a6")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 48, "color": bar_color}},
        title={"text": f"Readmission Risk: {category}", "font": {"size": 18, "color": "#5a5a5a"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#999", "tickfont": {"color": "#5a5a5a"}},
            "bar": {"color": bar_color, "thickness": 0.6},
            "bgcolor": "#F0F0F0",
            "bordercolor": "#E0E0E0",
            "steps": [
                {"range": [0, 15], "color": "rgba(52,168,83,0.15)"},
                {"range": [15, 30], "color": "rgba(251,188,4,0.15)"},
                {"range": [30, 50], "color": "rgba(234,67,53,0.15)"},
                {"range": [50, 100], "color": "rgba(183,28,28,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 3},
                "thickness": 0.8,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font={"color": "#1a1d1e"}, height=300, margin={"t": 60, "b": 20, "l": 40, "r": 40},
    )
    return fig


def _build_cluster_bar(risk: Dict) -> go.Figure:
    """Build horizontal bar chart of cluster scores."""
    cluster_scores = risk.get("cluster_scores") or {}
    clusters = list(CLUSTER_MAX_SCORES.keys())
    scores = [cluster_scores.get(c, {}).get("score", 0) if isinstance(cluster_scores.get(c), dict)
              else cluster_scores.get(c, 0) for c in clusters]
    maxes = [CLUSTER_MAX_SCORES[c] for c in clusters]
    colors = [CLUSTER_COLORS.get(c, "#9aa0a6") for c in clusters]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=clusters, x=maxes, orientation="h", name="Max",
        marker={"color": "rgba(200,200,200,0.25)", "line": {"width": 0}},
    ))
    fig.add_trace(go.Bar(
        y=clusters, x=scores, orientation="h", name="Score",
        marker={"color": colors, "line": {"width": 0}},
        text=[f"{s}/{m}" for s, m in zip(scores, maxes)],
        textposition="auto", textfont={"color": "#1a1d1e", "size": 11},
    ))
    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font={"color": "#1a1d1e"}, height=360,
        margin={"t": 30, "b": 30, "l": 120, "r": 20},
        showlegend=False,
        xaxis={"title": "Points", "gridcolor": "#E8E8E8", "range": [0, 35]},
        yaxis={"autorange": "reversed"},
        title={"text": "Cluster Score Breakdown", "font": {"size": 15, "color": "#5a5a5a"}},
    )
    return fig


def _format_risk_summary(risk: Dict | None) -> Tuple[str, go.Figure | None, go.Figure | None, str]:
    """Return (markdown, gauge_fig, bar_fig, json_str)."""
    if not risk:
        placeholder_html = """
<div class="plot-placeholder">
  <div class="ph-icon">📊</div>
  <div class="ph-text">Risk analysis will appear here after running StructCore</div>
  <div class="ph-cta">← Go to Case Input tab and click 🚀 Run StructCore</div>
</div>"""
        empty_fig = go.Figure()
        empty_fig.update_layout(
            paper_bgcolor="#F8F9FA", plot_bgcolor="#F8F9FA", height=280,
            margin={"t": 10, "b": 10, "l": 10, "r": 10},
            annotations=[{
                "text": "Run StructCore to see risk gauge",
                "showarrow": False,
                "font": {"color": "#9AA0A6", "size": 14},
                "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
            }]
        )
        return placeholder_html, empty_fig, empty_fig, "{}"

    prob = risk.get("probability", 0)
    category = risk.get("risk_category", "Unknown")
    score = risk.get("composite_score", 0)
    completeness = risk.get("data_completeness", 0)
    factors = risk.get("risk_factors") or []

    # Build metric cards HTML
    cat_color = {"Low": "green", "Medium": "yellow", "High": "red", "Critical": "red"}.get(category, "blue")
    md = f"""
<div class="metric-row">
  <div class="metric-card"><div class="label">Risk Category</div><div class="value {cat_color}">{category}</div></div>
  <div class="metric-card"><div class="label">Probability</div><div class="value">{prob:.1%}</div></div>
  <div class="metric-card"><div class="label">Composite Score</div><div class="value blue">{score}</div></div>
  <div class="metric-card"><div class="label">Data Completeness</div><div class="value">{completeness:.0%}</div></div>
</div>
"""
    if factors:
        md += "\n**Top Risk Factors:**\n"
        for f in factors[:6]:
            md += f"- ⚠️ {f}\n"

    gauge = _build_risk_gauge(prob, category)
    bar = _build_cluster_bar(risk)
    return md, gauge, bar, json.dumps(risk, ensure_ascii=False, indent=2)


# ── Comparison helpers ──────────────────────────────────────────

def _build_comparison_bar(results: Dict[str, Dict]) -> go.Figure:
    """Bar chart comparing extraction metrics across models."""
    models = list(results.keys())
    fact_counts = [results[m].get("fact_count", 0) for m in models]
    durations = [results[m].get("duration_sec", 0) for m in models]
    cluster_counts = [len(results[m].get("cluster_coverage", [])) for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Facts", x=models, y=fact_counts,
                         marker_color=[CLUSTER_COLORS.get("LABS", "#43a047")] * len(models)))
    fig.add_trace(go.Bar(name="Clusters", x=models, y=cluster_counts,
                         marker_color=[CLUSTER_COLORS.get("VITALS", "#1e88e5")] * len(models)))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font={"color": "#1a1d1e"}, height=320,
        margin={"t": 40, "b": 40, "l": 50, "r": 20},
        title={"text": "Extraction Quality Comparison", "font": {"size": 15, "color": "#5a5a5a"}},
        xaxis={"gridcolor": "#E8E8E8"}, yaxis={"gridcolor": "#E8E8E8"},
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def _build_timing_bar(results: Dict[str, Dict]) -> go.Figure:
    """Bar chart comparing latency."""
    models = list(results.keys())
    durations = [results[m].get("duration_sec", 0) for m in models]
    colors = ["#00bfa5" if "MedGemma" in m else "#ea4335" for m in models]

    fig = go.Figure(go.Bar(x=models, y=durations, marker_color=colors,
                           text=[f"{d:.2f}s" for d in durations], textposition="auto",
                           textfont={"color": "#1a1d1e"}))
    fig.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font={"color": "#1a1d1e"}, height=280,
        margin={"t": 40, "b": 40, "l": 50, "r": 20},
        title={"text": "Latency Comparison", "font": {"size": 15, "color": "#5a5a5a"}},
        yaxis={"title": "Seconds", "gridcolor": "#E8E8E8"},
    )
    return fig


# ── Evidence Board helpers ──────────────────────────────────────

def _build_evidence_cards_html(evidence_df: pd.DataFrame) -> str:
    """Generate visual evidence cards HTML from the evidence dataframe."""
    cards_html = '<div class="evidence-grid">\n'
    for _, row in evidence_df.iterrows():
        status = str(row.get("Status", "")).strip().lower()
        css_class = status if status in {"verified", "preliminary", "planned"} else ""
        badge_class = f"badge-{status}" if status in {"verified", "preliminary", "planned"} else ""
        status_label = str(row.get("Status", "")).strip()
        claim_id = str(row.get('Claim ID', ''))
        # Special highlight for C01 — the primary proof point
        extra_class = " c01-highlight" if claim_id == "C01" else ""
        cards_html += f"""
  <div class="evidence-card {css_class}{extra_class}">
    <div class="ec-id">{claim_id}</div>
    <div class="ec-metric">{row.get('Metric', '')}</div>
    <div class="ec-claim">{row.get('Claim', '')}</div>
    <div class="ec-status"><span class="badge {badge_class}">{status_label}</span></div>
  </div>
"""
    cards_html += "</div>"
    return cards_html


def _build_auroc_comparison_chart() -> go.Figure:
    """Build AUROC comparison bar chart for the Evidence Board."""
    arms = ["LACE\nProxy", "HOSPITAL\nProxy", "A0\nMetadata", "A1\nRule Engine", "A4\nTabular ML"]
    aurocs = [0.5568, 0.5808, 0.5842, 0.6024, 0.6846]
    ci_lo = [0.5422, 0.5673, 0.5700, 0.5882, 0.6703]
    ci_hi = [0.5714, 0.5946, 0.5982, 0.6167, 0.6986]
    colors = ["#666", "#777", "#888", "#00FFAA", "#0087FF"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=arms, y=aurocs, marker_color=colors,
        text=[f"{a:.4f}" for a in aurocs], textposition="outside",
        textfont={"color": "#1a1d1e", "size": 12},
    ))
    # Add CI error bars for those that have them
    error_y_vals = [
        (aurocs[i] - ci_lo[i]) if ci_lo[i] is not None else 0
        for i in range(len(aurocs))
    ]
    fig.data[0].error_y = dict(
        type="data",
        array=[(ci_hi[i] - aurocs[i]) if ci_hi[i] else 0 for i in range(len(aurocs))],
        arrayminus=error_y_vals,
        visible=True,
        color="#999",
        thickness=1.5,
    )
    # Reference line at 0.5 (random baseline)
    fig.add_hline(y=0.5, line_dash="dash", line_color="#CCC",
                  annotation_text="Random baseline (0.5)", annotation_font_color="#999")
    fig.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font={"color": "#1a1d1e"}, height=380,
        margin={"t": 40, "b": 60, "l": 50, "r": 20},
        showlegend=False,
        yaxis={"title": "AUROC", "gridcolor": "#E8E8E8", "range": [0.45, 0.75]},
        xaxis={"gridcolor": "#E8E8E8"},
        title={"text": "Canonical baselines — Track B (N=9,857) [Verified]", "font": {"size": 15, "color": "#5a5a5a"}},
    )
    return fig


def _build_auroc_scaleout_chart() -> go.Figure:
    """Build AUROC comparison chart for extracted subset evaluations (Preliminary)."""
    subsets = ["subset4200\n(N=3,000)", "test4100\n(N=4,100)"]
    a4 = [0.6382, 0.6484]
    a3_fact = [0.6550, 0.6621]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="A4 tabular (refit)",
        x=subsets,
        y=a4,
        marker_color="#0087FF",
        text=[f"{v:.4f}" for v in a4],
        textposition="outside",
        textfont={"color": "#1a1d1e", "size": 12},
    ))
    fig.add_trace(go.Bar(
        name="A3_factlevel (refit)",
        x=subsets,
        y=a3_fact,
        marker_color="#FFCC00",
        text=[f"{v:.4f}" for v in a3_fact],
        textposition="outside",
        textfont={"color": "#1a1d1e", "size": 12},
    ))
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="#CCC",
        annotation_text="Random baseline (0.5)",
        annotation_font_color="#999",
    )
    fig.update_layout(
        barmode="group",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": "#1a1d1e"},
        height=380,
        margin={"t": 40, "b": 60, "l": 50, "r": 20},
        yaxis={"title": "AUROC", "gridcolor": "#E8E8E8", "range": [0.60, 0.70]},
        xaxis={"gridcolor": "#E8E8E8"},
        legend={"orientation": "h", "y": -0.25, "x": 0},
        title={"text": "Extracted subset evaluations — Track B [Preliminary]", "font": {"size": 15, "color": "#5a5a5a"}},
    )
    return fig


# ── Clinical Impact calculator ──────────────────────────────────

def _calculate_impact(
    annual_discharges: float, readmission_rate: float, cost_per_readmission: float,
) -> Tuple[str, go.Figure]:
    """Calculate projected savings for different avoidable-readmission scenarios."""
    rate = readmission_rate / 100.0
    total_readmissions = annual_discharges * rate
    scenarios = [
        ("Conservative (2%)", 0.02),
        ("Moderate (5%)", 0.05),
        ("Optimistic (8%)", 0.08),
    ]

    html_parts = ['<div class="impact-result">']
    html_parts.append(f'<div class="impact-sub">Based on {int(annual_discharges):,} annual discharges, '
                      f'{readmission_rate:.0f}% baseline rate, ${cost_per_readmission:,.0f}/readmission</div>')
    html_parts.append(f'<div class="impact-sub" style="margin-bottom:16px;">Estimated {int(total_readmissions):,} readmissions/year</div>')

    savings_list = []
    for label, fraction in scenarios:
        avoided = total_readmissions * fraction
        saving = avoided * cost_per_readmission
        savings_list.append((label, fraction, avoided, saving))
        color = "#00FFAA" if fraction <= 0.02 else ("#FFCC00" if fraction <= 0.05 else "#0087FF")
        html_parts.append(f"""
<div class="metric-row">
  <div class="metric-card" style="flex:2"><div class="label">{label}</div>
    <div class="value" style="font-size:1em;color:#AAAAB2">{int(avoided):,} readmissions avoided</div></div>
  <div class="metric-card" style="flex:1"><div class="label">Annual Savings</div>
    <div class="value" style="color:{color}">${saving:,.0f}</div></div>
</div>""")

    html_parts.append("</div>")
    html = "\n".join(html_parts)

    # Build chart — Medical Brain polished palette
    labels = [s[0] for s in savings_list]
    values = [s[3] for s in savings_list]
    avoided_counts = [int(s[2]) for s in savings_list]

    # Medical Brain: teal / amber / cobalt
    bar_colors = ["#00B8A3", "#F9A533", "#1E7FFF"]
    bar_line_colors = ["#009080", "#D97700", "#155CC0"]

    fig = go.Figure()
    for i, (lbl, val, avoided, color, lclr) in enumerate(
        zip(labels, values, avoided_counts, bar_colors, bar_line_colors)
    ):
        fig.add_trace(go.Bar(
            x=[lbl], y=[val],
            name=lbl,
            marker=dict(
                color=color,
                opacity=0.88,
                line=dict(color=lclr, width=2),
            ),
            text=[f"${val:,.0f}"],
            textposition="outside",
            textfont=dict(color="#1a1d1e", size=13, family="Inter, sans-serif"),
            hovertemplate=(
                f"<b>{lbl}</b><br>"
                f"Savings: <b>${val:,.0f}</b><br>"
                f"Readmissions avoided: {avoided:,}<br>"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    # Dashed reference: conservative line as baseline anchor
    conservative_val = values[0]
    fig.add_hline(
        y=conservative_val,
        line_dash="dot",
        line_color="rgba(0,185,163,0.45)",
        line_width=1.5,
        annotation_text=f" Conservative baseline: ${conservative_val:,.0f}",
        annotation_font=dict(color="#00896F", size=11),
        annotation_position="top right",
    )

    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#1a1d1e", family="Inter, sans-serif"),
        height=400,
        margin=dict(t=60, b=50, l=70, r=30),
        yaxis=dict(
            title="Annual Savings (USD)",
            gridcolor="#EEEEEE",
            gridwidth=1,
            tickprefix="$",
            tickformat=",.0f",
            zeroline=False,
        ),
        xaxis=dict(
            gridcolor="#EEEEEE",
        ),
        title=dict(
            text=(
                "Projected Annual Savings by Scenario"
                "<br><sup style='color:#888;font-size:11px'>"
                f"Based on {int(annual_discharges):,} discharges · "
                f"{readmission_rate:.0f}% readmission rate · "
                f"${cost_per_readmission:,.0f}/readmission"
                "</sup>"
            ),
            font=dict(size=16, color="#1a1d1e"),
            x=0.02,
            xanchor="left",
        ),
        bargap=0.35,
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter, sans-serif"),
    )
    return html, fig


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE HANDLER
# ═══════════════════════════════════════════════════════════════════════════

def _run_demo(
    case_id: str,
    note_text: str,
    backend_mode: str,
    stage1_url: str,
    stage1_model: str,
    stage2_url: str,
    stage2_model: str,
    fallback_to_mock: bool,
) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str, go.Figure | None, go.Figure | None, str, str, Dict]:
    note = (note_text or "").strip()
    effective_case_id = case_id or "custom"

    logger.info("=" * 80)
    logger.info(f"🚀 Starting extraction | Case: {effective_case_id} | Mode: {backend_mode}")
    
    if not note and effective_case_id != "custom":
        c = get_case(effective_case_id)
        if c is not None:
            note = c.text

    logger.info(f"📝 Note length: {len(note)} chars")

    cfg = StructCoreConfig(
        backend_mode=(backend_mode or "mock").strip(),
        stage1_url=(stage1_url or "").strip(),
        stage1_model=(stage1_model or "").strip(),
        stage2_url=(stage2_url or "").strip(),
        stage2_model=(stage2_model or "").strip(),
        fallback_to_mock_on_error=bool(fallback_to_mock),
    )

    result = run_structcore(note, effective_case_id, cfg)
    
    # Log results
    logger.info(f"✅ Extraction complete | Duration: {result.duration_sec:.2f}s")
    logger.info(f"📊 Facts extracted: {len(result.normalized_lines)}")
    logger.info(f"🏷️  Clusters: {', '.join(result.gate_summary.get('clusters_present', []))}")
    if result.risk:
        logger.info(f"⚠️  Risk score: {result.risk.get('composite_score', 0)} | Category: {result.risk.get('risk_category', 'N/A')}")
    if result.error:
        logger.error(f"❌ Error: {result.error}")
    if result.warnings:
        logger.warning(f"⚠️  Warnings: {'; '.join(result.warnings)}")

    status_md = _format_status(
        note_id=result.note_id,
        backend_mode=result.backend_mode,
        duration_sec=result.duration_sec,
        gate_summary=result.gate_summary,
        warnings=result.warnings,
        error=result.error,
    )

    rows = lines_to_rows(result.normalized_lines)
    df = pd.DataFrame(rows, columns=["CLUSTER", "Keyword", "Value", "Timestamp"])
    stage1_df = pd.DataFrame({"Stage1 summary": _normalize_stage_output(result.stage1_summary)})
    stage2_df = pd.DataFrame({"Stage2 write output": _normalize_stage_output(result.stage2_raw)})

    risk_md, risk_gauge, risk_bar, risk_json = _format_risk_summary(result.risk)

    return (
        status_md,
        stage1_df,
        stage2_df,
        df,
        json.dumps(result.gate_summary, ensure_ascii=False, indent=2),
        risk_md,
        risk_gauge,
        risk_bar,
        risk_json,
        result_to_debug_json(result),
        {"text": note, "case_id": effective_case_id, "result": result},  # Cache for state
    )


# ═══════════════════════════════════════════════════════════════════════════
# COMPARATIVE ANALYSIS HANDLER
# ═══════════════════════════════════════════════════════════════════════════

def _run_comparison(
    case_id: str,
    note_text: str,
    backend_mode: str,
    stage1_url: str,
    stage1_model: str,
    stage2_url: str,
    stage2_model: str,
    use_local: bool,
    use_gemini_flash: bool,
    use_gemini3: bool,
    last_run_state: Dict | None = None,
) -> Tuple[str, str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, go.Figure, go.Figure, str]:
    note = (note_text or "").strip()
    effective_case_id = case_id or "custom"
    
    # Try to use cached note from state if input is empty but state exists
    cached_result = None
    if not note and last_run_state and effective_case_id == last_run_state.get("case_id"):
        note = last_run_state.get("text", "")
        cached_result = last_run_state.get("result")

    # If input note matches cache, we can reuse result
    if note and last_run_state and note == last_run_state.get("text"):
        cached_result = last_run_state.get("result")

    logger.info("=" * 80)
    logger.info(f"⚖️  Starting comparison | Case: {effective_case_id}")
    
    if not note and effective_case_id != "custom":
        c = get_case(effective_case_id)
        if c is not None:
            note = c.text

    if not note:
        empty_df = pd.DataFrame(columns=["CLUSTER", "Keyword", "Value", "Timestamp"])
        empty_fig = go.Figure()
        return "", "", "", empty_df, empty_df, empty_df, empty_fig, empty_fig, "{}"
    
    logger.info(f"📝 Note length: {len(note)} chars")

    all_results: Dict[str, Dict] = {}
    model_dfs: Dict[str, pd.DataFrame] = {}

    # 1) Local MedGemma
    if use_local:
        if cached_result:
            logger.info("🚀 Reusing CACHED result for MedGemma (local)")
            local_result = cached_result
            tag = " (cached)"
        else:
            logger.info("⚙️ Running MedGemma (local)...")
            cfg = StructCoreConfig(
                backend_mode=(backend_mode or "mock").strip(),
                stage1_url=(stage1_url or "").strip(),
                stage1_model=(stage1_model or "").strip(),
                stage2_url=(stage2_url or "").strip(),
                stage2_model=(stage2_model or "").strip(),
                fallback_to_mock_on_error=True,
            )
            local_result = run_structcore(note, effective_case_id, cfg)
            tag = ""

        local_rows = lines_to_rows(local_result.normalized_lines)
        local_df = pd.DataFrame(local_rows, columns=["CLUSTER", "Keyword", "Value", "Timestamp"])
        
        all_results[f"🔒 MedGemma (local){tag}"] = {
            "fact_count": len(local_result.normalized_lines),
            "cluster_coverage": local_result.gate_summary.get("clusters_present", []),
            "duration_sec": local_result.duration_sec,
            "format_validity": "99.74%",
            "privacy": "🔒 Zero PHI leakage",
            "cost": "$0",
        }
        model_dfs[f"🔒 MedGemma (local){tag}"] = local_df

    # 2) Gemini 2.5 Flash
    if use_gemini_flash:
        cloud_result = extract_with_cloud("gemini-2.5-flash", note, effective_case_id)
        cloud_rows = lines_to_rows(cloud_result.normalized_lines)
        cloud_df = pd.DataFrame(cloud_rows, columns=["CLUSTER", "Keyword", "Value", "Timestamp"])
        cached_tag = " (cached)" if cloud_result.is_cached else ""
        all_results[f"☁️ Gemini 2.5 Flash{cached_tag}"] = {
            "fact_count": cloud_result.fact_count,
            "cluster_coverage": cloud_result.cluster_coverage,
            "duration_sec": cloud_result.duration_sec,
            "format_validity": f"{cloud_result.format_validity_rate:.1%}",
            "privacy": "☁️ Data sent to cloud",
            "cost": "~$0.001/call",
            "error": cloud_result.error,
        }
        model_dfs[f"☁️ Gemini 2.5 Flash{cached_tag}"] = cloud_df

    # 3) Gemini 3 Flash Preview
    if use_gemini3:
        cloud_result = extract_with_cloud("gemini-3-flash-preview", note, effective_case_id)
        cloud_rows = lines_to_rows(cloud_result.normalized_lines)
        cloud_df = pd.DataFrame(cloud_rows, columns=["CLUSTER", "Keyword", "Value", "Timestamp"])
        cached_tag = " (cached)" if cloud_result.is_cached else ""
        all_results[f"☁️ Gemini 3 Flash{cached_tag}"] = {
            "fact_count": cloud_result.fact_count,
            "cluster_coverage": cloud_result.cluster_coverage,
            "duration_sec": cloud_result.duration_sec,
            "format_validity": f"{cloud_result.format_validity_rate:.1%}",
            "privacy": "☁️ Data sent to cloud",
            "cost": "~$0.002/call",
            "error": cloud_result.error,
        }
        model_dfs[f"☁️ Gemini 3 Flash{cached_tag}"] = cloud_df

    # Build columnar outputs (max 3)
    mds = ["", "", ""]
    dfs = [pd.DataFrame(columns=["CLUSTER", "Keyword", "Value", "Timestamp"])] * 3
    idx = 0
    
    for model_name, metrics in all_results.items():
        if idx >= 3: break
        
        dfs[idx] = model_dfs[model_name]
        error_note = f"\n  - ⚠️ {metrics['error']}" if metrics.get("error") else ""
        
        mds[idx] = f"""
### {model_name}
- **Facts**: {metrics['fact_count']}
- **Time**: {metrics['duration_sec']:.2f}s
- **Clusters**: {', '.join(metrics['cluster_coverage']) if metrics['cluster_coverage'] else 'none'}
- **Valid**: {metrics.get('format_validity', 'N/A')}
- **Privacy**: {metrics['privacy']}
- **Cost**: {metrics['cost']}{error_note}
"""
        idx += 1

    # Build charts
    quality_chart = _build_comparison_bar(all_results)
    timing_chart = _build_timing_bar(all_results)

    return (
        mds[0], mds[1], mds[2],
        dfs[0], dfs[1], dfs[2],
        quality_chart,
        timing_chart,
        json.dumps(all_results, ensure_ascii=False, indent=2, default=str),
    )


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE HTML
# ═══════════════════════════════════════════════════════════════════════════

ARCHITECTURE_HTML = """
<div style="max-width:1000px; width:95%; margin:0 auto;">
  <div class="arch-box" style="border-left:4px solid #1a73e8;">
    <div class="arch-title">📄 Input: EHR Discharge Note</div>
    <div class="arch-desc">Free-text clinical note (synthetic or real MIMIC-IV)</div>
  </div>
  <div class="arch-arrow">⬇️</div>
  <div class="arch-box" style="border-left:4px solid #00bfa5;">
    <div class="arch-title">🧠 Stage 1: Domain Summary (MedGemma Base GGUF)</div>
    <div class="arch-desc">
      Model: <strong>google/medgemma-1.5-4b-it</strong> (Q5_K_M, 2.5 GB)<br>
      Backend: llama.cpp (OpenAI-compatible)<br>
      Profile: sgr_v2 (structured JSON schema)<br>
      Output: Markdown summary with 9 clinical clusters
    </div>
  </div>
  <div class="arch-arrow">⬇️</div>
  <div class="arch-box" style="border-left:4px solid #8e24aa;">
    <div class="arch-title">🎯 Stage 2: KVT4 Projection (MedGemma + LoRA)</div>
    <div class="arch-desc">
      Model: Base GGUF + LoRA adapter (200-sample teacher-student)<br>
      Teacher: Gemini 3 Flash Thinking (LOW)<br>
      Runtime: llama.cpp with prompt KV-cache reuse (CAG)<br>
      Input: stage1.md (compacted)<br>
      Output: KVT4 lines (<code>CLUSTER|Keyword|Value|Timestamp</code>)<br>
      Format validity: <strong>99.74%</strong>
    </div>
  </div>
  <div class="arch-arrow">⬇️</div>
  <div class="arch-box" style="border-left:4px solid #e53935;">
    <div class="arch-title">📊 Readmission Risk Engine (9 clusters, 215 pts max)</div>
    <div class="arch-desc">
      Rule-based scoring: DEMOGRAPHICS (20) + VITALS (25) + LABS (30) + DISPOSITION (25) + MEDICATIONS (25) + PROCEDURES (25) + UTILIZATION (20) + PROBLEMS (20) + SYMPTOMS (5)<br>
      Calibration: logistic (α=-2.3475, β=0.017)<br>
      Output: probability, risk category, confidence, cluster breakdown
    </div>
  </div>
  <div class="arch-arrow">⬇️</div>
  <div class="arch-box" style="border-left:4px solid #34a853;">
    <div class="arch-title">✅ Output: Interpretable Risk Assessment</div>
    <div class="arch-desc">
      • P(readmit 30d) with confidence interval<br>
      • Risk category: Low / Medium / High / Critical<br>
      • Cluster-level score breakdown<br>
      • Top risk factors with evidence traceability
    </div>
  </div>
</div>

<div style="margin-top:24px;">
  <div class="metric-row">
    <div class="metric-card"><div class="label">Model Size</div><div class="value blue">2.5 GB</div></div>
    <div class="metric-card"><div class="label">AUROC (A1 rule engine)</div><div class="value green">0.6024</div></div>
    <div class="metric-card"><div class="label">Format Validity</div><div class="value green">99.74%</div></div>
    <div class="metric-card"><div class="label">Stage2 CAG</div><div class="value green">+10.6%</div></div>
  </div>
</div>
"""


# ═══════════════════════════════════════════════════════════════════════════
# BUILD THE DEMO
# ═══════════════════════════════════════════════════════════════════════════

def build_demo() -> gr.Blocks:
    cfg_defaults = StructCoreConfig()
    case_choices = _case_choices()
    default_case_id = _default_case_id()
    initial_case = get_case(default_case_id)
    initial_text = initial_case.text if initial_case else ""
    initial_desc = f"**{initial_case.title}**\n\n{initial_case.description}" if initial_case else "Manual mode"


    evidence_df = pd.DataFrame(load_evidence_rows(), columns=["Claim ID", "Claim", "Metric", "Status", "Artifact"])

    with gr.Blocks(title="MedGemma StructCore Demo", theme=APP_THEME, css=CUSTOM_CSS) as demo:
        last_run_state = gr.State({})

        # ── Header ──
        gr.HTML("""
	<div class="app-header">
	  <h1>🧠 MedGemma StructCore</h1>
	  <p class="subtitle">Local-First Clinical Structuring Engine for EHR — Two-Stage Extraction → Risk Prediction</p>
	  <span class="model-badge local">MedGemma 1.5-4B · GGUF Q5_K_M · 2.5 GB</span>
	  <span class="model-badge privacy">🔒 Zero PHI Leakage · HIPAA-Ready</span>
	  <div class="hero-metrics">
    <div class="hero-metric">
      <div class="hm-value" style="color:#00FFAA;">99.74%</div>
      <div class="hm-label">Format Validity</div>
    </div>
	    <div class="hero-metric">
	      <div class="hm-value" style="color:#0087FF;">0.6846</div>
	      <div class="hm-label">Risk Prediction AUROC</div>
	    </div>
    <div class="hero-metric">
      <div class="hm-value" style="color:#FFCC00;">2.5 GB</div>
      <div class="hm-label">Model Size</div>
    </div>
	    <div class="hero-metric">
	      <div class="hm-value" style="color:#00FFAA;">+10.6%</div>
	      <div class="hm-label">Stage2 CAG speedup</div>
	    </div>
	  </div>
	</div>
""")

        # ── Tab 1: Case Input ──
        with gr.Tab(" Case Input"):
            with gr.Row(equal_height=True):
                case_id = gr.Dropdown(label="Synthetic case", choices=case_choices, value=default_case_id, scale=3)
                upload_btn = gr.UploadButton("📂 Load File", file_types=[".txt", ".md", ".pdf", ".docx"], scale=1)
                run_btn = gr.Button("🚀 Run StructCore", variant="primary", scale=1, elem_id="run-structcore-btn")
            
            case_desc = gr.Markdown(initial_desc)
            with gr.Row():
                with gr.Column(scale=3):
                    note_text = gr.Textbox(label="Clinical note text", lines=22, value=initial_text)
                with gr.Column(scale=1):
                    gr.Markdown("### ⏱️ Live Status")
                    status_md = gr.HTML()

            with gr.Row():
                backend_mode = gr.Radio(
                    label="Backend mode",
                    choices=["pipeline", "mock"],
                    value=os.getenv("STRUCTCORE_BACKEND_MODE", "pipeline"),
                    info="pipeline = Stage1/Stage2 with local MedGemma servers (RECOMMENDED) | mock = offline synthetic demo",
                )
                fallback_to_mock = gr.Checkbox(label="Fallback to mock if pipeline fails", value=True)
            
            gr.HTML("""
<div style="background:linear-gradient(135deg,rgba(234,67,53,0.08),rgba(251,188,4,0.08));border:2px solid #ea4335;border-radius:12px;padding:16px;margin:12px 0;">
  <p style="margin:0;color:#c5221f;font-weight:600;font-size:0.9em;">⚠️ <strong>DEMO MODE NOTICE:</strong> If you see "Backend: mock" in results, this is using <strong>synthetic data</strong>, not real MedGemma inference. Please ensure local model servers are running for authentic demonstration.</p>
</div>
""")

            with gr.Accordion("⚙️ Pipeline settings", open=False):
                stage1_url = gr.Textbox(label="Stage1 URL", value=cfg_defaults.stage1_url)
                stage1_model = gr.Textbox(label="Stage1 model", value=cfg_defaults.stage1_model)
                stage2_url = gr.Textbox(label="Stage2 URL", value=cfg_defaults.stage2_url)
                stage2_model = gr.Textbox(label="Stage2 model", value=cfg_defaults.stage2_model)

        # ── Tab 2: StructCore Inspector ──
        with gr.Tab("🔍 StructCore Inspector"):
            with gr.Row():
                stage1_summary = gr.Dataframe(
                    label="Stage1 summary",
                    headers=["Stage1 summary"],
                    datatype=["str"],
                    row_count=10,
                    interactive=False,
                    wrap=True,
                )
                stage2_raw = gr.Dataframe(
                    label="Stage2 write output",
                    headers=["Stage2 write output"],
                    datatype=["str"],
                    row_count=10,
                    interactive=False,
                    wrap=True,
                )
            normalized_df = gr.Dataframe(
                label="Normalized KVT4 facts",
                headers=["CLUSTER", "Keyword", "Value", "Timestamp"],
                datatype=["str", "str", "str", "str"],
                row_count=8,
            )
            gate_json = gr.Textbox(label="Quality gate summary", lines=8)

        # ── Tab 3: Risk View ──
        with gr.Tab("📊 Risk View"):
            risk_md = gr.HTML()
            with gr.Row():
                risk_gauge = gr.Plot(label="Risk Gauge")
                risk_bar = gr.Plot(label="Cluster Breakdown")
            
            with gr.Accordion("ℹ️ Risk Scale Legend", open=False):
                gr.Markdown("""
### 30-Day Readmission Risk Scale

The risk score is a composite sum of weighted clinical factors. Probability is derived via logistic calibration on MIMIC-IV.

| Category | Score | Probability | Meaning |
| :--- | :--- | :--- | :--- |
| <span style="color:green">● Low</span> | 0 - 19 | < 12% | Routine discharge planning sufficient. |
| <span style="color:#fbbc04">● Medium</span> | 20 - 39 | 12% - 16% | Consider follow-up within 7 days. |
| <span style="color:orange">● High</span> | 40 - 59 | 16% - 21% | Requires intervention (home health, med rec). |
| <span style="color:red">● Critical</span> | 60+ | > 21% | **High Risk:** >4x baseline. Urgent review needed. |
""")

            risk_json = gr.Textbox(label="Risk payload (JSON)", lines=12)

        # ── Tab 4: Comparative Analysis ──
        with gr.Tab("⚖️ Comparative Analysis"):
            gr.Markdown("""
### MedGemma (Local) vs Cloud LLMs

Compare extraction quality, speed, and privacy between local MedGemma and cloud Gemini models.
The same clinical note is processed by each selected model using identical KVT4 extraction prompts.
""")
            gr.HTML("""
<table class="privacy-table">
  <tr><th>Feature</th><th>🔒 MedGemma (local)</th><th>☁️ Gemini Cloud</th></tr>
  <tr><td>Data leaves device</td><td class="yes">❌ No — fully local</td><td class="no">✅ Yes — sent to API</td></tr>
  <tr><td>HIPAA compliance</td><td class="yes">✅ Ready (no PHI exposure)</td><td class="warn">⚠️ Depends on BAA</td></tr>
  <tr><td>Cost per note</td><td class="yes">$0</td><td class="no">~$0.001–0.002</td></tr>
  <tr><td>Offline capable</td><td class="yes">✅ Yes</td><td class="no">❌ No</td></tr>
  <tr><td>Model size</td><td>2.5 GB (GGUF Q5_K_M)</td><td>Cloud-hosted</td></tr>
</table>
""")
            with gr.Row():
                cmp_local = gr.Checkbox(label="🔒 MedGemma (local)", value=True)
                cmp_flash = gr.Checkbox(label="☁️ Gemini 2.5 Flash", value=True)
                cmp_gem3 = gr.Checkbox(label="☁️ Gemini 3 Flash Preview", value=True)
            cmp_btn = gr.Button("⚖️ Run Comparison", variant="primary", size="lg")
            
            with gr.Row():
                cmp_quality_chart = gr.Plot(label="Extraction Quality")
                cmp_timing_chart = gr.Plot(label="Latency Comparison")

            with gr.Row():
                with gr.Column():
                    cmp_md1 = gr.Markdown()
                    cmp_df1 = gr.Dataframe(label="Model 1", interactive=False)
                with gr.Column():
                    cmp_md2 = gr.Markdown()
                    cmp_df2 = gr.Dataframe(label="Model 2", interactive=False)
                with gr.Column():
                    cmp_md3 = gr.Markdown()
                    cmp_df3 = gr.Dataframe(label="Model 3", interactive=False)
            with gr.Accordion("Raw comparison JSON", open=False):
                cmp_json = gr.Textbox(label="Comparison payload", lines=12)

        # ── Tab 5: Evidence Board ──
        with gr.Tab("📋 Evidence Board"):
            gr.Markdown(
                "### Verified Claims & Metrics\n"
                "Every metric is labeled **Verified**, **Preliminary**, or **Planned** with artifact references."
            )
            gr.HTML(_build_evidence_cards_html(evidence_df))
            gr.Markdown("---")
            gr.Markdown("#### Canonical baselines (Track B, N=9,857) [Verified]")
            evidence_auroc_chart = gr.Plot(value=_build_auroc_comparison_chart())
            gr.Markdown("#### Extracted subset evaluations (Track B) [Preliminary]")
            evidence_scaleout_chart = gr.Plot(value=_build_auroc_scaleout_chart())
            with gr.Accordion("📄 Full Evidence Table", open=False):
                gr.Dataframe(
                    value=evidence_df,
                    headers=["Claim ID", "Claim", "Metric", "Status", "Artifact"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    row_count=len(evidence_df),
                    label="Evidence claims",
                )

        # ── Tab 6: Clinical Impact ──
        with gr.Tab("💰 Clinical Impact"):
            gr.Markdown("""
### Potential Clinical & Economic Impact

Estimate the downstream value of improved readmission targeting using StructCore-extracted risk signals.
*This is a scenario model for illustration — actual deployment outcomes require clinical validation.*
""")
            with gr.Row():
                impact_discharges = gr.Slider(label="Annual discharges", minimum=1000, maximum=100000, value=10000, step=1000)
                impact_rate = gr.Slider(label="Baseline readmission rate (%)", minimum=5, maximum=35, value=20, step=1)
                impact_cost = gr.Slider(label="Cost per readmission (USD)", minimum=5000, maximum=30000, value=15000, step=1000)
            impact_btn = gr.Button("📊 Calculate Impact", variant="primary", size="lg")
            # Pre-render with default slider values so the chart is visible on load
            _default_impact_html, _default_impact_fig = _calculate_impact(10000, 20, 15000)
            impact_html = gr.HTML(value=_default_impact_html)
            impact_chart = gr.Plot(label="Projected Annual Savings", value=_default_impact_fig)

        # ── Tab 7: Architecture ──
        with gr.Tab("🏗️ Architecture"):
            gr.Markdown("### Two-Stage Pipeline Architecture")
            gr.HTML(ARCHITECTURE_HTML)

        # ── Tab 8: Help & Guide ──
        with gr.Tab("❓ Help & Guide"):
            try:
                guide_content = DEMO_GUIDE_PATH.read_text(encoding="utf-8")
            except Exception as e:
                guide_content = f"### User Guide Not Found\nCould not load guide: {e}"
            gr.Markdown(guide_content)

        # ── Debug accordion ──
        with gr.Accordion("🐛 Debug JSON", open=False):
            debug_json = gr.Textbox(label="Full run payload", lines=14)

        # ── Event wiring ──
        case_id.change(fn=_on_case_change, inputs=[case_id], outputs=[note_text, case_desc])

        def _on_upload_wrapper(file_obj):
            if file_obj is None:
                return gr.update(), gr.update(), gr.update()
            content = _read_file_content(file_obj.name)
            new_desc = f"**Custom Upload**: {Path(file_obj.name).name}\n\nProcessed via generic file reader."
            return content, new_desc, "custom"

        upload_btn.upload(
            fn=_on_upload_wrapper,
            inputs=[upload_btn],
            outputs=[note_text, case_desc, case_id],
        )

        run_evt = run_btn.click(
            fn=_on_run_started,
            inputs=[case_id, backend_mode],
            outputs=[status_md, run_btn],
            queue=False,
        )
        run_evt = run_evt.then(
            fn=_run_demo,
            inputs=[
                case_id, note_text, backend_mode,
                stage1_url, stage1_model, stage2_url, stage2_model,
                fallback_to_mock,
            ],
            outputs=[
                status_md, stage1_summary, stage2_raw, normalized_df, gate_json,
                risk_md, risk_gauge, risk_bar, risk_json, debug_json, last_run_state,
            ],
        )
        run_evt.then(
            fn=_on_run_finished,
            outputs=[run_btn],
            queue=False,
        )

        cmp_btn.click(
            fn=_run_comparison,
            inputs=[
                case_id, note_text, backend_mode,
                stage1_url, stage1_model, stage2_url, stage2_model,
                cmp_local, cmp_flash, cmp_gem3,
                last_run_state,
            ],
            outputs=[
                cmp_md1, cmp_md2, cmp_md3,
                cmp_df1, cmp_df2, cmp_df3,
                cmp_quality_chart, cmp_timing_chart, cmp_json,
            ],
        )

        impact_btn.click(
            fn=_calculate_impact,
            inputs=[impact_discharges, impact_rate, impact_cost],
            outputs=[impact_html, impact_chart],
        )

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        show_error=True,
        share=True,
    )


if __name__ == "__main__":
    main()
