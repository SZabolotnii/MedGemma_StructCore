from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Analysis_Readmission.readmission_risk_engine import ReadmissionRiskEngine
from kvt_utils import extract_kvt_fact_lines, normalize_readmission_kvt4_lines


REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_two_stage_structured_pipeline.py"

VALID_CLUSTERS = {
    "DEMOGRAPHICS",
    "VITALS",
    "LABS",
    "DISPOSITION",
    "MEDICATIONS",
    "PROCEDURES",
    "UTILIZATION",
    "PROBLEMS",
    "SYMPTOMS",
}


@dataclass
class StructCoreConfig:
    backend_mode: str = "pipeline"  # pipeline | mock (pipeline is PRODUCTION default)
    python_executable: str = sys.executable

    stage1_url: str = os.getenv("STRUCTCORE_STAGE1_URL", os.getenv("OPENAI_COMPAT_URL", "http://127.0.0.1:1245"))
    stage1_model: str = os.getenv("STRUCTCORE_STAGE1_MODEL", os.getenv("OPENAI_COMPAT_MODEL_STAGE1", "medgemma-base-q5_k_m"))
    stage1_profile: str = "sgr_v2"
    stage1_max_tokens: int = 1024
    stage1_temperature: float = 0.0

    stage2_url: str = os.getenv("STRUCTCORE_STAGE2_URL", os.getenv("OPENAI_COMPAT_URL", "http://127.0.0.1:1246"))
    stage2_model: str = os.getenv("STRUCTCORE_STAGE2_MODEL", os.getenv("OPENAI_COMPAT_MODEL_STAGE2", "medgemma-ft-lora-adapters-q5_k_m"))
    stage2_scope: str = "all"
    stage2_output_mode: str = "lines"
    stage2_max_tokens: int = 2048
    stage2_temperature: float = 0.0

    fallback_to_mock_on_error: bool = True


@dataclass
class StructCoreResult:
    backend_mode: str
    note_id: str
    stage1_summary: str
    stage2_raw: str
    stage2_lines: List[str]
    normalized_lines: List[str]
    normalization_stats: Dict[str, Any]
    gate_summary: Dict[str, Any]
    risk: Optional[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    duration_sec: float = 0.0


_ENGINE: Optional[ReadmissionRiskEngine] = None


def _get_engine() -> ReadmissionRiskEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ReadmissionRiskEngine()
    return _ENGINE


def run_structcore(note_text: str, note_id: str, cfg: StructCoreConfig) -> StructCoreResult:
    text = (note_text or "").strip()
    if not text:
        return StructCoreResult(
            backend_mode=cfg.backend_mode,
            note_id=note_id,
            stage1_summary="",
            stage2_raw="",
            stage2_lines=[],
            normalized_lines=[],
            normalization_stats={},
            gate_summary={"parse_success": False, "reason": "empty_input"},
            risk=None,
            warnings=["Input note is empty."],
            error="empty_input",
            duration_sec=0.0,
        )

    if cfg.backend_mode == "pipeline":
        try:
            return _run_pipeline_backend(text, note_id, cfg)
        except Exception as exc:  # noqa: BLE001
            if not cfg.fallback_to_mock_on_error:
                return StructCoreResult(
                    backend_mode="pipeline",
                    note_id=note_id,
                    stage1_summary="",
                    stage2_raw="",
                    stage2_lines=[],
                    normalized_lines=[],
                    normalization_stats={},
                    gate_summary={"parse_success": False, "reason": "pipeline_error"},
                    risk=None,
                    warnings=[],
                    error=f"pipeline_error: {exc}",
                    duration_sec=0.0,
                )
            mock = _run_mock_backend(text, note_id)
            mock.backend_mode = "mock (pipeline fallback)"
            mock.warnings.insert(0, f"Pipeline backend failed, fallback enabled: {exc}")
            return mock

    return _run_mock_backend(text, note_id)


def _run_pipeline_backend(note_text: str, note_id: str, cfg: StructCoreConfig) -> StructCoreResult:
    start = time.perf_counter()
    hadm_id = 990001

    with tempfile.TemporaryDirectory(prefix="structcore_demo_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        cohort_root = tmp_dir / "cohort"
        out_dir = tmp_dir / "out"

        hadm_dir = cohort_root / str(hadm_id)
        hadm_dir.mkdir(parents=True, exist_ok=True)
        (hadm_dir / f"ehr_{hadm_id}.txt").write_text(note_text, encoding="utf-8")

        stage1_cmd = [
            cfg.python_executable,
            str(PIPELINE_SCRIPT),
            "--cohort-root",
            str(cohort_root),
            "--out-dir",
            str(out_dir),
            "--hadm-ids",
            str(hadm_id),
            "--num-docs",
            "1",
            "--allow-missing-gt",
            "stage1",
            "--url",
            cfg.stage1_url,
            "--model",
            cfg.stage1_model,
            "--profile",
            cfg.stage1_profile,
            "--max-tokens",
            str(int(cfg.stage1_max_tokens)),
            "--temperature",
            str(float(cfg.stage1_temperature)),
            "--overwrite-stage1",
        ]

        stage2_cmd = [
            cfg.python_executable,
            str(PIPELINE_SCRIPT),
            "--cohort-root",
            str(cohort_root),
            "--out-dir",
            str(out_dir),
            "--hadm-ids",
            str(hadm_id),
            "--num-docs",
            "1",
            "--allow-missing-gt",
            "stage2",
            "--url",
            cfg.stage2_url,
            "--model",
            cfg.stage2_model,
            "--scope",
            cfg.stage2_scope,
            "--output-mode",
            cfg.stage2_output_mode,
            "--max-tokens",
            str(int(cfg.stage2_max_tokens)),
            "--temperature",
            str(float(cfg.stage2_temperature)),
            "--overwrite-stage2",
        ]

        _run_cmd(stage1_cmd)
        _run_cmd(stage2_cmd)

        per_dir = out_dir / str(hadm_id)
        stage1_summary = _read_optional(per_dir / "stage1.md")
        stage1_facts = _read_optional(per_dir / "stage1_facts.txt")  # VITALS + LABS
        stage2_raw = _read_optional(per_dir / "stage2_raw.txt")
        stage2_lines_text = _read_optional(per_dir / "stage2_facts.txt")

        # Merge Stage 1 (VITALS/LABS) + Stage 2 (semantic) facts
        stage1_kvt4 = extract_kvt_fact_lines(stage1_facts) if stage1_facts.strip() else []
        stage2_kvt4 = extract_kvt_fact_lines(stage2_lines_text if stage2_lines_text.strip() else stage2_raw)
        raw_lines = stage1_kvt4 + stage2_kvt4

    normalized_lines, normalization_stats = normalize_readmission_kvt4_lines(raw_lines)
    risk = _score_risk(normalized_lines)
    gate_summary = _build_gate_summary(note_text, stage1_summary, normalized_lines, normalization_stats)

    return StructCoreResult(
        backend_mode="pipeline",
        note_id=note_id,
        stage1_summary=stage1_summary,
        stage2_raw=stage2_raw,
        stage2_lines=raw_lines,
        normalized_lines=normalized_lines,
        normalization_stats=normalization_stats,
        gate_summary=gate_summary,
        risk=risk,
        warnings=[],
        error=None,
        duration_sec=round(time.perf_counter() - start, 3),
    )


def _run_cmd(cmd: List[str]) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        msg = stderr or stdout or f"Command failed with exit code {proc.returncode}"
        raise RuntimeError(msg)


def _read_optional(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _run_mock_backend(note_text: str, note_id: str) -> StructCoreResult:
    start = time.perf_counter()

    stage2_lines = _heuristic_extract_kvt(note_text)
    stage2_raw = "\n".join(stage2_lines)
    stage1_summary = _render_stage1_like_summary(stage2_lines)

    normalized_lines, normalization_stats = normalize_readmission_kvt4_lines(stage2_lines)
    risk = _score_risk(normalized_lines)
    gate_summary = _build_gate_summary(note_text, stage1_summary, normalized_lines, normalization_stats)

    warnings: List[str] = []
    if not normalized_lines:
        warnings.append("No valid KVT4 facts after normalization.")

    return StructCoreResult(
        backend_mode="mock",
        note_id=note_id,
        stage1_summary=stage1_summary,
        stage2_raw=stage2_raw,
        stage2_lines=stage2_lines,
        normalized_lines=normalized_lines,
        normalization_stats=normalization_stats,
        gate_summary=gate_summary,
        risk=risk,
        warnings=warnings,
        error=None,
        duration_sec=round(time.perf_counter() - start, 3),
    )


def _score_risk(normalized_lines: List[str]) -> Optional[Dict[str, Any]]:
    if not normalized_lines:
        return None
    engine = _get_engine()
    result = engine.score_from_toon("\n".join(normalized_lines))
    return asdict(result)


def _build_gate_summary(
    note_text: str,
    stage1_summary: str,
    normalized_lines: List[str],
    normalization_stats: Dict[str, Any],
) -> Dict[str, Any]:
    # Calculate stats
    clusters_found = set()
    cluster_counts = {}
    valid_count = 0
    for line in normalized_lines:
        parts = line.split("|")
        if len(parts) >= 3:
            c = parts[0].strip()
            clusters_found.add(c)
            cluster_counts[c] = cluster_counts.get(c, 0) + 1
            valid_count += 1

    valid_ratio = 0.0
    if len(normalized_lines) > 0:
        valid_ratio = valid_count / len(normalized_lines)

    return {
        "parse_success": bool(normalized_lines),
        "input_char_len": len(note_text),
        "stage1_tokens": len(stage1_summary) // 4,  # Approx
        "lines_extracted": len(normalized_lines),
        "clusters_present": sorted(list(clusters_found)),
        "cluster_counts": cluster_counts,
        "valid_ratio": valid_ratio,
        "all_clusters_valid": all(c in VALID_CLUSTERS for c in clusters_found),
        "duplicates_after_dedup": int(normalization_stats.get("duplicates_after_dedup", 0)) if isinstance(normalization_stats, dict) else 0,
        "canonical_keyword_rate_strict": normalization_stats.get("canonical_keyword_rate_strict") if isinstance(normalization_stats, dict) else None,
        "numeric_only_rate_vitals_labs": normalization_stats.get("numeric_only_rate_vitals_labs") if isinstance(normalization_stats, dict) else None,
    }


def _render_stage1_like_summary(lines: List[str]) -> str:
    grouped: Dict[str, List[Tuple[str, str, str]]] = {}
    for line in lines:
        parts = line.split("|")
        if len(parts) != 4:
            continue
        cluster, key, value, ts = [p.strip() for p in parts]
        grouped.setdefault(cluster.upper(), []).append((key, value, ts))

    ordered_clusters = [
        "DEMOGRAPHICS",
        "VITALS",
        "LABS",
        "DISPOSITION",
        "MEDICATIONS",
        "PROCEDURES",
        "UTILIZATION",
        "PROBLEMS",
        "SYMPTOMS",
    ]

    out: List[str] = []
    for cluster in ordered_clusters:
        items = grouped.get(cluster, [])
        if not items:
            continue
        out.append(f"## {cluster}")
        for key, value, ts in items:
            out.append(f"- {key}={value} ({ts})")
        out.append("")

    return "\n".join(out).strip()


def _heuristic_extract_kvt(note_text: str) -> List[str]:
    text = note_text or ""
    lowered = text.lower()
    lines: List[str] = []
    seen = set()

    def add(cluster: str, keyword: str, value: str, timestamp: str) -> None:
        key = (cluster, keyword)
        if key in seen:
            return
        seen.add(key)
        lines.append(f"{cluster}|{keyword}|{value}|{timestamp}")

    def m1(pattern: str) -> Optional[str]:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        return m.group(1) if m else None

    age = m1(r"\b(\d{1,3})\s*(?:y/o|yo|year-old|years old)\b")
    if age:
        add("DEMOGRAPHICS", "Age", age, "Admission")

    if re.search(r"\bfemale\b", lowered):
        add("DEMOGRAPHICS", "Sex", "female", "Admission")
    elif re.search(r"\bmale\b", lowered):
        add("DEMOGRAPHICS", "Sex", "male", "Admission")

    hr = m1(r"(?:heart\s*rate|\bhr\b|pulse)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)")
    if hr:
        add("VITALS", "Heart Rate", hr, "Admission")

    bp = re.search(r"(?:blood\s*pressure|\bbp\b)\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})", text, flags=re.IGNORECASE)
    if bp:
        add("VITALS", "Systolic BP", bp.group(1), "Admission")
        add("VITALS", "Diastolic BP", bp.group(2), "Admission")

    rr = m1(r"(?:respiratory\s*rate|\brr\b|\bresp\b)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)")
    if rr:
        add("VITALS", "Respiratory Rate", rr, "Admission")

    temp = m1(r"(?:temperature|\btemp\b)\s*[:=]?\s*(\d{2}(?:\.\d+)?)")
    if temp:
        add("VITALS", "Temperature", temp, "Admission")

    spo2 = m1(r"(?:spo2|o2\s*sat|oxygen\s*saturation)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*%?")
    if spo2:
        add("VITALS", "SpO2", spo2, "Admission")

    weight = m1(r"\bweight\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)")
    if weight:
        add("VITALS", "Weight", weight, "Admission")

    lab_patterns = [
        ("Hemoglobin", r"(?:hemoglobin|\bhgb\b)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)"),
        ("Hematocrit", r"(?:hematocrit|\bhct\b)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)"),
        ("WBC", r"\bwbc\b\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)"),
        ("Platelet", r"(?:platelet|\bplt\b)\s*[:=]?\s*(\d{2,4}(?:\.\d+)?)"),
        ("Sodium", r"(?:sodium|\bna\b)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)"),
        ("Potassium", r"(?:potassium|\bk\b)\s*[:=]?\s*(\d(?:\.\d+)?)"),
        ("Creatinine", r"(?:creatinine|\bcr\b)\s*[:=]?\s*(\d(?:\.\d+)?)"),
        ("BUN", r"\bbun\b\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)"),
        ("Glucose", r"\bglucose\b\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)"),
        ("Bicarbonate", r"(?:bicarbonate|\bhco3\b|bicarb)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)"),
    ]
    for keyword, pattern in lab_patterns:
        val = m1(pattern)
        if val:
            add("LABS", keyword, val, "Admission")

    prior_adm = m1(r"(\d+)\s*(?:prior|previous)\s*admissions?\s*(?:in|within)?\s*12\s*months")
    if prior_adm:
        add("UTILIZATION", "Prior Admissions 12mo", prior_adm, "Past")

    ed_visits = m1(r"(\d+)\s*(?:ed|er|emergency)\s*visits?\s*(?:in|within)?\s*(?:last\s*)?6\s*months")
    if ed_visits:
        add("UTILIZATION", "ED Visits 6mo", ed_visits, "Past")

    days_last = m1(r"days\s*since\s*last\s*admission\s*[:=]?\s*(\d+)")
    if days_last:
        add("UTILIZATION", "Days Since Last Admission", days_last, "Past")

    los = m1(r"(?:length\s*of\s*stay|\blos\b)\s*[:=]?\s*(\d+)")
    if los:
        add("UTILIZATION", "Current Length of Stay", los, "Admission")

    if "skilled nursing" in lowered or "snf" in lowered:
        add("DISPOSITION", "Discharge Disposition", "Skilled Nursing Facility", "Discharge")
    elif "home" in lowered:
        add("DISPOSITION", "Discharge Disposition", "Home", "Discharge")

    if re.search(r"confus|disorient", lowered):
        add("DISPOSITION", "Mental Status", "Altered", "Discharge")
    elif re.search(r"alert and oriented|a&o", lowered):
        add("DISPOSITION", "Mental Status", "Normal", "Discharge")

    if re.search(r"warfarin|apixaban|rivaroxaban|heparin|anticoag", lowered):
        add("MEDICATIONS", "Anticoagulation", "yes", "Discharge")
    if re.search(r"insulin", lowered):
        add("MEDICATIONS", "Insulin Therapy", "yes", "Discharge")
    if re.search(r"opioid|morphine|oxycodone|hydromorphone|fentanyl", lowered):
        add("MEDICATIONS", "Opioid Therapy", "yes", "Discharge")
    if re.search(r"diuretic|furosemide|torsemide|bumetanide", lowered):
        add("MEDICATIONS", "Diuretic Therapy", "yes", "Discharge")
    if re.search(r"antidepressant|ssri|sertraline|fluoxetine|escitalopram", lowered):
        add("MEDICATIONS", "Antidepressant", "yes", "Discharge")
    if re.search(r"anxiolytic|benzodiazepine|lorazepam|diazepam", lowered):
        add("MEDICATIONS", "Anxiolytic", "yes", "Discharge")
    if re.search(r"corticosteroid|prednisone|dexamethasone|oral corticosteroid", lowered):
        add("MEDICATIONS", "Corticosteroid", "yes", "Discharge")
    if re.search(r"albuterol|nebulizer|bronchodilator", lowered):
        add("MEDICATIONS", "Bronchodilator", "yes", "Discharge")

    if re.search(r"mechanical ventilation|intubat", lowered):
        add("PROCEDURES", "Mechanical Ventilation", "yes", "Admission")
    if re.search(r"dialysis", lowered):
        add("PROCEDURES", "Dialysis", "yes", "Admission")
    if re.search(r"surgery|operative|operation", lowered):
        add("PROCEDURES", "Surgery", "yes", "Admission")

    problem_terms = {
        "heart failure": "Heart Failure",
        "chf": "Heart Failure",
        "ckd": "Chronic Kidney Disease",
        "copd": "COPD",
        "atrial fibrillation": "Atrial Fibrillation",
        "diabetes": "Diabetes Mellitus",
        "hypertension": "Hypertension",
        "asthma": "Asthma",
        "depression": "Depression",
        "suicidal": "Suicidal Ideation",
    }
    for token, label in problem_terms.items():
        if token in lowered:
            add("PROBLEMS", label, "chronic", "Past")

    symptom_terms = {
        "shortness of breath": "Dyspnea",
        "dyspnea": "Dyspnea",
        "chest pain": "Chest Pain",
        "fever": "Fever",
        "wheezing": "Wheezing",
        "edema": "Edema",
    }
    for token, label in symptom_terms.items():
        if token in lowered:
            add("SYMPTOMS", label, "present", "Admission")

    return lines


def lines_to_rows(lines: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in lines:
        parts = line.split("|")
        if len(parts) != 4:
            continue
        rows.append(
            {
                "CLUSTER": parts[0].strip(),
                "Keyword": parts[1].strip(),
                "Value": parts[2].strip(),
                "Timestamp": parts[3].strip(),
            }
        )
    return rows


def result_to_debug_json(result: StructCoreResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False, indent=2)
