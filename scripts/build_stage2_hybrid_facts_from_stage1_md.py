#!/usr/bin/env python3
"""Build a hybrid Stage2 facts file by combining:

1) deterministic sanitization of Stage2 raw outputs (via shared KVT parser + sanitizer), and
2) deterministic extraction of supplements from Stage1.md (MEDICATIONS/PROBLEMS, optional VITALS/LABS).

Motivation:
- Stage2 sometimes emits JSON/fenced/drift outputs (including 3-part lines) that need robust deterministic recovery.
- Stage2 may omit high-signal fields that Stage1.md already contains.
- This script provides a reproducible offline post-process step to recover those facts
  without re-running any LLM.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running as a script from anywhere without requiring `PYTHONPATH=.`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_two_stage_structured_pipeline import _sanitize_stage2_lines  # noqa: E402
from kvt_utils import extract_kvt_fact_lines  # noqa: E402


_TEXT_PLACEHOLDERS = {"not stated", "...", "<not stated>", "___", "__"}
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_CANON_VITALS = [
    "Heart Rate",
    "Systolic BP",
    "Diastolic BP",
    "Respiratory Rate",
    "Temperature",
    "SpO2",
    "Weight",
]
_CANON_LABS = [
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


def _load_hadm_ids(src_out_dir: Path, hadm_ids_json: Optional[Path]) -> List[int]:
    if hadm_ids_json is not None:
        obj = json.loads(hadm_ids_json.read_text(encoding="utf-8"))
        return [int(x) for x in obj]
    default = src_out_dir / "hadm_ids.json"
    if default.exists():
        obj = json.loads(default.read_text(encoding="utf-8"))
        return [int(x) for x in obj]
    ids: List[int] = []
    for p in sorted(src_out_dir.iterdir()):
        if p.is_dir() and p.name.isdigit():
            ids.append(int(p.name))
    return ids


def _parse_stage1_md_sections(md: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    cur: Optional[str] = None
    buf: List[str] = []
    for ln in md.splitlines():
        if ln.startswith("## "):
            if cur is not None:
                sections[cur] = "\n".join(buf).strip()
            cur = ln[3:].strip().upper()
            buf = []
        else:
            buf.append(ln)
    if cur is not None:
        sections[cur] = "\n".join(buf).strip()
    return sections


def _split_items(raw: str, *, limit: int) -> List[str]:
    s = (raw or "").strip()
    if not s:
        return []
    if s.casefold() in _TEXT_PLACEHOLDERS:
        return []
    out: List[str] = []
    # Split by semicolon/newline, then comma.
    for seg in re.split(r"[;\n]+", s):
        seg = seg.strip()
        if not seg:
            continue
        for item in seg.split(","):
            it = " ".join(item.strip().split())
            if not it:
                continue
            out.append(it)
            if len(out) >= limit:
                return out
    return out


def _extract_stage1_medication_lines(block: str) -> List[str]:
    allowed = {
        "Medication Count",
        "New Medications Count",
        "Polypharmacy",
        "Anticoagulation",
        "Insulin Therapy",
        "Opioid Therapy",
        "Diuretic Therapy",
    }
    out: List[str] = []
    for ln in (block or "").splitlines():
        ln = ln.strip()
        if not ln or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k not in allowed:
            continue
        if v.casefold() in _TEXT_PLACEHOLDERS:
            continue
        out.append(f"MEDICATIONS|{k}|{v}|Admission")
    return out


def _extract_stage1_problem_lines(block: str) -> List[str]:
    # Stage1.md aggregates
    # - Discharge Dx => acute|Discharge
    # - Working Dx => exist|Discharge
    # - Complications => acute|Discharge
    # - PMH/Comorbidities => chronic|Past
    key_map: Dict[str, Tuple[str, str, int]] = {
        "Discharge Dx": ("acute", "Discharge", 12),
        "Working Dx": ("exist", "Discharge", 12),
        "Complications": ("acute", "Discharge", 12),
        "PMH/Comorbidities": ("chronic", "Past", 12),
        "PMH": ("chronic", "Past", 12),
    }
    out: List[str] = []
    for ln in (block or "").splitlines():
        ln = ln.strip()
        if not ln or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        k = k.strip()
        v = v.strip()
        km = key_map.get(k)
        if not km:
            continue
        status, ts, limit = km
        for it in _split_items(v, limit=limit):
            out.append(f"PROBLEMS|{it}|{status}|{ts}")
    return out


def _extract_numeric_value_for_keyword(keyword: str, raw_value: str) -> Optional[str]:
    s = (raw_value or "").strip()
    if not s:
        return None
    if s.casefold() in _TEXT_PLACEHOLDERS:
        return None
    nums = _NUM_RE.findall(s)
    if not nums:
        return None
    if keyword == "Diastolic BP" and "/" in s and len(nums) >= 2:
        return nums[1]
    return nums[0]


def _extract_stage1_objective_lines(block: str, *, cluster: str, ts_policy: str) -> List[str]:
    if cluster == "VITALS":
        ordered_keys = list(_CANON_VITALS)
    elif cluster == "LABS":
        ordered_keys = list(_CANON_LABS)
    else:
        return []
    allowed = set(ordered_keys)
    by_key: Dict[str, Dict[str, str]] = {k: {} for k in ordered_keys}
    for ln in (block or "").splitlines():
        src = ln.strip()
        if not src:
            continue
        ts = "Unknown"
        src_upper = src.upper()
        if src_upper.startswith("ADM:"):
            ts = "Admission"
            src = src[4:].strip()
        elif src_upper.startswith("DC:"):
            ts = "Discharge"
            src = src[3:].strip()
        elif src_upper.startswith("ADMISSION:"):
            ts = "Admission"
            src = src[len("ADMISSION:") :].strip()
        elif src_upper.startswith("DISCHARGE:"):
            ts = "Discharge"
            src = src[len("DISCHARGE:") :].strip()

        for seg in re.split(r"[;\n]+", src):
            part = seg.strip()
            if not part or "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = " ".join(key.strip().split())
            if key not in allowed:
                continue
            num = _extract_numeric_value_for_keyword(key, value.strip())
            if num is None:
                continue
            by_key[key][ts] = num

    out: List[str] = []
    for key in ordered_keys:
        ts_map = by_key.get(key, {})
        if not ts_map:
            continue
        if ts_policy == "both":
            for ts in ("Admission", "Discharge", "Unknown"):
                v = ts_map.get(ts)
                if v is not None:
                    out.append(f"{cluster}|{key}|{v}|{ts}")
            continue
        if ts_policy == "discharge":
            prefs = ("Discharge", "Admission", "Unknown")
        else:
            prefs = ("Admission", "Discharge", "Unknown")
        for ts in prefs:
            v = ts_map.get(ts)
            if v is not None:
                out.append(f"{cluster}|{key}|{v}|{ts}")
                break
    return out


def _line_ck(line: str) -> Optional[Tuple[str, str]]:
    parts = [x.strip() for x in (line or "").split("|")]
    if len(parts) != 4:
        return None
    c, k, _v, _t = parts
    if not c or not k:
        return None
    return c, k


def _extract_stage2_lines_from_raw(path: Path, *, scope: str) -> List[str]:
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    extracted = extract_kvt_fact_lines(raw_text)
    return _sanitize_stage2_lines(extracted, scope=scope)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-out-dir", required=True, help="Existing two-stage output dir with per-doc stage1.md and stage2_raw.txt")
    ap.add_argument("--dst-out-dir", required=True, help="Output dir to write hybrid facts into (<hadm>/stage2_facts_hybrid.txt)")
    ap.add_argument("--hadm-ids-json", default="", help="Optional hadm_ids.json override")
    ap.add_argument("--scope", default="all", choices=["objective", "all"])
    ap.add_argument("--facts-out-name", default="stage2_facts_hybrid.txt")
    ap.add_argument("--recover-missing-timestamp", action="store_true", help="Enable Stage2 3-part recovery for missing Timestamp")
    ap.add_argument("--from-stage2-raw", action="store_true", help="Use stage2_raw.txt as base (sanitized deterministically)")
    ap.add_argument("--supplement-medications", action="store_true")
    ap.add_argument("--supplement-problems", action="store_true")
    ap.add_argument(
        "--supplement-objective",
        action="store_true",
        help="Supplement missing VITALS/LABS from stage1.md objective sections (ADM/DC).",
    )
    ap.add_argument(
        "--objective-ts-policy",
        default="admission",
        choices=["admission", "discharge", "both"],
        help="Timestamp policy when supplementing objective keys from stage1.md.",
    )
    ap.add_argument("--summary-json", default="", help="Optional summary JSON path.")
    args = ap.parse_args()

    src_out_dir = Path(args.src_out_dir).expanduser().resolve()
    dst_out_dir = Path(args.dst_out_dir).expanduser().resolve()
    dst_out_dir.mkdir(parents=True, exist_ok=True)

    hadm_ids_json = Path(args.hadm_ids_json).expanduser().resolve() if str(args.hadm_ids_json).strip() else None
    hadm_ids = _load_hadm_ids(src_out_dir, hadm_ids_json)

    # Mirror hadm_ids.json for downstream evaluators.
    (dst_out_dir / "hadm_ids.json").write_text(json.dumps(hadm_ids, indent=2) + "\n", encoding="utf-8")

    # Configure sanitizer behavior via env (as used by run_two_stage_structured_pipeline.py).
    old_env = dict(os.environ)
    summary = {
        "src_out_dir": str(src_out_dir),
        "dst_out_dir": str(dst_out_dir),
        "n_hadm": len(hadm_ids),
        "n_stage2_raw_base_docs": 0,
        "n_stage2_facts_base_docs": 0,
        "n_docs_with_stage1_md": 0,
        "n_supplemented_objective_keys": 0,
        "n_supplemented_medprob_lines": 0,
        "n_supplement_skipped_existing_key": 0,
    }
    try:
        if args.recover_missing_timestamp:
            os.environ["MEDGEMMA_STAGE2_RECOVER_3PART_LINES"] = "1"

        for hid in hadm_ids:
            src_per = src_out_dir / str(hid)
            dst_per = dst_out_dir / str(hid)
            dst_per.mkdir(parents=True, exist_ok=True)

            base_lines: List[str] = []
            if args.from_stage2_raw:
                raw_path = src_per / "stage2_raw.txt"
                if raw_path.exists():
                    base_lines = _extract_stage2_lines_from_raw(raw_path, scope=str(args.scope))
                    summary["n_stage2_raw_base_docs"] += 1
            else:
                facts_path = src_per / "stage2_facts.txt"
                if facts_path.exists():
                    base_lines = [ln.strip() for ln in facts_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
                    summary["n_stage2_facts_base_docs"] += 1

            # Stage1 supplements
            add_lines: List[str] = []  # MEDICATIONS/PROBLEMS supplements
            add_objective_lines: List[str] = []  # VITALS/LABS supplements
            stage1_md_path = src_per / "stage1.md"
            if stage1_md_path.exists() and (
                args.supplement_medications
                or args.supplement_problems
                or args.supplement_objective
            ):
                summary["n_docs_with_stage1_md"] += 1
                sec = _parse_stage1_md_sections(stage1_md_path.read_text(encoding="utf-8", errors="ignore"))
                if args.supplement_medications:
                    add_lines.extend(_extract_stage1_medication_lines(sec.get("MEDICATIONS", "")))
                if args.supplement_problems:
                    add_lines.extend(_extract_stage1_problem_lines(sec.get("PROBLEMS", "")))
                if args.supplement_objective:
                    add_objective_lines.extend(
                        _extract_stage1_objective_lines(
                            sec.get("VITALS", ""),
                            cluster="VITALS",
                            ts_policy=str(args.objective_ts_policy),
                        )
                    )
                    add_objective_lines.extend(
                        _extract_stage1_objective_lines(
                            sec.get("LABS", ""),
                            cluster="LABS",
                            ts_policy=str(args.objective_ts_policy),
                        )
                    )

            # Merge:
            # - MEDICATIONS/PROBLEMS: dedup by exact line.
            # - Objective supplements: add only if (CLUSTER, Keyword) is currently missing.
            seen = set(base_lines)
            seen_ck = {_line_ck(ln) for ln in base_lines}
            out_lines = list(base_lines)
            for ln in add_lines:
                if ln not in seen:
                    out_lines.append(ln)
                    seen.add(ln)
                    ck = _line_ck(ln)
                    if ck is not None:
                        seen_ck.add(ck)
                    summary["n_supplemented_medprob_lines"] += 1
            for ln in add_objective_lines:
                ck = _line_ck(ln)
                if ck is None:
                    continue
                if ck in seen_ck:
                    summary["n_supplement_skipped_existing_key"] += 1
                    continue
                if ln not in seen:
                    out_lines.append(ln)
                    seen.add(ln)
                seen_ck.add(ck)
                summary["n_supplemented_objective_keys"] += 1

            out_lines.sort(key=lambda s: (s.split("|", 1)[0], s.split("|", 2)[1]))
            (dst_per / str(args.facts_out_name)).write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    if str(args.summary_json).strip():
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
