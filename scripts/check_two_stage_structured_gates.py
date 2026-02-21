#!/usr/bin/env python3
"""
Regression gates for the two-stage structured pipeline output directory.

This script is intentionally lightweight so it can run in sandbox without model inference.
It reads artifacts written by scripts/run_two_stage_structured_pipeline.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _has_placeholders(obj: Dict[str, Any]) -> bool:
    s = json.dumps(obj, ensure_ascii=False).casefold()
    return "___" in s


def _dedupe_rate(lines: List[str]) -> Tuple[int, int]:
    raw = [ln.strip() for ln in lines if ln.strip()]
    return len(raw), len(set(raw))


_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_MENTAL_STATUS_ALLOWLIST = {"alert", "confused", "oriented", "lethargic", "not stated"}


def _load_stage1_obj(per_dir: Path) -> Dict[str, Any]:
    norm_path = per_dir / "stage1_normalized.json"
    if norm_path.exists():
        obj = _read_json(norm_path)
        # stage1_normalized.json is a wrapper: {"normalized": {...}, "hygiene_stats": {...}}
        if isinstance(obj, dict) and isinstance(obj.get("normalized"), dict):
            return obj["normalized"]
        if isinstance(obj, dict):
            return obj
    raw_path = per_dir / "stage1.json"
    if raw_path.exists():
        obj = _read_json(raw_path)
        return obj if isinstance(obj, dict) else {}
    return {}


def _get_mental_status(stage1: Dict[str, Any]) -> str:
    disp = str(stage1.get("DISPOSITION", "") or "")
    for ln in disp.splitlines():
        if "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        if k.strip().casefold() == "mental status":
            return v.strip().casefold()
    return ""


def _vitals_labs_numeric_ok(stage1: Dict[str, Any]) -> bool:
    for key in ["VITALS", "LABS"]:
        block = str(stage1.get(key, "") or "")
        if not block:
            continue
        # Hard red flags anywhere.
        # Detect "RA" as a standalone token (avoid false positives like "Respiratory").
        if key == "VITALS" and re.search(r"(?i)\bRA\b", block):
            return False
        if "%" in block:
            return False

        for ln in block.splitlines():
            if ":" in ln:
                _prefix, rest = ln.split(":", 1)
            else:
                rest = ln
            parts = [p.strip() for p in rest.split(";") if p.strip()]
            for p in parts:
                if "=" not in p:
                    continue
                _k, v = p.split("=", 1)
                vv = v.strip().casefold()
                if vv == "not stated":
                    continue
                if not _NUM_RE.match(vv):
                    return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True, help="Output dir produced by run_two_stage_structured_pipeline.py")
    p.add_argument("--min-stage1-json-parse-rate", type=float, default=0.99)
    p.add_argument("--max-stage1-placeholder-rate", type=float, default=0.0)
    p.add_argument("--max-stage1-bad-mental-status-rate", type=float, default=0.0)
    p.add_argument("--max-stage1-nonnumeric-rate", type=float, default=0.0)
    p.add_argument("--min-stage2-parse-success-rate", type=float, default=0.95)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    hadm_ids_path = run_dir / "hadm_ids.json"
    if not hadm_ids_path.exists():
        raise SystemExit(f"Missing hadm_ids.json in: {run_dir}")
    hadm_ids = json.loads(hadm_ids_path.read_text(encoding="utf-8"))

    n = len(hadm_ids)
    stage1_parse_ok = 0
    stage1_placeholders_model = 0
    stage1_placeholders_sanitized = 0
    stage1_placeholders_normalized = 0
    stage1_bad_mental = 0
    stage1_nonnumeric = 0
    stage2_parse_ok = 0
    stage2_dup_bad = 0

    for hadm in hadm_ids:
        per = run_dir / str(hadm)
        meta_path = per / "stage1_meta.json"
        json_path = per / "stage1.json"
        if meta_path.exists():
            meta = _read_json(meta_path)
            if meta.get("json_parse_ok"):
                stage1_parse_ok += 1
        # Placeholders can be removed deterministically before Stage2.
        # Track both:
        # - model drift signal: stage1_raw_model.txt (if present)
        # - effective pipeline hygiene: stage1_raw.txt (sanitized, used downstream)
        raw_model_txt = per / "stage1_raw_model.txt"
        if raw_model_txt.exists():
            s = raw_model_txt.read_text(encoding="utf-8", errors="ignore").casefold()
            if "___" in s:
                stage1_placeholders_model += 1

        raw_txt = per / "stage1_raw.txt"
        if raw_txt.exists():
            s = raw_txt.read_text(encoding="utf-8", errors="ignore").casefold()
            if "___" in s:
                stage1_placeholders_sanitized += 1

        if json_path.exists() or (per / "stage1_normalized.json").exists():
            stage1_obj = _load_stage1_obj(per)
            if _has_placeholders(stage1_obj):
                stage1_placeholders_normalized += 1

            ms = _get_mental_status(stage1_obj)
            if ms and ms not in _MENTAL_STATUS_ALLOWLIST:
                stage1_bad_mental += 1

            if not _vitals_labs_numeric_ok(stage1_obj):
                stage1_nonnumeric += 1

        norm_path = per / "stage2_normalized.json"
        facts_path = per / "stage2_facts.txt"
        if norm_path.exists():
            norm = _read_json(norm_path).get("normalized") or []
            if isinstance(norm, list) and len(norm) > 0:
                stage2_parse_ok += 1
        if facts_path.exists():
            raw_lines = facts_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            total, unique = _dedupe_rate(raw_lines)
            if total and unique < total:
                stage2_dup_bad += 1

    stage1_parse_rate = stage1_parse_ok / n if n else 1.0
    stage1_placeholder_rate_model = stage1_placeholders_model / n if n else 0.0
    stage1_placeholder_rate_sanitized = stage1_placeholders_sanitized / n if n else 0.0
    stage1_placeholder_rate_normalized = stage1_placeholders_normalized / n if n else 0.0
    stage1_bad_mental_rate = stage1_bad_mental / n if n else 0.0
    stage1_nonnumeric_rate = stage1_nonnumeric / n if n else 0.0
    stage2_parse_rate = stage2_parse_ok / n if n else 1.0

    print(f"Docs: {n}")
    print(f"Stage1 JSON parse rate: {stage1_parse_rate:.2%} ({stage1_parse_ok}/{n})")
    print(f"Stage1 model placeholder rate (___ in stage1_raw_model.txt): {stage1_placeholder_rate_model:.2%} ({stage1_placeholders_model}/{n})")
    print(f"Stage1 sanitized placeholder rate (___ in stage1_raw.txt): {stage1_placeholder_rate_sanitized:.2%} ({stage1_placeholders_sanitized}/{n})")
    print(f"Stage1 normalized placeholder rate (___): {stage1_placeholder_rate_normalized:.2%} ({stage1_placeholders_normalized}/{n})")
    print(f"Stage1 bad mental status rate: {stage1_bad_mental_rate:.2%} ({stage1_bad_mental}/{n})")
    print(f"Stage1 non-numeric VITALS/LABS rate: {stage1_nonnumeric_rate:.2%} ({stage1_nonnumeric}/{n})")
    print(f"Stage2 parse success rate (>=1 normalized fact): {stage2_parse_rate:.2%} ({stage2_parse_ok}/{n})")
    print(f"Stage2 raw-duplicate docs: {stage2_dup_bad}/{n}")

    ok = True
    if stage1_parse_rate < float(args.min_stage1_json_parse_rate):
        ok = False
        print(f"FAIL: stage1_parse_rate < {args.min_stage1_json_parse_rate}")
    if stage1_placeholder_rate_sanitized > float(args.max_stage1_placeholder_rate):
        ok = False
        print(f"FAIL: stage1_sanitized_placeholder_rate > {args.max_stage1_placeholder_rate}")
    if stage1_bad_mental_rate > float(args.max_stage1_bad_mental_status_rate):
        ok = False
        print(f"FAIL: stage1_bad_mental_status_rate > {args.max_stage1_bad_mental_status_rate}")
    if stage1_nonnumeric_rate > float(args.max_stage1_nonnumeric_rate):
        ok = False
        print(f"FAIL: stage1_nonnumeric_rate > {args.max_stage1_nonnumeric_rate}")
    if stage2_parse_rate < float(args.min_stage2_parse_success_rate):
        ok = False
        print(f"FAIL: stage2_parse_rate < {args.min_stage2_parse_success_rate}")

    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
