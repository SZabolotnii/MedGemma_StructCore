#!/usr/bin/env python3
"""
Sequential per-document runner for two-stage structured extraction.

Operational intent:
  - Stage 1: BASE GGUF (no LoRA)
  - Stage 2: BASE GGUF + LoRA adapter (scope=all by default)
  - Process documents one-by-one (resume-friendly, low-memory friendly)

This script orchestrates existing `run_two_stage_structured_pipeline.py`
in single-document mode for each hadm_id.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_two_stage_structured_pipeline.py"


def _pick_python(python_override: str | None) -> str:
    if python_override and python_override.strip():
        return python_override.strip()

    for cand in (
        REPO_ROOT / "venv" / "bin" / "python3",
        REPO_ROOT / "venv" / "bin" / "python",
    ):
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)

    return sys.executable


def _run(cmd: List[str], dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _env_truthy(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _post_json(url: str, path: str, obj: object, *, timeout: float = 10.0) -> None:
    base = (url or "").rstrip("/")
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(obj).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
        resp.read()


def _maybe_set_lora(
    *,
    control_url: str,
    adapter_id: int,
    scale: float,
    timeout: float,
    last_scale: float | None,
) -> float | None:
    """
    Best-effort LoRA control for llama-server:
    - If scale <= 0: disable all adapters (POST [] to /lora-adapters)
    - Else: enable a specific adapter id with given scale
    """
    if not control_url:
        return last_scale

    want = float(scale)
    if last_scale is not None and abs(last_scale - want) < 1e-9:
        return last_scale

    if want <= 0.0:
        body = []
    else:
        body = [{"id": int(adapter_id), "scale": want}]
    _post_json(control_url, "/lora-adapters", body, timeout=float(timeout))
    return want


def _parse_hadm_ids_from_labels_csv(path: Path) -> List[int]:
    ids: List[int] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "hadm_id" not in (reader.fieldnames or []):
            raise SystemExit(f"labels csv must contain `hadm_id`: {path}")
        for row in reader:
            raw = str(row.get("hadm_id", "")).strip()
            if raw.isdigit():
                ids.append(int(raw))
    if not ids:
        raise SystemExit(f"no hadm_id values found in: {path}")
    return ids


def _discover_hadm_ids_ehr_only(cohort_root: Path, n: int) -> List[int]:
    ids: List[int] = []
    for p in sorted(cohort_root.iterdir()):
        if not p.is_dir() or not p.name.isdigit():
            continue
        hadm = int(p.name)
        ehr = p / f"ehr_{hadm}.txt"
        if ehr.exists():
            ids.append(hadm)
        if n > 0 and len(ids) >= n:
            break
    return ids


def _unique_keep_order(items: Iterable[int]) -> List[int]:
    seen: set[int] = set()
    out: List[int] = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _safe_load_hadm_ids_json(path: Path) -> List[int]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    out: List[int] = []
    if not isinstance(obj, list):
        return out
    for x in obj:
        s = str(x).strip()
        if s.isdigit():
            out.append(int(s))
    return _unique_keep_order(out)


def _as_csv_ints(raw: str) -> List[int]:
    out: List[int] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        if not t.isdigit():
            raise SystemExit(f"invalid hadm id: {t}")
        out.append(int(t))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort-root", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--hadm-ids", type=str, default="", help="Comma-separated hadm_ids.")
    p.add_argument("--labels-csv", type=str, default="", help="Optional labels csv with hadm_id column.")
    p.add_argument("--num-docs", type=int, default=0, help="Used only for auto-discovery; 0 means all.")
    p.add_argument("--python", type=str, default="", help="Python interpreter for pipeline calls.")

    p.add_argument(
        "--stage1-url",
        type=str,
        default=os.getenv("OPENAI_COMPAT_URL", os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234")),
    )
    p.add_argument(
        "--stage1-model",
        type=str,
        default=os.getenv("OPENAI_COMPAT_MODEL_STAGE1", os.getenv("LMSTUDIO_MODEL", "medgemma-base-q5_k_m")),
    )
    p.add_argument(
        "--stage1-profile",
        type=str,
        default=os.getenv("STAGE1_PROFILE", "sgr_v2"),
        choices=["strings_v1", "sgr_v1", "sgr_v2", "sgr_v2_compact", "sgr_v3", "sgr_v4"],
    )
    p.add_argument("--schema-path", type=str, default="schemas/readmission_domain_summary.schema.json")
    p.add_argument("--stage1-max-tokens", type=int, default=int(os.getenv("STAGE1_MAX_TOKENS", "1536")))
    p.add_argument("--stage1-temperature", type=float, default=0.0)

    p.add_argument(
        "--stage2-url",
        type=str,
        default=os.getenv("OPENAI_COMPAT_URL", os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234")),
    )
    p.add_argument(
        "--stage2-model",
        type=str,
        default=os.getenv("OPENAI_COMPAT_MODEL_STAGE2", "medgemma-ft-lora-adapters-q5_k_m"),
    )
    p.add_argument("--stage2-max-tokens", type=int, default=768)
    p.add_argument("--stage2-temperature", type=float, default=0.0)
    p.add_argument("--stage2-repetition-penalty", type=float, default=None,
                    help="Override Stage2 repetition penalty (default: 1.10 for scope=all).")
    p.add_argument("--stage2-scope", type=str, default="all", choices=["objective", "all"])
    p.add_argument("--overwrite-stage1", action="store_true")
    p.add_argument("--overwrite-stage2", action="store_true")

    # Optional llama-server LoRA control for single-backend runs.
    # When set, this script toggles adapters via POST /lora-adapters:
    # - before stage1: scale = --lora-stage1-scale (default 0.0, base-only)
    # - before stage2: scale = --lora-stage2-scale (default 1.0, base+LoRA)
    p.add_argument("--lora-control-url", type=str, default="")
    p.add_argument("--lora-adapter-id", type=int, default=0)
    p.add_argument("--lora-stage1-scale", type=float, default=0.0)
    p.add_argument("--lora-stage2-scale", type=float, default=1.0)
    p.add_argument("--lora-timeout", type=float, default=10.0)

    p.add_argument("--sleep-seconds", type=float, default=0.0)
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    py = _pick_python(args.python)
    cohort_root = Path(args.cohort_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_profile = str(args.stage1_profile).strip() or "sgr_v2"
    allow_sgr_v4 = _env_truthy("MEDGEMMA_ALLOW_SGR_V4_STAGE1", "0")
    if stage1_profile == "sgr_v4" and not allow_sgr_v4:
        print("[guardrail] Stage1 profile sgr_v4 is disabled for production; forcing sgr_v2", flush=True)
        stage1_profile = "sgr_v2"

    stage1_schema_path = str(args.schema_path).strip()
    if (
        not allow_sgr_v4
        and stage1_profile == "sgr_v2"
        and stage1_schema_path.endswith("readmission_domain_summary_sgr_v4.schema.json")
    ):
        print("[guardrail] Stage1 schema sgr_v4 is disabled for production; forcing sgr_v2 schema", flush=True)
        stage1_schema_path = "schemas/readmission_domain_summary_sgr_v2.schema.json"

    if stage1_profile == "sgr_v1" and stage1_schema_path.strip() == "schemas/readmission_domain_summary.schema.json":
        stage1_schema_path = "schemas/readmission_domain_summary_sgr_v1.schema.json"
    if stage1_profile in ("sgr_v2", "sgr_v2_compact") and stage1_schema_path.strip() == "schemas/readmission_domain_summary.schema.json":
        stage1_schema_path = "schemas/readmission_domain_summary_sgr_v2.schema.json"
    if stage1_profile == "sgr_v3" and stage1_schema_path.strip() == "schemas/readmission_domain_summary.schema.json":
        stage1_schema_path = "schemas/readmission_domain_summary_sgr_v3.schema.json"
    if stage1_profile == "sgr_v4" and stage1_schema_path.strip() == "schemas/readmission_domain_summary.schema.json":
        stage1_schema_path = "schemas/readmission_domain_summary_sgr_v4.schema.json"

    hadm_ids: List[int] = []
    if args.hadm_ids.strip():
        hadm_ids.extend(_as_csv_ints(args.hadm_ids))
    if args.labels_csv.strip():
        hadm_ids.extend(_parse_hadm_ids_from_labels_csv(Path(args.labels_csv).expanduser().resolve()))
    if not hadm_ids:
        hadm_ids = _discover_hadm_ids_ehr_only(cohort_root, int(args.num_docs))
    hadm_ids = _unique_keep_order(hadm_ids)
    if not hadm_ids:
        raise SystemExit("no hadm_ids to process")

    # IMPORTANT: run_two_stage_structured_pipeline.py overwrites <out_dir>/hadm_ids.json on each call
    # (often with a single hadm_id). Keep a stable copy and also re-write hadm_ids.json after each doc.
    #
    # If out_dir already has a broader hadm_ids.json (e.g. full cohort), keep that broader list
    # to avoid accidentally shrinking evaluation scope during subset reruns.
    hadm_ids_path = out_dir / "hadm_ids.json"
    existing_hadm_ids = _safe_load_hadm_ids_json(hadm_ids_path)
    if existing_hadm_ids and set(hadm_ids).issubset(set(existing_hadm_ids)):
        hadm_ids_stable = existing_hadm_ids
    else:
        hadm_ids_stable = hadm_ids
    (out_dir / "hadm_ids_sequential.json").write_text(json.dumps(hadm_ids, indent=2), encoding="utf-8")
    (out_dir / "hadm_ids_stable.json").write_text(json.dumps(hadm_ids_stable, indent=2), encoding="utf-8")
    hadm_ids_path.write_text(json.dumps(hadm_ids_stable, indent=2), encoding="utf-8")

    ok = 0
    skipped = 0
    failed = 0
    lora_scale: float | None = None

    timings_csv = out_dir / "timings_two_stage_sequential.csv"
    timings_rows: List[dict] = []

    for idx, hadm in enumerate(hadm_ids, 1):
        per_dir = out_dir / str(hadm)
        stage1_md = per_dir / "stage1.md"
        stage2_facts = per_dir / "stage2_facts.txt"

        if stage2_facts.exists() and not args.overwrite_stage2:
            skipped += 1
            print(f"[{idx}/{len(hadm_ids)}] HADM {hadm} | skip (stage2 exists)", flush=True)
            timings_rows.append(
                {
                    "hadm_id": hadm,
                    "status": "skipped",
                    "stage1_s": "",
                    "stage2_s": "",
                    "total_s": "",
                }
            )
            hadm_ids_path.write_text(json.dumps(hadm_ids_stable, indent=2), encoding="utf-8")
            continue

        try:
            t_total0 = time.perf_counter()
            t_stage1: float | None = None
            t_stage2: float | None = None

            if args.overwrite_stage1 or not stage1_md.exists():
                lora_scale = _maybe_set_lora(
                    control_url=str(args.lora_control_url),
                    adapter_id=int(args.lora_adapter_id),
                    scale=float(args.lora_stage1_scale),
                    timeout=float(args.lora_timeout),
                    last_scale=lora_scale,
                )
                stage1_cmd = [
                    py,
                    str(PIPELINE_SCRIPT),
                    "--cohort-root",
                    str(cohort_root),
                    "--out-dir",
                    str(out_dir),
                    "--hadm-ids",
                    str(hadm),
                    "stage1",
                    "--url",
                    str(args.stage1_url),
                    "--model",
                    str(args.stage1_model),
                    "--profile",
                    str(stage1_profile),
                    "--schema-path",
                    str(stage1_schema_path),
                    "--max-tokens",
                    str(int(args.stage1_max_tokens)),
                    "--temperature",
                    str(float(args.stage1_temperature)),
                ]
                t0 = time.perf_counter()
                _run(stage1_cmd, args.dry_run)
                t_stage1 = time.perf_counter() - t0
            else:
                print(f"[{idx}/{len(hadm_ids)}] HADM {hadm} | reuse stage1.md", flush=True)

            lora_scale = _maybe_set_lora(
                control_url=str(args.lora_control_url),
                adapter_id=int(args.lora_adapter_id),
                scale=float(args.lora_stage2_scale),
                timeout=float(args.lora_timeout),
                last_scale=lora_scale,
            )
            stage2_cmd = [
                py,
                str(PIPELINE_SCRIPT),
                "--cohort-root",
                str(cohort_root),
                "--out-dir",
                str(out_dir),
                "--hadm-ids",
                str(hadm),
                "stage2",
                "--url",
                str(args.stage2_url),
                "--model",
                str(args.stage2_model),
                "--max-tokens",
                str(int(args.stage2_max_tokens)),
                "--temperature",
                str(float(args.stage2_temperature)),
                "--scope",
                str(args.stage2_scope),
            ]
            if args.stage2_repetition_penalty is not None:
                stage2_cmd.extend(["--repetition-penalty", str(args.stage2_repetition_penalty)])
            if args.overwrite_stage2:
                stage2_cmd.append("--overwrite-stage2")
            t1 = time.perf_counter()
            _run(stage2_cmd, args.dry_run)
            t_stage2 = time.perf_counter() - t1
            t_total = time.perf_counter() - t_total0

            ok += 1
            print(f"[{idx}/{len(hadm_ids)}] HADM {hadm} | done", flush=True)
            timings_rows.append(
                {
                    "hadm_id": hadm,
                    "status": "ok",
                    "stage1_s": "" if t_stage1 is None else f"{t_stage1:.3f}",
                    "stage2_s": "" if t_stage2 is None else f"{t_stage2:.3f}",
                    "total_s": f"{t_total:.3f}",
                }
            )
            hadm_ids_path.write_text(json.dumps(hadm_ids_stable, indent=2), encoding="utf-8")
            if args.sleep_seconds > 0:
                time.sleep(float(args.sleep_seconds))
        except subprocess.CalledProcessError:
            failed += 1
            print(f"[{idx}/{len(hadm_ids)}] HADM {hadm} | FAILED", flush=True)
            timings_rows.append(
                {
                    "hadm_id": hadm,
                    "status": "failed",
                    "stage1_s": "",
                    "stage2_s": "",
                    "total_s": "",
                }
            )
            hadm_ids_path.write_text(json.dumps(hadm_ids_stable, indent=2), encoding="utf-8")
            if not args.continue_on_error:
                raise

    # Write timings CSV (best-effort, even on partial runs).
    with timings_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["hadm_id", "status", "stage1_s", "stage2_s", "total_s"])
        w.writeheader()
        for row in timings_rows:
            w.writerow(row)

    summary = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": len(hadm_ids),
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "hadm_ids_run_count": len(hadm_ids),
        "hadm_ids_stable_count": len(hadm_ids_stable),
        "cohort_root": str(cohort_root),
        "out_dir": str(out_dir),
        "stage1_model": args.stage1_model,
        "stage1_profile": stage1_profile,
        "stage1_schema_path": stage1_schema_path,
        "stage2_model": args.stage2_model,
        "stage2_scope": args.stage2_scope,
        "lora_control_url": str(args.lora_control_url),
        "lora_adapter_id": int(args.lora_adapter_id),
        "lora_stage1_scale": float(args.lora_stage1_scale),
        "lora_stage2_scale": float(args.lora_stage2_scale),
    }
    (out_dir / "summary_two_stage_sequential.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
