#!/usr/bin/env python3
"""
One-command smoke runner for the two-stage structured pipeline.

Flow (low-memory friendly):
  1) Run Stage1 (BASE, JSON schema) into a fixed output dir
  2) Pause so you can restart your OpenAI-compatible backend with Stage2 weights
  3) Run Stage2 (FT LoRA / merged) reusing the same output dir
  4) Run regression gates (no LLM)

Default target doc: HADM 20629020 (EHR_test_data convention).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: List[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _list_models(url: str) -> Optional[set[str]]:
    try:
        import json
        import urllib.request

        with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=10) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        data = obj.get("data") if isinstance(obj, dict) else None
        if not isinstance(data, list):
            return None
        ids = set()
        for it in data:
            if isinstance(it, dict) and it.get("id"):
                ids.add(str(it["id"]))
        return ids
    except Exception:
        return None


def _pick_python(python_override: str | None) -> str:
    if python_override and python_override.strip():
        return python_override.strip()

    # Prefer local venv if present (common for this repo).
    for cand in (
        REPO_ROOT / "venv" / "bin" / "python3",
        REPO_ROOT / "venv" / "bin" / "python",
    ):
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)

    return sys.executable


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hadm-id", type=int, default=20629020)
    p.add_argument("--cohort-root", type=str, default="EHR_test_data")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--python", type=str, default="", help="Python interpreter to run pipeline scripts (optional).")
    p.add_argument("--url", type=str, default=_env("OPENAI_COMPAT_URL", _env("LMSTUDIO_URL", "http://127.0.0.1:1234")))
    p.add_argument("--stage1-model", type=str, default=_env("OPENAI_COMPAT_MODEL_STAGE1", _env("LMSTUDIO_MODEL", "medgemma-base-q5_k_m")))
    p.add_argument("--stage2-model", type=str, default=_env("OPENAI_COMPAT_MODEL_STAGE2", "medgemma-ft-lora-adapters-q5_k_m"))
    p.add_argument("--schema-path", type=str, default="schemas/readmission_domain_summary.schema.json")
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dry-run", action="store_true", help="Print commands and exit without calling the backend.")
    p.add_argument("--no-prompt", action="store_true", help="Do not wait for Enter between Stage1 and Stage2.")
    p.add_argument("--stop-after-stage1", action="store_true", help="Run only Stage1, then exit 0 with instructions.")
    p.add_argument("--skip-gates", action="store_true", help="Skip check_two_stage_structured_gates.py.")
    args = p.parse_args()

    hadm = int(args.hadm_id)
    cohort_root = args.cohort_root
    out_dir = Path(args.out_dir) if args.out_dir.strip() else (REPO_ROOT / "results" / f"two_stage_smoke_{hadm}")
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    py = _pick_python(args.python)

    common = [
        py,
        str(REPO_ROOT / "scripts" / "run_two_stage_structured_pipeline.py"),
        "--cohort-root",
        str(cohort_root),
        "--hadm-ids",
        str(hadm),
        "--out-dir",
        str(out_dir),
    ]

    stage1_cmd = common + [
        "stage1",
        "--url",
        args.url,
        "--model",
        args.stage1_model,
        "--schema-path",
        args.schema_path,
        "--max-tokens",
        str(int(args.max_tokens)),
        "--temperature",
        str(float(args.temperature)),
    ]
    stage2_cmd = common + [
        "stage2",
        "--url",
        args.url,
        "--model",
        args.stage2_model,
        "--max-tokens",
        str(int(args.max_tokens)),
        "--temperature",
        str(float(args.temperature)),
    ]
    gates_cmd = [
        py,
        str(REPO_ROOT / "scripts" / "check_two_stage_structured_gates.py"),
        "--run-dir",
        str(out_dir),
        "--min-stage1-json-parse-rate",
        "1.0",
        "--max-stage1-placeholder-rate",
        "0.0",
        "--min-stage2-parse-success-rate",
        "1.0",
    ]

    if args.dry_run:
        print("+ " + " ".join(stage1_cmd))
        print("# restart backend for stage2")
        print("+ " + " ".join(stage2_cmd))
        print("+ " + " ".join(gates_cmd))
        return

    try:
        _run(stage1_cmd)
    except subprocess.CalledProcessError as e:
        print("", flush=True)
        print("Stage1 failed.", flush=True)
        print(f"- If this is a dependency issue, try running with the repo venv: `--python venv/bin/python3`.", flush=True)
        raise e

    # If Stage1 produced invalid JSON, stop early (no point running Stage2).
    meta_path = out_dir / str(hadm) / "stage1_meta.json"
    if meta_path.exists():
        try:
            import json

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not bool(meta.get("json_parse_ok")):
                print("", flush=True)
                print("Stage1 JSON parse failed; stopping before Stage2.", flush=True)
                print(f"- Inspect: {out_dir / str(hadm) / 'stage1_raw.txt'}", flush=True)
                retry_path = out_dir / str(hadm) / "stage1_raw_retry1.txt"
                if retry_path.exists():
                    print(f"- Retry output: {retry_path}", flush=True)
                sys.exit(2)
        except Exception:
            pass

    print("", flush=True)
    print("Stage1 done.", flush=True)
    print("Now restart your OpenAI-compatible backend with Stage2 weights (FT merged or base+LoRA).", flush=True)
    print(f"- Stage2 model id to use: {args.stage2_model}", flush=True)
    print(f"- URL: {args.url}", flush=True)
    print("", flush=True)

    if args.stop_after_stage1:
        print(f"Stopping after Stage1 as requested. Re-run with the same --out-dir to continue Stage2: {out_dir}", flush=True)
        return

    if not args.no_prompt:
        try:
            input("Press Enter to run Stage2...")
        except EOFError:
            # Non-interactive environment: continue.
            pass

    model_ids = _list_models(args.url)
    if model_ids is not None and args.stage2_model not in model_ids:
        print("WARNING: Stage2 model id not present in /v1/models; backend may ignore it and use a default model.", flush=True)
        print(f"- Requested: {args.stage2_model}", flush=True)
        print(f"- Available (sample): {', '.join(sorted(list(model_ids))[:6])}", flush=True)
        print("", flush=True)

    try:
        _run(stage2_cmd)
    except subprocess.CalledProcessError as e:
        print("", flush=True)
        print("Stage2 failed.", flush=True)
        print(f"- If this is a dependency issue, try running with the repo venv: `--python venv/bin/python3`.", flush=True)
        raise e

    if args.skip_gates:
        print("Skipping gates as requested.", flush=True)
        return

    print("", flush=True)
    _run(gates_cmd)

    print("", flush=True)
    print(f"OK: smoke run complete. Artifacts in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
