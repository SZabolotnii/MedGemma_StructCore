#!/usr/bin/env python3
"""
Helper to prepare/track weights for the two-stage structured pipeline:
Stage1: base GGUF
Stage2: FT (merged GGUF) and/or LoRA adapter for llama.cpp

Default mode is PLAN-ONLY (prints a reproducible command plan and writes a manifest).
Use --execute for running only lightweight steps; heavy steps (merge/convert/build)
still require local toolchains and dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _find_llama_convert_script(llama_dir: Path) -> Optional[Path]:
    candidates = [
        llama_dir / "convert_hf_to_gguf.py",
        llama_dir / "convert_hf_to_gguf.py.inp",  # defensive (some forks)
        llama_dir / "convert_hf_to_gguf.py.py",  # defensive
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: search for any python file containing the name.
    for p in sorted(llama_dir.glob("*.py")):
        if "convert" in p.name and "gguf" in p.name and "hf" in p.name:
            return p
    return None


def _find_llama_quantize_binary(llama_dir: Path) -> Optional[Path]:
    # Common locations:
    candidates = [
        llama_dir / "llama-quantize",
        llama_dir / "build" / "bin" / "llama-quantize",
        llama_dir / "build" / "llama-quantize",
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
    return None


def _lora_base_id_from_config(lora_dir: Path) -> Optional[str]:
    cfg = lora_dir / "adapter_config.json"
    if not cfg.exists():
        return None
    obj = _read_json(cfg)
    if not isinstance(obj, dict):
        return None
    for key in ("base_model_name_or_path", "model", "base_model"):
        v = str(obj.get(key) or "").strip()
        if v:
            return v
    return None


def _is_mlx_lora_checkpoint(lora_dir: Path) -> bool:
    # Heuristic: this repo's MLX LoRA training writes these files.
    return (lora_dir / "mlx_lora_config.yaml").exists() or (lora_dir / "training_metadata.json").exists()


def _assert_lora_checkpoint_sane(lora_dir: Path) -> Tuple[bool, List[str]]:
    warnings: List[str] = []
    ok = True

    cfg = lora_dir / "adapter_config.json"
    if not cfg.exists():
        warnings.append("Missing adapter_config.json in lora dir.")
        ok = False

    weights = None
    for name in ("adapter_model.safetensors", "adapters.safetensors", "adapters.npz"):
        p = lora_dir / name
        if p.exists():
            weights = p
            break
    if not weights:
        # Any safetensors is acceptable.
        safes = sorted(lora_dir.glob("*.safetensors"))
        weights = safes[0] if safes else None

    if not weights:
        warnings.append("Missing adapter weights (*.safetensors or *.npz).")
        ok = False
    else:
        sz = weights.stat().st_size
        if sz < 1024 * 1024:
            warnings.append(f"Adapter weights look too small: {weights} ({sz} bytes).")
            ok = False

    return ok, warnings


@dataclass(frozen=True)
class Manifest:
    ts: str
    base_repo: str
    base_revision: Optional[str]
    lora_dir: str
    lora_base_id: Optional[str]
    stage2_merge_method: str
    gguf_out_dir: str
    hf_cache_dir: Optional[str]
    llama_cpp_dir: str
    quant: str
    plan_only: bool
    notes: List[str]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-repo", type=str, default="google/medgemma-1.5-4b-it")
    p.add_argument("--base-revision", type=str, default="", help="Optional: HF commit/tag for reproducibility.")
    p.add_argument("--lora-dir", type=str, required=True, help="Path to LoRA checkpoint dir (local).")
    p.add_argument("--gguf-out-dir", type=str, required=True, help="Where to place GGUF outputs (recommend outside repo).")
    p.add_argument("--hf-cache-dir", type=str, default="", help="Where to download HF base/merged directories.")
    p.add_argument("--llama-cpp-dir", type=str, default="", help="Path to llama.cpp checkout (optional).")
    p.add_argument("--quant", type=str, default="Q5_K_M", help="Quant type, e.g. Q5_K_M, Q4_K_S.")
    p.add_argument(
        "--stage2-merge-method",
        type=str,
        default="",
        help="Stage2 merge method: 'mlx_fuse' (recommended for this repo) or 'peft_merge'. Default: auto-detect.",
    )
    p.add_argument(
        "--manifest-path",
        type=str,
        default="results/two_stage_weights_manifest.json",
        help="Where to write a small JSON manifest (default: ignored by git).",
    )
    p.add_argument("--plan-only", action="store_true", help="Print steps + write manifest only (default).")
    p.add_argument("--execute", action="store_true", help="Run only lightweight validation steps.")
    args = p.parse_args()

    plan_only = bool(args.plan_only or not args.execute)

    lora_dir = Path(args.lora_dir).expanduser().resolve()
    gguf_out_dir = Path(args.gguf_out_dir).expanduser().resolve()
    hf_cache_dir = Path(args.hf_cache_dir).expanduser().resolve() if args.hf_cache_dir.strip() else None
    llama_cpp_dir = Path(args.llama_cpp_dir).expanduser().resolve() if args.llama_cpp_dir.strip() else None
    base_repo = args.base_repo.strip()
    base_revision = args.base_revision.strip() or None
    quant = args.quant.strip()
    merge_method = args.stage2_merge_method.strip().lower()

    if not lora_dir.exists():
        raise SystemExit(f"--lora-dir does not exist: {lora_dir}")

    ok, warnings = _assert_lora_checkpoint_sane(lora_dir)
    lora_base_id = _lora_base_id_from_config(lora_dir)
    if not lora_base_id:
        warnings.append("Could not infer base id from adapter_config.json (expected key: base_model_name_or_path/model/base_model).")
        ok = False

    if lora_base_id and base_repo and base_repo not in lora_base_id:
        warnings.append(f"LoRA base id mismatch: base_repo='{base_repo}' not in lora_base_id='{lora_base_id}'")
        ok = False

    is_mlx = _is_mlx_lora_checkpoint(lora_dir)
    if not merge_method:
        merge_method = "mlx_fuse" if is_mlx else "peft_merge"
    if merge_method not in {"mlx_fuse", "peft_merge"}:
        raise SystemExit("--stage2-merge-method must be 'mlx_fuse' or 'peft_merge'")

    notes: List[str] = []
    if plan_only:
        notes.append("PLAN-ONLY: no downloads/merges/conversions performed.")
    else:
        notes.append("EXECUTE: only lightweight validation performed; heavy steps still manual.")
    if is_mlx:
        notes.append("Detected MLX LoRA checkpoint (mlx_lm.lora). Recommended Stage2 merge: mlx_lm.fuse.")

    manifest = Manifest(
        ts=datetime.now().isoformat(timespec="seconds"),
        base_repo=base_repo,
        base_revision=base_revision,
        lora_dir=str(lora_dir),
        lora_base_id=lora_base_id,
        stage2_merge_method=merge_method,
        gguf_out_dir=str(gguf_out_dir),
        hf_cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
        llama_cpp_dir=str(llama_cpp_dir) if llama_cpp_dir else "",
        quant=quant,
        plan_only=plan_only,
        notes=notes,
    )

    out_manifest = Path(args.manifest_path).expanduser()
    if not out_manifest.is_absolute():
        out_manifest = (REPO_ROOT / out_manifest).resolve()
    _write_json(out_manifest, asdict(manifest))

    print("== Two-stage weights prep plan ==")
    print(f"Manifest: {out_manifest}")
    print(f"LoRA dir: {lora_dir}")
    print(f"LoRA base id: {lora_base_id or 'UNKNOWN'}")
    print(f"Base repo: {base_repo}")
    if base_revision:
        print(f"Base revision: {base_revision}")
    print(f"GGUF out: {gguf_out_dir}")
    print(f"Quant: {quant}")
    print(f"Stage2 merge method: {merge_method}")
    print("")

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")
        print("")

    if not ok:
        print("FAIL: LoRA checkpoint sanity check failed. Fix issues above before converting/merging.")
        sys.exit(2)

    # Command plan (copy/paste oriented).
    print("Step 1) (Optional) HF download base (pinned revision recommended):")
    if hf_cache_dir:
        base_dir = hf_cache_dir / "medgemma-1.5-4b-it"
        rev_arg = f', revision="{base_revision}"' if base_revision else ""
        print("  python3 -m pip install -U huggingface_hub")
        print("  python3 - <<'PY'")
        print("  from huggingface_hub import snapshot_download")
        print("  snapshot_download(")
        print(f'    repo_id="{base_repo}",{rev_arg}')
        print(f'    local_dir="{base_dir}",')
        print("    local_dir_use_symlinks=False,")
        print("  )")
        print("  print('OK')")
        print("  PY")
    else:
        print("  (set --hf-cache-dir to get a concrete command)")
    print("")

    print("Step 2) Build Stage2 merged HF-like dir:")
    if hf_cache_dir:
        base_dir = hf_cache_dir / "medgemma-1.5-4b-it"
        merged_dir = hf_cache_dir / "medgemma-1.5-4b-it-ft-merged"
        if merge_method == "mlx_fuse":
            print("  # Recommended for this repo (LoRA trained via mlx_lm.lora)")
            print("  python3 -m pip install -U mlx mlx-lm pyyaml")
            print("  python3 -m mlx_lm.fuse \\")
            print(f"    --model {base_dir} \\")
            print(f"    --adapter-path {str(lora_dir)} \\")
            print(f"    --save-path {merged_dir}")
            print("  # Note: for strict reproducibility, ensure `base_dir` is a pinned HF snapshot (see Step 1).")
        else:
            print("  # PEFT merge (ONLY if LoRA is in PEFT format)")
            print("  python3 -m pip install -U transformers peft safetensors")
            print("  python3 - <<'PY'")
            print("  from transformers import AutoModelForCausalLM, AutoTokenizer")
            print("  from peft import PeftModel")
            print(f"  base_dir = {base_dir!r}")
            print(f"  lora_dir = {str(lora_dir)!r}")
            print(f"  out_dir  = {merged_dir!r}")
            print("  tok = AutoTokenizer.from_pretrained(base_dir)")
            print("  tok.save_pretrained(out_dir)")
            print("  m = AutoModelForCausalLM.from_pretrained(base_dir, device_map='cpu', torch_dtype='auto')")
            print("  m = PeftModel.from_pretrained(m, lora_dir)")
            print("  m = m.merge_and_unload()")
            print("  m.save_pretrained(out_dir, safe_serialization=True)")
            print("  print('OK:', out_dir)")
            print("  PY")
    else:
        print("  (set --hf-cache-dir to get a concrete command)")
    print("")

    print("Step 3) llama.cpp: convert HF → GGUF (f16), then quantize:")
    if llama_cpp_dir:
        convert = _find_llama_convert_script(llama_cpp_dir)
        quant_bin = _find_llama_quantize_binary(llama_cpp_dir)
        print(f"  llama.cpp dir: {llama_cpp_dir}")
        print(f"  convert script: {convert or 'NOT FOUND'}")
        print(f"  quantize bin: {quant_bin or 'NOT FOUND (build first)'}")
    else:
        print("  (set --llama-cpp-dir to get path checks)")
    if hf_cache_dir:
        base_dir = hf_cache_dir / "medgemma-1.5-4b-it"
        merged_dir = hf_cache_dir / "medgemma-1.5-4b-it-ft-merged"
        base_f16 = gguf_out_dir / "medgemma-base-f16.gguf"
        base_q = gguf_out_dir / f"medgemma-base-{quant.casefold()}.gguf"
        merged_f16 = gguf_out_dir / "medgemma-ft-merged-f16.gguf"
        merged_q = gguf_out_dir / f"medgemma-ft-merged-{quant.casefold()}.gguf"
        print("  python3 /PATH/TO/llama.cpp/convert_hf_to_gguf.py \\")
        print(f"    {base_dir} --outfile {base_f16} --outtype f16")
        print("  python3 /PATH/TO/llama.cpp/convert_hf_to_gguf.py \\")
        print(f"    {merged_dir} --outfile {merged_f16} --outtype f16")
        print("  /PATH/TO/llama-quantize \\")
        print(f"    {base_f16} {base_q} {quant}")
        print("  /PATH/TO/llama-quantize \\")
        print(f"    {merged_f16} {merged_q} {quant}")
    else:
        print("  (set --hf-cache-dir to get concrete convert/quantize commands)")
    print("")

    if not plan_only:
        # Lightweight checks only.
        if shutil.which("git"):
            try:
                tracked = _run(["git", "ls-files", "-z", "--", "**/*.gguf", "**/*.safetensors"], check=False).stdout
                tracked_paths = [p for p in tracked.split("\0") if p.strip()]
                if tracked_paths:
                    print("WARNING: some weight-like files are tracked by git in this repo:")
                    for pth in tracked_paths[:10]:
                        print(f"- {pth}")
            except Exception:
                pass

    print("Done.")


if __name__ == "__main__":
    main()
