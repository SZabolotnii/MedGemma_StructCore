#!/usr/bin/env python3
"""
Stage2 CAG A/B smoke gate (llama.cpp prompt cache).

Goal:
- Verify that enabling llama.cpp prompt cache does NOT change Stage2 outputs.

Approach:
- Run Stage2 twice on the *same* Stage1 artifacts (Stage1 JSON+MD):
    A) baseline (e.g., no-cache server)
    B) CAG (cache-prompt enabled server)
- Compare per-doc Stage2 artifacts and fail fast on any diffs.

This script is dependency-free and intentionally does NOT commit any artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_two_stage_structured_pipeline.py"


def _pick_python(python_override: str | None) -> str:
    if python_override and python_override.strip():
        return python_override.strip()
    for cand in (REPO_ROOT / "venv" / "bin" / "python3", REPO_ROOT / "venv" / "bin" / "python"):
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return sys.executable


def _as_csv_ints(raw: str) -> List[int]:
    out: List[int] = []
    for token in (raw or "").split(","):
        t = token.strip()
        if not t:
            continue
        if not t.isdigit():
            raise SystemExit(f"invalid hadm id: {t}")
        out.append(int(t))
    return out


def _load_json_list(path: Path) -> List[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise SystemExit(f"expected JSON list: {path}")
    out: List[int] = []
    for x in obj:
        s = str(x).strip()
        if s.isdigit():
            out.append(int(s))
    return out


def _discover_hadm_ids_from_src(
    src_out_dir: Path,
    *,
    stage1_json_name: str,
    stage1_md_name: str,
    num_docs: int,
) -> List[int]:
    ids: List[int] = []
    for p in sorted(src_out_dir.iterdir()):
        if not p.is_dir() or not p.name.isdigit():
            continue
        hadm = int(p.name)
        if not (p / stage1_json_name).exists():
            continue
        if not (p / stage1_md_name).exists():
            continue
        ids.append(hadm)
        if num_docs > 0 and len(ids) >= num_docs:
            break
    if not ids:
        raise SystemExit(f"no hadm_ids with Stage1 artifacts found under: {src_out_dir}")
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


def _copy_stage1_artifacts(
    *,
    src_out_dir: Path,
    dst_out_dir: Path,
    hadm_ids: List[int],
    stage1_json_name: str,
    stage1_md_name: str,
    stage1_meta_name: str,
) -> None:
    dst_out_dir.mkdir(parents=True, exist_ok=True)
    for hadm in hadm_ids:
        src_dir = src_out_dir / str(hadm)
        if not src_dir.exists():
            raise SystemExit(f"missing src hadm dir: {src_dir}")

        src_json = src_dir / stage1_json_name
        src_md = src_dir / stage1_md_name
        if not src_json.exists() or not src_md.exists():
            raise SystemExit(f"missing Stage1 artifacts for hadm_id={hadm} under: {src_dir}")

        dst_dir = dst_out_dir / str(hadm)
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_json, dst_dir / "stage1.json")
        shutil.copy2(src_md, dst_dir / "stage1.md")

        src_meta = src_dir / stage1_meta_name
        if src_meta.exists():
            shutil.copy2(src_meta, dst_dir / "stage1_meta.json")


def _parse_env_overrides(pairs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in pairs or []:
        if "=" not in raw:
            raise SystemExit(f"--env must be KEY=VALUE, got: {raw}")
        k, v = raw.split("=", 1)
        k = k.strip()
        if not k:
            raise SystemExit(f"--env invalid key: {raw}")
        out[k] = v
    return out


def _run_stage2(
    *,
    python: str,
    cohort_root: str,
    out_dir: Path,
    hadm_ids: List[int],
    url: str,
    model: str,
    scope: str,
    output_mode: str,
    max_tokens: int,
    temperature: float,
    overwrite_stage2: bool,
    env_overrides: Dict[str, str],
    extra_args: List[str],
) -> None:
    cmd = [
        python,
        str(PIPELINE_SCRIPT),
        "--cohort-root",
        str(cohort_root),
        "--out-dir",
        str(out_dir),
        "--hadm-ids",
        ",".join(str(h) for h in hadm_ids),
        "stage2",
        "--url",
        str(url),
        "--model",
        str(model),
        "--scope",
        str(scope),
        "--output-mode",
        str(output_mode),
        "--max-tokens",
        str(int(max_tokens)),
        "--temperature",
        str(float(temperature)),
    ]
    if overwrite_stage2:
        cmd.append("--overwrite-stage2")
    cmd.extend(extra_args or [])

    env = os.environ.copy()
    env.update(env_overrides)

    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_stage2_normalized_json(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    return {
        "normalized": obj.get("normalized"),
        "normalization_stats": obj.get("normalization_stats"),
        "format_stats": obj.get("format_stats"),
        "prompt_template_id": obj.get("prompt_template_id"),
        "prompt_prefix_sha256": obj.get("prompt_prefix_sha256"),
        "generation_params": obj.get("generation_params"),
    }


@dataclass
class DiffItem:
    hadm_id: int
    relpath: str
    sha_a: str
    sha_b: str
    note: str


def _compare_pair(
    *,
    dir_a: Path,
    dir_b: Path,
    hadm_ids: List[int],
    strict_stage2_normalized_json: bool,
) -> Tuple[List[DiffItem], Dict[str, Any]]:
    diffs: List[DiffItem] = []

    def _cmp_file(hadm: int, rel: str) -> None:
        pa = dir_a / str(hadm) / rel
        pb = dir_b / str(hadm) / rel
        if not pa.exists() or not pb.exists():
            diffs.append(
                DiffItem(
                    hadm_id=hadm,
                    relpath=rel,
                    sha_a=_sha256(pa) if pa.exists() else "",
                    sha_b=_sha256(pb) if pb.exists() else "",
                    note="missing_file",
                )
            )
            return
        if _sha256(pa) != _sha256(pb):
            diffs.append(DiffItem(hadm_id=hadm, relpath=rel, sha_a=_sha256(pa), sha_b=_sha256(pb), note="content_diff"))

    def _cmp_stage2_normalized(hadm: int) -> None:
        rel = "stage2_normalized.json"
        pa = dir_a / str(hadm) / rel
        pb = dir_b / str(hadm) / rel
        if not pa.exists() or not pb.exists():
            diffs.append(
                DiffItem(
                    hadm_id=hadm,
                    relpath=rel,
                    sha_a=_sha256(pa) if pa.exists() else "",
                    sha_b=_sha256(pb) if pb.exists() else "",
                    note="missing_file",
                )
            )
            return

        if strict_stage2_normalized_json:
            if _sha256(pa) != _sha256(pb):
                diffs.append(DiffItem(hadm_id=hadm, relpath=rel, sha_a=_sha256(pa), sha_b=_sha256(pb), note="json_bytes_diff"))
            return

        ja = _normalize_stage2_normalized_json(_load_json(pa))
        jb = _normalize_stage2_normalized_json(_load_json(pb))
        if ja != jb:
            diffs.append(DiffItem(hadm_id=hadm, relpath=rel, sha_a=_sha256(pa), sha_b=_sha256(pb), note="json_semantic_diff"))

    for hadm in hadm_ids:
        for rel in ("stage2_raw.txt", "stage2_facts.txt", "stage2_metrics.json"):
            _cmp_file(hadm, rel)
        # Optional retry artifact: require presence/content to match if either side produced it.
        pa_retry = dir_a / str(hadm) / "stage2_raw_retry1.txt"
        pb_retry = dir_b / str(hadm) / "stage2_raw_retry1.txt"
        if pa_retry.exists() or pb_retry.exists():
            _cmp_file(hadm, "stage2_raw_retry1.txt")
        _cmp_stage2_normalized(hadm)

    # Compare run-level prompt signature (ignore timestamp).
    meta_a = dir_a / "meta_stage2.json"
    meta_b = dir_b / "meta_stage2.json"
    meta_note = {}
    if meta_a.exists() and meta_b.exists():
        try:
            ja = _load_json(meta_a)
            jb = _load_json(meta_b)
            keys = ["prompt_template_id", "prompt_prefix_sha256", "prompt_prefix_chars", "model", "url", "scope", "output_mode"]
            meta_note = {k: {"a": ja.get(k), "b": jb.get(k)} for k in keys}
        except Exception:
            meta_note = {"error": "failed_to_parse_meta_stage2.json"}

    summary = {
        "n_hadm": len(hadm_ids),
        "n_diffs": len(diffs),
        "strict_stage2_normalized_json": bool(strict_stage2_normalized_json),
        "meta_stage2_compare": meta_note,
    }
    return diffs, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-out-dir", type=str, required=True, help="Source directory containing per-hadm Stage1 artifacts.")
    ap.add_argument("--cohort-root", type=str, required=True, help="EHR_test_data root (used only for optional GT lookup).")
    ap.add_argument("--out-root", type=str, required=True, help="Output root; creates <out-root>/<label-a> and <out-root>/<label-b>.")
    ap.add_argument("--python", type=str, default="", help="Python interpreter for pipeline calls.")

    ap.add_argument("--hadm-ids", type=str, default="", help="Comma-separated hadm_ids. Optional if --hadm-ids-json or discovery.")
    ap.add_argument("--hadm-ids-json", type=str, default="", help="JSON list of hadm_ids.")
    ap.add_argument("--num-docs", type=int, default=20, help="If no hadm ids given: discover first N from --src-out-dir.")

    ap.add_argument("--stage1-json-name", type=str, default="stage1.json")
    ap.add_argument("--stage1-md-name", type=str, default="stage1.md")
    ap.add_argument("--stage1-meta-name", type=str, default="stage1_meta.json")

    ap.add_argument("--label-a", type=str, default="baseline", help="Subdir name under --out-root for run A.")
    ap.add_argument("--label-b", type=str, default="cag", help="Subdir name under --out-root for run B.")
    ap.add_argument("--url-a", type=str, default="", help="Stage2 backend URL for run A.")
    ap.add_argument("--url-b", type=str, default="", help="Stage2 backend URL for run B.")
    ap.add_argument("--model", type=str, required=True, help="Stage2 model alias (must exist in /v1/models).")

    ap.add_argument("--scope", type=str, default="all", choices=["objective", "all"])
    ap.add_argument("--output-mode", type=str, default="lines", choices=["lines", "json"])
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--overwrite-stage2", action="store_true")
    ap.add_argument("--strict-stage2-normalized-json", action="store_true", help="Require byte-identical stage2_normalized.json.")

    ap.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip Stage2 runs and only compare existing outputs under --out-root/<label-a|label-b>.",
    )
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra env for BOTH runs (repeatable). Format: KEY=VALUE. Example: MEDGEMMA_STAGE2_TRAINING_MATCH_PROMPT=1",
    )
    ap.add_argument("--extra-stage2-arg", action="append", default=[], help="Extra args forwarded to stage2 subcommand (repeatable).")
    args = ap.parse_args()

    python = _pick_python(args.python)
    src_out_dir = Path(args.src_out_dir)
    out_root = Path(args.out_root)
    out_a = out_root / str(args.label_a)
    out_b = out_root / str(args.label_b)

    hadm_ids: List[int] = []
    if args.hadm_ids.strip():
        hadm_ids = _as_csv_ints(args.hadm_ids)
    if args.hadm_ids_json.strip():
        hadm_ids.extend(_load_json_list(Path(args.hadm_ids_json)))
    hadm_ids = _unique_keep_order(hadm_ids)
    if not hadm_ids:
        hadm_ids = _discover_hadm_ids_from_src(
            src_out_dir,
            stage1_json_name=str(args.stage1_json_name),
            stage1_md_name=str(args.stage1_md_name),
            num_docs=int(args.num_docs),
        )

    env_overrides = _parse_env_overrides(list(args.env or []))

    if not args.compare_only:
        if not args.url_a.strip() or not args.url_b.strip():
            raise SystemExit("--url-a and --url-b are required unless --compare-only is set")
        # Prepare Stage1 artifacts for both runs.
        _copy_stage1_artifacts(
            src_out_dir=src_out_dir,
            dst_out_dir=out_a,
            hadm_ids=hadm_ids,
            stage1_json_name=str(args.stage1_json_name),
            stage1_md_name=str(args.stage1_md_name),
            stage1_meta_name=str(args.stage1_meta_name),
        )
        _copy_stage1_artifacts(
            src_out_dir=src_out_dir,
            dst_out_dir=out_b,
            hadm_ids=hadm_ids,
            stage1_json_name=str(args.stage1_json_name),
            stage1_md_name=str(args.stage1_md_name),
            stage1_meta_name=str(args.stage1_meta_name),
        )

        _run_stage2(
            python=python,
            cohort_root=str(args.cohort_root),
            out_dir=out_a,
            hadm_ids=hadm_ids,
            url=str(args.url_a),
            model=str(args.model),
            scope=str(args.scope),
            output_mode=str(args.output_mode),
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            overwrite_stage2=bool(args.overwrite_stage2),
            env_overrides=env_overrides,
            extra_args=list(args.extra_stage2_arg or []),
        )
        _run_stage2(
            python=python,
            cohort_root=str(args.cohort_root),
            out_dir=out_b,
            hadm_ids=hadm_ids,
            url=str(args.url_b),
            model=str(args.model),
            scope=str(args.scope),
            output_mode=str(args.output_mode),
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            overwrite_stage2=bool(args.overwrite_stage2),
            env_overrides=env_overrides,
            extra_args=list(args.extra_stage2_arg or []),
        )

    diffs, summary = _compare_pair(
        dir_a=out_a,
        dir_b=out_b,
        hadm_ids=hadm_ids,
        strict_stage2_normalized_json=bool(args.strict_stage2_normalized_json),
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "cag_ab_smoke_summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "hadm_ids": hadm_ids,
                "diffs": [asdict(d) for d in diffs],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if diffs:
        print(f"[cag_ab_smoke] FAIL: n_diffs={len(diffs)} (see {out_root}/cag_ab_smoke_summary.json)", flush=True)
        for d in diffs[:25]:
            print(f"[diff] hadm_id={d.hadm_id} file={d.relpath} note={d.note}", flush=True)
        if len(diffs) > 25:
            print(f"[diff] ... {len(diffs) - 25} more", flush=True)
        return 1

    print(f"[cag_ab_smoke] PASS: n_hadm={len(hadm_ids)} (see {out_root}/cag_ab_smoke_summary.json)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

