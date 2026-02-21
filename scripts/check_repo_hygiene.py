#!/usr/bin/env python3
"""
Repo hygiene gate for PUBLIC repositories.

This script is intentionally strict: it fails if any *git-tracked* file looks like
clinical note text, restricted datasets, model weights, or run artifacts.

It is designed to run in CI before public pushes/merges.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


FORBIDDEN_PATH_PREFIXES = [
    "results/",
    "EHR_test_data/",
    "Curated_EHR_Test_Sets/",
    "Data_MIMIC/",
    "physionet.org/",
]

FORBIDDEN_FILENAME_PREFIXES = [
    "ehr_",
]

FORBIDDEN_SUFFIXES = [
    ".ndjson",
    ".ndjson.gz",
    ".gguf",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".h5",
]


@dataclass(frozen=True)
class Finding:
    path: str
    reason: str


def _run_git(args: List[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=str(REPO_ROOT), stderr=subprocess.STDOUT).decode("utf-8", errors="replace")


def _git_ls_files() -> List[str]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=str(REPO_ROOT))
    parts = raw.split(b"\x00")
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        out.append(p.decode("utf-8", errors="replace"))
    return out


def _is_forbidden_path(rel_posix: str) -> Optional[str]:
    for pref in FORBIDDEN_PATH_PREFIXES:
        if rel_posix.startswith(pref):
            return f"forbidden path prefix: {pref}"

    name = Path(rel_posix).name
    for pref in FORBIDDEN_FILENAME_PREFIXES:
        if name.startswith(pref) and name.endswith(".txt"):
            return "looks like raw note text filename pattern (ehr_*.txt)"

    for suf in FORBIDDEN_SUFFIXES:
        if rel_posix.endswith(suf):
            return f"forbidden artifact/weight suffix: {suf}"

    return None


def _is_binary(path: Path) -> bool:
    try:
        data = path.read_bytes()[:2048]
    except Exception:
        return False
    return b"\x00" in data


def check_repo(*, max_bytes: int) -> List[Finding]:
    findings: List[Finding] = []
    files = _git_ls_files()
    for rel in files:
        rel_posix = rel.replace("\\", "/")
        reason = _is_forbidden_path(rel_posix)
        if reason:
            findings.append(Finding(path=rel_posix, reason=reason))
            continue

        full = REPO_ROOT / rel
        if not full.exists():
            # Should not happen, but don't crash CI.
            findings.append(Finding(path=rel_posix, reason="tracked file missing on disk"))
            continue

        try:
            size = full.stat().st_size
        except Exception:
            size = None
        if size is not None and size > max_bytes:
            findings.append(Finding(path=rel_posix, reason=f"file too large for public repo: {size} bytes > {max_bytes}"))
            continue

        if _is_binary(full):
            findings.append(Finding(path=rel_posix, reason="binary-like content detected (NUL byte)"))

    return findings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bytes", type=int, default=int(os.getenv("REPO_HYGIENE_MAX_BYTES", "2000000")))
    args = ap.parse_args()

    try:
        _run_git(["rev-parse", "--is-inside-work-tree"])
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Not a git repository (or git unavailable): {e}") from e

    findings = check_repo(max_bytes=int(args.max_bytes))
    if findings:
        print("Repo hygiene check FAILED. Findings:", file=sys.stderr)
        for f in findings:
            print(f"- {f.path}: {f.reason}", file=sys.stderr)
        raise SystemExit(2)

    print("Repo hygiene check OK.")


if __name__ == "__main__":
    main()

