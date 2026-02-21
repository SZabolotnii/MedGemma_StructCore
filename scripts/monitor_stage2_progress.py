#!/usr/bin/env python3
"""
Small, dependency-free progress monitor for Stage2 runs.

Motivation:
- macOS doesn't ship with `watch` by default.
- We want an easy way to see a "live" progress bar while a background
  supervisor/watchdog is running.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime


def load_hadm_ids(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return [int(x) for x in data]


def stage2_done(out_dir: str, hadm_id: int) -> bool:
    return os.path.exists(os.path.join(out_dir, str(hadm_id), "stage2_facts.txt"))


def count_done(out_dir: str, hadm_ids: list[int]) -> int:
    return sum(1 for h in hadm_ids if stage2_done(out_dir, h))


def render_progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[?]"
    done = max(0, min(done, total))
    filled = int(round(width * (done / total)))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {done}/{total}"


def read_text(path: str, max_bytes: int = 8192) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
        return data.decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


_CHUNK_RE = re.compile(r"hadm_ids=([0-9,]+)")
_RETRY_RE = re.compile(r"retry=([0-9]+)")


def tail_lines(path: str, n: int) -> list[str]:
    text = read_text(path, max_bytes=128 * 1024)
    if not text:
        return []
    lines = text.splitlines()
    return lines[-n:]


def parse_inflight(progress_text: str) -> tuple[str, str]:
    """
    Best-effort parse of the last reported chunk from _stage2_progress.txt.

    Returns (hadm_ids_csv, retry) where values may be empty strings.
    """
    hadm_ids_csv = ""
    retry = ""
    for line in reversed(progress_text.splitlines()):
        if not hadm_ids_csv:
            m = _CHUNK_RE.search(line)
            if m:
                hadm_ids_csv = m.group(1)
        if not retry:
            m = _RETRY_RE.search(line)
            if m:
                retry = m.group(1)
        if hadm_ids_csv and retry:
            break
    return hadm_ids_csv, retry


def newest_done_file(out_dir: str) -> tuple[str, float]:
    """
    Returns (hadm_id, mtime_epoch) for the newest stage2_facts.txt.
    If none: ("", 0.0)
    """
    newest_hadm = ""
    newest_mtime = 0.0
    try:
        with os.scandir(out_dir) as it:
            for ent in it:
                if not ent.is_dir():
                    continue
                if not ent.name.isdigit():
                    continue
                p = os.path.join(ent.path, "stage2_facts.txt")
                try:
                    st = os.stat(p)
                except FileNotFoundError:
                    continue
                if st.st_mtime > newest_mtime:
                    newest_mtime = st.st_mtime
                    newest_hadm = ent.name
    except FileNotFoundError:
        return ("", 0.0)
    return (newest_hadm, newest_mtime)


def done_file_mtime_range(out_dir: str) -> tuple[str, float, str, float]:
    """
    Returns (oldest_hadm, oldest_mtime, newest_hadm, newest_mtime) for stage2_facts.txt files.
    If none: ("", 0.0, "", 0.0)
    """
    oldest_hadm = ""
    oldest_mtime = 0.0
    newest_hadm = ""
    newest_mtime = 0.0
    try:
        with os.scandir(out_dir) as it:
            for ent in it:
                if not ent.is_dir():
                    continue
                if not ent.name.isdigit():
                    continue
                p = os.path.join(ent.path, "stage2_facts.txt")
                try:
                    st = os.stat(p)
                except FileNotFoundError:
                    continue
                if oldest_mtime == 0.0 or st.st_mtime < oldest_mtime:
                    oldest_mtime = st.st_mtime
                    oldest_hadm = ent.name
                if st.st_mtime > newest_mtime:
                    newest_mtime = st.st_mtime
                    newest_hadm = ent.name
    except FileNotFoundError:
        return ("", 0.0, "", 0.0)
    return (oldest_hadm, oldest_mtime, newest_hadm, newest_mtime)


def clear_screen() -> None:
    # Avoid ANSI when not a TTY (e.g., piping).
    if sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hadm-ids-json", required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--progress-file", type=str, default="")
    ap.add_argument("--tail-log", type=str, default="")
    ap.add_argument("--tail-lines", type=int, default=10)
    ap.add_argument("--no-clear", action="store_true")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    hadm_ids = load_hadm_ids(args.hadm_ids_json)
    total = len(hadm_ids)
    progress_file = args.progress_file.strip() or os.path.join(args.out_dir, "_stage2_progress.txt")

    start_ts = time.time()
    start_done = count_done(args.out_dir, hadm_ids)

    while True:
        done = count_done(args.out_dir, hadm_ids)
        bar = render_progress_bar(done, total)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        percent = (100.0 * done / total) if total else 0.0
        elapsed_s = max(1e-6, time.time() - start_ts)
        rate_per_min = (done - start_done) / elapsed_s * 60.0
        eta_min = ((total - done) / rate_per_min) if rate_per_min > 0 else float("inf")

        if not args.no_clear:
            clear_screen()

        print(f"[monitor] {now}")
        print(f"[monitor] out_dir={args.out_dir}")
        print(f"[monitor] progress {bar} ({percent:.1f}%)")
        if rate_per_min > 0:
            print(f"[monitor] rate_window={rate_per_min:.2f} docs/min eta_window={eta_min:.1f} min")
        else:
            print("[monitor] rate_window=0.00 docs/min eta_window=?")

        progress_text = read_text(progress_file)
        inflight_hadm, inflight_retry = parse_inflight(progress_text) if progress_text else ("", "")
        if inflight_hadm:
            retry_s = f" retry={inflight_retry}" if inflight_retry else ""
            print(f"[monitor] inflight hadm_ids={inflight_hadm}{retry_s}")

        oldest_hadm, oldest_mtime, newest_hadm, newest_mtime = done_file_mtime_range(args.out_dir)
        if newest_hadm:
            age_s = max(0.0, time.time() - newest_mtime)
            print(f"[monitor] last_done hadm_id={newest_hadm} age={age_s:.0f}s")
        if oldest_hadm and oldest_mtime > 0.0 and done >= 2:
            run_elapsed_s = max(1e-6, time.time() - oldest_mtime)
            rate_avg = (done / run_elapsed_s) * 60.0
            eta_avg = ((total - done) / rate_avg) if rate_avg > 0 else float("inf")
            print(f"[monitor] rate_avg={rate_avg:.2f} docs/min eta_avg={eta_avg:.1f} min (since hadm_id={oldest_hadm})")

        try:
            pf_mtime = os.stat(progress_file).st_mtime
            pf_age = max(0.0, time.time() - pf_mtime)
            print(f"[monitor] progress_file_age={pf_age:.0f}s")
        except FileNotFoundError:
            pass

        if progress_text.strip():
            print("\n[monitor] progress_file:")
            # Show only a few last lines to avoid huge spam.
            for line in progress_text.splitlines()[-5:]:
                print(line)

        if args.tail_log.strip():
            print(f"\n[monitor] tail {args.tail_lines} lines: {args.tail_log}")
            for line in tail_lines(args.tail_log, args.tail_lines):
                print(line)

        if done >= total:
            print("\n[monitor] done")
            return 0

        if args.once:
            return 0

        time.sleep(max(0.2, float(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
