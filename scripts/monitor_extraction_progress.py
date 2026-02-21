#!/usr/bin/env python3
"""
Live progress monitor for two-stage extraction runs (Track B scale-out blocks).

Goal:
- Provide an at-a-glance progress view (counts + rates + ETA) while extraction runs.
- Write lightweight artifacts under the run directory (ignored `results/`), so progress
  can be inspected later or viewed in a browser.

Outputs (under <block>/monitor/):
- progress.json                (latest snapshot)
- progress_timeseries.jsonl    (append-only snapshots)
- progress.html                (simple visualization that auto-refreshes)

This script does NOT read EHR note text; it only counts files.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Optional[object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_hadm_ids(run_dir: Path) -> List[str]:
    for name in ["hadm_ids_stable.json", "hadm_ids.json", "hadm_ids_sequential.json"]:
        p = run_dir / name
        if p.exists():
            obj = _read_json(p)
            if isinstance(obj, list) and obj:
                return [str(x).strip() for x in obj if str(x).strip()]
    return []


def _count_exists(paths: Iterable[Path]) -> int:
    return sum(1 for p in paths if p.exists())


def _iter_hadm_dirs(raw_stage_traces: Path) -> Iterable[Path]:
    if not raw_stage_traces.exists():
        return []
    for p in raw_stage_traces.iterdir():
        if p.is_dir() and p.name.isdigit():
            yield p


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def _format_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 0:
        return "n/a"
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    if h > 0:
        return f"{h:d}h{m:02d}m"
    if m > 0:
        return f"{m:d}m{ss:02d}s"
    return f"{ss:d}s"


@dataclass(frozen=True)
class Snapshot:
    ts_utc: str
    run_dir: str
    raw_stage_traces: str
    n_total: int
    stage1_ok: int
    stage2_ok: int
    stage1_errors: int
    stage2_errors: int
    stage2_retry1_present: int

    # Rates
    stage1_rate_per_min: float
    stage2_rate_per_min: float
    eta_stage1_s: Optional[float]
    eta_stage2_s: Optional[float]

    # Inferred runtime
    started_at_utc: Optional[str]
    elapsed_s: Optional[float]


def _infer_started_at(raw_stage_traces: Path) -> Optional[datetime]:
    """
    Best-effort: use meta_stage1.json if present, else fallback to earliest mtime of stage1.json.
    """
    meta = raw_stage_traces / "meta_stage1.json"
    if meta.exists():
        obj = _read_json(meta)
        if isinstance(obj, dict):
            ts = str(obj.get("ts") or "").strip()
            # meta ts is local time without tz; treat as local and only use for ordering, not absolute.
            # We avoid parsing to prevent locale issues; prefer file mtime as reliable fallback.
            # Returning None here is fine; elapsed will use mtime fallback.
    # Fallback: earliest stage1.json mtime.
    mtimes: List[float] = []
    for per in _iter_hadm_dirs(raw_stage_traces):
        p = per / "stage1.json"
        if p.exists():
            try:
                mtimes.append(p.stat().st_mtime)
            except Exception:
                pass
    if not mtimes:
        return None
    return datetime.fromtimestamp(min(mtimes), tz=timezone.utc)


def _compute_snapshot(run_dir: Path, *, stage_subdir: str = "raw_stage_traces") -> Snapshot:
    raw_stage_traces = run_dir / stage_subdir
    hadm_ids = _load_hadm_ids(raw_stage_traces)
    n_total = len(hadm_ids)

    stage1_ok = 0
    stage2_ok = 0
    stage1_errors = 0
    stage2_errors = 0
    stage2_retry1_present = 0

    for per in _iter_hadm_dirs(raw_stage_traces):
        if (per / "stage1.json").exists():
            stage1_ok += 1
        if (per / "stage2_facts.txt").exists():
            stage2_ok += 1
        if (per / "stage1_error.json").exists():
            stage1_errors += 1
        if (per / "stage2_error.json").exists():
            stage2_errors += 1
        if (per / "stage2_raw_retry1.txt").exists():
            stage2_retry1_present += 1

    started = _infer_started_at(raw_stage_traces)
    now = datetime.now(timezone.utc)
    elapsed_s = (now - started).total_seconds() if started else None

    stage1_rate_per_min = _safe_div(stage1_ok, elapsed_s / 60.0) if elapsed_s else 0.0
    stage2_rate_per_min = _safe_div(stage2_ok, elapsed_s / 60.0) if elapsed_s else 0.0

    eta_stage1_s = None
    if stage1_rate_per_min > 0 and n_total > stage1_ok:
        eta_stage1_s = (n_total - stage1_ok) / stage1_rate_per_min * 60.0
    eta_stage2_s = None
    if stage2_rate_per_min > 0 and n_total > stage2_ok:
        eta_stage2_s = (n_total - stage2_ok) / stage2_rate_per_min * 60.0

    return Snapshot(
        ts_utc=_utc_now_iso(),
        run_dir=str(run_dir),
        raw_stage_traces=str(raw_stage_traces),
        n_total=n_total,
        stage1_ok=stage1_ok,
        stage2_ok=stage2_ok,
        stage1_errors=stage1_errors,
        stage2_errors=stage2_errors,
        stage2_retry1_present=stage2_retry1_present,
        stage1_rate_per_min=stage1_rate_per_min,
        stage2_rate_per_min=stage2_rate_per_min,
        eta_stage1_s=eta_stage1_s,
        eta_stage2_s=eta_stage2_s,
        started_at_utc=started.isoformat(timespec="seconds") if started else None,
        elapsed_s=elapsed_s,
    )


def _write_snapshot(monitor_dir: Path, snap: Snapshot) -> None:
    monitor_dir.mkdir(parents=True, exist_ok=True)
    obj = {
        "ts_utc": snap.ts_utc,
        "run_dir": snap.run_dir,
        "raw_stage_traces": snap.raw_stage_traces,
        "n_total": snap.n_total,
        "stage1_ok": snap.stage1_ok,
        "stage2_ok": snap.stage2_ok,
        "stage1_errors": snap.stage1_errors,
        "stage2_errors": snap.stage2_errors,
        "stage2_retry1_present": snap.stage2_retry1_present,
        "stage1_rate_per_min": snap.stage1_rate_per_min,
        "stage2_rate_per_min": snap.stage2_rate_per_min,
        "eta_stage1_s": snap.eta_stage1_s,
        "eta_stage2_s": snap.eta_stage2_s,
        "started_at_utc": snap.started_at_utc,
        "elapsed_s": snap.elapsed_s,
    }
    (monitor_dir / "progress.json").write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    with (monitor_dir / "progress_timeseries.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _write_html(monitor_dir: Path) -> None:
    monitor_dir.mkdir(parents=True, exist_ok=True)
    html = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>MedGemma Extraction Progress</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; color: #111; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; max-width: 980px; }
      .card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 16px; background: #fff; }
      .k { color: #6b7280; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
      .v { font-size: 28px; font-weight: 650; margin-top: 6px; }
      .small { font-size: 13px; color: #374151; margin-top: 8px; line-height: 1.35; }
      .bar { height: 12px; background: #f3f4f6; border-radius: 999px; overflow: hidden; margin-top: 10px; }
      .bar > div { height: 100%; background: #2563eb; width: 0%; transition: width .35s ease; }
      .bar2 > div { background: #16a34a; }
      code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }
      .muted { color: #6b7280; }
      .row { display:flex; justify-content:space-between; gap: 10px; margin-top: 8px; font-size: 13px; color: #374151; }
      .warn { color: #b45309; }
    </style>
  </head>
  <body>
    <h1 style="margin:0 0 10px 0;">Extraction progress</h1>
    <div class="muted" id="meta"></div>
    <div class="grid" style="margin-top:16px;">
      <div class="card">
        <div class="k">Stage1</div>
        <div class="v" id="s1v">-</div>
        <div class="bar"><div id="s1bar"></div></div>
        <div class="row"><div id="s1rate" class="muted"></div><div id="s1eta" class="muted"></div></div>
      </div>
      <div class="card">
        <div class="k">Stage2</div>
        <div class="v" id="s2v">-</div>
        <div class="bar bar2"><div id="s2bar"></div></div>
        <div class="row"><div id="s2rate" class="muted"></div><div id="s2eta" class="muted"></div></div>
      </div>
      <div class="card">
        <div class="k">Errors</div>
        <div class="v" id="errv">-</div>
        <div class="small">Stage1 errors: <span id="e1">-</span><br/>Stage2 errors: <span id="e2">-</span></div>
      </div>
      <div class="card">
        <div class="k">Retries</div>
        <div class="v" id="retryv">-</div>
        <div class="small">Docs with <code>stage2_raw_retry1.txt</code></div>
      </div>
    </div>

    <p class="muted" style="margin-top:18px;">Auto-refresh every 10s. Source: <code>progress.json</code></p>

    <script>
      function fmtPct(x) { return (Math.max(0, Math.min(1, x)) * 100).toFixed(1) + '%'; }
      function fmtRate(x) { return (x || 0).toFixed(2) + '/min'; }
      function fmtEta(s) {
        if (s === null || s === undefined || !isFinite(s)) return 'ETA: n/a';
        s = Math.max(0, Math.floor(s));
        const h = Math.floor(s/3600);
        const m = Math.floor((s%3600)/60);
        const ss = s%60;
        if (h>0) return `ETA: ${h}h${String(m).padStart(2,'0')}m`;
        if (m>0) return `ETA: ${m}m${String(ss).padStart(2,'0')}s`;
        return `ETA: ${ss}s`;
      }
      async function tick() {
        const r = await fetch('./progress.json', { cache: 'no-store' });
        const d = await r.json();
        const n = d.n_total || 0;
        document.getElementById('meta').textContent = `Updated: ${d.ts_utc} | Run: ${d.run_dir}`;
        document.getElementById('s1v').textContent = `${d.stage1_ok}/${n} (${fmtPct(n? d.stage1_ok/n : 0)})`;
        document.getElementById('s2v').textContent = `${d.stage2_ok}/${n} (${fmtPct(n? d.stage2_ok/n : 0)})`;
        document.getElementById('s1bar').style.width = fmtPct(n? d.stage1_ok/n : 0);
        document.getElementById('s2bar').style.width = fmtPct(n? d.stage2_ok/n : 0);
        document.getElementById('s1rate').textContent = `Rate: ${fmtRate(d.stage1_rate_per_min)}`;
        document.getElementById('s2rate').textContent = `Rate: ${fmtRate(d.stage2_rate_per_min)}`;
        document.getElementById('s1eta').textContent = fmtEta(d.eta_stage1_s);
        document.getElementById('s2eta').textContent = fmtEta(d.eta_stage2_s);
        document.getElementById('e1').textContent = d.stage1_errors;
        document.getElementById('e2').textContent = d.stage2_errors;
        document.getElementById('errv').textContent = `${(d.stage1_errors||0) + (d.stage2_errors||0)}`;
        document.getElementById('retryv').textContent = `${d.stage2_retry1_present||0}`;
      }
      tick().catch(() => {});
      setInterval(() => tick().catch(() => {}), 10000);
    </script>
  </body>
</html>
"""
    (monitor_dir / "progress.html").write_text(html, encoding="utf-8")


def _print_console(snap: Snapshot) -> None:
    n = snap.n_total
    s1 = snap.stage1_ok
    s2 = snap.stage2_ok
    e1 = snap.stage1_errors
    e2 = snap.stage2_errors
    r1 = snap.stage2_retry1_present
    s1_pct = _safe_div(s1, n) * 100.0 if n else 0.0
    s2_pct = _safe_div(s2, n) * 100.0 if n else 0.0

    line = (
        f"[{snap.ts_utc}] "
        f"S1 {s1:>5}/{n:<5} ({s1_pct:>5.1f}%) | "
        f"S2 {s2:>5}/{n:<5} ({s2_pct:>5.1f}%) | "
        f"err {e1+e2:<3} (s1={e1}, s2={e2}) | "
        f"retry1 {r1:<3} | "
        f"rate S1 {snap.stage1_rate_per_min:>5.2f}/m ETA { _format_eta(snap.eta_stage1_s):>8} | "
        f"rate S2 {snap.stage2_rate_per_min:>5.2f}/m ETA { _format_eta(snap.eta_stage2_s):>8}"
    )
    print(line, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--block-dir",
        required=True,
        help="Block directory (contains raw_stage_traces/) OR directly raw_stage_traces dir.",
    )
    ap.add_argument("--stage-subdir", default="raw_stage_traces")
    ap.add_argument("--interval-s", type=float, default=10.0)
    ap.add_argument("--once", action="store_true", help="Print once and exit.")
    args = ap.parse_args()

    block_dir = Path(args.block_dir).expanduser().resolve()
    run_dir = block_dir
    # If user passes raw_stage_traces directly, treat parent as run_dir for monitor artifacts.
    if block_dir.name == args.stage_subdir and block_dir.is_dir():
        run_dir = block_dir.parent

    monitor_dir = run_dir / "monitor"
    _write_html(monitor_dir)

    if args.once:
        snap = _compute_snapshot(run_dir, stage_subdir=args.stage_subdir)
        _write_snapshot(monitor_dir, snap)
        _print_console(snap)
        return

    # Main loop
    while True:
        snap = _compute_snapshot(run_dir, stage_subdir=args.stage_subdir)
        _write_snapshot(monitor_dir, snap)
        _print_console(snap)

        if snap.n_total and snap.stage2_ok >= snap.n_total:
            # One final write and exit on completion.
            return
        time.sleep(max(1.0, float(args.interval_s)))


if __name__ == "__main__":
    main()
