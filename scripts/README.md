# Scripts

Scripts for **two-stage** structured extraction (Stage 1 → Stage 2) via an OpenAI-compatible backend
(llama.cpp `llama-server`, LM Studio, vLLM, etc.).

This repository ships **no data and no model weights**. You must run your own backend and point the scripts to your local `--cohort-root`.

## Core (recommended)

- `run_two_stage_structured_sequential.py` — Stage 1/Stage 2 orchestrator for batch runs (resume-friendly).
- `run_two_stage_structured_pipeline.py` — direct 2-pass runner (manual mode: `stage1` → restart backend → `stage2`).
- `smoke_two_stage_structured.py` — single-document smoke check.

## Quality / stability tools

- `check_two_stage_structured_gates.py` — offline gates (format/contract) without running an LLM.
- `check_stage2_cag_ab_smoke.py` — A/B smoke check that Stage 2 CAG (prompt cache) does not change outputs.
- `build_stage2_hybrid_facts_from_stage1_md.py` — build `stage2_facts_hybrid.txt` (offline) from Stage1.md + Stage2 raw.

## Progress monitoring

- `monitor_extraction_progress.py` — live progress/ETA + HTML (does not read note text; counts files only).
- `monitor_stage2_progress.py` — lightweight Stage 2 facts monitor.

## Other

- `prepare_two_stage_weights.py` — helper for preparing a consistent Stage 1/Stage 2 weight pair (weights are not included in this repo).
- `kaggle_one_cell_launcher.py` — validated one-cell Kaggle GPU launcher for Stage1/Stage2 + Gradio demo.

## Output artifacts

By default, outputs are written to `results/...` (and **ignored** via `.gitignore`) to reduce the risk of accidentally publishing derived artifacts.
