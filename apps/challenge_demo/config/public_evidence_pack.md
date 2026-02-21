# Public evidence pack (data-free)

This public repository is intentionally **data-free** (no clinical notes, no restricted datasets, no model weights, no run artifacts).

It still ships **aggregated, non-identifying benchmark summaries** for transparency in the demo UI:

- `apps/challenge_demo/config/benchmark_public_snapshot.json`

## What is (and is not) included

- Included:
  - Aggregated metrics only (e.g., AUROC with optional confidence intervals and test set size).
  - Reproducible code paths and scripts to run the pipeline on a local cohort (you must supply your own allowed data).
- Not included:
  - Any restricted cohort files or derived per-encounter artifacts.
  - Any `results/benchmark/...` folders from private/internal experimentation.

## Stage2 CAG A/B (prompt cache) — public summary

We provide a data-free A/B correctness summary for Stage2 CAG (llama.cpp prompt cache):

- `apps/challenge_demo/config/cag_ab_summary_public.json`

This is intended to support the claim that enabling prompt caching does not change outputs (byte-identical artifacts) on the evaluated split, while improving throughput.

## Reproduction notes

Exact downstream benchmark reproduction typically requires access to the same (restricted) clinical cohort and label definitions. This repository provides the pipeline and evaluation scripts, but does not ship the underlying datasets.

