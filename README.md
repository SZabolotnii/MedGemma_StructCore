# MedGemma StructCore (public, data-free)

StructCore is a minimal, local-first **two-stage structured extraction** stack for clinical notes:

- **Stage 1 (BASE model):** produce a compact, schema-shaped clinical summary (JSON → Markdown).
- **Stage 2 (FT/LoRA model):** consume only Stage 1 Markdown and emit **KVT4 facts**:
  `CLUSTER|Keyword|Value|Timestamp`

This repo is intentionally **code + prompts + schemas only**. It ships **no datasets, no note text, no weights**. A small sanitized benchmark snapshot (aggregated metrics only) is included for demo UI transparency.

## What you get

- OpenAI-compatible client: `openai_compat.py`
- KVT4 parsing/normalization (DSPy-free): `kvt_utils.py`
- Prompt profiles + mappings: `prompts/`
- JSON schemas (Stage 1 / Stage 2): `schemas/`
- Public benchmark snapshot (aggregated only): `apps/challenge_demo/config/benchmark_public_snapshot.json`
- Runners + tooling: `scripts/`
  - `scripts/run_two_stage_structured_sequential.py` (recommended)
  - `scripts/run_two_stage_structured_pipeline.py` (manual 2-pass mode)
  - `scripts/monitor_extraction_progress.py` (progress + ETA + HTML)
  - `scripts/build_stage2_hybrid_facts_from_stage1_md.py` (offline hybrid facts)

## How it works (one page)

**Stage 1 (base model)** → **Stage 2 (fine-tuned / LoRA model)**

- Stage 1 input: raw note text
- Stage 1 output: a domain summary as **Structured JSON** (schema-constrained when supported) and a deterministic **Markdown** rendering
- Stage 2 input: **Stage 1 Markdown only** (never the raw note text)
- Stage 2 output: **KVT4** fact lines: `CLUSTER|Keyword|Value|Timestamp`

Expected local cohort layout (not provided in this repo):
```text
<cohort_root>/
  <hadm_id>/
    ehr_<hadm_id>.txt
    ground_truth_<hadm_id>.json   # optional (curated eval only)
```

Output artifacts (under `--out-dir`):
```text
<out_dir>/
  raw_stage_traces/
    <hadm_id>/
      stage1.json
      stage1.md
      stage2_raw.txt
      stage2_facts.txt
      stage2_normalized.json
      stage2_metrics.json
```

Where to look:
- Contract and allowlists: `ONTOLOGY.md`
- “Run this / inspect that” checklist: `JURY_CHECKLIST.md`

## Safety / data hygiene

Do **not** commit clinical note text or restricted datasets (even if de-identified).
See `DATA_HYGIENE.md`.

## Quickstart (local llama.cpp)

1) Start an OpenAI-compatible backend for **Stage 1** (base weights):

```bash
llama-server -m /path/to/base.gguf --alias medgemma-base --host 127.0.0.1 --port 1245 --ctx-size 8192
```

## Model weights (optional)

This repository does not include model weights. For convenience, our two-stage GGUF releases are published on Hugging Face:
- https://huggingface.co/DocUA/medgemma-1.5-4b-it-gguf-q5-k-m-two-stage

Use is subject to upstream MedGemma terms; see `MODEL_TERMS.md`.

Published artifacts (as of 2026-02-19):
- Base GGUF: `medgemma-base-q5_k_m.gguf`
- Stage2 LoRA adapter (GGUF): `lora_stage2_all_hard200_20260207/lora_stage2_all_hard200_20260207-f16.gguf`

2) Start an OpenAI-compatible backend for **Stage 2** (base + LoRA adapter).
Enable prompt caching (CAG) for speed:

```bash
llama-server -m /path/to/medgemma-base-q5_k_m.gguf --lora /path/to/lora_stage2_all_hard200_20260207-f16.gguf \
  --alias medgemma-stage2 --host 127.0.0.1 --port 1246 --ctx-size 8192 \
  --cache-prompt --cache-reuse 256
```

3) Run sequential extraction over your local cohort directory (no data included here):

```bash
python3 scripts/run_two_stage_structured_sequential.py \
  --cohort-root /path/to/EHR_test_data \
  --out-dir results/run_YYYYMMDD_HHMMSS \
  --stage1-url http://127.0.0.1:1245 --stage1-model medgemma-base --stage1-profile sgr_v2 \
  --stage2-url http://127.0.0.1:1246 --stage2-model medgemma-stage2
```

`results/` is ignored by default to prevent accidental publication of derived artifacts.

## Jury quickstart (synthetic note, no data required)

Create a tiny local cohort with a synthetic note:

```bash
mkdir -p local_cohort/10000001
cat > local_cohort/10000001/ehr_10000001.txt << 'EOF'
DISCHARGE SUMMARY (synthetic)
Vitals on admission: HR 92, BP 120/80, RR 18, Temp 98.6 F, SpO2 98%.
Labs: Sodium 138, Potassium 4.0, Creatinine 1.2, BUN 18, Glucose 110.
Discharge disposition: Home. Mental status: alert.
EOF
```

Then run on that single document:

```bash
python3 scripts/run_two_stage_structured_sequential.py \
  --cohort-root local_cohort \
  --out-dir results/jury_smoke \
  --hadm-ids 10000001 \
  --stage1-url http://127.0.0.1:1245 --stage1-model medgemma-base --stage1-profile sgr_v2 \
  --stage2-url http://127.0.0.1:1246 --stage2-model medgemma-stage2
```

Artifacts will be written under `results/jury_smoke/raw_stage_traces/10000001/`.

## Demo UI (optional)

Run the Gradio demo app (synthetic cases + local pipeline mode):

```bash
python3 -m pip install -r requirements-demo.txt
python3 apps/challenge_demo/app_challenge.py
```

Open: `http://127.0.0.1:7863`

Tip: for an offline-only demo (no backends), set `STRUCTCORE_BACKEND_MODE=mock`.

## Backend sanity check

Both runners validate backend readiness via `GET /v1/models`. You can confirm manually:

```bash
curl -s http://127.0.0.1:1245/v1/models | head
```

For llama.cpp, the model id is the `--alias` you started the server with.

## Monitoring

While extraction runs:

```bash
python3 scripts/monitor_extraction_progress.py --block-dir results/run_YYYYMMDD_HHMMSS
```

Open `results/run_YYYYMMDD_HHMMSS/monitor/progress.html` in a browser.

## Development

Run tests locally:

```bash
python3 -m pip install -r requirements-dev.txt
pytest -q
```

Run the public-repo hygiene gate:

```bash
python3 scripts/check_repo_hygiene.py
```

## License

- Code: Apache-2.0 (`LICENSE`)
- Documentation: CC BY 4.0 (`LICENSE-DOCS`)

Model weights remain governed by upstream terms; see `MODEL_TERMS.md`.
