# Jury checklist (quick verification)

This is a short, **data-free** checklist to validate that the repository is:
- runnable end-to-end on a synthetic note,
- reproducible (tests + hygiene gate),
- aligned with a local/edge deployment workflow (llama.cpp),
- stable under Stage 2 CAG (prompt cache).

## 0) Repo hygiene (public-safety gate)

```bash
python3 -m pip install -r requirements-dev.txt
pytest -q
python3 scripts/check_repo_hygiene.py
```

Expected:
- tests pass
- hygiene gate prints `Repo hygiene check OK.`

## 1) Start local backends (llama.cpp examples)

Model weights (optional, not included in this repo):
- https://huggingface.co/DocUA/medgemma-1.5-4b-it-gguf-q5-k-m-two-stage

Stage 1 (base model):
```bash
llama-server -m /path/to/base.gguf --alias medgemma-base --host 127.0.0.1 --port 1245 --ctx-size 8192
```

Stage 2 (fine-tuned / LoRA / merged model) with CAG enabled:
```bash
llama-server -m /path/to/medgemma-base-q5_k_m.gguf --lora /path/to/lora_stage2_all_hard200_20260207-f16.gguf \
  --alias medgemma-stage2 --host 127.0.0.1 --port 1246 --ctx-size 8192 \
  --cache-prompt --cache-reuse 256
```

Sanity check (both servers must list the alias in `/v1/models`):
```bash
curl -s http://127.0.0.1:1245/v1/models | head
curl -s http://127.0.0.1:1246/v1/models | head
```

## 2) Run a full end-to-end smoke test (synthetic note)

Create a tiny synthetic cohort (no data required):
```bash
mkdir -p local_cohort/10000001
cat > local_cohort/10000001/ehr_10000001.txt << 'EOF'
DISCHARGE SUMMARY (synthetic)
Vitals on admission: HR 92, BP 120/80, RR 18, Temp 98.6 F, SpO2 98%.
Labs: Sodium 138, Potassium 4.0, Creatinine 1.2, BUN 18, Glucose 110.
Discharge disposition: Home. Mental status: alert.
EOF
```

Run extraction:
```bash
python3 scripts/run_two_stage_structured_sequential.py \
  --cohort-root local_cohort \
  --out-dir results/jury_smoke \
  --hadm-ids 10000001 \
  --stage1-url http://127.0.0.1:1245 --stage1-model medgemma-base --stage1-profile sgr_v2 \
  --stage2-url http://127.0.0.1:1246 --stage2-model medgemma-stage2
```

## 2b) (Optional) Run the demo UI (Gradio)

This is a visual inspector for Stage1/Stage2 outputs, gates, and the rule-based risk engine.

```bash
python3 -m pip install -r requirements-demo.txt
python3 apps/challenge_demo/app_challenge.py
```

Open:
- `http://127.0.0.1:7863`

## 3) What to inspect in artifacts

All per-document artifacts are under:
- `results/jury_smoke/raw_stage_traces/10000001/`

Core files:
- `stage1.json`:
  - should be valid JSON
  - should contain the 9 domain keys (DEMOGRAPHICS/VITALS/LABS/PROBLEMS/SYMPTOMS/MEDICATIONS/PROCEDURES/UTILIZATION/DISPOSITION)
- `stage1.md`:
  - should be a deterministic Markdown rendering of `stage1.json`
  - must not contain the `|` character (to avoid KVT4 leakage into Stage 2 prompts)
- `stage2_raw.txt`:
  - raw model output (for drift/debug)
- `stage2_facts.txt`:
  - sanitized KVT4 fact lines
  - each line must match: `CLUSTER|Keyword|Value|Timestamp`
- `stage2_normalized.json`:
  - normalized facts + normalization stats + format stats (the evaluation contract)

Quick checks on `stage2_facts.txt`:
- numeric-only enforcement for `VITALS/LABS/UTILIZATION` (no units, no words)
- timestamps constrained to: `Past | Admission | Discharge | Unknown`
- no duplicates for objective/integral clusters (at most one per `(CLUSTER, Keyword)`)

## 4) Monitor progress (agentic / operational UX)

While a batch run is executing (or after), run:
```bash
python3 scripts/monitor_extraction_progress.py --block-dir results/jury_smoke --once
python3 scripts/monitor_extraction_progress.py --block-dir results/jury_smoke
```

Open:
- `results/jury_smoke/monitor/progress.html`

This monitor only counts files (it does not read note text).

## 5) Stage 2 CAG correctness (no-diffs A/B)

Goal: verify that enabling llama.cpp prompt caching (CAG) does **not** change Stage 2 outputs.

Prereq: you need two Stage 2 backends pointing to the same model:
- one with CAG enabled (e.g. `--cache-prompt --cache-reuse 256`)
- one without CAG

Run A/B on the same Stage 1 artifacts:
```bash
python3 scripts/check_stage2_cag_ab_smoke.py \
  --src-out-dir results/jury_smoke \
  --cohort-root local_cohort \
  --out-root results/jury_cag_ab \
  --hadm-ids 10000001 \
  --url-a http://127.0.0.1:1246 \
  --url-b http://127.0.0.1:1247 \
  --model medgemma-stage2 \
  --strict-stage2-normalized-json
```

Expected:
- `stage2_facts.txt` and `stage2_normalized.json` are byte-identical between A and B

## 6) Common failure modes (fast triage)

- `Failed to verify model availability via /v1/models`:
  - confirm you can `curl <url>/v1/models`
  - confirm your `--stage1-model` / `--stage2-model` equals the backend model id (llama.cpp: `--alias`)
- Empty `stage2_facts.txt`:
  - inspect `stage2_raw.txt` to determine whether the backend returned an empty generation vs a parsing/sanitization drop
  - check that Stage 2 is using the intended fine-tuned weights (model id mismatch can silently degrade outputs)
