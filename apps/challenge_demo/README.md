# MedGemma StructCore Demo App

This is the implementation-focused demo app for:

**MedGemma StructCore: Local-First Clinical Structuring Engine for EHR**

## Run

```bash
python3 apps/challenge_demo/app_challenge.py
```

Open: `http://localhost:7863`

## Deployment (Current)

- **Primary demo path:** Kaggle notebook + local `llama-server` backends.
- **Secondary path:** local workstation demo for recording/review.
- **HF Zero bundle path (`hf_zero`) is retired** from active competition flow.

## Model Artifacts (Source of Truth)

Two-stage model repository:

- `https://huggingface.co/DocUA/medgemma-1.5-4b-it-gguf-q5-k-m-two-stage`

Upload/update artifacts:

```bash
python3 scripts/hf_upload_two_stage_models.py \
  --repo-id DocUA/medgemma-1.5-4b-it-gguf-q5-k-m-two-stage \
  --stage1-file /absolute/path/to/stage1.gguf \
  --stage2-file /absolute/path/to/stage2.gguf
```

## Runtime Modes

- `mock`:
  - offline deterministic extraction (fast, no model server required),
  - useful for demo recording and UI development.

- `pipeline`:
  - runs real Stage1/Stage2 using existing runners,
  - requires local OpenAI-compatible model servers.

If pipeline mode fails and fallback is enabled, app falls back to mock mode.

Runtime env vars:

- `STRUCTCORE_STAGE1_URL`, `STRUCTCORE_STAGE1_MODEL`
- `STRUCTCORE_STAGE2_URL`, `STRUCTCORE_STAGE2_MODEL`
- optional: `STRUCTCORE_BACKEND_MODE=mock|pipeline`

## Kaggle Secrets (Cloud Comparison)

For Gemini comparison blocks in public notebooks:

- store `GEMINI_API_KEY` in Kaggle Secrets,
- do not persist keys in repository files,
- avoid writing `.env` with real credentials in public artifacts.

## Architecture

- `app_challenge.py`: Gradio UI and orchestration glue.
- `services/structcore_service.py`: execution modes, normalization, risk scoring.
- `services/case_library.py`: synthetic demo cases.
- `services/evidence_service.py`: claim/evidence board data.
- `config/evidence_claims.json`: status-labeled claims.
- `data/synthetic_cases.json`: synthetic note samples.

## Notes

- This demo is extraction-first.
- Readmission risk is presented as a downstream use case.
- Public demos should use synthetic notes only.
- `Run StructCore` now shows immediate processing state in Live Status for clear user feedback.
