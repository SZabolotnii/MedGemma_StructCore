# Data Hygiene (public repo)

This repository is intended to be **public**. It must remain **data-free**.

## Do not commit

- Any clinical note text (raw or derived), including:
  - `EHR_test_data/`, `Curated_EHR_Test_Sets/`
  - files like `ehr_*.txt`, `*.note.txt`, `*.discharge_summary.txt`
- Any MIMIC / PhysioNet dataset files (or processed derivatives), including:
  - `physionet.org/`, `Data_MIMIC/`
  - `*.ndjson`, `*.ndjson.gz`, large cohort `*.csv` exports, etc.
- Model weights and adapters:
  - `*.gguf`, `*.safetensors`, `*.bin`, `*.pt`, ...
- Extraction run artifacts:
  - `results/` (even if it “only” contains metrics, it may still encode restricted IDs or derived distributions)

## Allowed

- Code, prompts, schemas, documentation.
- Small synthetic examples that are clearly synthetic and non-clinical.

## Guardrails

- `.gitignore` excludes common data and artifact paths.
- CI will run `scripts/check_repo_hygiene.py` (added in this repo) to block forbidden paths/patterns.

If you’re unsure whether a file is safe to publish, assume **it is not** and keep it out of this repo.

