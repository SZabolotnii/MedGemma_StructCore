# Prompts (Stage 1 / Stage 2)

This folder contains prompts and mappings for the **two-stage** extraction pipeline:

- **Stage 1:** schema-guided domain summary (JSON → Markdown)
- **Stage 2:** projection of Stage 1 Markdown → **KVT4 facts**

Prompts are used by:
- `scripts/run_two_stage_structured_pipeline.py`
- `scripts/run_two_stage_structured_sequential.py`

## KVT4 contract (Stage 2 output)

One fact per line:

```
CLUSTER|Keyword|Value|Timestamp
```

Examples:

```
VITALS|Heart Rate|92|Admission
LABS|Creatinine|1.4|Admission
DISPOSITION|Discharge Disposition|Home|Discharge
```

Notes:
- `VITALS/LABS` values must be **numeric-only** (no units).
- Timestamp is normalized to: `Past | Admission | Discharge | Unknown`.

## Profiles

Stage 1 supports profiles (system prompt + schema), e.g. `sgr_v2`.
Stage 2 is a fixed “lines-only” contract with sanitizer/normalizer logic in `kvt_utils.py`.

## Mapping files

JSON mappings (disposition/medications/problems/procedures/symptoms) are used
to normalize values to the canonical ontology (see `ONTOLOGY.md`).
