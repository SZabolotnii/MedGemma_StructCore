# ONTOLOGY.md — readmission extraction ontology (v1)

**Goal:** define a *stable* structured extraction contract used by StructCore for downstream readmission feature engineering and evaluation.

This ontology specifies:
- what we extract (clusters + keywords),
- how it is represented (KVT4 lines),
- which values are allowed,
- which deterministic normalization/dedup rules are canonical.

This is a **public, data-free** repository. Do not add or commit any note text or restricted datasets.

---

## 1) Data representations (contracts)

### 1.1) Final fact-level contract (KVT4)

One fact per line:

```
CLUSTER|Keyword|Value|Timestamp
```

**Strict requirements**
- exactly 4 fields (exactly 3 `|` characters)
- `CLUSTER` must be one of the 9 clusters in §2
- `Timestamp` must be one of: `Past | Admission | Discharge | Unknown`
- for `VITALS/LABS/UTILIZATION`: `Value` must be **numeric-only** (no units, no words)
- duplicates are forbidden for objective/integral clusters (see §4.2)

**Canonical parser/normalizer**
- `kvt_utils.py:extract_kvt_fact_lines()`
- `kvt_utils.py:normalize_readmission_kvt4_lines()`

### 1.2) Stage 1 contract (Structured JSON → Markdown)

Stage 1 returns a JSON object constrained by a schema (when supported by the backend):
- schema files: `schemas/readmission_domain_summary*.schema.json`
- required top keys (always present):
  `DEMOGRAPHICS, VITALS, LABS, PROBLEMS, SYMPTOMS, MEDICATIONS, PROCEDURES, UTILIZATION, DISPOSITION`

Each value is a short evidence-based string. Stage 1 JSON is then deterministically rendered into Markdown sections:

```
## DEMOGRAPHICS
...
## VITALS
...
```

**Stage 2 consumes ONLY Stage 1 Markdown** (never the raw note text).

---

## 2) Clusters (9)

The ontology uses the same 9 clusters in Stage 1 and Stage 2:

1) `DEMOGRAPHICS` — baseline demographics  
2) `VITALS` — objective vital signs (numeric-only)  
3) `LABS` — objective lab values (numeric-only)  
4) `PROBLEMS` — diagnoses / conditions (semantic)  
5) `SYMPTOMS` — symptoms / complaints (semantic)  
6) `MEDICATIONS` — integral medication flags/counters  
7) `PROCEDURES` — integral procedure/intervention flags  
8) `UTILIZATION` — integral healthcare utilization / administrative counts  
9) `DISPOSITION` — discharge outcome / state at discharge

Rule of thumb:
- `VITALS/LABS/DISPOSITION/UTILIZATION` are the most stable for objective gates.
- `PROBLEMS/SYMPTOMS` are high-value but require stronger safeguards against false positives.

---

## 3) Cluster specifications

### 3.1) DEMOGRAPHICS

**Keywords (MVP)**
- `Sex` — `male|female` (lowercase)
- `Age` — numeric-only (years)

**Timestamp (recommended):** `Admission`

### 3.2) VITALS (numeric-only, canonical keywords)

**Keywords (STRICT, exact match)**
- `Heart Rate`
- `Systolic BP`
- `Diastolic BP`
- `Respiratory Rate`
- `Temperature`
- `SpO2`
- `Weight`

**Value:** numeric-only (`-?\d+(\.\d+)?`)

**Timestamp:** typically `Admission` or `Discharge` (use `Unknown` only when unavoidable).

**Canonical extraction/sanitization notes**
- If BP appears as `120/80`, emit two facts: SBP=120 and DBP=80 (same timestamp).
- `SpO2` must be a number without `%` and without `"RA"`.
- `Temperature` must be numeric-only (no `F`/`C` suffix).

**QA ranges (soft flags, not hard drops)**
- HR: 30–220
- SBP: 50–260
- DBP: 20–160
- RR: 5–60
- SpO2: 50–100
- Weight: >0

### 3.3) LABS (numeric-only, canonical keywords)

**Keywords (STRICT, exact match)**
- `Hemoglobin`
- `Hematocrit`
- `WBC`
- `Platelet`
- `Sodium`
- `Potassium`
- `Creatinine`
- `BUN`
- `Glucose`
- `Bicarbonate`

**Value:** numeric-only.

**Timestamp:** typically `Admission` or `Discharge`.

**QA ranges (soft flags)**
- Sodium: 100–180
- Potassium: 1.5–8.0
- Creatinine: 0.1–20
- BUN: 1–200
- Glucose: 20–1000

### 3.4) DISPOSITION (high-signal, integral)

**Keywords (MVP)**
- `Discharge Disposition`
- `Mental Status`

**Discharge Disposition (allowlist, exact)**
- `Home`
- `Home with Services`
- `SNF`
- `Rehab`
- `LTAC`
- `Hospice`
- `AMA`

**Mental Status (allowlist, exact)**
- `alert`
- `confused`
- `oriented`
- `lethargic`

**Timestamp:** `Discharge`

Mappings used by prompts/normalization:
- `prompts/disposition_mapping.json`

### 3.5) UTILIZATION (numeric-only, integral)

**Keywords (STRICT, numeric-only)**
- `Prior Admissions 12mo`
- `ED Visits 6mo`
- `Days Since Last Admission`
- `Current Length of Stay`

**Value:** numeric-only.

**Timestamp (recommended)**
- `Past` for historical counts/intervals
- `Admission` or `Unknown` for current LOS (depending on context)

Important: in many end-to-end pipelines, additional utilization fields are best sourced from structured data (FHIR/Encounter) rather than extracted from notes. This ontology keeps the note-extraction contract conservative.

### 3.6) MEDICATIONS (integral)

**Keywords (STRICT set)**
- `Medication Count` (numeric)
- `New Medications Count` (numeric)
- `Polypharmacy` (`yes|no`)
- `Anticoagulation` (`yes|no`)
- `Insulin Therapy` (`yes|no`)
- `Opioid Therapy` (`yes|no`)
- `Diuretic Therapy` (`yes|no`)

**Evidence-only policy (critical)**
- allow `yes` only when there is explicit evidence in Stage 1 Markdown
- if evidence is missing: omit the fact (do not default to `no`)

Mappings used by prompts/normalization:
- `prompts/medications_mapping.json`

### 3.7) PROCEDURES (integral)

**Keywords (STRICT set)**
- `Any Procedure` (`yes|no`)
- `Surgery` (`yes|no`)
- `Dialysis` (`decided|started|done|cancelled|no`)
- `Mechanical Ventilation` (`no` or numeric days)

**Evidence-only:** same principle as MEDICATIONS.

Mappings used by prompts/normalization:
- `prompts/procedures_mapping.json`

### 3.8) PROBLEMS (semantic)

**Keyword:** condition/diagnosis name (free text or normalized label).

**Value (allowlist)**
- `chronic` (typically Past / PMH)
- `acute` (typically Discharge / final dx)
- `exist` (neutral “present”)
- `not exist` (explicit negation only)

**Timestamp (policy)**
- PMH/comorbidities → `Past` + `chronic`
- discharge/final dx → `Discharge` + `acute`
- presenting diagnosis → `Admission` + `acute`

Mappings used by prompts/normalization:
- `prompts/problems_mapping.json`

### 3.9) SYMPTOMS (semantic)

**Keyword:** symptom name.

**Value (allowlist)**
- `yes`
- `no` (explicit negation only)
- `severe`

**Timestamp:** usually `Admission` (sometimes `Discharge`).

Mappings used by prompts/normalization:
- `prompts/symptoms_mapping.json`

---

## 4) Normalization, dedupe, stability

### 4.1) Deterministic normalization

Normalization exists to make datasets and evaluation reproducible without hiding model drift:
- enforce numeric-only for `VITALS/LABS/UTILIZATION`
- normalize `Sex` to lowercase `male|female`
- normalize timestamps into the allowed set (optionally filling `Unknown` deterministically)

### 4.2) Canonical dedupe

Short-term gates typically require:
- objective/integral clusters: **at most 1** fact per `(CLUSTER, Keyword)`
- semantic clusters (`PROBLEMS/SYMPTOMS`): allow multiple facts, but forbid exact duplicate lines

The runner’s source of truth for dedupe is the Stage 2 sanitizer:
- `scripts/run_two_stage_structured_pipeline.py:_sanitize_stage2_lines()`

### 4.3) Scope modes (why it matters)

Stage 2 supports two practical scopes:
- `objective`: extract only stable objective/integral clusters for format + signal gates  
  (typically `DEMOGRAPHICS,VITALS,LABS,UTILIZATION,DISPOSITION`)
- `all`: extract all 9 clusters for dataset building (expect higher drift/FP in semantic clusters)

---

## 5) Versioning / governance

Source-of-truth locations in this repo:
- pipeline prompt templates (Stage 1 + Stage 2): `prompts/optimized_prompt.py`
- strict keyword sets + normalization rules: `kvt_utils.py`
- stage 2 sanitizer rules: `scripts/run_two_stage_structured_pipeline.py`

Any ontology change (new keywords, value allowlists, normalization behavior) should include:
1) updating this file,
2) updating strict keyword sets / schemas as needed,
3) running offline gates and a small regression extraction run.
