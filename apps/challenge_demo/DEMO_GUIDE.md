# MedGemma StructCore — Demo User Guide

Welcome to **MedGemma StructCore** — a local-first clinical NLP engine that extracts structured facts from free-text EHR discharge notes and predicts 30-day hospital readmission risk.

> **Privacy guarantee:** In local (pipeline) mode, patient data **never leaves your machine**. No cloud API, no PHI leakage.

---

## 🚀 Getting Started

### Step 1 — Case Input tab

| Action | How |
|---|---|
| **Pre-loaded synthetic case** | Choose from the **Synthetic case** dropdown (e.g., *High Risk: Decompensated Heart Failure*) |
| **Upload your own note** | Click **📂 Load File** — supports `.txt`, `.pdf`, `.docx` |
| **Paste custom text** | Select *"✍️ Paste Custom Note"* from the dropdown, then type directly in the text area |

> ⚠️ **PDF disclaimer:** Only PDFs with **embedded (selectable) text** are supported. Scanned documents or image-based PDFs require OCR pre-processing — **OCR is not supported in this demo**. If your PDF shows empty extraction results, convert it to `.txt` first.

### Step 2 — Run the pipeline

Click the **🚀 Run StructCore** button (top-right of the Case Input tab).

The two-stage pipeline runs locally:
1. **Stage 1 — Domain Summary** (MedGemma base, sgr_v2 schema) → Markdown clinical summary with 9 clusters  
2. **Stage 2 — KVT4 Projection** (MedGemma + LoRA adapter, CAG) → Structured `CLUSTER|Keyword|Value|Timestamp` facts  
3. **Risk Engine** → Deterministic rule-based score calibrated on MIMIC-IV (α=−2.3475, β=0.017)

Progress is visible in the **⏱️ Live Status** panel on the right.

---

## 📊 Interpreting Results

### Tab 2 — 🔍 StructCore Inspector

Transparency into what the model extracted:
- **Stage1 output** — raw Markdown domain summary (9 clinical clusters)
- **Stage2 output** — raw KVT4 lines before normalization
- **Normalized KVT4 facts** — cleaned `[CLUSTER | Keyword | Value | Timestamp]` table
- **Quality Gate JSON** — parse success, cluster counts, format validity score

### Tab 3 — 📊 Risk View

Main decision-support dashboard:

| Element | Meaning |
|---|---|
| **Risk Gauge** | 30-day readmission probability (0–100%) |
| **Cluster Breakdown** | Which clinical domains contributed most (max 215 points total) |
| **Risk Category** | Low (<12%) → Medium (12–16%) → High (16–21%) → Critical (>21%) |
| **Risk Factors** | Top contributing KVT4 facts (e.g., *Heart Rate=118, Tachycardia +3*) |

### Tab 4 — ⚖️ Comparative Analysis

Side-by-side comparison of **MedGemma (local)** vs **Gemini 2.5 Flash** vs **Gemini 3 Flash Preview**:
- Select which models to include with the checkboxes
- Click **⚖️ Run Comparison** — the local result is reused from cache if you already ran Step 2
- Privacy table shows PHI exposure risk per model

### Tab 5 — 📋 Evidence Board

Scientific validation of every system claim:
- Cards are color-coded: **Verified** (green) | **Preliminary** (amber) | **Planned** (pink)
- **C01 (99.74% format validity)** is the primary extraction proof point
- AUROC charts compare MedGemma extraction against LACE/HOSPITAL and tabular ML baselines
- All metrics from the MIMIC-IV Track B benchmark (N=9,857)

### Tab 6 — 💰 Clinical Impact

ROI scenario calculator:
- Adjust sliders: **Annual discharges**, **Baseline readmission rate (%)**, **Cost per readmission (USD)**
- Click **📊 Calculate Impact** to update projections
- Three scenarios: Conservative (2% reduction) / Moderate (5%) / Optimistic (8%)

### Tab 7 — 🏗️ Architecture

Visual overview of the two-stage pipeline (image + annotated text blocks).  
Shows model names, sizes, backends (llama.cpp), and output formats at each stage.

---

## ⚙️ Backend Modes

| Mode | Description |
|---|---|
| **pipeline** (recommended) | Calls local Stage 1 + Stage 2 MedGemma servers via OpenAI-compatible API |
| **mock** | Offline synthetic demo — no model required, shows pre-generated outputs |
| **Fallback to mock** | If enabled, automatically falls back to mock if the local servers are unreachable |

> To run in pipeline mode, both Stage 1 and Stage 2 llama.cpp servers must be running on ports 8081/8082 (configurable in the ⚙️ Pipeline settings accordion).

---

## ❓ FAQ

**Q: What does AUROC 0.6024 / 0.6846 mean?**  
A: AUROC measures ranking accuracy. 0.5 = random. The rule-engine (A1) achieves 0.6024 on the full MIMIC-IV Track B test (N=9,857). The tabular ML model (A4) achieves 0.6846 on a subset. Both exceed all proxy baselines (LACE, HOSPITAL).

**Q: Why is >21% risk "Critical"?**  
A: The baseline 30-day readmission rate for stable patients is ~5–10%. A probability of >21% represents a 4×+ increase in risk — statistically significant and clinically actionable.

**Q: Is my data sent to the cloud?**  
A: **No**, in `pipeline` or `mock` backend modes. Data stays on your machine. If you run Comparative Analysis with Gemini models enabled, those notes are sent to the Gemini API.

**Q: Can I use my own clinical notes?**  
A: Yes — use **📂 Load File** or paste directly. Avoid real patient data (PHI) in public demo deployments; use synthetic or de-identified notes.

**Q: What is the KVT4 format?**  
A: `CLUSTER|Keyword|Value|Timestamp` — a compact, parseable line format for clinical facts. Example: `VITALS|Heart Rate|118.0|Admission`. The pipe-delimited format enables deterministic rule-based scoring.
