# Kaggle demo (GPU notebook) — MedGemma StructCore

This is an optional deployment path for **interactive demos and GPU smoke tests** using Kaggle Notebooks.

Kaggle is great for:
- free/low-cost GPU access,
- quick reproducible runs,
- recording a short demo video.

Kaggle is *not* ideal for:
- a long-lived public web app (sessions time out),
- handling any real clinical note text (do not upload PHI).

## Working one-cell launcher (canonical)

The repository includes a working Kaggle one-cell launcher copied from a validated notebook:

- `scripts/kaggle_one_cell_launcher.py`

Recommended use in Kaggle:

1. Open this file in GitHub.
2. Copy all content into one Kaggle notebook cell.
3. Run the cell in a GPU notebook.

## Kaggle secrets (optional, for cloud comparison)

To enable Gemini comparison in the demo UI:

1. In Kaggle notebook, open **Add-ons -> Secrets**.
2. Create secret:
   - Name: `GEMINI_API_KEY`
   - Value: your Gemini API key
3. The launcher `scripts/kaggle_one_cell_launcher.py` loads it automatically via `UserSecretsClient`.
4. The launcher keeps the key in process environment only (no key persistence to `.env` or datasets).

If the secret is absent, local pipeline and mock modes still work.

## 0) Notebook settings

- Create a new Kaggle Notebook.
- In the right panel: **Settings → Accelerator → GPU**.
- Confirm GPU availability:

```bash
!nvidia-smi
```

## 1) Clone the public repo

```bash
!git clone https://github.com/SZabolotnii/MedGemma_StructCore.git
%cd MedGemma_StructCore
```

## 2) Install Python dependencies

```bash
!python3 -m pip install -r requirements-demo.txt -q
```

If you only want the CLI runners (no UI), you can install minimal deps (stdlib + repo code) and skip this step.

## 3) Get GGUF weights (Stage 1 + Stage 2)

This repo ships **no weights**. Download your GGUF artifacts from Hugging Face.

Model repo:
- `DocUA/medgemma-1.5-4b-it-gguf-q5-k-m-two-stage`

Published artifacts (as of 2026-02-19):
- Base model GGUF: `medgemma-base-q5_k_m.gguf`
- Stage2 LoRA adapter (GGUF): `lora_stage2_all_hard200_20260207/lora_stage2_all_hard200_20260207-f16.gguf`

Example:

```python
from huggingface_hub import hf_hub_download

REPO_ID = "DocUA/medgemma-1.5-4b-it-gguf-q5-k-m-two-stage"

stage1_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="medgemma-base-q5_k_m.gguf",
)
stage2_base_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="medgemma-base-q5_k_m.gguf",
)
stage2_lora_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="lora_stage2_all_hard200_20260207/lora_stage2_all_hard200_20260207-f16.gguf",
)

print(stage1_path)
print(stage2_base_path)
print(stage2_lora_path)
```

If the model repo is gated/private, add `token=...` or set `HF_TOKEN` in Kaggle secrets.

## 4) Build llama.cpp (CUDA) and start local OpenAI-compatible servers

### Option A (recommended): two servers at once (Stage1 + Stage2)

This is the most convenient for `scripts/run_two_stage_structured_sequential.py` and the demo UI.

```bash
!git clone https://github.com/ggerganov/llama.cpp.git
%cd llama.cpp
# Kaggle images sometimes ship an older CMake; if you hit errors about missing CUDA targets,
# upgrade CMake first:
!python3 -m pip install -q -U cmake ninja

# Build with CUDA (modern flag is GGML_CUDA)
!cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
!cmake --build build -j
%cd ..
```

Start Stage 1 and Stage 2 servers in the background:

```python
import os
import subprocess
import time
import urllib.request

LLAMA_SERVER = "./llama.cpp/build/bin/llama-server"

def wait_models(url: str, timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception as e:
            last_err = e
        time.sleep(1)
    raise RuntimeError(f"Server not ready: {url} (last error: {last_err})")

stage1 = subprocess.Popen(
    [
        LLAMA_SERVER,
        "-m", stage1_path,
        "--alias", "medgemma-base",
        "--host", "127.0.0.1",
        "--port", "1245",
        "--ctx-size", "8192",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

stage2 = subprocess.Popen(
    [
        LLAMA_SERVER,
        "-m", stage2_base_path,
        "--lora", stage2_lora_path,
        "--alias", "medgemma-stage2",
        "--host", "127.0.0.1",
        "--port", "1246",
        "--ctx-size", "8192",
        "--cache-prompt",
        "--cache-reuse", "256",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

wait_models("http://127.0.0.1:1245", 120)
wait_models("http://127.0.0.1:1246", 120)
print("Stage1/Stage2 servers are ready.")
```

### Option B: manual 2-pass mode (lower VRAM)

If running two servers simultaneously does not fit in GPU memory:

1) Start Stage1 server only → run `stage1` pass
2) Stop Stage1 server → start Stage2 server only → run `stage2` pass

This uses `scripts/run_two_stage_structured_pipeline.py` directly:

```bash
python3 scripts/run_two_stage_structured_pipeline.py stage1 --help
python3 scripts/run_two_stage_structured_pipeline.py stage2 --help
```

## 5) Run the demo UI (Gradio) in the notebook

```python
import os
from apps.challenge_demo.app_challenge import build_demo

os.environ["STRUCTCORE_BACKEND_MODE"] = "pipeline"
os.environ["STRUCTCORE_STAGE1_URL"] = "http://127.0.0.1:1245"
os.environ["STRUCTCORE_STAGE1_MODEL"] = "medgemma-base"
os.environ["STRUCTCORE_STAGE2_URL"] = "http://127.0.0.1:1246"
os.environ["STRUCTCORE_STAGE2_MODEL"] = "medgemma-stage2"

demo = build_demo()
demo.launch(share=False)
```

If you want an offline-only demo (no servers), use:

```python
os.environ["STRUCTCORE_BACKEND_MODE"] = "mock"
```

## 6) Run the CLI smoke test (synthetic, no PHI)

```bash
mkdir -p local_cohort/10000001
cat > local_cohort/10000001/ehr_10000001.txt << 'EOF'
DISCHARGE SUMMARY (synthetic)
Vitals on admission: HR 92, BP 120/80, RR 18, Temp 98.6 F, SpO2 98%.
Labs: Sodium 138, Potassium 4.0, Creatinine 1.2, BUN 18, Glucose 110.
Discharge disposition: Home. Mental status: alert.
EOF

python3 scripts/run_two_stage_structured_sequential.py \
  --cohort-root local_cohort \
  --out-dir results/kaggle_smoke \
  --hadm-ids 10000001 \
  --stage1-url http://127.0.0.1:1245 --stage1-model medgemma-base --stage1-profile sgr_v2 \
  --stage2-url http://127.0.0.1:1246 --stage2-model medgemma-stage2
```

## Notes / pitfalls

- Kaggle sessions time out; keep runs short and save artifacts you need.
- Do not upload or publish restricted clinical data.
- `--alias` must match `--stage1-model` / `--stage2-model` (OpenAI `/v1/models` id).
- If you see `Failed to verify model availability via /v1/models`, open the URL from inside the notebook:
  - `curl -s http://127.0.0.1:1245/v1/models | head`
  - `curl -s http://127.0.0.1:1246/v1/models | head`
