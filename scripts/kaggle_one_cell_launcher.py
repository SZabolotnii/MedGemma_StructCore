# =============================================================================
# MedGemma StructCore — Kaggle ONE-CELL launcher
#
# Binary acquisition priority (fastest to slowest):
#   1. Already in working dir         -> skip           (~0 sec)
#   2. Kaggle Dataset input attached  -> copy binary    (~5 sec)
#   3. Prebuilt from ai-dock GitHub   -> download       (~1 min)
#   4. Compile from source (fallback) -> cmake + ninja  (10-30 min)
#
# After step 3 or 4 the binary is saved to Kaggle Dataset automatically,
# so subsequent sessions always use path 2.
#
# Requirements:
#   - Kaggle notebook with 2x GPU (Tesla T4)
#   - Dataset "zabolotnii/llama-server-cuda" attached as Input (after first run)
#   - Optional secret: GEMINI_API_KEY (for Comparative Analysis cloud models)
# =============================================================================

import datetime
import json
import os
import re
import subprocess
import tarfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# =============================================================================
# CONFIG  — edit only this section if needed
# =============================================================================

HF_REPO = "DocUA/medgemma-1.5-4b-it-gguf-q5-k-m-two-stage"
REPO_URL = "https://github.com/SZabolotnii/MedGemma_StructCore.git"
KAGGLE_DATASET_OWNER = "zabolotnii"
KAGGLE_DATASET_SLUG = "llama-server-cuda"

# =============================================================================
# PATHS
# =============================================================================

WORK = Path("/kaggle/working").resolve()
REPO_DIR = WORK / "MedGemma_StructCore"
LLAMA_DIR = REPO_DIR / "llama.cpp"
BUILD_DIR = LLAMA_DIR / "build"
LLAMA_BIN = BUILD_DIR / "bin" / "llama-server"
PREBUILT_DIR = BUILD_DIR / "bin"
# Kaggle mounts datasets at TWO possible paths depending on notebook version:
#   Old: /kaggle/input/{slug}
#   New: /kaggle/input/datasets/{owner}/{slug}
_INPUT_NEW = Path(f"/kaggle/input/datasets/{KAGGLE_DATASET_OWNER}/{KAGGLE_DATASET_SLUG}")
_INPUT_OLD = Path(f"/kaggle/input/{KAGGLE_DATASET_SLUG}")
KAGGLE_INPUT_DIR = _INPUT_NEW if _INPUT_NEW.exists() else _INPUT_OLD
DIST_DIR = WORK / "llama_server_dist"


# =============================================================================
# HELPERS
# =============================================================================

def sh(cmd: str) -> None:
    """Run a shell command, print it, raise on failure."""
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def wait_server(url: str, timeout_s: int = 300) -> None:
    """Block until llama-server /v1/models returns HTTP 200."""
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    print(f"Ready: {url}", flush=True)
                    return
        except Exception as e:
            last_err = e
        time.sleep(1)
    raise RuntimeError(f"Server not ready: {url}  (last error: {last_err})")


def wait_servers_parallel(*urls: str, timeout_s: int = 300) -> None:
    """Wait for multiple servers concurrently."""
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        futures = [pool.submit(wait_server, u, timeout_s) for u in urls]
        for f in futures:
            f.result()


def load_kaggle_secret(name: str) -> str | None:
    """Load a secret from Kaggle Secrets and mirror it into environment."""
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except Exception:
        return None
    try:
        value = UserSecretsClient().get_secret(name)
    except Exception:
        return None
    if value:
        os.environ[name] = value
        print(f"Loaded Kaggle secret: {name}", flush=True)
        return value
    return None


def detect_cuda_and_arch() -> tuple[str, str]:
    """
    Detect CUDA major version and GPU SM architecture.
    Returns e.g. ('12', '75') for Kaggle Tesla T4 + CUDA 12.x.
    """
    cuda_major = "12"
    try:
        out = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT, text=True)
        m = re.search(r"release (\d+)\.", out)
        if m:
            cuda_major = m.group(1)
    except Exception:
        pass

    sm_arch = "75"  # Tesla T4 default
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        cap = out.strip().split("\n")[0].strip().replace(".", "")  # "7.5" -> "75"
        if cap.isdigit():
            sm_arch = cap
    except Exception:
        pass

    return cuda_major, sm_arch


def llama_git_commit() -> str:
    """Return short git hash of the cloned llama.cpp, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(LLAMA_DIR), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


# =============================================================================
# BINARY ACQUISITION — strategy 3: prebuilt from ai-dock/llama.cpp-cuda
# =============================================================================

def try_download_prebuilt(bin_dir: Path) -> bool:
    """
    Download prebuilt llama-server from ai-dock/llama.cpp-cuda GitHub Releases.
    Picks the best asset matching current CUDA version and GPU SM arch.
    Returns True on success, False on any error (caller falls back to compile).

    Note: official ggml-org/llama.cpp does NOT publish Linux CUDA binaries —
    only Windows DLLs. ai-dock fills that gap.
    """
    cuda_major, sm_arch = detect_cuda_and_arch()
    print(f"Detected: CUDA {cuda_major}.x  SM arch {sm_arch}", flush=True)

    api_url = "https://api.github.com/repos/ai-dock/llama.cpp-cuda/releases/latest"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            release = json.loads(r.read())
    except Exception as e:
        print(f"GitHub API error: {e} -> will compile.", flush=True)
        return False

    assets = release.get("assets", [])
    if not assets:
        print("No release assets found -> will compile.", flush=True)
        return False

    def score(name: str) -> int:
        """Score an asset name by how well it matches linux + cuda + sm_arch."""
        n = name.lower()
        if "linux" not in n:
            return 0
        s = 0
        if f"cu{cuda_major}" in n:
            s += 2
        if f"sm_{sm_arch}" in n or f"sm{sm_arch}" in n:
            s += 2
        if n.endswith((".tar.gz", ".tgz", ".zip")):
            s += 1
        return s

    best = max(assets, key=lambda a: score(a["name"]))
    if score(best["name"]) == 0:
        print(f"No suitable Linux CUDA asset (best candidate: {best['name']}) -> will compile.", flush=True)
        return False

    asset_name = best["name"]
    asset_url = best["browser_download_url"]
    print(f"Downloading prebuilt: {asset_name} ...", flush=True)

    bin_dir.mkdir(parents=True, exist_ok=True)
    archive_path = bin_dir / asset_name
    try:
        urllib.request.urlretrieve(asset_url, archive_path)
    except Exception as e:
        print(f"Download failed: {e} -> will compile.", flush=True)
        return False

    try:
        if asset_name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path) as t:
                t.extractall(bin_dir)
        else:
            with zipfile.ZipFile(archive_path) as z:
                z.extractall(bin_dir)
    except Exception as e:
        print(f"Extraction failed: {e} -> will compile.", flush=True)
        return False
    finally:
        archive_path.unlink(missing_ok=True)

    # llama-server may be nested inside the archive — find it recursively
    candidates = list(bin_dir.rglob("llama-server"))
    if not candidates:
        print("llama-server not found in archive -> will compile.", flush=True)
        return False

    server_bin = candidates[0]
    server_bin.chmod(0o755)
    target = bin_dir / "llama-server"
    if server_bin != target:
        server_bin.rename(target)

    print(f"Prebuilt llama-server ready: {target}", flush=True)
    return True


# =============================================================================
# BINARY ACQUISITION — strategy 4: compile from source (fallback)
# =============================================================================

def build_from_source() -> None:
    """Compile llama-server with CUDA support using cmake + ninja."""
    if BUILD_DIR.exists():
        sh(f"rm -rf {BUILD_DIR}")

    cmake_cmd = (
        f"cmake -S {LLAMA_DIR} -B {BUILD_DIR} "
        f"-DBUILD_SHARED_LIBS=OFF "
        f"-DGGML_CUDA=ON "
        f"-DGGML_CUDA_FORCE_CUBLAS=ON "
        f"-DGGML_CUDA_FA=OFF "
        f"-DGGML_CUDA_FA_ALL_QUANTS=OFF "
        f"-DGGML_CUDA_NO_VMM=ON "
        f"-DGGML_CUDA_GRAPHS=OFF "
        f"-DCMAKE_BUILD_TYPE=Release "
        f"-DCUDAToolkit_ROOT=/usr/local/cuda"
    )
    try:
        sh(cmake_cmd)
    except subprocess.CalledProcessError:
        # Fallback for Kaggle images with stubs-only CUDA lib path
        sh(cmake_cmd + " -DCMAKE_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs")

    jobs = min(os.cpu_count() or 1, 2)  # -j2 is safe on Kaggle; use -j1 if OOM
    print(f"Building llama-server with -j{jobs} (first build: 10-30 min) ...", flush=True)
    sh(f"cmake --build {BUILD_DIR} --target llama-server -j{jobs}")


# =============================================================================
# BINARY PERSISTENCE — save to Kaggle Dataset after build/download
# =============================================================================

def save_to_kaggle_dataset(bin_path: Path) -> None:
    """
    Upload llama-server binary directly to Kaggle Dataset (no tar.gz).
    Called only when the binary was freshly built or downloaded (not restored).
    """
    sh("python3 -m pip install -q kaggle")

    commit = llama_git_commit()
    _, sm = detect_cuda_and_arch()

    DIST_DIR.mkdir(exist_ok=True)
    dest_bin = DIST_DIR / "llama-server"
    sh(f"cp {bin_path} {dest_bin}")
    dest_bin.chmod(0o755)

    # Also bundle any shared libraries (.so) next to the binary
    bin_parent = bin_path.parent
    for so_file in bin_parent.glob("*.so*"):
        sh(f"cp -P {so_file} {DIST_DIR}/")

    print(f"Binary ready for upload: {dest_bin} ({dest_bin.stat().st_size // 1024 // 1024} MB)", flush=True)

    # Human-readable build info
    (DIST_DIR / "VERSION.txt").write_text(
        f"llama.cpp commit : {commit}\n"
        f"CUDA             : 12.x\n"
        f"SM arch          : {sm} (Tesla T4)\n"
        f"Built            : {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n"
    )

    # Kaggle Dataset metadata
    (DIST_DIR / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": KAGGLE_DATASET_SLUG,
                "id": f"{KAGGLE_DATASET_OWNER}/{KAGGLE_DATASET_SLUG}",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
    )

    print(f"Uploading to Kaggle Dataset '{KAGGLE_DATASET_OWNER}/{KAGGLE_DATASET_SLUG}' ...", flush=True)
    sh(
        f"kaggle datasets version "
        f"--path {DIST_DIR} "
        f"--message 'llama-server {commit} (CUDA 12, SM{sm})'"
    )
    print(
        "Dataset updated.\n"
        f"  Attach it as Input in future sessions:\n"
        f"  Add Data -> {KAGGLE_DATASET_OWNER}/{KAGGLE_DATASET_SLUG}",
        flush=True,
    )


# =============================================================================
# STEP 1 — Clone application repo
# =============================================================================

os.chdir(WORK)
print("CWD:", Path.cwd(), flush=True)
sh("nvidia-smi -L || true")

if not REPO_DIR.exists():
    sh(f"git clone {REPO_URL} {REPO_DIR}")
os.chdir(REPO_DIR)

# Optional: enable cloud model comparison tab without hardcoding API keys
load_kaggle_secret("GEMINI_API_KEY")


# =============================================================================
# STEP 2 — Install Python dependencies
# =============================================================================

sh("python3 -m pip install -q -U pip")
sh("python3 -m pip install -q -U cmake ninja huggingface_hub")
sh("python3 -m pip install -q -r requirements-demo.txt")


# =============================================================================
# STEP 3 — Download model weights from HuggingFace
# =============================================================================

from huggingface_hub import hf_hub_download  # noqa: E402

stage1_path = hf_hub_download(
    repo_id=HF_REPO,
    filename="medgemma-base-q5_k_m.gguf",
)
stage2_lora_path = hf_hub_download(
    repo_id=HF_REPO,
    filename="lora_stage2_all_hard200_20260207/lora_stage2_all_hard200_20260207-f16.gguf",
)
# Stage-2 base weights are identical to Stage-1 — reuse the cached file
stage2_base_path = stage1_path

print("stage1_path     :", stage1_path, flush=True)
print("stage2_base_path:", stage2_base_path, flush=True)
print("stage2_lora_path:", stage2_lora_path, flush=True)


# =============================================================================
# STEP 4 — Clone llama.cpp source (needed for build; skipped if already cloned)
# =============================================================================

if not LLAMA_DIR.exists():
    sh(f"git clone --depth=1 https://github.com/ggml-org/llama.cpp.git {LLAMA_DIR}")


# =============================================================================
# STEP 5 — Obtain llama-server binary (4-tier priority)
# =============================================================================

_need_save = False  # set True when binary is freshly obtained (not from cache)

if LLAMA_BIN.exists():
    # Already compiled in this session — skip everything
    print(f"llama-server already present, skipping acquisition: {LLAMA_BIN}", flush=True)

elif KAGGLE_INPUT_DIR.exists():
    # Restore from attached Kaggle Dataset (~5 sec)
    print(f"Dataset contents ({KAGGLE_INPUT_DIR}):", flush=True)
    for p in sorted(KAGGLE_INPUT_DIR.rglob("*")):
        kind = "DIR" if p.is_dir() else f"FILE ({p.stat().st_size // 1024}K)"
        print(f"   {p.relative_to(KAGGLE_INPUT_DIR)}  [{kind}]", flush=True)

    candidates = sorted(KAGGLE_INPUT_DIR.rglob("llama-server*"))
    candidates = [p for p in candidates if p.is_file() and "VERSION" not in p.name]
    if candidates:
        src_bin = candidates[-1]
        print(f"Restoring from Kaggle Dataset: {src_bin}", flush=True)
        ver = src_bin.parent / "VERSION.txt"
        if not ver.exists():
            ver = KAGGLE_INPUT_DIR / "VERSION.txt"
        if ver.exists():
            print(ver.read_text(), flush=True)
        PREBUILT_DIR.mkdir(parents=True, exist_ok=True)
        sh(f"cp {src_bin} {PREBUILT_DIR}/llama-server")
        (PREBUILT_DIR / "llama-server").chmod(0o755)
        for so_file in src_bin.parent.glob("*.so*"):
            sh(f"cp -P {so_file} {PREBUILT_DIR}/")
        os.environ["LD_LIBRARY_PATH"] = str(PREBUILT_DIR) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

        # Verify binary health (catches missing .so dependencies)
        try:
            subprocess.run(
                [str(PREBUILT_DIR / "llama-server"), "--version"],
                check=True,
                capture_output=True,
                timeout=10,
                env={
                    **os.environ,
                    "LD_LIBRARY_PATH": str(PREBUILT_DIR) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
                },
            )
            print("Binary ready (restored from Dataset)", flush=True)
        except Exception as e:
            print(f"Restored binary failed health check: {e}", flush=True)
            print("  Rebuilding with static linking ...", flush=True)
            build_from_source()
            _need_save = True
    else:
        print("Dataset attached but llama-server binary not found -> compiling ...", flush=True)
        build_from_source()
        _need_save = True

elif try_download_prebuilt(PREBUILT_DIR):
    # Prebuilt binary downloaded from ai-dock/llama.cpp-cuda (~1 min)
    _need_save = True

else:
    # Last resort: compile from source (10-30 min)
    build_from_source()
    _need_save = True

# Persist freshly obtained binary to Kaggle Dataset for future sessions
if _need_save and LLAMA_BIN.exists():
    try:
        save_to_kaggle_dataset(LLAMA_BIN)
    except Exception as exc:
        print(f"Could not save to Dataset: {exc}  (continuing anyway)", flush=True)

LLAMA_SERVER = str(LLAMA_BIN.resolve())
print("LLAMA_SERVER:", LLAMA_SERVER, flush=True)


# =============================================================================
# STEP 6 — Start Stage-1 and Stage-2 inference servers on separate GPUs
# =============================================================================

LOG1 = REPO_DIR / "llama_stage1_1245.log"
LOG2 = REPO_DIR / "llama_stage2_1246.log"

env0 = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "0",
    "LD_LIBRARY_PATH": str(PREBUILT_DIR) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
}
env1 = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "1",
    "LD_LIBRARY_PATH": str(PREBUILT_DIR) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
}

# tee mirrors output to both the log file and the notebook console
stage1_cmd = (
    f"{LLAMA_SERVER} "
    f"-m {stage1_path} "
    f"--alias medgemma-base "
    f"--host 127.0.0.1 --port 1245 "
    f"--ctx-size 8192 -ngl 999 "
    f"2>&1 | tee {LOG1}"
)

stage2_cmd = (
    f"{LLAMA_SERVER} "
    f"-m {stage2_base_path} "
    f"--lora {stage2_lora_path} "
    f"--alias medgemma-stage2 "
    f"--host 127.0.0.1 --port 1246 "
    f"--ctx-size 8192 -ngl 999 "
    f"--cache-prompt --cache-reuse 256 "
    f"2>&1 | tee {LOG2}"
)

stage1 = subprocess.Popen(["bash", "-c", stage1_cmd], text=True, env=env0)
stage2 = subprocess.Popen(["bash", "-c", stage2_cmd], text=True, env=env1)

# Wait for both servers in parallel to save time
wait_servers_parallel(
    "http://127.0.0.1:1245",
    "http://127.0.0.1:1246",
    timeout_s=300,
)
print("Stage-1 / Stage-2 servers are ready.", flush=True)


# =============================================================================
# STEP 7 — Launch Gradio demo
# =============================================================================

os.environ["STRUCTCORE_BACKEND_MODE"] = "pipeline"
os.environ["STRUCTCORE_STAGE1_URL"] = "http://127.0.0.1:1245"
os.environ["STRUCTCORE_STAGE1_MODEL"] = "medgemma-base"
os.environ["STRUCTCORE_STAGE2_URL"] = "http://127.0.0.1:1246"
os.environ["STRUCTCORE_STAGE2_MODEL"] = "medgemma-stage2"

from apps.challenge_demo.app_challenge import build_demo  # noqa: E402

demo = build_demo()
demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)
