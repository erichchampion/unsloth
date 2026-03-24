# Chapter 4: Installing Unsloth — Studio, Core, and Docker

> *"The right install path depends on what you plan to build."*

---

## Introduction

Unsloth offers several installation paths, each targeting a different user profile. A researcher who just wants to fine-tune a model in a Jupyter notebook needs different tooling than a team deploying Studio as an internal web service. This chapter walks through every option, explains what happens under the hood at each step, and helps you choose the right path for your situation.

The installation process is more complex than a typical `pip install` because Unsloth must coordinate GPU drivers, CUDA versions, PyTorch builds, and xformers wheels — all of which must match exactly. The installer scripts exist to automate this coordination.

### What You'll Learn

- The five installation paths and when to use each one
- How `install.sh` detects your platform and wires up dependencies
- What `unsloth studio setup` does behind the scenes
- Docker container setup with GPU passthrough
- Troubleshooting common installation issues

### Prerequisites

- A machine with Python 3.9–3.14
- For training: an NVIDIA, AMD, or Intel GPU with appropriate drivers
- For inference-only: any platform (including CPU-only and Apple Silicon)

---

## 4.1 Choosing an Installation Path

| Path | Command | What You Get | Best For |
|------|---------|-------------|----------|
| **One-liner** | `curl -fsSL https://unsloth.ai/install.sh \| sh` | Full Studio + Core | First-time users, quick setup |
| **Developer** | `uv pip install unsloth --torch-backend=auto` | Core + CLI | Script-based workflows, notebooks |
| **Editable** | `git clone` + `uv pip install -e .` | Source install | Contributing, debugging |
| **Docker** | `docker run ... unsloth/unsloth` | Pre-built container | Reproducible environments |
| **Colab/Kaggle** | `pip install "unsloth[colab-new]"` | Minimal Core | Cloud notebooks |

The key distinction is between **Studio** (web UI with frontend build) and **Core** (Python library only). A Core install is a standard `pip install`. A Studio install additionally requires Node.js to build the React frontend and cmake/git to compile the GGUF inference engine (llama.cpp).

---

## 4.2 The One-Line Installer: `install.sh`

The fastest path to a working installation is the one-line script:

```bash
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/install.sh | sh
```

This 265-line shell script performs the following steps:

### Step 1: Platform Detection

```bash
# install.sh (lines 98-104)
OS="linux"
if [ "$(uname)" = "Darwin" ]; then
    OS="macos"
elif grep -qi microsoft /proc/version 2>/dev/null; then
    OS="wsl"
fi
```

The script distinguishes between native Linux, macOS (Apple Silicon or Intel), and Windows Subsystem for Linux (WSL). Each platform has different system dependency requirements.

### Step 2: System Dependency Check

The script checks for `cmake`, `git`, and — on Linux — `build-essential` and `libcurl4-openssl-dev`. These are needed to compile the GGUF inference engine (llama.cpp) during Studio setup. On macOS, it verifies Xcode Command Line Tools are installed.

If packages are missing on Linux, the script attempts `apt-get install` first without `sudo`, then escalates to `sudo` only if needed — prompting the user for explicit consent:

```bash
# install.sh (lines 57-63)
echo "    WARNING: We require sudo elevated permissions to install:"
echo "    $_STILL_MISSING"
echo "    If you accept, we'll run sudo now, and it'll prompt your password."
printf "    Accept? [Y/n] "
```

### Step 3: Install `uv` Package Manager

The script installs `uv` (Astral's fast Python package manager) if it's not already present or is below the minimum version (`0.7.14`). `uv` is used instead of standard pip because it resolves dependencies significantly faster and handles the complex xformers wheel matrix more reliably.

### Step 4: Create Virtual Environment

```bash
# install.sh (lines 228-234)
uv venv "$VENV_NAME" --python "$PYTHON_VERSION"
```

The script creates a dedicated `unsloth_studio` virtual environment with Python 3.13. If the venv already exists with a valid interpreter, this step is skipped.

### Step 5: Install Unsloth

```bash
uv pip install --python "$VENV_NAME/bin/python" "unsloth>=2026.3.11" --torch-backend=auto
```

The `--torch-backend=auto` flag tells `uv` to automatically detect and install the correct PyTorch build for the system's CUDA version. This is the critical step that resolves the CUDA × PyTorch × xformers compatibility matrix described in Chapter 3.

### Step 6: Studio Setup

Finally, the script runs `unsloth studio setup`, which invokes `studio/setup.sh` (24K) to install the Studio backend's Python requirements, build the React frontend with npm, and compile the llama.cpp GGUF inference engine.

---

## 4.3 The Windows Installer: `install.ps1`

The Windows PowerShell installer (`install.ps1`, 16K) follows the same logical sequence but handles Windows-specific concerns:

- Installs Python 3.13 via the official Windows installer if needed
- Uses `uv` for package management
- Calls `studio/setup.ps1` (73K) for Studio setup
- Handles Windows path separators and environment variable differences

---

## 4.4 Developer Install with `uv`

For users who want Core without Studio, or who need an editable install for development:

```bash
# Core only (no Studio frontend)
uv pip install unsloth --torch-backend=auto

# With specific CUDA + PyTorch version
uv pip install "unsloth[cu126-torch270]"

# With Flash Attention (Ampere+ GPUs)
uv pip install "unsloth[cu126-ampere-torch270]"

# Editable install from source
git clone https://github.com/unslothai/unsloth.git
cd unsloth
uv pip install -e . --torch-backend=auto
```

The extras syntax (e.g., `[cu126-torch270]`) selects the correct xformers wheel from the `pyproject.toml` dependency matrix. If you're unsure which CUDA version you have, `--torch-backend=auto` handles the detection.

---

## 4.5 Colab and Kaggle

Cloud notebook environments have pre-installed PyTorch and CUDA, so the installation is simpler:

```python
# Google Colab
!pip install "unsloth[colab-new]"

# Kaggle
!pip install "unsloth[kaggle-new]"
```

The `colab-new` extra installs a minimal dependency set that avoids reinstalling PyTorch (which is already present in the Colab runtime). It includes `unsloth_zoo`, `transformers`, `datasets`, `bitsandbytes`, and `triton`.

---

## 4.6 Docker

For reproducible environments, Unsloth provides Docker images:

```bash
docker run --gpus all -p 8888:8888 unsloth/unsloth
```

The container includes all dependencies pre-installed and configured. GPU passthrough requires the NVIDIA Container Toolkit (`nvidia-container-toolkit`) on the host.

---

## 4.7 Studio Setup: What `unsloth studio setup` Does

When you run `unsloth studio setup`, the CLI invokes `studio/setup.sh` (or `setup.ps1` on Windows). This script:

1. **Detects the GPU platform** — NVIDIA CUDA, AMD ROCm, Intel XPU, Apple Metal, or CPU-only
2. **Installs Python backend requirements** — from `studio/backend/requirements/` directory
3. **Installs Node.js** — needed to build the React frontend
4. **Builds the frontend** — runs `npm install` and `npm run build` in `studio/frontend/`
5. **Compiles llama.cpp** — builds the GGUF inference engine for local model execution
6. **Configures platform-specific backends** — links vLLM for NVIDIA, MLX for Apple Silicon, etc.

---

## 4.8 Launching

After installation, you can use Unsloth in two ways:

```bash
# Launch Studio (web interface)
source unsloth_studio/bin/activate
unsloth studio -H 0.0.0.0 -p 8888
# Then open http://localhost:8888 in your browser

# Use Core in a Python script
python -c "import unsloth; print('Unsloth loaded successfully')"
```

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| macOS/Linux installer | `install.sh` |
| Windows installer | `install.ps1` |
| Studio setup (Linux) | `studio/setup.sh` |
| Studio setup (Windows) | `studio/setup.ps1` |
| Python stack installer | `studio/install_python_stack.py` |
| CLI studio command | `unsloth_cli/commands/studio.py` |
| Package config + extras | `pyproject.toml` |
