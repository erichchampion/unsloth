# Chapter 4: Installing Unsloth — Studio, Core, and Docker

---

## Introduction

Unsloth offers multiple installation paths to match different hardware, OS, and usage patterns. This chapter walks through each one and explains what happens under the hood.

### What You'll Learn

- One-line installer scripts for macOS, Linux, WSL, and Windows
- Developer installs with `uv` and Python 3.13
- Docker container setup
- The difference between Studio and Core installs

---

## Notes & Key Points

### 4.1 Installation Paths

| Path | Command | What You Get |
|------|---------|-------------|
| **One-liner** | `curl -fsSL https://unsloth.ai/install.sh \| sh` | Full Studio setup |
| **Developer** | `uv pip install unsloth --torch-backend=auto` | Core + Studio |
| **Nightly** | `git clone` + `uv pip install -e .` | Editable install from source |
| **Docker** | `docker run ... unsloth/unsloth` | Pre-built container with GPU support |
| **Core only** | `uv pip install unsloth --torch-backend=auto` (in a venv) | No Studio UI |

### 4.2 The Install Scripts

- `install.sh` (8.7KB) — handles macOS/Linux/WSL: installs uv, creates venv, pip installs, runs `unsloth studio setup`
- `install.ps1` (15.8KB) — handles Windows PowerShell: installs Python 3.13, uv, creates venv
- `studio/setup.sh` (23.9KB) — Studio-specific setup: installs Node.js, builds frontend
- `studio/setup.ps1` (72.8KB) — Windows Studio setup

### 4.3 `unsloth studio setup`

- The `studio` subcommand in the CLI invokes `studio/setup.sh` or `setup.ps1`
- Installs backend Python requirements and builds the React frontend
- Configures platform-specific dependencies (CUDA, ROCm, Metal)

### 4.4 Launching

```bash
# Studio mode
unsloth studio -H 0.0.0.0 -p 8888

# Core mode — use in scripts
from unsloth import FastLanguageModel
```

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| macOS/Linux installer | `install.sh` |
| Windows installer | `install.ps1` |
| Studio setup | `studio/setup.sh`, `studio/setup.ps1` |
| CLI studio command | `unsloth_cli/commands/studio.py` |
| Package config | `pyproject.toml` |
