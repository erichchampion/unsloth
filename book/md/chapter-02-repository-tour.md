# Chapter 2: Repository Tour — From Root to Source Tree

> *"Know the terrain before you march."*

---

## Introduction

Before diving into any code, it pays to walk the full directory tree. Unsloth's repository is not a single library — it is three interlocking subsystems housed under one roof: the **core library** (`unsloth/`), the **CLI layer** (`unsloth_cli/`), and the **Studio web application** (`studio/`). A fourth component, **unsloth_zoo**, lives in a separate pip package but is required at runtime.

This chapter provides a guided tour of every significant file and directory. By the end you will know exactly where to look when you need to understand a kernel, a model adapter, a CLI command, or a Studio API route.

### What You'll Learn

- The top-level directory layout and the purpose of every root-level file
- How the three subsystems (`unsloth/`, `unsloth_cli/`, `studio/`) relate to each other
- Where kernels, model implementations, and utilities live inside the core library
- The role of `pyproject.toml`, `build.sh`, and the installer scripts
- How the model registry maps friendly names to Hugging Face repositories

### Prerequisites

- Comfortable navigating a Python project
- Basic familiarity with `pyproject.toml` and pip extras
- Understanding of the high-level concepts from Chapter 1 (Core vs. Studio, training vs. inference)

---

## 2.1 Top-Level Layout

When you clone the repository, here is what you see:

```
unsloth/                           # Root
├── unsloth/                       # Core Python library  (Apache 2.0)
├── unsloth_cli/                   # CLI layer            (Apache 2.0)
├── studio/                        # Web UI               (AGPL-3.0)
├── scripts/                       # Helper scripts
├── tests/                         # Test suite
├── images/                        # README graphics
├── book/                          # This book's source
├── pyproject.toml                 # Package config + dependency matrix
├── build.sh                       # Build helper
├── install.sh / install.ps1       # One-line installers (Linux / Windows)
├── cli.py / unsloth-cli.py        # Standalone CLI entry points
├── README.md                      # Project documentation
├── LICENSE                        # Apache 2.0
├── COPYING                        # Full license text
├── CODE_OF_CONDUCT.md             # Community guidelines
├── CONTRIBUTING.md                # Contributor guide
└── .pre-commit-config.yaml        # Code quality hooks
```

The three directories that matter most are `unsloth/`, `unsloth_cli/`, and `studio/`. Everything else is either build infrastructure, documentation, or support tooling. Let's explore each one.

---

## 2.2 The Core Library: `unsloth/`

This is the heart of the project — the importable Python package that provides model loading, training, kernels, and saving. When your script says `import unsloth`, Python executes `unsloth/__init__.py`, which triggers the entire import-time patching sequence described in Chapter 1.

```
unsloth/
├── __init__.py              # Import-time patching, dependency checks (331 lines)
├── import_fixes.py          # 25+ monkey patches for transformers, trl, peft (65K)
├── device_type.py           # GPU/CPU detection and capability flags
├── models/                  # Model-specific Fast* classes + loader (22 files)
├── kernels/                 # Custom Triton kernels (11 files + moe/ subdir)
├── registry/                # Model name → HF path registry (9 files)
├── utils/                   # Attention dispatch, packing, HF hub helpers
├── dataprep/                # Raw text + synthetic data generation
├── save.py                  # Model saving, LoRA merging, GGUF export (120K)
├── trainer.py               # UnslothTrainer + TRL backwards compatibility
├── chat_templates.py        # Chat template definitions (120K+)
├── tokenizer_utils.py       # Tokenizer fixes and utilities (43K)
└── ollama_template_mappers.py  # Ollama ↔ HF template translation (83K)
```

### 2.2.1 The `models/` Directory

The `models/` directory is the largest subsystem in the core library, containing 22 files that handle model loading, architecture-specific optimizations, and inference/training setup:

| File | Size | Purpose |
|------|------|---------|
| `llama.py` | 143K | FastLlamaModel — the base class most architectures derive from |
| `_utils.py` | 107K | Shared utilities: version checks, gradient checkpointing, `fast_inference_setup()` |
| `loader.py` | 69K | `FastLanguageModel.from_pretrained()` — the main model loading entry point |
| `rl.py` | 82K | Reinforcement learning support (GRPO, DPO patching) |
| `rl_replacements.py` | 69K | RL-specific function replacements for memory savings |
| `sentence_transformer.py` | 85K | Embedding model training via Sentence Transformers |
| `vision.py` | 64K | Vision-language model support (LLaVA, Qwen-VL, etc.) |
| `mapper.py` | 49K | Model name → architecture mapping for auto-dispatch |
| `loader_utils.py` | 16K | FP8 quantization, device map preparation, model name resolution |
| `gemma.py` / `gemma2.py` | 19K / 25K | FastGemmaModel, FastGemma2Model |
| `qwen2.py` / `qwen3.py` / `qwen3_moe.py` | 4K / 17K / 10K | FastQwen* implementations |
| `falcon_h1.py` | 29K | Falcon-H1 model support |
| `granite.py` | 23K | FastGraniteModel |
| `mistral.py` | 18K | FastMistralModel |
| `cohere.py` | 19K | FastCohereModel |

The pattern is consistent: each architecture file defines a `Fast*Model` class with `from_pretrained()` and `get_peft_model()` static methods. The loader in `loader.py` reads the model's `config.json`, identifies the `model_type` field (e.g., `"llama"`, `"gemma2"`, `"qwen3"`), and dispatches to the correct `Fast*Model`.

### 2.2.2 The `kernels/` Directory

Custom Triton GPU kernels live here. Each file implements one or two fused operations that replace slower PyTorch/transformers equivalents:

| File | Size | Kernel |
|------|------|--------|
| `cross_entropy_loss.py` | 15K | Chunked cross-entropy that avoids materializing the full V×T logit tensor |
| `fast_lora.py` | 21K | Fused dequantization + matmul + LoRA forward/backward |
| `rope_embedding.py` | 14K | Fused rotary position embedding (cos/sin in one pass) |
| `rms_layernorm.py` | 10K | RMSNorm: single kernel launch instead of three separate operations |
| `layernorm.py` | 7K | Standard LayerNorm variant |
| `swiglu.py` | 4K | SwiGLU activation (gate + silu + multiply fused) |
| `geglu.py` | 9K | GeGLU activation variant |
| `flex_attention.py` | 7K | PyTorch 2.5+ Flex Attention integration |
| `fp8.py` | 24K | FP8 quantization kernels for memory-efficient training |
| `utils.py` | 34K | Shared kernel utilities, numerics, and autotuning |
| `moe/` | — | Mixture-of-Experts kernel sub-package |

### 2.2.3 The `registry/` Directory

The registry maps user-friendly model names to their Hugging Face repository paths and pre-quantized variants:

```
registry/
├── registry.py       # Core ModelRegistry class and lookup logic
├── _llama.py         # Llama family entries (Llama 3, 3.1, 3.2, 3.3)
├── _qwen.py          # Qwen family entries (Qwen2, Qwen2.5, Qwen3)
├── _gemma.py         # Gemma family entries
├── _mistral.py       # Mistral family entries
├── _deepseek.py      # DeepSeek family entries
├── _phi.py           # Phi family entries
├── __init__.py       # Registry initialization
└── REGISTRY.md       # Human-readable model listing
```

When you call `FastLanguageModel.from_pretrained("unsloth/Llama-3.2-1B-Instruct")`, the registry resolves this name to the actual Hugging Face repo, potentially substituting a pre-quantized 4-bit variant (`-bnb-4bit` suffix) if `load_in_4bit=True`.

### 2.2.4 Supporting Modules

- **`utils/`** — Four files covering attention dispatch (`attention_dispatch.py`, 13K), sample packing (`packing.py`, 14K), Hugging Face Hub helpers (`hf_hub.py`, 2K), and a shared `__init__.py`.
- **`dataprep/`** — Raw text data loading (`raw_text.py`, 13K), synthetic dataset generation (`synthetic.py`, 16K), and config templates (`synthetic_configs.py`, 4K).

---

## 2.3 The CLI Layer: `unsloth_cli/`

The CLI is a thin Typer-based wrapper registered as the `unsloth` console script via `pyproject.toml`:

```toml
[project.scripts]
unsloth = "unsloth_cli:app"
```

```
unsloth_cli/
├── __init__.py       # Typer app initialization and command registration
├── config.py         # Training configuration dataclass (5.6K)
├── options.py        # Shared CLI options and parameter validation (5.5K)
└── commands/
    ├── train.py      # `unsloth train` — fine-tuning from CLI (4.8K)
    ├── inference.py  # `unsloth inference` — model inference (2.5K)
    ├── export.py     # `unsloth export` — GGUF/safetensors export (4.5K)
    ├── studio.py     # `unsloth studio` — Studio management (7.5K)
    └── ui.py         # `unsloth ui` — UI-related commands (3.5K)
```

Each command file defines a Typer sub-command that validates arguments, instantiates the appropriate Core library objects, and runs the operation. The CLI is intentionally stateless — it does not maintain sessions or caches. Every invocation is a fresh run.

---

## 2.4 The Studio Web Application: `studio/`

Studio is a full-stack web application with a FastAPI backend and a React frontend. It is licensed under AGPL-3.0, separate from the Apache-licensed core:

```
studio/
├── backend/
│   ├── main.py               # FastAPI application factory (12K)
│   ├── run.py                 # Server startup and configuration (14K)
│   ├── routes/                # API endpoint modules
│   ├── core/                  # Business logic + data recipe engine
│   │   └── data_recipe/       # Node-based dataset builder
│   ├── models/                # Pydantic data models
│   ├── auth/                  # Authentication
│   ├── state/                 # Application state management
│   ├── plugins/               # Plugin system
│   ├── loggers/               # Logging configuration
│   ├── utils/                 # Backend utilities
│   ├── tests/                 # Backend test suite
│   ├── assets/                # Static assets
│   └── requirements/          # Python requirements files
├── frontend/                  # React SPA (built by npm)
├── setup.sh                   # Linux/macOS setup script (24K)
├── setup.ps1                  # Windows setup script (73K)
├── install_python_stack.py    # Python dependency installer (15K)
├── Unsloth_Studio_Colab.ipynb # Google Colab launcher
└── LICENSE.AGPL-3.0           # AGPL license
```

The backend's `main.py` creates a FastAPI application that serves both the REST API and the built frontend. Under the hood it calls the same Core library functions — `FastLanguageModel.from_pretrained()`, `UnslothTrainer`, `unsloth_save_model()` — ensuring performance parity between the CLI and web interfaces.

---

## 2.5 External Companion Package: `unsloth_zoo`

`unsloth_zoo` is a separate pip package that Unsloth imports at startup. It contains shared utilities that are maintained on a faster release cadence than the main library:

- **RL environments** — `check_python_modules()`, `create_locked_down_function()`, `execute_with_time_limit()`, `Benchmarker`
- **GGUF conversion wrappers** — Used by `save.py` during export
- **Training utilities** — `Version` class, dtype helpers, HF utility functions
- **Device type detection** — `is_hip()`, `get_device_type()`, `DEVICE_TYPE`, `DEVICE_COUNT`

The minimum required version is declared in `pyproject.toml`:

```toml
"unsloth_zoo>=2026.3.5"
```

The `__init__.py` enforces this at import time — if `unsloth_zoo` is missing or too old, import fails with a clear error message.

---

## 2.6 Key Configuration Files

### `pyproject.toml`

At nearly 1,200 lines, this is one of the most complex `pyproject.toml` files you will encounter in a Python project. Its size comes from the **dependency matrix**: every combination of CUDA version (cu118, cu121, cu124, cu126, cu128, cu130) × PyTorch version (2.0 through 2.10) requires a specific xformers wheel URL. The file defines 50+ optional dependency groups (extras) like `cu126-torch270`, `colab-ampere`, and `kaggle-new`.

The key structural sections are:

```toml
[build-system]          # setuptools 80.9.0 + setuptools-scm
[project]               # Package metadata, Python >=3.9,<3.15
[project.scripts]       # CLI entry point: unsloth = "unsloth_cli:app"
[project.optional-dependencies]   # The massive extras matrix
[tool.setuptools]       # Package data globs for Studio frontend files
```

### `build.sh`

A build helper script (2K) that orchestrates packaging. Used primarily for creating releases.

### `install.sh` / `install.ps1`

One-line installer scripts for Linux (9K) and Windows (16K). These detect the platform, CUDA version, and Python version, then install the correct combination of extras.

---

## 2.7 File Size Distribution

One practical observation: the codebase's complexity is not evenly distributed. A handful of files contain the bulk of the logic:

| File | Lines | Role |
|------|-------|------|
| `models/llama.py` | ~3,500 | The "reference" model implementation |
| `save.py` | ~3,200 | All saving, merging, and export logic |
| `chat_templates.py` | ~3,000+ | Template definitions for 100+ models |
| `models/_utils.py` | ~2,800 | Shared model utilities |
| `import_fixes.py` | ~1,700 | All monkey patches |
| `models/rl.py` | ~2,100 | RL training support |
| `ollama_template_mappers.py` | ~2,000+ | Ollama template support |

Understanding `llama.py + _utils.py + loader.py` gives you roughly 60% of the core library's behavior.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Core library root | `unsloth/__init__.py` |
| Import-time fixes | `unsloth/import_fixes.py` |
| Kernel implementations | `unsloth/kernels/*.py` |
| Model implementations | `unsloth/models/*.py` |
| Model registry | `unsloth/registry/registry.py` |
| CLI commands | `unsloth_cli/commands/*.py` |
| CLI entry point | `unsloth_cli/__init__.py` |
| Studio backend | `studio/backend/main.py` |
| Studio frontend | `studio/frontend/` |
| Package config | `pyproject.toml` |
| Build / install | `build.sh`, `install.sh`, `install.ps1` |
