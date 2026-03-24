# Chapter 3: Technology Stack and Key Dependencies

> *"Standing on the shoulders of giants — and patching them at import time."*

---

## Introduction

Unsloth does not exist in isolation. It is built on top of a rich ecosystem of machine learning libraries, and understanding those dependencies is essential for grasping how the codebase works and why certain design decisions were made. Some dependencies provide the computational foundation (PyTorch, Triton). Others define the user-facing API surface (transformers, TRL, PEFT). And a few solve narrow but critical problems (bitsandbytes for 4-bit quantization, xformers for memory-efficient attention).

This chapter catalogs every significant dependency, explains what role it plays, and — critically — shows how Unsloth manages the explosive combinatorics of version compatibility across CUDA generations, PyTorch releases, and Python versions.

### What You'll Learn

- The five-layer dependency stack from PyTorch to Unsloth
- What each Hugging Face library contributes and where Unsloth patches it
- How the `pyproject.toml` manages 50+ optional dependency groups
- Version gate constants and how they control feature availability
- The role of `unsloth_zoo` as a fast-moving companion package

### Prerequisites

- Basic familiarity with pip, extras, and dependency resolution
- Understanding of GPU computing concepts (CUDA, compute capability)
- The repository layout from Chapter 2

---

## 3.1 The Five-Layer Stack

Unsloth's dependencies form a layered stack. Each layer builds on the one below it:

```
┌─────────────────────────────────────────────┐
│  Layer 5: Unsloth                           │
│    unsloth, unsloth_zoo, unsloth_cli        │
├─────────────────────────────────────────────┤
│  Layer 4: Training Frameworks               │
│    TRL (SFTTrainer, GRPOTrainer)            │
│    PEFT (LoRA, QLoRA adapters)              │
│    accelerate (device placement)            │
├─────────────────────────────────────────────┤
│  Layer 3: Model Framework                   │
│    transformers (model architectures)       │
│    datasets (data loading)                  │
│    huggingface_hub (download/upload)        │
│    sentencepiece (tokenization)             │
├─────────────────────────────────────────────┤
│  Layer 2: GPU Acceleration                  │
│    Triton (custom GPU kernels)              │
│    bitsandbytes (4-bit quantization)        │
│    xformers (memory-efficient attention)    │
│    flash-attn (Flash Attention 2)           │
├─────────────────────────────────────────────┤
│  Layer 1: Foundation                        │
│    PyTorch (computation, autograd, GPU)     │
│    CUDA / ROCm / XPU (hardware drivers)     │
│    NumPy, packaging, protobuf               │
└─────────────────────────────────────────────┘
```

When you `import unsloth`, the library patches Layers 3 and 4 (transformers, TRL, PEFT) using the custom kernels from Layer 2 (Triton). The user's code continues to use the standard Layer 3/4 APIs, but the hot paths now run through Unsloth's optimized implementations.

---

## 3.2 Layer 1: Foundation — PyTorch and Hardware

**PyTorch** is the computational bedrock. Every tensor operation, every gradient computation, every GPU memory allocation flows through PyTorch. Unsloth does not contain its own autograd engine or memory allocator — it relies entirely on PyTorch for these fundamentals.

The hardware abstraction layer supports three backends:

| Backend | Hardware | Status |
|---------|----------|--------|
| **CUDA** | NVIDIA GPUs (RTX 30/40/50, A100, H100, Blackwell, DGX) | Full support |
| **ROCm (HIP)** | AMD Instinct GPUs | Training + inference |
| **XPU** | Intel GPUs | Training + inference |

The `device_type.py` module (and its counterpart in `unsloth_zoo`) detects the available backend at import time and exposes constants used throughout the codebase:

```python
# From unsloth_zoo/device_type.py — used everywhere
DEVICE_TYPE           # "cuda", "hip", or "xpu"
DEVICE_TYPE_TORCH     # "cuda" (for hip too, since ROCm uses CUDA API)
DEVICE_COUNT          # Number of available GPUs
ALLOW_PREQUANTIZED_MODELS   # False for AMD GPUs with bitsandbytes < 0.49.2
ALLOW_BITSANDBYTES    # False when bitsandbytes is unstable (AMD)
```

For NVIDIA GPUs, compute capability determines which features are available. Ampere (SM 8.0) and above get bfloat16 support — older GPUs fall back to float16:

```python
# unsloth/__init__.py (lines 199-201)
if DEVICE_TYPE == "cuda":
    major_version, minor_version = torch.cuda.get_device_capability()
    SUPPORTS_BFLOAT16 = major_version >= 8
```

---

## 3.3 Layer 2: GPU Acceleration Libraries

### Triton

Triton is OpenAI's language for writing GPU kernels in Python. Every custom kernel in `unsloth/kernels/` is a Triton program — typically 50-200 lines of decorated Python that compiles to GPU machine code at first use. Triton handles thread blocking, memory coalescing, and register allocation automatically, making it dramatically easier to write fast GPU code compared to raw CUDA C++.

Unsloth requires `triton >= 3.0.0` on Linux, with a Windows-specific `triton-windows` package:

```toml
# pyproject.toml
triton = [
    "triton>=3.0.0 ; ('linux' in sys_platform)",
    "triton-windows ; (sys_platform == 'win32') ...",
]
```

### bitsandbytes

`bitsandbytes` provides the `Linear4bit` layer type that makes QLoRA possible. When `load_in_4bit=True`, model weights are stored in NF4 (4-bit NormalFloat) format and dequantized on-the-fly during the forward pass. This reduces VRAM usage by roughly 4× compared to float16 weights.

Unsloth pins `bitsandbytes>=0.45.5` and excludes known-broken versions (`!=0.46.0`, `!=0.48.0`).

### xformers

Meta's xformers library provides memory-efficient attention implementations. Unsloth uses it as a fallback when Flash Attention is not available. The `pyproject.toml` contains an enormous matrix of xformers wheel URLs — one for every combination of CUDA version, PyTorch version, and Python version. This is the primary reason the file is nearly 1,200 lines long.

### Flash Attention

Flash Attention (`flash-attn >= 2.6.3`) is optional but recommended for Ampere+ GPUs. It provides the fastest and most memory-efficient attention implementation, and is required for some features like Gemma 2's attention softcapping. Unsloth detects its presence at import time:

```python
# From models/_utils.py — checked during model loading
HAS_FLASH_ATTENTION              # True if flash-attn is installed
HAS_FLASH_ATTENTION_SOFTCAPPING  # True if flash-attn >= 2.6.3
```

---

## 3.4 Layer 3: The Hugging Face Ecosystem

Unsloth is deeply integrated with the Hugging Face ecosystem. Here is what each library provides:

| Library | Version Constraint | Role in Unsloth |
|---------|-------------------|-----------------|
| **transformers** | `>=4.51.3, <=5.3.0` (with 12 excluded versions) | Base model classes (`LlamaModel`, `AutoConfig`), tokenization |
| **PEFT** | `>=0.18.0, !=0.11.0` | LoRA/QLoRA adapter framework (`PeftModel`, `PeftConfig`) |
| **TRL** | `>=0.18.2, !=0.19.0, <=0.24.0` | Training loop implementations (`SFTTrainer`, `GRPOTrainer`) |
| **datasets** | `>=3.4.1, <4.4.0` | Data loading and preprocessing |
| **huggingface_hub** | `>=0.34.0` | Model download/upload, token management, `HfFileSystem` |
| **accelerate** | `>=0.34.1` | Distributed training and device placement |
| **hf_transfer** | (latest) | Fast parallel downloads from the Hub |
| **sentence-transformers** | (latest) | Embedding model training interface |
| **diffusers** | (latest) | Diffusion model support |

The transformers version constraint is particularly strict — notice the long list of excluded versions (`!=4.52.0`, `!=4.52.1`, ..., `!=5.1.0`). Each exclusion represents a release that introduced a regression or breaking change that Unsloth's patches could not safely work around.

---

## 3.5 Layer 4: Training Frameworks

**TRL** (Transformer Reinforcement Learning) provides the trainer classes that users interact with directly. Unsloth patches these trainers at import time to inject its optimizations:

- `SFTTrainer` — Supervised Fine-Tuning. Unsloth patches this to add automatic sample packing, padding-free batching, and embedding learning rate separation.
- `GRPOTrainer` — Group Relative Policy Optimization. Unsloth's implementation uses 80% less VRAM than the standard version through careful memory management.
- `DPOTrainer` — Direct Preference Optimization, also patched for memory efficiency.

**PEFT** (Parameter-Efficient Fine-Tuning) provides the LoRA adapter framework. Unsloth patches PEFT's `LoraLayer` to use its custom fast LoRA kernels instead of standard PyTorch matmuls.

---

## 3.6 Version Gate Constants

The `loader.py` file defines a set of boolean constants that gate feature availability based on the installed transformers version. These are checked during model loading to provide clear error messages when a user tries to load a model that requires a newer transformers:

```python
# unsloth/models/loader.py (lines 70-83)
SUPPORTS_FOURBIT    = transformers_version >= Version("4.37")
SUPPORTS_GEMMA      = transformers_version >= Version("4.38")
SUPPORTS_GEMMA2     = transformers_version >= Version("4.42")
SUPPORTS_LLAMA31    = transformers_version >= Version("4.43.2")
SUPPORTS_LLAMA32    = transformers_version >  Version("4.45.0")
SUPPORTS_GRANITE    = transformers_version >= Version("4.46.0")
SUPPORTS_QWEN3      = transformers_version >= Version("4.50.3")
SUPPORTS_QWEN3_MOE  = transformers_version >= Version("4.50.3")
SUPPORTS_FALCON_H1  = transformers_version >= Version("4.53.0")
SUPPORTS_GEMMA3N    = transformers_version >= Version("4.53.0")
SUPPORTS_GPTOSS     = transformers_version >= Version("4.55.0")
_NEEDS_ROPE_FIX     = transformers_version >= Version("5.0.0")
```

The `_NEEDS_ROPE_FIX` gate is particularly interesting — transformers v5 changed how models are initialized on the "meta" device, which corrupted RoPE positional embedding buffers. Unsloth detects this and applies a fix that recomputes `inv_freq` from stored parameters (see `_fix_rope_inv_freq()` in `loader.py`).

---

## 3.7 The `pyproject.toml` Dependency Matrix

The sheer scale of the dependency matrix deserves explanation. The `[project.optional-dependencies]` section defines 50+ extras groups. Here is how they are structured:

```
Base groups:
  triton           → Platform-specific Triton wheels
  huggingfacenotorch → HF ecosystem without PyTorch
  huggingface      → huggingfacenotorch + unsloth_zoo + torchvision + triton

CUDA × PyTorch combinations:
  cu118onlytorch230  → xformers wheel for CUDA 11.8 + PyTorch 2.3.0
  cu121onlytorch240  → xformers wheel for CUDA 12.1 + PyTorch 2.4.0
  cu126onlytorch270  → xformers wheel for CUDA 12.6 + PyTorch 2.7.0
  ... (40+ such groups)

User-facing extras:
  cu126-torch270         → huggingface + bitsandbytes + xformers
  cu126-ampere-torch270  → same + flash-attn
  colab-new              → Minimal Colab dependencies
  kaggle                 → Kaggle-specific configuration
```

This design means users can install exactly the right combination for their environment:

```bash
# CUDA 12.6, PyTorch 2.7.0, with Flash Attention
pip install "unsloth[cu126-ampere-torch270]"

# Google Colab (auto-detects CUDA/PyTorch)
pip install "unsloth[colab-new]"

# Kaggle
pip install "unsloth[kaggle-new]"
```

---

## 3.8 Supporting Libraries

Beyond the core ML stack, several utility libraries play important roles:

| Library | Purpose |
|---------|---------|
| **Typer** | CLI framework for `unsloth_cli` (powers `unsloth train`, `unsloth studio`, etc.) |
| **Pydantic** | Data validation for CLI config and Studio API models |
| **FastAPI / Uvicorn** | Studio web backend server |
| **React** | Studio web frontend (shipped as pre-built static files) |
| **vLLM** | Optional high-performance inference backend (`fast_inference=True`) |
| **PyYAML** | Configuration file parsing |
| **nest-asyncio** | Allows nested event loops (needed for Jupyter/Colab environments) |
| **protobuf** | Required for some tokenizer and model config formats |
| **psutil** | System resource monitoring (CPU, memory, GPU) |

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Dependency definitions | `pyproject.toml` |
| Version gates | `unsloth/models/loader.py` (lines 70-83) |
| Import-time patches | `unsloth/__init__.py`, `unsloth/import_fixes.py` |
| Device detection | `unsloth/device_type.py`, `unsloth_zoo/device_type.py` |
| Kernel implementations | `unsloth/kernels/*.py` |
| Companion package | `unsloth_zoo` (separate pip package) |
