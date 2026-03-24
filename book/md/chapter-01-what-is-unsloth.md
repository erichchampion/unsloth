# Chapter 1: What is Unsloth?

> *"Run and train AI models with a unified local interface."*

---

## Introduction

Unsloth is an open-source toolkit for running and fine-tuning large language models on your own hardware. It provides two interfaces -- **Unsloth Studio**, a web-based UI, and **Unsloth Core**, a Python library -- both backed by the same optimized engine. The project's central claim is concrete: up to 2x faster training with 70% less VRAM, achieved through custom Triton kernels and deep integration with the Hugging Face ecosystem.

Unlike wrapper libraries that simply expose existing training APIs, Unsloth rewrites the hot paths. When you `import unsloth`, the library patches `transformers`, `trl`, and `peft` at import time, replacing standard PyTorch operations with fused GPU kernels. The result is that your existing training scripts run faster and use less memory without any code changes beyond adding that single import.

The project is maintained by Daniel Han-Chen and the Unsloth team. The core library is Apache 2.0 licensed; the Studio web UI is AGPL-3.0.

### What You'll Learn

- What separates Unsloth from other fine-tuning tools
- The dual-interface design: library (Core) vs. web UI (Studio)
- Key performance claims and how they are achieved
- The supported hardware and platform matrix
- How a single `import unsloth` changes everything

### Prerequisites

- Comfortable reading Python
- Basic familiarity with LLM concepts (models, fine-tuning, quantization, LoRA)
- Some experience with PyTorch or Hugging Face `transformers`

---

## Notes & Key Points

### 1.1 The Two Modes: Core and Studio

Unsloth can be used in two fundamentally different ways, and understanding this split early is important because the codebase is organized around it.

**Unsloth Core** is a Python library installed via `pip install unsloth`. You use it in scripts, notebooks, or the command line. A typical Core workflow looks like this:

```python
import unsloth  # Must be the first import!
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)
```

**Unsloth Studio** is a web application with a FastAPI backend and React frontend. You launch it from the command line and interact through your browser:

```bash
unsloth studio -H 0.0.0.0 -p 8888
```

Studio provides a visual interface for chatting with models, configuring training runs, monitoring GPU usage in real time, and building datasets through a node-based "Data Recipe" editor. Under the hood, Studio calls the same Core library functions -- `FastLanguageModel.from_pretrained()`, `UnslothTrainer`, `unsloth_save_model()` -- so the performance characteristics are identical.

The **CLI** (`unsloth_cli/`) bridges both worlds. It is a Typer-based command-line tool registered as the `unsloth` console script, exposing commands for training, inference, export, and Studio management:

```
unsloth train          # Fine-tune from command line
unsloth inference      # Run inference
unsloth export         # Export to GGUF or other formats
unsloth studio         # Launch or configure Studio
unsloth studio setup   # Install Studio dependencies
```

### 1.2 Key Features -- Inference

Unsloth is not just a training tool. It provides a complete local inference pipeline:

- **Search, download, and run models** -- supports GGUF files, LoRA adapters, and safetensors weights. The model registry (`unsloth/registry/`) maps friendly names to Hugging Face repository paths.
- **Export models** -- save trained models as GGUF (for llama.cpp / Ollama), 16-bit safetensors, or 4-bit quantized formats.
- **Tool calling** -- LLMs can call external tools with self-healing: if a tool call produces malformed JSON, the model retries with a corrected version.
- **Code execution** -- models can write and test code in sandboxed environments, similar to ChatGPT's Code Interpreter.
- **Auto-tune inference parameters** -- Studio automatically adjusts temperature, top-p, and other parameters.
- **Multi-format uploads** -- chat with images, audio, PDFs, code files, and DOCX documents.

### 1.3 Key Features -- Training

Training is where Unsloth's optimizations have the most impact:

- **500+ models supported** with up to 2x speed gains and 70% less VRAM.
- **Custom Triton kernels** replace PyTorch's built-in operations for cross-entropy loss, LoRA forward/backward passes, rotary position embeddings (RoPE), SwiGLU activations, and layer normalization.
- **Data Recipes** -- auto-create training datasets from PDFs, CSVs, DOCX, and other files using an LLM to generate instruction-response pairs.
- **Reinforcement Learning** -- the most VRAM-efficient GRPO implementation, using 80% less memory than standard approaches. Supports FP8 RL training for additional savings.
- **Training modes** -- full fine-tuning, LoRA, QLoRA (4-bit), FP8, 8-bit, and 16-bit.
- **Observability** -- live training loss graphs, GPU memory tracking, and customizable metric dashboards in Studio.
- **Multi-GPU** -- available now, with "major improvements coming soon."

### 1.4 How Unsloth Achieves Its Speed

The performance gains come from four techniques layered on top of each other:

**1. Import-time monkey patching.** When you write `import unsloth`, the library's `__init__.py` runs a sequence of 25+ patches against `transformers`, `trl`, `peft`, and other libraries. This is the mechanism that makes everything else work -- it replaces slow implementations with fast ones before any model code runs. The `__init__.py` file even checks whether critical modules were imported first:

```python
# unsloth/__init__.py (lines 24-70)
critical_modules = ["trl", "transformers", "peft"]
already_imported = [mod for mod in critical_modules if mod in sys.modules]

if already_imported:
    warnings.warn(
        f"WARNING: Unsloth should be imported before "
        f"[{', '.join(already_imported)}] to ensure all "
        f"optimizations are applied.",
        stacklevel=2,
    )
```

**2. Custom Triton kernels.** The `unsloth/kernels/` directory contains hand-written GPU kernels for the operations that dominate training time:

| Kernel | What It Replaces | Memory Savings |
|--------|-----------------|----------------|
| Cross-entropy loss | `nn.CrossEntropyLoss` | Avoids materializing V×T logit tensor |
| Fast LoRA | PEFT's LoRA forward | Fuses dequant + matmul + LoRA addition |
| RoPE embedding | `transformers` RoPE | Fused cos/sin computation |
| SwiGLU / GeGLU | Separate gate + activation | Eliminates intermediate allocations |
| RMSNorm | `nn.LayerNorm` | Single kernel launch vs. three |

**3. Optimized training strategies.** The trainer automatically enables sample packing (concatenating multiple short examples into one sequence) and padding-free batching (variable-length sequences without wasted padding tokens). Both techniques improve GPU utilization without changing the training outcome.

**4. Gradient checkpointing.** Unsloth's custom gradient checkpointing strategy (`use_gradient_checkpointing="unsloth"`) trades compute for memory more efficiently than the standard HF implementation.

### 1.5 Supported Platforms

Unsloth runs on a broad range of hardware, but the feature set varies by platform:

| Platform | Chat / Inference | Training |
|----------|-----------------|----------|
| NVIDIA CUDA (RTX 30/40/50, Blackwell, DGX) | ✅ Full support | ✅ Full support |
| macOS (Apple Silicon) | ✅ Chat + Data Recipes | 🔜 MLX training coming soon |
| AMD (ROCm) | ✅ Chat + Data Recipes | ✅ Via Core CLI |
| Intel (XPU) | ✅ Chat + Data Recipes | ✅ Via Core CLI |
| CPU-only | ✅ Chat + Data Recipes | ❌ Not supported |

NVIDIA GPUs with compute capability ≥ 8.0 (Ampere and above) get the full optimization stack including bfloat16 support and Flash Attention. Older GPUs still work but fall back to float16 and standard attention.

### 1.6 The Public API Surface

Despite the complexity under the hood, Unsloth's user-facing API is intentionally small. Most users interact with exactly three functions:

```python
import unsloth
from unsloth import FastLanguageModel

# 1. Load a model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 2. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
)

# 3. Train (using standard HF/TRL APIs)
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(model=model, ...)
trainer.train()

# 4. Save
model.save_pretrained("my-model")
```

The key insight is that after `import unsloth` and `FastLanguageModel.from_pretrained()`, you use standard Hugging Face APIs for everything else. The optimizations are applied transparently.

### 1.7 Licensing

The codebase uses a dual-license model:

- **Apache 2.0** -- the core library (`unsloth/`), CLI (`unsloth_cli/`), and all kernel code
- **AGPL-3.0** -- the Studio web interface (`studio/`)

This means you can freely use the Core library in proprietary projects, but the Studio UI carries copyleft obligations if you distribute modifications.

### 1.8 A Training Run's Journey Through Every Layer

To understand why the codebase is structured the way it is, it helps to trace what happens during a complete fine-tuning workflow. This is the roadmap for the rest of the book -- each numbered step corresponds to one or more chapters.

```
User Script
    |
    v
+----------------------+  Ch 1
| import unsloth       |  - 25+ patches applied to transformers, trl, peft
+----------+-----------+  - device detection, Triton linking
           v
+----------------------+  Ch 7, 8, 9
| from_pretrained()    |  - registry lookup → HF download
|                      |  - architecture detection → Fast*Model dispatch
|                      |  - quantization (4-bit, FP8, 16-bit)
+----------+-----------+  - RoPE fix, attention config
           v
+----------------------+  Ch 12
| get_peft_model()     |  - LoRA adapter injection
|                      |  - gradient checkpointing setup
+----------+-----------+  - fast LoRA kernel attachment
           v
+----------------------+  Ch 14, 18
| UnslothTrainer       |  - auto packing / padding-free
|                      |  - data loading + preprocessing
+----------+-----------+  - TRL backwards compatibility
           v
+--------------------------------------------------+
|              TRAINING LOOP (Ch 14)               |
|                                                  |
|  +--------------------------------------------+  |
|  | Forward Pass                                |  |
|  |  - RoPE kernel (Ch 25)                     |  |
|  |  - Attention (Ch 40: Flash/SDPA/manual)    |  |
|  |  - SwiGLU kernel (Ch 26)                   |  |
|  |  - RMSNorm kernel (Ch 27)                  |  |
|  |  - Cross-entropy kernel (Ch 23)            |  |
|  +--------------------------------------------+  |
|  | Backward Pass                               |  |
|  |  - Fast LoRA backward (Ch 24)              |  |
|  |  - Gradient checkpointing (Ch 12)          |  |
|  +--------------------------------------------+  |
|  | Optimizer Step                              |  |
|  |  - Embedding LR separation (Ch 14)         |  |
|  +--------------------------------------------+  |
|                                                  |
|  Repeat for N steps...                           |
+--------------------------------------------------+
           |
           v
+----------------------+  Ch 19, 20, 21
| Save / Export        |  - LoRA adapters only
|                      |  - Merged 16-bit (for GGUF)
|                      |  - Push to Hugging Face Hub
+----------------------+  - GGUF quantization
```

Every box in this diagram is a chapter in this book. By the time you reach Chapter 40 you'll have traced a model from download through training to export, understanding every optimization along the way.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Package entry + patching | `unsloth/__init__.py` |
| Import-time fixes | `unsloth/import_fixes.py` |
| CLI entry | `unsloth_cli/__init__.py` |
| README | `README.md` |
| Build config | `pyproject.toml` |
| Studio backend | `studio/backend/main.py` |
| Studio frontend | `studio/frontend/` |
