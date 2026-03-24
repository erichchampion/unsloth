# Chapter 12: LoRA and QLoRA — Parameter-Efficient Fine-Tuning

> *"Train 1% of the parameters, get 99% of the quality."*

---

## Introduction

Fine-tuning a 7-billion parameter model requires loading all those parameters into GPU memory, computing gradients for every one, and storing optimizer states for each. For a 7B model in FP16, that's 14 GB just for the weights — plus another 14 GB for gradients and roughly 28 GB for Adam optimizer states. A single fine-tuning run would need 56+ GB of VRAM, far beyond what most consumer GPUs provide.

LoRA (Low-Rank Adaptation) and its quantized variant QLoRA solve this by training only a small number of additional parameters injected into the model's existing layers. Unsloth builds on these techniques with architecture-specific optimizations, custom Triton kernels for LoRA operations, and a streamlined API through `get_peft_model()`.

### What You'll Learn

- The mathematical foundation of LoRA and QLoRA
- How `get_peft_model()` applies LoRA adapters in Unsloth
- Target module selection and rank configuration
- LoRA-specific Triton kernel optimizations
- Memory savings compared to full fine-tuning
- RSLoRA and DoRA variants

### Prerequisites

- Matrix algebra basics (matrix multiplication, rank)
- The model dispatch system from Chapter 9
- Understanding of gradient computation in neural networks

---

## 12.1 The LoRA Idea

Instead of updating a full weight matrix **W** ∈ ℝ^(d×k), LoRA freezes **W** and trains two small matrices **A** ∈ ℝ^(d×r) and **B** ∈ ℝ^(r×k), where r ≪ min(d, k):

```
Original:    y = Wx
LoRA:       y = Wx + (α/r) · BAx
```

For a typical attention layer with d=4096, k=4096, and r=16:
- Full weight: 4096 × 4096 = **16.7M parameters**
- LoRA adapters: (4096 × 16) + (16 × 4096) = **131K parameters** (0.8%)

The scaling factor α/r (called `lora_alpha / r`) controls how much the LoRA update influences the output.

---

## 12.2 QLoRA: 4-bit Quantized LoRA

QLoRA takes LoRA further by quantizing the frozen base weights **W** to 4-bit NF4 (Normal Float 4) or FP4 format using bitsandbytes:

```
Memory comparison (7B model):
  Full FP16:     14 GB (weights) + 14 GB (gradients) + 28 GB (Adam) = 56 GB
  LoRA FP16:     14 GB (weights, frozen) + ~0.1 GB (adapters) + ~0.2 GB (Adam) = 14.3 GB
  QLoRA 4-bit:   3.5 GB (weights, 4-bit) + ~0.1 GB (adapters) + ~0.2 GB (Adam) = 3.8 GB
```

In Unsloth, QLoRA is the default mode (`load_in_4bit=True`). The base weights are quantized using `BitsAndBytesConfig`:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Nested quantization
)
```

---

## 12.3 get_peft_model() in Unsloth

After loading a model with `from_pretrained()`, LoRA adapters are applied via `get_peft_model()`:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                  # LoRA rank
    lora_alpha = 16,         # Scaling factor
    lora_dropout = 0,        # Dropout (Unsloth recommends 0)
    target_modules = [       # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_rslora = False,      # Rank-Stabilized LoRA
    use_dora = False,        # Weight-Decomposed LoRA
    bias = "none",           # No bias adaptation
)
```

### Target Modules

The `target_modules` parameter specifies which linear layers receive LoRA adapters:

| Module | Layer | Impact |
|--------|-------|--------|
| `q_proj`, `k_proj`, `v_proj` | Attention queries, keys, values | Core attention behavior |
| `o_proj` | Attention output projection | Output routing |
| `gate_proj`, `up_proj`, `down_proj` | MLP (SwiGLU) | Knowledge and reasoning |
| `embed_tokens` | Input embeddings | Vocabulary specialization |
| `lm_head` | Output head | Generation probabilities |

By default, Unsloth targets all seven attention + MLP modules. Adding `embed_tokens` and `lm_head` is recommended when teaching the model new vocabulary or domains.

### Rank Selection

| Rank (r) | Parameters Added | Use Case |
|----------|-----------------|----------|
| 8 | ~28M (7B model) | Simple tasks, small datasets |
| 16 | ~56M | General fine-tuning (default) |
| 32 | ~112M | Complex tasks, large datasets |
| 64 | ~224M | Maximum adaptation |
| 128+ | ~448M+ | Near full fine-tuning flexibility |

---

## 12.4 Unsloth's LoRA Optimizations

Unsloth doesn't just use PEFT's standard LoRA — it replaces the forward pass with custom Triton kernels:

### Fused Dequant + Matmul + LoRA

Standard LoRA requires three separate operations: dequantize W, compute Wx, compute BAx. Unsloth fuses these into a single Triton kernel:

```python
# Standard (3 kernel launches):
W_fp16 = dequantize(W_4bit)     # Kernel 1
y = W_fp16 @ x                   # Kernel 2
y += (alpha/r) * (B @ (A @ x))   # Kernel 3

# Unsloth (1 kernel launch):
y = fast_lora_forward(W_4bit, A, B, x, alpha, r)  # Single fused kernel
```

### Post-PEFT Re-Patching

PEFT's `get_peft_model()` wraps the model in a `PeftModel`, which can undo some of Unsloth's patches. The `patch_peft_model()` static method re-applies the Triton replacements:

```python
# Internal flow:
model = peft.get_peft_model(model, lora_config)  # PEFT wrapping
FastLlamaModel.patch_peft_model(model)            # Re-apply Unsloth patches
```

---

## 12.5 RSLoRA and DoRA

Unsloth supports two LoRA variants:

- **RSLoRA** (Rank-Stabilized LoRA): Scales LoRA by 1/√r instead of 1/r, providing more stable training at higher ranks. Enable with `use_rslora=True`.
- **DoRA** (Weight-Decomposed LoRA): Decomposes the weight update into magnitude and direction components, improving convergence. Enable with `use_dora=True`.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| get_peft_model() | `unsloth/models/llama.py` → `FastLlamaModel.get_peft_model()` |
| LoRA forward kernels | `unsloth/kernels/fast_lora.py` |
| BitsAndBytes config | `unsloth/models/loader_utils.py` |
| PEFT integration | `peft` (external) |
