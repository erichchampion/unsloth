# Chapter 9: Model Dispatch — Architecture-Specific Fast Paths

---

## Introduction

Once `from_pretrained()` identifies the model architecture, it dispatches to a `Fast*Model` class. Each class applies architecture-specific optimizations by patching the Hugging Face model in-place.

### What You'll Learn

- The `Fast*Model` class hierarchy
- How `from_pretrained()` works on each dispatch class
- The patching pattern: replace attention, MLP, and norm layers
- `get_peft_model()` for adding LoRA adapters

---

## Notes & Key Points

### 9.1 The Fast*Model Family

All `Fast*Model` classes share a common pattern inherited from `FastLlamaModel`:
- Static `from_pretrained()` → loads model + tokenizer, applies patches
- Static `get_peft_model()` → wraps model with LoRA adapters
- Static `patch_peft_model()` → re-patches after PEFT wrapping

### 9.2 What "Fast" Means

The `Fast*Model` classes replace standard PyTorch operations with:
- **Triton kernels** for cross-entropy, RoPE, LayerNorm, SwiGLU
- **Fused operations** that combine multiple steps into one GPU kernel
- **Optimized attention** via FlexAttention or Flash Attention
- **Gradient checkpointing** with Unsloth's custom strategy

### 9.3 Architecture-Specific Differences

- **Llama** (`llama.py`, 143K) — The reference; includes RoPE scaling, LongRoPE, Llama 3.1+ fixes
- **Gemma** (`gemma.py`, 19K) — Soft-capping attention, fixed embeddings
- **Gemma 2** (`gemma2.py`, 25K) — Attention soft-capping with Flash Attention support
- **Qwen 2** (`qwen2.py`, 3.7K) — Thin wrapper around Llama
- **Qwen 3** (`qwen3.py`, 17K) — Custom attention with QK-norms
- **Mistral** (`mistral.py`, 18K) — Sliding window attention
- **Qwen 3 MoE** (`qwen3_moe.py`, 9.5K) — Mixture of Experts routing

### 9.4 The Fallback: FastModel

- When architecture is unrecognized, `FastModel.from_pretrained()` is called
- Uses `torch.compile` for optimization instead of hand-written kernels
- Supports full fine-tuning, 8-bit, and 16-bit modes

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Llama dispatch | `unsloth/models/llama.py` |
| Gemma dispatch | `unsloth/models/gemma.py`, `gemma2.py` |
| Qwen dispatch | `unsloth/models/qwen2.py`, `qwen3.py` |
| Mistral dispatch | `unsloth/models/mistral.py` |
| Model utilities | `unsloth/models/_utils.py` (107K) |
