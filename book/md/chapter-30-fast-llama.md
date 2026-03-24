# Chapter 30: FastLlamaModel — The Reference Implementation

---

## Introduction

`FastLlamaModel` is the most complete and well-developed `Fast*Model` class. At 143K, it serves as the reference for all other architectures. This chapter dives into its patching strategy.

### What You'll Learn

- The `from_pretrained()` flow for Llama models
- How attention, MLP, and norm layers are patched
- RoPE variants: standard, linear, Llama3, LongRoPE
- The `get_peft_model()` and `patch_peft_model()` methods

---

## Notes & Key Points

### 30.1 Patching Strategy

- Each transformer layer's self-attention is replaced with Unsloth's optimized version
- RoPE computation uses the custom Triton kernel
- MLP gates use fused SwiGLU kernel
- Forward pass is rewritten to use fused operations

### 30.2 RoPE Variants

- Standard: `base=10000`, no scaling
- Linear scaling: divide frequencies by scale factor
- Llama 3 style: complex frequency scaling with `low_freq_wavelen` and `high_freq_wavelen`
- LongRoPE (Phi-3.5): separate short/long inv_freq tensors

### 30.3 Multi-GPU Cos/Sin Caching

- `multi_gpu_cos_cached` and `multi_gpu_sin_cached`: per-device caches
- Avoids cross-device transfers for RoPE embeddings

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| FastLlamaModel | `unsloth/models/llama.py` |
| Model utilities | `unsloth/models/_utils.py` |
