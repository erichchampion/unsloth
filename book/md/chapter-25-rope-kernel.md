# Chapter 25: RoPE Embedding Kernel

---

## Introduction

Rotary Position Embedding (RoPE) is the position encoding used by Llama, Qwen, Mistral, and most modern LLMs. Unsloth's custom kernel speeds up the rotation computation.

### What You'll Learn

- How RoPE encodes position information
- The Triton kernel implementation
- Scaling methods: linear, dynamic, YaRN, Llama 3-style
- The `_fix_rope_inv_freq()` fix for transformers v5

---

## Notes & Key Points

### 25.1 RoPE Basics

- Applies rotation matrix to Q and K vectors based on position
- `inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))`
- Cos/sin caches are precomputed for efficiency

### 25.2 Scaling Methods

- Standard RoPE
- Linear scaling: `inv_freq /= scale_factor`
- LongRoPE (Phi-3.5): separate short/long inv_freq with scaling factors
- Llama 3.1 "llama3" type: requires transformers ≥ 4.43.2

### 25.3 The v5 Fix

- Transformers v5 uses meta-device loading, corrupting `inv_freq` buffers
- `_fix_rope_inv_freq()` recomputes from stored `base` and `dim`
- Critical: without this fix, training loss is 5-11x higher

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| RoPE kernel | `unsloth/kernels/rope_embedding.py` |
| RoPE fix | `unsloth/models/loader.py` → `_fix_rope_inv_freq()` |
