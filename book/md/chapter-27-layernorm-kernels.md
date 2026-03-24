# Chapter 27: LayerNorm and RMSNorm Kernels

---

## Introduction

Layer normalization is applied before every attention and MLP block. Unsloth replaces PyTorch's implementations with fused Triton kernels.

### What You'll Learn

- LayerNorm vs. RMSNorm (used by Llama, Gemma, Qwen)
- Triton kernel implementations
- Forward and backward pass fusion

---

## Notes & Key Points

### 27.1 RMSNorm

- Used by Llama, Qwen, Mistral — no mean subtraction
- `rms_layernorm.py` (10K) — Triton kernel
- `RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight`

### 27.2 LayerNorm

- `layernorm.py` (7K) — Standard LayerNorm with mean subtraction
- `LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias`

### 27.3 Fusion Benefits

- Standard: 3 separate kernel launches (mean, variance, normalize)
- Fused: single kernel launch, data stays in GPU registers

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| RMSNorm kernel | `unsloth/kernels/rms_layernorm.py` |
| LayerNorm kernel | `unsloth/kernels/layernorm.py` |
