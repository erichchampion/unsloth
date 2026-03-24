# Chapter 28: FlexAttention and FP8 Kernels

---

## Introduction

FlexAttention provides flexible attention patterns (used by gpt-oss), while the FP8 kernel enables 8-bit floating-point quantization during training.

### What You'll Learn

- FlexAttention: custom attention masks and score modifiers
- FP8 quantization: per-tensor and block-level modes
- How these integrate with the training loop

---

## Notes & Key Points

### 28.1 FlexAttention

- `flex_attention.py` (7K) — Wraps PyTorch's FlexAttention API
- Enables custom attention score modifiers (e.g., causal masking, document masking)
- Used by gpt-oss models
- Currently blocks padding-free training due to compatibility issues

### 28.2 FP8 Kernels

- `fp8.py` (24K) — FP8 quantization/dequantization
- Supports per-tensor and per-block quantization
- Uses TorchAO's float8 backend
- Enables training on consumer GPUs with 8-bit weights

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| FlexAttention | `unsloth/kernels/flex_attention.py` |
| FP8 kernel | `unsloth/kernels/fp8.py` |
