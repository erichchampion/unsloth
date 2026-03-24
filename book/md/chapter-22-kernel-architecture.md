# Chapter 22: Kernel Architecture Overview

---

## Introduction

Unsloth's speed gains come from custom Triton kernels that replace standard PyTorch operations. This chapter provides an overview of the kernel architecture before diving into individual kernels in subsequent chapters.

### What You'll Learn

- Why custom kernels matter for training speed
- The kernel directory structure and module organization
- How kernels are selected and applied during model loading
- The relationship between kernels and `Fast*Model` patching

---

## Notes & Key Points

### 22.1 The Kernel Directory

```
unsloth/kernels/
├── __init__.py            # Exports, utility functions
├── cross_entropy_loss.py  # Fused cross-entropy loss
├── fast_lora.py           # LoRA forward/backward kernels
├── flex_attention.py      # FlexAttention support
├── fp8.py                 # FP8 quantization kernels
├── geglu.py               # GeGLU activation kernel
├── layernorm.py           # LayerNorm Triton kernel
├── rms_layernorm.py       # RMSNorm Triton kernel
├── rope_embedding.py      # Rotary Position Embedding kernel
├── swiglu.py              # SwiGLU activation kernel
├── utils.py               # Shared kernel utilities (34K)
└── moe/                   # Mixture of Experts kernels
    ├── autotune_cache.py
    └── grouped_gemm/      # Grouped GEMM for MoE
```

### 22.2 How Kernels Are Applied

1. `Fast*Model.from_pretrained()` loads the base model from HF
2. Model layers are iterated and specific operations are replaced:
   - `nn.LayerNorm` → Triton RMSNorm kernel
   - `CrossEntropyLoss` → Fused cross-entropy kernel
   - Attention → Patched with RoPE + FlexAttention
   - MLP → Patched with SwiGLU/GeGLU kernel
3. LoRA adapters get the `fast_lora` kernel treatment

### 22.3 Kernel Utilities

- `utils.py` (34K) — Shared helpers: `fast_dequantize()`, `QUANT_STATE`, `get_lora_parameters_bias()`
- Quantization state management for bitsandbytes 4-bit weights
- Triton autotuning configuration

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Kernel exports | `unsloth/kernels/__init__.py` |
| Kernel utilities | `unsloth/kernels/utils.py` |
