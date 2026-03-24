# Chapter 24: Fast LoRA Kernels

---

## Introduction

The fast LoRA kernel fuses the base weight dequantization and LoRA adapter application into a single GPU operation, eliminating intermediate memory allocations.

### What You'll Learn

- How standard LoRA forward pass works: W·x + s·B·A·x
- How the fused kernel combines dequant + matmul + LoRA in one pass
- The `get_lora_parameters_bias()` utility

---

## Notes & Key Points

### 24.1 Standard LoRA vs. Fused LoRA

```
Standard: x → dequant(W)·x → (result) + s·B·(A·x) → output
Fused:    x → kernel(W, A, B, s, x) → output  (one kernel launch)
```

### 24.2 Key Functions

- `fast_lora.py` (21K) — Main Triton kernel implementations
- Forward and backward passes are both fused
- Handles 4-bit NF4 dequantization inline

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Fast LoRA kernel | `unsloth/kernels/fast_lora.py` |
| LoRA parameter extraction | `unsloth/kernels/utils.py` |
