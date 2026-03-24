# Chapter 26: SwiGLU, GeGLU, and Activation Kernels

---

## Introduction

Modern LLMs use gated activation functions in their MLP layers. Unsloth provides fused Triton kernels for SwiGLU and GeGLU that eliminate intermediate allocations.

### What You'll Learn

- SwiGLU: `SiLU(gate) * up` — used by Llama, Mistral, Qwen
- GeGLU: `GELU(gate) * up` — used by some architectures
- How fusing reduces memory usage

---

## Notes & Key Points

### 26.1 SwiGLU

```python
# Standard: two separate operations
gate = F.silu(gate_proj(x))
up = up_proj(x)
output = gate * up

# Fused: single kernel
output = swiglu_kernel(gate_proj_w, up_proj_w, x)
```

### 26.2 GeGLU

- Same fusion pattern but with GELU activation
- `geglu.py` (9K) — more complex due to GELU approximation variants

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| SwiGLU kernel | `unsloth/kernels/swiglu.py` |
| GeGLU kernel | `unsloth/kernels/geglu.py` |
