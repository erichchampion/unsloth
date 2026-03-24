# Chapter 27: LayerNorm and RMSNorm Kernels

> *"Normalize in one pass, not three."*

---

## Introduction

Every transformer block applies layer normalization twice — before attention and before the MLP. For a 32-layer model, that's 64 normalization operations per forward pass. Standard PyTorch implements normalization as a sequence of separate operations (compute mean, compute variance, normalize, scale). Unsloth replaces both LayerNorm and RMSNorm with fused Triton kernels that perform all operations in a single kernel launch, keeping intermediate values in GPU registers.

### What You'll Learn

- The mathematical difference between LayerNorm and RMSNorm
- Why RMSNorm replaced LayerNorm in modern LLMs
- The Triton kernel: single-pass normalization
- Forward and backward pass fusion
- Which models use which normalization

### Prerequisites

- The kernel architecture from Chapter 22
- Understanding of normalization in neural networks

---

## 27.1 RMSNorm vs. LayerNorm

### RMSNorm (Root Mean Square Normalization)

Used by Llama, Qwen, Mistral, and most modern LLMs. Simpler than LayerNorm — no mean subtraction:

```
RMSNorm(x) = x / √(mean(x²) + ε) × weight

Where:
  mean(x²) = (1/d) Σ xᵢ²     (no centering)
  ε = 1e-6                     (numerical stability)
  weight = learnable scale     (no bias term)
```

### LayerNorm (Full Layer Normalization)

Used by older models and some architectures like Falcon:

```
LayerNorm(x) = (x - mean(x)) / √(var(x) + ε) × weight + bias

Where:
  mean(x) = (1/d) Σ xᵢ         (centering)
  var(x) = (1/d) Σ (xᵢ - mean)²  (variance)
  weight, bias = learnable       (both scale and shift)
```

### Why RMSNorm Won

RMSNorm removes the mean computation and bias term, reducing computational cost by ~15% with negligible quality difference. For a hidden dimension of 4096, RMSNorm saves:
- 1 reduction operation (mean computation)
- 1 subtraction (centering)
- 4096 bias additions

---

## 27.2 Standard vs. Fused Implementation

### Standard PyTorch RMSNorm (3 kernel launches)

```python
# Launch 1: compute sum of squares
variance = x.pow(2).mean(-1, keepdim=True)

# Launch 2: normalize
x_norm = x * torch.rsqrt(variance + eps)

# Launch 3: scale
output = x_norm * weight
```

### Fused Triton Kernel (1 launch)

```python
@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, output_ptr, eps, d: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, d)

    # Load entire row into registers
    x = tl.load(x_ptr + row * d + offsets)

    # Compute RMS in-register
    x_sq = x * x
    mean_sq = tl.sum(x_sq) / d
    rms = tl.math.rsqrt(mean_sq + eps)

    # Normalize and scale in-register
    weight = tl.load(weight_ptr + offsets)
    output = x * rms * weight

    # Single write to global memory
    tl.store(output_ptr + row * d + offsets, output)
```

---

## 27.3 Implementation Details

### rms_layernorm.py (339 lines)

| Function | Purpose |
|----------|---------|
| `fast_rms_layernorm()` | Main entry point, dispatches to Triton kernel |
| `_rms_layernorm_forward` | Forward pass kernel |
| `_rms_layernorm_backward` | Backward pass kernel |
| Autotuning configs | Block sizes: 1024, 2048, 4096 |

### layernorm.py (227 lines)

| Function | Purpose |
|----------|---------|
| `fast_layernorm()` | Full LayerNorm with mean centering and bias |
| `_layernorm_forward` | Forward pass kernel |
| `_layernorm_backward` | Backward pass kernel |

---

## 27.4 Which Models Use Which

| Normalization | Models |
|--------------|--------|
| RMSNorm | Llama 2/3, Qwen 2/3, Mistral, Gemma |
| LayerNorm | Falcon, some Cohere variants |
| RMSNorm + QK-Norm | Qwen 3 (additional norm on Q and K) |
| RMSNorm + post-norm | Gemma 2 (pre- and post- normalization) |

---

## 27.5 Memory and Performance

| Metric | Standard | Fused | Improvement |
|--------|----------|-------|-------------|
| Kernel launches (per norm) | 3 | 1 | 3× fewer |
| Global memory reads | 3d | d | 3× less |
| Global memory writes | 3d | d | 3× less |
| Intermediate allocations | 2 tensors | 0 | 100% reduction |

For a 32-layer model with 2 norms per layer, this adds up to 192 eliminated kernel launches and 128d eliminated memory transactions per forward pass.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| RMSNorm kernel | `unsloth/kernels/rms_layernorm.py` |
| LayerNorm kernel | `unsloth/kernels/layernorm.py` |
| Norm patching | `unsloth/models/llama.py` |
