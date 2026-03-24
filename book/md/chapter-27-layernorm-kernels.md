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

For a 32-layer model with 2 norms per layer, this adds up to 192 eliminated kernel launches. The savings are per-layer and apply to both `input_layernorm` and `post_attention_layernorm`, so the total is doubled.

### Source Code Walkthrough: The Forward Kernel

The actual Triton kernel from `rms_layernorm.py` — annotated:

```python
@triton.jit
def _rms_layernorm_forward(
    Y, Y_row_stride,     # Output tensor + stride
    X, X_row_stride,     # Input tensor + stride
    W, W_row_stride,     # Weight tensor + stride
    r, r_row_stride,     # Inverse variance (saved for backward)
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Each program instance processes one row (one token)."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load entire row into registers (hidden_dim fits in one block)
    X_row = tl.load(X + row_idx*X_row_stride + col_offsets,
                    mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)
    
    # Compute RMS: sqrt(mean(x²))
    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)   # 1/sqrt(var + eps)
    tl.store(r + row_idx, inv_var)            # Save for backward pass
    
    # Normalize and scale
    normed = X_row * inv_var                  # x̂ = x / RMS(x)
    normed = normed.to(W_row.dtype)           # Match weight dtype
    output = normed * W_row                   # y = x̂ * γ
    
    tl.store(Y + row_idx*Y_row_stride + col_offsets, output, mask=mask)
```

### Gemma Variant: The +1 Offset

Gemma's RMSNorm adds 1.0 to the weight, so initially-zero weights produce an identity transform:

```python
@triton.jit
def _gemma_rms_layernorm_forward(...):
    # Same as standard RMSNorm, except the final line:
    output = normed * (W_row + 1.0)    # Note: +1.0 !
    # Standard:  output = normed * W_row
    # Gemma:     output = normed * (W_row + 1.0)
```

This is handled via the `gemma` flag in `fast_rms_layernorm()`, which selects the correct kernel variant. The backward kernel similarly branches:

```python
# In backward:
if GEMMA:
    dY_W = dY_row * (W_row + 1.0)   # Chain rule through (W + 1)
else:
    dY_W = dY_row * W_row
```

For a 32-layer model this means 128 fewer kernel launches just for normalization.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| RMSNorm kernel | `unsloth/kernels/rms_layernorm.py` |
| LayerNorm kernel | `unsloth/kernels/layernorm.py` |
| Norm patching | `unsloth/models/llama.py` |
