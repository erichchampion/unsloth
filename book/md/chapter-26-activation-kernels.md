# Chapter 26: SwiGLU, GeGLU, and Activation Kernels

> *"Three matrix multiplies, one kernel launch."*

---

## Introduction

The MLP block in a modern transformer doesn't use a simple ReLU — it uses gated activation functions like SwiGLU (Llama, Qwen, Mistral) or GeGLU (some research models). These functions involve three weight matrices (`gate_proj`, `up_proj`, `down_proj`) and an element-wise gating operation. Standard PyTorch computes these as separate operations with separate memory allocations. Unsloth's activation kernels (`swiglu.py`, 143 lines and `geglu.py`, 290 lines) fuse the gate and up projections with the activation function.

### What You'll Learn

- SwiGLU: the activation function used by most modern LLMs
- GeGLU: the GELU-based variant
- How fusion eliminates the intermediate activation tensor
- Forward and backward pass implementations

### Prerequisites

- The kernel architecture from Chapter 22
- Understanding of MLP layers in transformers
- SiLU (Swish) and GELU activation functions

---

## 26.1 SwiGLU — The Standard

SwiGLU (Sigmoid Linear Unit Gated Linear Unit) is the MLP activation used by Llama, Qwen, Mistral, and Gemma:

```python
# Standard PyTorch (3 operations):
gate = gate_proj(x)        # [batch, seq, hidden] → [batch, seq, intermediate]
gate = F.silu(gate)        # SiLU activation (element-wise)
up = up_proj(x)            # [batch, seq, hidden] → [batch, seq, intermediate]
output = gate * up          # Element-wise multiply
output = down_proj(output)  # [batch, seq, intermediate] → [batch, seq, hidden]
```

The problem: the intermediate tensors `gate` and `up` are each `[batch, seq, intermediate_size]`. For Llama 7B with `intermediate_size=11008`, a single batch element at 2048 sequence length uses:
- `gate`: 2048 × 11008 × 2 bytes = **43 MB**
- `up`: 2048 × 11008 × 2 bytes = **43 MB**
- Total intermediate memory: **86 MB per sample**

### Fused SwiGLU

```python
# Fused kernel: gate, silu, and multiply in one pass
output = fast_swiglu_fg(gate_proj_weight, up_proj_weight, x)
# Internally:
#   1. Load chunk of x
#   2. Compute gate_proj(x) in-register
#   3. Apply SiLU in-register
#   4. Compute up_proj(x) in-register
#   5. Multiply gate * up in-register
#   6. Store result — one write instead of three
```

Memory saved: **86 MB per sample** (the two intermediate tensors are never materialized).

### Source Code Walkthrough: The SwiGLU Forward Kernel

The actual Triton kernel from `swiglu.py` — annotated:

```python
@triton.jit
def _fg_kernel(e, g, h, n_elements,
               BLOCK_SIZE: tl.constexpr, LONG_INDEXING: tl.constexpr):
    """
    e = gate_proj output,  g = up_proj output,  h = output buffer
    Computes: h = SiLU(e) * g = (e * sigmoid(e)) * g
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)  # FP32 accumulation
    g_row = tl.load(g + offsets, mask=mask, other=0)
    
    f_row = e_row * tl.sigmoid(e_row)  # SiLU = x * σ(x) — computed in-register
    f_row = f_row.to(g_row.dtype)      # Cast back to model dtype (bf16/fp16)
    h_row = f_row * g_row              # Gate × Up — the GLU operation
    
    tl.store(h + offsets, h_row, mask=mask)  # Single write
```

### Source Code Walkthrough: The SwiGLU Backward Kernel

The backward kernel is more complex — it computes three related values in a single pass and reuses buffers to avoid allocations:

```python
@triton.jit
def _DWf_DW_dfg_kernel(DW, e, g, n_elements, ...):
    """
    Given upstream gradient DW, computes:
      h  = SiLU(e) * g              (forward recomputation)
      df = DW * SiLU(e)             (gradient for up_proj)
      de = DW * g * σ(e) * (1 + e*(1-σ(e)))  (gradient for gate_proj)
    
    Critically: stores results IN THE INPUT BUFFERS to avoid allocations:
      DW buffer ← h    (reused for forward output)
      e  buffer ← df   (reused for gate gradient)
      g  buffer ← de   (reused for up gradient)
    """
    se_row = tl.sigmoid(e_row)                           # σ(e)
    f_row  = se_row * e_row                              # SiLU(e)
    h_row  = f_row * g_row                               # SiLU(e) * g
    df_row = DW_row * f_row                              # ∂L/∂up
    dg_row = DW_row * g_row                              # intermediate
    de_row = dg_row * se_row * (1.0 + e_row * (1.0 - se_row))  # ∂L/∂gate
    
    # Overwrite input buffers — zero allocations for gradients!
    tl.store(DW + offsets, h_row,  mask=mask)  # DW ← h
    tl.store(e  + offsets, df_row, mask=mask)  # e  ← df
    tl.store(g  + offsets, de_row, mask=mask)  # g  ← de
```

The backward kernel's buffer reuse is a key optimization — it computes three outputs without allocating any new memory.

---

## 26.2 GeGLU — The GELU Variant

GeGLU replaces SiLU with GELU (Gaussian Error Linear Unit):

```python
# GeGLU:  GELU(gate_proj(x)) * up_proj(x)
# SwiGLU: SiLU(gate_proj(x)) * up_proj(x)
```

The `geglu.py` kernel is larger (290 lines) because GELU's approximation is more complex:

```python
# GELU exact: x * Φ(x) where Φ is the CDF of standard normal
# GELU tanh approx: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

The Triton kernel implements the tanh approximation inline, avoiding the cost of calling a separate GELU function.

---

## 26.3 Forward and Backward Fusion

Both kernels fuse the forward and backward passes:

### Forward Pass

| Step | Standard | Fused |
|------|----------|-------|
| gate = gate_proj(x) | Kernel 1 + write | In-register |
| gate = silu(gate) | Kernel 2 + write | In-register |
| up = up_proj(x) | Kernel 3 + write | In-register |
| output = gate * up | Kernel 4 + write | Single write |
| **Total** | **4 launches, 4 writes** | **1 launch, 1 write** |

### Backward Pass

The backward pass must compute gradients for both `gate_proj` and `up_proj` weights:

```python
# d_output is given
d_gate = d_output * up                    # Scale by up
d_gate = d_gate * silu_backward(gate)     # Through SiLU derivative
d_up = d_output * silu(gate)              # Scale by activated gate
d_gate_proj = d_gate @ x.T                # Weight gradient
d_up_proj = d_up @ x.T                    # Weight gradient
```

The fused backward kernel recomputes `silu(gate)` from the saved input rather than storing it, trading a small amount of compute for significant memory savings (gradient checkpointing at the kernel level).

---

## 26.4 Which Models Use Which

| Activation | Models |
|-----------|--------|
| SwiGLU | Llama 2/3, Qwen 2/3, Mistral, Cohere |
| GeGLU | Gemma (uses GEGLU approx), some research models |
| Standard MLP (ReLU/GELU) | Older models (GPT-2, BERT) — not fused |

## 26.5 Tiled MLP Computation

For very large models, even the fused kernel's output can be too large to hold in memory at once. Unsloth's `Fast*Model` classes support **tiled MLP** computation — processing the MLP in chunks along the sequence dimension:

```python
# Instead of processing entire sequence at once:
output = swiglu(gate_proj(x), up_proj(x))

# Process in tiles of size T:
for i in range(0, seq_len, tile_size):
    tile = x[:, i:i+tile_size, :]
    output[:, i:i+tile_size, :] = swiglu(gate_proj(tile), up_proj(tile))
```

This reduces peak memory further at the cost of slightly more kernel launches.

### Memory Savings Summary

| Model Size | intermediate_size | Standard | Fused | Savings |
|-----------|-------------------|----------|-------|---------|
| 1B | 3072 | 24 MB | 0 MB | 24 MB/layer |
| 7B | 11008 | 86 MB | 0 MB | 86 MB/layer |
| 70B | 28672 | 225 MB | 0 MB | 225 MB/layer |

For a 7B model with 32 layers: 86 MB × 32 = **2.75 GB** of intermediate memory eliminated.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| SwiGLU kernel | `unsloth/kernels/swiglu.py` |
| GeGLU kernel | `unsloth/kernels/geglu.py` |
| MLP patching | `unsloth/models/llama.py` |
