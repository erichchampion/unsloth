# Chapter 25: RoPE Embedding Kernel

> *"Position is everything — and it must be encoded fast."*

---

## Introduction

Rotary Position Embedding (RoPE) is the position encoding used by nearly every modern LLM — Llama, Qwen, Mistral, Gemma, and their derivatives. Unlike absolute or learned position embeddings, RoPE encodes position by rotating query and key vectors in the complex plane, allowing the attention mechanism to naturally decay with distance while being compatible with arbitrary sequence lengths.

Unsloth's RoPE kernel (`rope_embedding.py`, 465 lines) fuses the cos/sin computation and rotation into a single Triton kernel, and provides critical fixes for bugs introduced by newer versions of the transformers library.

### What You'll Learn

- How RoPE encodes positional information mathematically
- The Triton kernel: fused cos/sin rotation
- Scaling methods: standard, linear, dynamic, LongRoPE, Llama 3
- The critical `_fix_rope_inv_freq()` fix for transformers v5
- Performance impact of the fused kernel

### Prerequisites

- The kernel architecture from Chapter 22
- Complex number basics (rotation in 2D)
- Understanding of attention mechanisms (Q, K, V)

---

## 25.1 RoPE Mathematics

RoPE splits each query/key vector into pairs of dimensions and rotates each pair by an angle proportional to the sequence position:

```
For position p and dimension pair (2i, 2i+1):
  q'[2i]   = q[2i] * cos(p * θᵢ) - q[2i+1] * sin(p * θᵢ)
  q'[2i+1] = q[2i] * sin(p * θᵢ) + q[2i+1] * cos(p * θᵢ)

Where θᵢ = 1 / (base^(2i/dim))
```

The `inv_freq` tensor stores these frequencies:

```python
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
# For dim=128, base=10000: [1.0, 0.794, 0.631, ..., 0.0001]
```

### Key Property

The dot product `q'·k'` depends only on the **relative** position `(p₁ - p₂)`, not the absolute positions. This makes RoPE naturally support relative position encoding.

---

## 25.2 Standard vs. Fused Kernel

```python
# Standard PyTorch (4 operations, 4 kernel launches):
cos = torch.cos(position * inv_freq)        # Launch 1
sin = torch.sin(position * inv_freq)        # Launch 2
q_rot = q * cos - q_flip * sin              # Launch 3
k_rot = k * cos - k_flip * sin              # Launch 4

# Fused Triton kernel (1 launch):
q_rot, k_rot = fast_rope_embedding(q, k, position, inv_freq)
# cos/sin computed in-register, rotation applied in-place
```

The fused kernel avoids materializing the cos/sin tensors (each of which is `[seq_len, dim/2]`), saving memory and eliminating three kernel launch overheads.

### Source Code Walkthrough: The Rotation Kernel

The actual Triton kernel from `rope_embedding.py` — annotated:

```python
def _rope_embedding(
    Q, Q_row_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seqlen, head_dim, n_heads,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RoPE is Q * cos + rotate_half(Q) * sin
    Each program processes one (row, head_group) pair.
    """
    ROPE_GROUP_SIZE = 4
    row_position  = tl.program_id(0)   # Which (batch, seq) position
    group_head    = tl.program_id(1)   # Which group of 4 heads
    half_head_dim = head_dim // 2
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < half_head_dim
    
    # Load cos/sin for this position (position = row % seq_len)
    sin1 = tl.load(sin + (row_position % seqlen) * sin_row_stride + col_offsets,
                   mask=mask, other=0)
    cos1 = tl.load(cos + (row_position % seqlen) * cos_row_stride + col_offsets,
                   mask=mask, other=0)
    
    # For backward pass: negate sin (rotation in opposite direction)
    if BACKWARD_PASS: sin1 = -sin1
    
    # Process 4 heads per thread block (10% speedup from PR #238)
    head_start = group_head * ROPE_GROUP_SIZE
    head_end   = min(head_start + ROPE_GROUP_SIZE, n_heads)
    
    for k in range(head_start, head_end):
        # Load the two halves of this head's Q vector
        Q1 = tl.load(Q + row_position * Q_row_stride + k * head_dim + col_offsets,
                     mask=mask, other=0)
        Q2 = tl.load(Q + row_position * Q_row_stride + k * head_dim + col_offsets
                     + half_head_dim, mask=mask, other=0)
        
        # Apply rotation: [q0', q1'] = [q0*cos - q1*sin, q1*cos + q0*sin]
        tl.store(Q + ... + col_offsets,                Q1*cos1 - Q2*sin1, mask=mask)
        tl.store(Q + ... + col_offsets + half_head_dim, Q2*cos1 + Q1*sin1, mask=mask)
```

Key design decisions:
- **Grouped heads** — processes 4 attention heads per program instance, amortizing the cos/sin load
- **In-place** — modifies Q and K tensors directly, avoiding output allocation
- **Shared backward** — same kernel with negated sin handles the backward pass

---

## 25.3 Scaling Methods

Different models extend RoPE to support longer context lengths:

| Method | Models | How It Works |
|--------|--------|-------------|
| Standard | Llama 2, Qwen 2 | No scaling, fixed context window |
| Linear | Generic | `inv_freq /= scale_factor` (simple interpolation) |
| Dynamic NTK | CodeLlama | Dynamically adjust base frequency |
| LongRoPE | Phi-3.5 | Separate short/long inv_freq with learned factors |
| Llama 3.1 | Llama 3.1+ | `rope_type="llama3"` with smooth interpolation |
| YaRN | Mistral variants | Yet Another RoPE Extension |

Each scaling method modifies the `inv_freq` tensor. Unsloth's kernel handles all methods by accepting pre-computed `inv_freq` and applying the same rotation algorithm.

---

## 25.4 The Transformers v5 Fix

This is one of the most critical fixes in Unsloth. Transformers v5 introduced meta-device loading, which corrupts the `inv_freq` buffers:

```python
def _fix_rope_inv_freq(model):
    """Recompute inv_freq from stored base and dim parameters."""
    # Problem: meta-device loading zeros out inv_freq
    # Symptom: training loss is 5-11x higher than expected
    # Fix: recompute from config.rope_theta and head_dim
    
    for layer in model.model.layers:
        rotary = layer.self_attn.rotary_emb
        base = getattr(rotary, "base", 10000.0)
        dim = rotary.dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
        rotary.inv_freq = inv_freq
```

Without this fix, the model trains with essentially random position encodings, producing dramatically worse results that are difficult to diagnose.

---

## 25.5 Performance Impact

| Component | Standard (ms) | Fused (ms) | Speedup |
|-----------|--------------|------------|---------|
| cos/sin computation | 0.15 | included | — |
| Q rotation | 0.08 | included | — |
| K rotation | 0.08 | included | — |
| **Total RoPE** | **0.31** | **0.09** | **3.4×** |

*Per layer, batch_size=4, seq_len=2048, dim=128, measured on RTX 4090*

## 25.6 Cos/Sin Caching

For efficiency, the cos and sin values are precomputed and cached for the maximum expected sequence length:

```python
# During model initialization:
max_seq_length = 4096  # or user-specified
positions = torch.arange(max_seq_length)
freqs = torch.outer(positions, inv_freq)     # [max_seq, dim/2]
cos_cache = torch.cos(freqs)                  # Cached
sin_cache = torch.sin(freqs)                  # Cached

# During forward pass: just look up by position
cos = cos_cache[position_ids]  # No recomputation needed
sin = sin_cache[position_ids]
```

Unsloth's kernel takes this further by computing the lookup inline, avoiding the need to store the full cache in GPU memory for very long sequences.

### Dynamic Sequence Extension

When a sequence exceeds the cached length (e.g., scaling from 4K to 32K tokens), the cache is recomputed on-the-fly with the appropriate scaling method applied to `inv_freq`.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| RoPE Triton kernel | `unsloth/kernels/rope_embedding.py` |
| RoPE inv_freq fix | `unsloth/models/loader.py` → `_fix_rope_inv_freq()` |
| RoPE scaling configs | `unsloth/models/llama.py` |
