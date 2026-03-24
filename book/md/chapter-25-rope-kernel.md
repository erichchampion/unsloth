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
