# Chapter 28: FlexAttention and FP8 Kernels

> *"Flexible masks, compact numbers."*

---

## Introduction

This chapter covers two specialized kernel modules that extend Unsloth's capabilities beyond the core transformer operations. FlexAttention (`flex_attention.py`, 187 lines) wraps PyTorch's FlexAttention API to enable custom attention patterns without manual masking. The FP8 kernels (`fp8.py`, 624 lines) implement 8-bit floating-point quantization/dequantization for memory-efficient training and inference.

### What You'll Learn

- FlexAttention: custom attention score modifiers and masking
- FP8 formats: E4M3 and E5M2 for training
- Per-tensor and per-block quantization strategies
- Integration with TorchAO's float8 backend
- When and why these kernels are used

### Prerequisites

- The kernel architecture from Chapter 22
- FP8 training concepts from Chapter 13
- Understanding of attention masking

---

## 28.1 FlexAttention

FlexAttention is PyTorch's API (introduced in 2.5) for expressing custom attention patterns without explicitly constructing attention masks. Instead of creating a `[seq_len, seq_len]` mask tensor, you define a `score_mod` function that modifies attention scores on the fly:

```python
from torch.nn.attention.flex_attention import flex_attention

def causal_score_mod(score, b, h, q_idx, kv_idx):
    """Causal masking: mask future tokens."""
    return torch.where(q_idx >= kv_idx, score, -float('inf'))

output = flex_attention(query, key, value, score_mod=causal_score_mod)
```

### Unsloth's FlexAttention Wrapper

`flex_attention.py` provides:

| Function | Purpose |
|----------|---------|
| `UnslothFlexAttention` | Wrapper class for FlexAttention with Unsloth compatibility |
| Score modifier helpers | Pre-built modifiers for common patterns |
| Compatibility checks | Version detection (requires PyTorch ≥ 2.5) |

### Current Limitations

- **gpt_oss models only** — currently the primary consumer of FlexAttention in Unsloth
- **Padding-free incompatible** — FlexAttention doesn't correctly handle the padding-free token layout, so `gpt_oss` is in the padding-free blocklist
- **Softcapping** — Gemma 2's attention softcapping uses a different mechanism

---

## 28.2 FP8 Kernels

The FP8 module provides quantization infrastructure for 8-bit training:

### FP8 Formats

| Format | Bits | Exponent | Mantissa | Range | Primary Use |
|--------|------|----------|----------|-------|------------|
| E4M3 | 8 | 4 | 3 | ±448 | Weights and activations (forward) |
| E5M2 | 8 | 5 | 2 | ±57344 | Gradients (backward) |

E4M3 has more precision (3 mantissa bits) but less range, making it better for weights. E5M2 has more range (5 exponent bits) but less precision, better suited for gradients which have larger dynamic range.

### Quantization Strategies

The FP8 module supports two quantization granularities:

```python
# Per-tensor quantization
# One scale factor for the entire tensor
scale = max(abs(tensor)) / max_fp8
quantized = tensor / scale

# Per-block quantization
# One scale factor per block (e.g., per 128 elements)
for block in tensor.chunks(block_size):
    block_scale = max(abs(block)) / max_fp8
    quantized_block = block / block_scale
```

Per-block quantization provides better accuracy because each block gets its own scale factor, but adds storage overhead for the scale factors.

### Key Functions

| Function | Purpose |
|----------|---------|
| `fp8_quantize()` | Quantize FP16/BF16 weights to FP8 E4M3 |
| `fp8_dequantize()` | Dequantize FP8 back to FP16/BF16 |
| `_offline_quantize_to_fp8()` | Batch quantize a full model offline |
| `fp8_matmul()` | FP8 matrix multiplication with TorchAO |

### TorchAO Integration

The FP8 kernels build on TorchAO's float8 backend:

```python
from torchao.float8 import float8_linear_utils
# TorchAO provides:
# - Efficient FP8 GEMM kernels
# - Automatic scaling factor computation
# - Integration with torch.compile
```

---

## 28.3 Attention Score Modification Patterns

FlexAttention enables several attention patterns without explicit mask construction:

| Pattern | score_mod Implementation |
|---------|------------------------|
| Causal | `score if q_idx >= kv_idx else -inf` |
| Sliding window | `score if abs(q_idx - kv_idx) <= window else -inf` |
| Document masking | `score if doc_id[q_idx] == doc_id[kv_idx] else -inf` |
| Softcapping (Gemma 2) | `tanh(score / cap) * cap` |
| ALiBi | `score - slope * abs(q_idx - kv_idx)` |

## 28.4 When These Kernels Are Used

| Kernel | Trigger |
|--------|---------|
| FlexAttention | `gpt_oss` architecture detection |
| FP8 quantize | `load_in_fp8=True` in `from_pretrained()` |
| FP8 matmul | During FP8 training forward/backward passes |

## 28.5 FP8 Hardware Requirements

FP8 support depends on GPU hardware capability:

| GPU | FP8 Support | Notes |
|-----|------------|-------|
| H100 | ✅ Native | Full hardware FP8 support |
| RTX 4090 | ✅ Emulated | Via TorchAO software emulation |
| A100 | ⚠️ Partial | Software emulation, slower than H100 |
| RTX 3090 | ❌ No | Ampere lacks FP8 tensor cores |
| T4 | ❌ No | Turing lacks FP8 support |

On GPUs without native FP8 support, `load_in_fp8=True` will fail or fall back to BF16/FP16.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| FlexAttention wrapper | `unsloth/kernels/flex_attention.py` |
| FP8 kernels | `unsloth/kernels/fp8.py` |
| FP8 model loading | `unsloth/models/loader_utils.py` |
| TorchAO integration | `torchao.float8` (external) |
