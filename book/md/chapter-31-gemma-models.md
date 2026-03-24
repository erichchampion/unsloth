# Chapter 31: Gemma, Gemma 2, and Gemma 3 Support

> *"Google's Gemma family — where every generation introduces a new challenge."*

---

## Introduction

Google's Gemma model family represents three distinct generations, each introducing architectural features that require special handling in Unsloth. Gemma 1 has a unique embedding scaling factor. Gemma 2 introduces attention soft-capping that requires specific Flash Attention versions. Gemma 3 adds multimodal (vision) support and has SDPA correctness issues that force fallback to alternative attention implementations. This chapter explains how Unsloth handles each generation.

### What You'll Learn

- Gemma 1: fixed embedding scaling, RMSNorm offset, GeGLU activation
- Gemma 2: attention soft-capping, alternating window/full attention
- Gemma 3: SDPA workarounds, multimodal support, system prompt handling
- Version requirements for each generation

### Prerequisites

- The FastLlamaModel reference from Chapter 30
- Attention dispatch from Chapters 9 and 22
- Vision model support from Chapter 16

---

## 31.1 Gemma 1 — The Foundation

**File:** `gemma.py` (19K)

Gemma 1 differs from Llama in several ways:

### Embedding Scaling

Gemma multiplies the embedding output by `√hidden_size` before feeding it into the transformer blocks:

```python
# Gemma embedding (differs from Llama)
hidden_states = self.embed_tokens(input_ids)
hidden_states = hidden_states * (self.config.hidden_size ** 0.5)  # Scaling!
```

This is a fixed scaling factor, not a learnable parameter. Without this scaling, the model produces garbage.

### RMSNorm Offset

Gemma's RMSNorm adds 1 to the weight parameter: `output = x * rms(x) * (weight + 1)` instead of the standard `output = x * rms(x) * weight`. This offset means the initial weight values can be close to zero while still producing near-identity normalization.

### GeGLU Activation

While Llama uses SwiGLU, Gemma uses an approximation of GeGLU (GELU-gated linear unit), requiring the fused GeGLU kernel from Chapter 26.

---

## 31.2 Gemma 2 — Soft-Capping Attention

**File:** `gemma2.py` (25K)

Gemma 2's most distinctive feature is **attention soft-capping** — a numerical stability mechanism that prevents attention scores from growing unboundedly:

```python
# Standard attention:
scores = (Q @ K.T) / sqrt(head_dim)

# Gemma 2 soft-capped attention:
scores = (Q @ K.T) / sqrt(head_dim)
scores = tanh(scores / softcap_value) * softcap_value  # Soft-cap!
```

### Flash Attention Requirements

Soft-capping requires specific Flash Attention support:

| Flash Attention Version | Soft-Capping Support |
|------------------------|---------------------|
| < 2.6.3 | ❌ Not supported |
| ≥ 2.6.3 | ✅ Native support |
| SDPA | ❌ Not supported |

Without Flash Attention ≥ 2.6.3, Unsloth falls back to a custom "slow" attention implementation that applies soft-capping manually.

### Alternating Attention Pattern

Gemma 2 alternates between **full attention** and **sliding window attention** across layers:

```python
# Even layers: full causal attention
# Odd layers: sliding window attention (window_size=4096)
if layer_idx % 2 == 0:
    attention = full_causal_attention(Q, K, V)
else:
    attention = sliding_window_attention(Q, K, V, window=4096)
```

### Padding-Free Blocklist

Gemma 2 is in the padding-free blocklist because `slow_attention_softcapping` has `torch.compile` issues (see Chapter 14).

---

## 31.3 Gemma 3 — Multimodal and SDPA Workarounds

Gemma 3 introduces multimodal (image + text) capabilities and requires several workarounds.

### SDPA Disabling

SDPA (Scaled Dot-Product Attention) produces incorrect results for Gemma 3. Unsloth explicitly disables it:

```python
# loader.py
DISABLE_SDPA_MODEL_NAMES = "gemma3,"  # Note trailing comma to avoid matching "gemma3n"
```

The trailing comma is critical — without it, the pattern would also match `gemma3n` (a different model).

### System Prompt Handling

Gemma 3's chat template doesn't support a `system` role directly. Instead, the system prompt is prepended to the first user message:

```python
# Gemma 3 template (simplified)
{% if message.role == "system" %}
    # Not a separate turn — prepended to next user turn
{% endif %}
```

### Vision Support

Gemma 3 supports image inputs via a SigLIP vision encoder with a linear projection connector. See Chapter 16 for vision fine-tuning details.

---

## 31.4 Version Requirements

| Model | Minimum transformers | Flash Attention |
|-------|---------------------|-----------------|
| Gemma 1 | ≥ 4.38 | Optional |
| Gemma 2 | ≥ 4.42 | ≥ 2.6.3 (for soft-capping) |
| Gemma 3 | ≥ 4.49 | Recommended |

### Note on Gemma 3n

Gemma 3n is a separate model family (lightweight nano variant) that uses a different architecture. The SDPA blocklist uses `"gemma3,"` (with trailing comma) specifically to avoid matching `gemma3n`, which has its own handling.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| FastGemmaModel | `unsloth/models/gemma.py` |
| FastGemma2Model | `unsloth/models/gemma2.py` |
| Gemma 3 vision | `unsloth/models/vision.py` |
| SDPA blocklist | `unsloth/models/loader.py` → `DISABLE_SDPA_MODEL_NAMES` |
