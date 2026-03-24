# Chapter 31: Gemma, Gemma 2, and Gemma 3 Support

---

## Introduction

Google's Gemma family requires special handling due to unique architectural features like fixed embeddings, soft-capping attention, and GDN (Gated Depthwise Normalization).

### What You'll Learn

- Gemma 1: fixed embedding layer, RMSNorm with offset
- Gemma 2: attention soft-capping, flash-attn integration
- How Unsloth disables SDPA for Gemma 3 due to correctness issues

---

## Notes & Key Points

### 31.1 Gemma-Specific Features

- Embedding weights are fixed (not trained) in some modes
- RMSNorm uses a +1 offset compared to standard implementations
- Requires `transformers >= 4.38` (Gemma 1) / `>= 4.42` (Gemma 2)

### 31.2 Attention Soft-Capping (Gemma 2)

- Gemma 2 applies `tanh(score / cap) * cap` to attention scores
- Flash Attention ≥ 2.6.3 supports this natively
- Without flash-attn, falls back to custom slow implementation

### 31.3 Gemma 3 and SDPA

- SDPA (Scaled Dot-Product Attention) is disabled for Gemma 3
- `DISABLE_SDPA_MODEL_NAMES` includes `"gemma3,"` (with comma to avoid matching gemma3n)
- Forces use of alternative attention implementation

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| FastGemmaModel | `unsloth/models/gemma.py` |
| FastGemma2Model | `unsloth/models/gemma2.py` |
| SDPA blocklist | `unsloth/models/loader.py` → `DISABLE_SDPA_MODEL_NAMES` |
