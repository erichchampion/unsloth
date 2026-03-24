# Chapter 33: Mistral, Cohere, Granite, and Falcon H1

---

## Introduction

Unsloth supports several additional architectures beyond Llama, Gemma, and Qwen. Some are fully optimized, others are in development.

### What You'll Learn

- Mistral: sliding window attention
- Cohere: temporarily disabled optimizations
- Granite: IBM's LLM family
- Falcon H1: hybrid architecture (temporarily disabled)
- GLM-4 MoE: Chinese LLM with MoE

---

## Notes & Key Points

### 33.1 Mistral

- `mistral.py` (18K) — Sliding window attention support
- Architecture similar to Llama but with windowed attention
- Includes Ministral (3B) support

### 33.2 Temporarily Disabled Architectures

Some architectures have `Fast*Model` implementations but are currently disabled in the dispatch:

```python
# From loader.py:
# Temporary disable optimized Cohere until errors match
# elif model_type == "cohere":
#     dispatch_model = FastCohereModel

# Temporary disable optimized Granite until errors match
# elif model_type == "granite":
#     dispatch_model = FastGraniteModel
```

- These fall back to `FastModel` (torch.compile-based optimization)

### 33.3 Falcon H1

- `falcon_h1.py` (29K) — Hybrid architecture support
- Currently commented out in dispatch (requires transformers ≥ 4.53.0)

### 33.4 GLM-4 MoE

- `glm4_moe.py` (15K) — Chinese LLM with MoE routing
- Dispatched through `FastModel` fallback

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| FastMistralModel | `unsloth/models/mistral.py` |
| FastCohereModel | `unsloth/models/cohere.py` |
| FastGraniteModel | `unsloth/models/granite.py` |
| FastFalconH1Model | `unsloth/models/falcon_h1.py` |
| GLM-4 MoE | `unsloth/models/glm4_moe.py` |
