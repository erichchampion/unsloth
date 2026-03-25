# Chapter 33: Mistral, Cohere, Granite, and Falcon H1

> *"Supporting the long tail of architectures."*

---

## Introduction

Beyond the major families (Llama, Gemma, Qwen), Unsloth supports several additional architectures at varying levels of optimization. Mistral has a fully optimized `FastMistralModel` with sliding window attention. Cohere, Granite, and Falcon H1 have `Fast*Model` implementations that are currently disabled in the dispatch map due to correctness issues, falling back to the generic `FastModel` path. This chapter catalogs each architecture, its unique features, and its optimization status.

### What You'll Learn

- Mistral: sliding window attention and Ministral variants
- Cohere: current status and planned optimizations
- Granite: IBM's LLM family
- Falcon H1: hybrid attention + state-space architecture
- GLM-4 MoE: THUDM's Mixture of Experts model
- The FastModel fallback path

### Prerequisites

- The model dispatch system from Chapter 9
- The FastLlamaModel reference from Chapter 30

---

## 33.1 Mistral — Sliding Window Attention

**File:** `mistral.py` (18K) — Fully optimized and dispatched

Mistral's key architectural innovation is **sliding window attention** — each layer only attends to the most recent `window_size` tokens instead of the full context:

```python
# Full causal attention (Llama):
scores[i][j] = Q[i] · K[j]  for all j <= i

# Sliding window attention (Mistral):
scores[i][j] = Q[i] · K[j]  for all max(0, i-window) <= j <= i
```

### Benefits and Trade-offs

| Feature | Full Attention | Sliding Window |
|---------|---------------|----------------|
| Memory | O(n²) | O(n × window) |
| Long-range dependencies | ✅ Full | ⚠️ Transitive only |
| Speed | Slower for long seq | Faster for long seq |
| Default window | N/A | 4096 tokens |

Information beyond the window is accessible through **transitive attention** — layer N can see tokens that layer N-1 attended to, effectively extending the receptive field across the full network depth.

### Attention Mask Handling

Mistral requires different attention mask construction than Llama:
- **2D masks** — used in some configurations
- **4D masks** — required for sliding window computation
- Unsloth handles the mask format detection and conversion automatically

### Ministral Support

The smaller Ministral (3B) model uses the same architecture and shares the `FastMistralModel` code path.

---

## 33.2 Cohere — Temporarily Disabled

**File:** `cohere.py` (19K) — Implemented but dispatch disabled

Cohere's `Command R` family uses custom architectural features:
- Layerwise scaling factors
- Custom tokenizer handling
- Different attention mask conventions

```python
# loader.py — currently commented out:
# elif model_type == "cohere":
#     dispatch_model = FastCohereModel
```

The `FastCohereModel` implementation exists and contains Triton kernel patches, but is disabled until output correctness matches the reference implementation exactly. Cohere models fall back to `FastModel` (torch.compile optimization).

---

## 33.3 Granite — IBM's Architecture

**File:** `granite.py` (23K) — Implemented but dispatch disabled

IBM's Granite family has custom initialization patterns and architectural variations:
- Custom weight initialization scheme
- Different normalization placement
- Architecture-specific attention patterns

Like Cohere, Granite is temporarily disabled in the dispatch map and uses the `FastModel` fallback.

---

## 33.4 Falcon H1 — Hybrid Architecture

**File:** `falcon_h1.py` (29K) — Implemented but dispatch disabled

Falcon H1 is the most architecturally unique model Unsloth supports. It combines traditional transformer attention with **state-space model (SSM)** layers, similar to Mamba:

```python
# Falcon H1 hybrid layer:
if is_attention_layer(layer_idx):
    output = transformer_attention(x)     # Standard attention
else:
    output = state_space_layer(x)         # SSM (Mamba-like)
```

This hybrid design uses attention for global context and SSM for efficient local processing. The dispatch is disabled pending `transformers ≥ 4.53.0` support.

---

## 33.5 GLM-4 MoE

**File:** `glm4_moe.py` (15K)

THUDM's GLM-4 MoE uses a standard Mixture of Experts architecture with:
- 16 experts, top-4 routing
- Standard transformer attention
- Chinese and English bilingual support

Currently dispatched through the `FastModel` fallback.

---

## 33.6 The FastModel Fallback

When a model architecture doesn't have an optimized `Fast*Model` class (or its dispatch is disabled), Unsloth falls back to `FastModel` in `_utils.py`:

```python
class FastModel:
    @staticmethod
    def from_pretrained(...):
        # No hand-written Triton kernels
        # Uses torch.compile for automatic optimization
        # Supports: full fine-tuning, 8-bit, 16-bit
        # Typical speedup: ~1.5× (vs. 2-3× for optimized)
```

The fallback ensures every Hugging Face model works with Unsloth, even without architecture-specific optimizations.

---

## 33.7 Architecture Support Summary

| Architecture | File | Status | Optimization Level |
|-------------|------|--------|-------------------|
| Mistral | `mistral.py` | ✅ Active | Full (Triton kernels) |
| Cohere | `cohere.py` | ⚠️ Disabled | Fallback (torch.compile) |
| Granite | `granite.py` | ⚠️ Disabled | Fallback (torch.compile) |
| Falcon H1 | `falcon_h1.py` | ⚠️ Disabled | Fallback (torch.compile) |
| GLM-4 MoE | `glm4_moe.py` | ⚠️ Fallback | Fallback (torch.compile) |
| DeepSeek | `_deepseek.py` | ✅ Active | Full (MoE + Triton) |

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| FastMistralModel | `unsloth/models/mistral.py` |
| FastCohereModel | `unsloth/models/cohere.py` |
| FastGraniteModel | `unsloth/models/granite.py` |
| FastFalconH1Model | `unsloth/models/falcon_h1.py` |
| GLM-4 MoE | `unsloth/models/glm4_moe.py` |
| FastModel fallback | `unsloth/models/_utils.py` |
