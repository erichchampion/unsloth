# Chapter 40: Attention Dispatch and Memory Optimization

> *"The right attention for the right hardware — automatically."*

---

## Introduction

Attention computation dominates both the time and memory cost of transformer training and inference. Unsloth dynamically selects the best attention implementation based on hardware capabilities, model requirements, and library availability. Beyond attention dispatch, Unsloth combines multiple memory optimization techniques — quantization, gradient checkpointing, kernel fusion, sample packing, padding-free batching, and tiled MLP — into a layered stack that enables training 7B models on 6GB GPUs and 70B models on 80GB GPUs.

This final chapter ties together the optimization strategies discussed throughout the book into a unified picture.

### What You'll Learn

- Attention backend selection: Flash Attention, SDPA, FlexAttention, manual
- The `attention_dispatch.py` module
- The full memory optimization stack and how techniques compose
- Long-context training: 500K+ tokens
- Packing vs. padding-free: when to use each

### Prerequisites

- All kernel chapters (Part VI)
- The trainer from Chapter 14
- FastLlamaModel from Chapter 30

---

## 40.1 Attention Dispatch

`utils/attention_dispatch.py` (13K) selects the best attention implementation at model load time:

```
Decision tree:
  Is Flash Attention 2 installed?
    ├─ Yes: Is GPU Ampere or newer (SM ≥ 80)?
    │   ├─ Yes: Use Flash Attention 2    ← Fastest, most memory efficient
    │   └─ No: Fall through
    └─ No: Fall through

  Does model need FlexAttention? (gpt_oss)
    ├─ Yes: Is PyTorch ≥ 2.5?
    │   ├─ Yes: Use FlexAttention         ← Custom attention patterns
    │   └─ No: Fall through
    └─ No: Fall through

  Is model in SDPA blocklist? (Gemma 3, etc.)
    ├─ Yes: Use manual attention loop      ← Slowest but always correct
    └─ No: Use PyTorch SDPA               ← Good default
```

### Backend Comparison

| Backend | Speed | Memory | Requirements | Limitations |
|---------|-------|--------|-------------|-------------|
| Flash Attention 2 | ★★★★★ | ★★★★★ | Ampere+ GPU, Linux | No Windows |
| FlexAttention | ★★★★ | ★★★★ | PyTorch ≥ 2.5 | gpt_oss only |
| PyTorch SDPA | ★★★ | ★★★ | PyTorch ≥ 2.0 | No soft-capping |
| Manual loop | ★★ | ★★ | None | Slowest |

### SDPA Blocklist

Some models produce incorrect results with SDPA:

```python
DISABLE_SDPA_MODEL_NAMES = "gemma3,"
# Trailing comma prevents matching "gemma3n"
```

---

## 40.2 The Memory Optimization Stack

Unsloth's memory optimizations are **composable** — they stack multiplicatively:

```
Base requirement (7B, full FP16):                    ~28 GB
  └─ + 4-bit quantization (QLoRA):                   ~7 GB   (75% reduction)
      └─ + Gradient checkpointing:                   ~4 GB   (43% reduction)
          └─ + Fused cross-entropy kernel:            ~3 GB   (25% reduction)
              └─ + Fused SwiGLU/RMSNorm:              ~2.8 GB (7% reduction)
                  └─ + Sample packing:                 ~2.5 GB (10% reduction)
                      └─ + Padding-free:               ~2.3 GB (8% reduction)

Final: 7B model trainable on a 6GB GPU!
```

### Technique Details

| Technique | Chapter | Memory Saved | Speed Impact |
|-----------|---------|-------------|-------------|
| 4-bit quantization | Ch 12 | ~75% | Slightly slower (dequant) |
| Gradient checkpointing | Ch 30 | ~50% of activations | ~30% slower |
| Fused cross-entropy | Ch 23 | Up to 1 GB | Faster (fewer launches) |
| Fused kernels (all) | Ch 22-29 | ~10-20% | 2× faster |
| Sample packing | Ch 14 | Variable | Faster (less padding) |
| Padding-free | Ch 14 | Variable | Faster (zero padding) |
| FP8 training | Ch 28 | ~25% vs FP16 | Similar speed |

---

## 40.3 Packing vs. Padding-Free

Two approaches to eliminate wasted computation on padding tokens:

### Sample Packing

Concatenates multiple short sequences into one long sequence with attention masks to prevent cross-contamination:

```
Before packing:  [A A A PAD PAD] [B B B B PAD] [C C PAD PAD PAD]
After packing:   [A A A B B B B] [C C PAD PAD PAD PAD PAD PAD]  ← fewer sequences
```

### Padding-Free

Removes padding entirely and packs tokens into a 1D stream with position tracking:

```
Padding-free:    [A A A B B B B C C]  ← zero padding, pure tokens
Position IDs:    [0 1 2 0 1 2 3 0 1]  ← reset per sample
```

### When to Use Each

| Scenario | Recommended | Why |
|----------|------------|-----|
| Variable-length data | Padding-free | Maximum efficiency |
| Equal-length data | Either | No padding waste |
| Gemma 2 models | Packing | Padding-free incompatible |
| gpt_oss models | Packing | FlexAttention blocks padding-free |
| GRPO/RL training | Packing | RL requires per-sample boundaries |

---

## 40.4 Long-Context Training

Unsloth enables training on sequences of 500K+ tokens through the combination of:

1. **RoPE scaling** (Chapter 25) — extends positional encoding
2. **Flash Attention** — O(n) memory instead of O(n²)
3. **Gradient checkpointing** — recompute activations instead of storing
4. **Tiled MLP** — process MLP in chunks along sequence dimension

```python
# Long-context example:
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 131072,  # 128K context
    load_in_4bit = True,
)
```

### Memory Scaling

| Context Length | GPU Memory (7B, QLoRA) | Requires |
|---------------|----------------------|----------|
| 2,048 | ~6 GB | Any GPU |
| 8,192 | ~12 GB | RTX 4070+ |
| 32,768 | ~24 GB | RTX 4090 |
| 131,072 | ~48 GB | A40/A6000 |
| 524,288 | ~80 GB | A100/H100 |

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Attention dispatch | `unsloth/utils/attention_dispatch.py` |
| Packing | `unsloth/utils/packing.py` |
| Padding-free | `unsloth/utils/__init__.py` |
| Tiled MLP | `unsloth_zoo/tiled_mlp.py` (external) |
| Gradient checkpointing | `unsloth/models/_utils.py` |
