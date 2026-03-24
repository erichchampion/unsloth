# Chapter 40: Attention Dispatch and Memory Optimization

---

## Introduction

Unsloth dynamically selects attention implementations based on hardware capabilities and model requirements. This chapter covers the attention dispatch system and broader memory optimization strategies.

### What You'll Learn

- Attention backend selection: Flash Attention, SDPA, manual
- The `attention_dispatch.py` module
- Memory optimization techniques: gradient checkpointing, tiled MLP, packing
- Long-context training: 500K+ tokens

---

## Notes & Key Points

### 40.1 Attention Dispatch

- `utils/attention_dispatch.py` (13K) — Selects best attention implementation
- Priority: Flash Attention → SDPA → manual loop
- Some models blocklisted from SDPA (`DISABLE_SDPA_MODEL_NAMES`)
- Flash Attention requires Ampere+ GPU and Linux

### 40.2 Memory Optimization Stack

| Technique | Module | Memory Savings |
|-----------|--------|---------------|
| 4-bit quantization (QLoRA) | `bitsandbytes` | ~75% |
| Gradient checkpointing | `_utils.py` | ~50% |
| Fused cross-entropy | `kernels/cross_entropy_loss.py` | ~15-30% |
| Sample packing | `utils/packing.py` | Variable |
| Padding-free batching | `utils/__init__.py` | Variable |
| Tiled MLP | `unsloth_zoo/tiled_mlp.py` | ~10-20% |
| FP8 training | `kernels/fp8.py` | ~25% |

### 40.3 Long-Context Training

- 500K+ context windows on 80GB GPUs
- Achieved through layered memory optimizations
- 7x longer context RL via custom batching algorithms
- RoPE scaling for extended context support

### 40.4 Packing and Padding-Free

- `utils/packing.py` (14K) — Concatenates multiple samples into one sequence
- `configure_sample_packing()` and `enable_sample_packing()` — setup functions
- `configure_padding_free()` and `enable_padding_free_metadata()` — alternative approach
- Both avoid wasting compute on padding tokens

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Attention dispatch | `unsloth/utils/attention_dispatch.py` |
| Packing | `unsloth/utils/packing.py` |
| Tiled MLP | `unsloth_zoo/tiled_mlp.py` (external) |
| Gradient checkpointing | `unsloth/models/_utils.py` |
