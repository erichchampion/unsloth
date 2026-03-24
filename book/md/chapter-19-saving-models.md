# Chapter 19: Saving Models — LoRA, Merged 16-bit, and 4-bit

---

## Introduction

`unsloth_save_model()` is the central function for persisting trained models. It supports three save methods and handles everything from LoRA adapter exports to full weight merging.

### What You'll Learn

- The three save methods: `"lora"`, `"merged_16bit"`, `"merged_4bit"`
- How LoRA weights are merged with `_merge_lora()`
- Memory management: VRAM → RAM → disk spill strategy
- Sharding and `safe_serialization` trade-offs

---

## Notes & Key Points

### 19.1 Save Methods

| Method | What It Saves | When to Use |
|--------|--------------|-------------|
| `"lora"` | Only adapter weights (~100MB) | Default. Fast. Resume training later. |
| `"merged_16bit"` | Full model in float16 | Required for GGUF export. |
| `"merged_4bit"` | Full model in 4-bit | For inference-only deployment. |

### 19.2 LoRA Merging

- `_merge_lora()` — Dequantizes base weights, adds `s * A @ B`, re-quantizes
- Uses `fast_dequantize()` for efficient 4-bit → float32 conversion
- Checks for infinity/NaN after merge
- Handles bias terms

### 19.3 Memory Spill Strategy

1. Try to fit merged weights in **GPU VRAM**
2. If VRAM full, try **system RAM** (limited due to memory leaks)
3. If RAM full, save to **disk** via temporary `.pt` files, then `mmap` them back

### 19.4 Tokenizer Saving

- Sets `padding_side = "left"` for inference compatibility
- Saves tokenizer alongside model weights
- Reverts padding side after saving

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Save function | `unsloth/save.py` → `unsloth_save_model()` |
| LoRA merge | `unsloth/save.py` → `_merge_lora()` |
| Fast dequantize | `unsloth/kernels/__init__.py` |
