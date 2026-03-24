# Chapter 23: Cross-Entropy Loss Kernel

---

## Introduction

The cross-entropy loss kernel is one of Unsloth's most impactful optimizations. It fuses the logit computation and loss calculation into a single GPU kernel, avoiding the materialization of the full logit tensor.

### What You'll Learn

- Why standard cross-entropy is memory-inefficient
- How the fused kernel avoids materializing the logit tensor
- Triton implementation details
- Memory savings for large vocabularies

---

## Notes & Key Points

### 23.1 The Problem

- Standard flow: logits (V×T tensor) → softmax → log → NLL loss
- For Llama with V=128K vocabulary: logit tensor alone is ~1GB per batch
- This is often the memory bottleneck during training

### 23.2 The Solution

- Fused kernel computes loss chunk-by-chunk without materializing full logits
- Processes one vocabulary slice at a time
- Reduces peak memory from O(V×T) to O(chunk_size×T)

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Cross-entropy kernel | `unsloth/kernels/cross_entropy_loss.py` |
