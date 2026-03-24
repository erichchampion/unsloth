# Chapter 32: Qwen 2, Qwen 3, and Qwen 3 MoE

---

## Introduction

The Qwen family from Alibaba is one of the most popular model families supported by Unsloth. Qwen 3 introduces QK-norms, and Qwen 3 MoE adds sparse expert routing.

### What You'll Learn

- Qwen 2: thin wrapper around Llama architecture
- Qwen 3: custom attention with QK normalization
- Qwen 3 MoE: Mixture of Experts with grouped GEMM

---

## Notes & Key Points

### 32.1 Qwen 2

- `qwen2.py` (3.7K) — Very thin wrapper around FastLlamaModel
- Architecture is essentially Llama with minor differences
- Shares all Llama optimizations

### 32.2 Qwen 3

- `qwen3.py` (17K) — Custom attention implementation
- Adds QK-norms (query and key normalization before attention)
- Requires `transformers >= 4.50.3`

### 32.3 Qwen 3 MoE

- `qwen3_moe.py` (9.5K) — Sparse MoE variant
- Uses the grouped GEMM kernels from `kernels/moe/`
- Expert routing with top-K selection

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| FastQwen2Model | `unsloth/models/qwen2.py` |
| FastQwen3Model | `unsloth/models/qwen3.py` |
| FastQwen3MoeModel | `unsloth/models/qwen3_moe.py` |
