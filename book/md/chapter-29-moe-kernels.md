# Chapter 29: MoE Grouped GEMM Kernels

---

## Introduction

Mixture of Experts (MoE) models like DeepSeek, GLM, and Qwen3 MoE use sparse expert routing. Unsloth provides custom grouped GEMM kernels for 12x faster MoE training.

### What You'll Learn

- How MoE routing works
- The grouped GEMM optimization
- Autotune caching for Triton kernels
- Integration with MoE model architectures

---

## Notes & Key Points

### 29.1 MoE Architecture

- Instead of one MLP, MoE has N expert MLPs
- A router selects top-K experts per token
- Only selected experts compute, making it sparse
- Challenge: varying expert batch sizes make GPU utilization uneven

### 29.2 Grouped GEMM

- `kernels/moe/grouped_gemm/` — Custom Triton kernel for batched expert computation
- Groups all expert computations into a single fused kernel
- Avoids the overhead of launching N separate GEMM kernels

### 29.3 Autotune Caching

- `moe/autotune_cache.py` (17K) — Caches Triton autotuning results
- First run tunes kernel parameters for the specific GPU
- Subsequent runs use cached values for faster startup

### 29.4 Supported MoE Architectures

- DeepSeek V2/V3/R1
- Qwen3 MoE
- GLM-4 MoE
- gpt-oss

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| MoE kernel directory | `unsloth/kernels/moe/` |
| Grouped GEMM | `unsloth/kernels/moe/grouped_gemm/` |
| Autotune cache | `unsloth/kernels/moe/autotune_cache.py` |
| MoE model docs | `unsloth/kernels/moe/README.md` |
