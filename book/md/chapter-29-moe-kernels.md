# Chapter 29: MoE Grouped GEMM Kernels

> *"When one expert isn't enough."*

---

## Introduction

Mixture of Experts (MoE) models like DeepSeek V2/V3, Qwen 3 MoE, and GLM-4 MoE replace the standard MLP with a collection of expert MLPs and a router that selects which experts process each token. This sparsity allows models to have enormous total parameter counts (e.g., DeepSeek V3's 671B) while only activating a fraction per token. The challenge is efficiency — naive implementations launch separate matrix multiplications for each expert, wasting GPU throughput.

Unsloth's MoE kernels (`kernels/moe/`, including a grouped GEMM implementation) solve this by batching all expert computations into a single fused kernel, achieving up to 12× faster MoE training.

### What You'll Learn

- How MoE routing works: top-K expert selection
- The grouped GEMM optimization: batching variable-size expert computations
- Triton autotuning and caching for MoE kernels
- Supported MoE architectures in Unsloth

### Prerequisites

- The kernel architecture from Chapter 22
- Understanding of matrix multiplication (GEMM)
- Basic MoE concepts (routers, experts, load balancing)

---

## 29.1 MoE Architecture Basics

In a standard transformer MLP, every token passes through the same set of weights. In an MoE MLP, a router network selects a subset of experts for each token:

```
Standard MLP:
  x → [gate_proj, up_proj, down_proj] → output    (all tokens use same weights)

MoE MLP:
  x → Router → scores for N experts
            → select top-K experts (e.g., K=2 of N=64)
            → x → Expert[i] → weighted output
            → x → Expert[j] → weighted output
            → sum weighted outputs
```

### Parameter Efficiency

| Model | Total Params | Active Params/Token | Experts |
|-------|-------------|--------------------|---------| 
| Llama 3 70B | 70B | 70B | 1 (dense) |
| DeepSeek V3 | 671B | ~37B | 256 (top-8) |
| Qwen 3 MoE (30B) | 30B | ~3.5B | 128 (top-8) |

The model has 671B total parameters but only activates 37B per token — getting the quality of a much larger model at the inference cost of a smaller one.

---

## 29.2 The Problem: Variable-Size Batches

When the router sends different tokens to different experts, each expert receives a different number of tokens. Naive execution launches separate GEMM operations for each expert:

```
Expert 0: processes 15 tokens → GEMM(15, d, h)   ← small, inefficient
Expert 1: processes 3 tokens  → GEMM(3, d, h)    ← tiny, very inefficient
Expert 2: processes 28 tokens → GEMM(28, d, h)   ← medium
...
Expert 63: processes 7 tokens → GEMM(7, d, h)    ← small
```

With 64 experts, that's 64 separate kernel launches, most with small batch sizes that underutilize the GPU's massive parallelism.

---

## 29.3 Grouped GEMM Solution

The grouped GEMM kernel batches all expert computations into a single kernel launch:

```python
# Instead of 64 separate launches:
for expert_id in range(num_experts):
    output[expert_id] = expert_weight[expert_id] @ input[expert_id]

# Single grouped GEMM launch:
all_outputs = grouped_gemm(
    expert_weights,     # [N, d, h] — all expert weights stacked
    grouped_inputs,     # Variable-size inputs, sorted by expert
    group_sizes,        # [15, 3, 28, ..., 7] — tokens per expert
)
```

The Triton kernel parallelizes across both the expert dimension and the matrix multiplication:

```
Thread block assignment:
  ├─ Block (0,0): Expert 0, rows 0-15
  ├─ Block (0,1): Expert 1, rows 0-3
  ├─ Block (0,2): Expert 2, rows 0-15
  ├─ Block (0,3): Expert 2, rows 16-28
  └─ ...
```

---

## 29.4 Autotuning Cache

MoE kernels have many tunable parameters (block sizes, number of warps, pipeline stages). The `autotune_cache.py` module (500 lines) persists Triton autotuning results:

```python
# First run: benchmark all configurations
autotune_config = triton.autotune(
    configs=[
        Config(BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, num_warps=4),
        Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4),
        Config(BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_warps=8),
        # ... many more configs
    ],
    key=["M", "N", "K"],
)

# autotune_cache.py saves results to disk
# Subsequent runs load cached optimal configs instantly
```

The cache is keyed by GPU model, data dimensions, and kernel parameters. This avoids the expensive benchmarking phase (which can take 30+ seconds) on subsequent runs.

---

## 29.5 Supported MoE Architectures

| Architecture | File | Experts | Top-K | Special Features |
|-------------|------|---------|-------|-----------------|
| DeepSeek V2/V3/R1 | `deepseek.py` | 160/256 | 6/8 | Shared expert + routed experts |
| Qwen 3 MoE | `qwen3_moe.py` | 128 | 8 | Standard top-K routing |
| GLM-4 MoE | `glm.py` | 16 | 4 | IBM's architecture |
| gpt-oss | `gpt_oss.py` | Variable | Variable | FlexAttention + MoE |

### DeepSeek's Shared Expert

DeepSeek models use a unique architecture with both **shared experts** (always active) and **routed experts** (selected by the router). The shared expert ensures a baseline of computation happens for every token, while routed experts provide specialization.

### Load Balancing

MoE routing can degenerate if the router always selects the same experts. An auxiliary load balancing loss encourages uniform expert utilization:

```python
# Load balancing loss (simplified)
expert_counts = count_tokens_per_expert(router_output)
balance_loss = coefficient * variance(expert_counts)
total_loss = language_model_loss + balance_loss
```

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| MoE kernel directory | `unsloth/kernels/moe/` |
| Grouped GEMM | `unsloth/kernels/moe/grouped_gemm/` |
| Autotuning cache | `unsloth/kernels/moe/autotune_cache.py` |
| DeepSeek MoE model | `unsloth/models/deepseek.py` |
| Qwen 3 MoE model | `unsloth/models/qwen3_moe.py` |
