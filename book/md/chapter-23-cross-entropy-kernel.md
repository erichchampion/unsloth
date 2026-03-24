# Chapter 23: Cross-Entropy Loss Kernel

> *"The single biggest memory saving in the entire codebase."*

---

## Introduction

Cross-entropy loss is computed at every training step, and for large-vocabulary models it creates the single largest memory bottleneck. A model with a 128K vocabulary and a batch of 2,048 tokens would normally materialize a logit tensor of shape [2048, 128K] — roughly **1 GB** of peak memory — just for the loss computation. Unsloth's fused cross-entropy kernel (`cross_entropy_loss.py`, 463 lines) eliminates this bottleneck by computing the loss chunk-by-chunk without ever materializing the full logit tensor.

### What You'll Learn

- Why standard cross-entropy is memory-inefficient for large vocabularies
- How the chunked/fused kernel avoids materializing the logit tensor
- The Triton implementation: online softmax and chunked reduction
- Memory savings analysis for different vocabulary sizes
- Gradient computation in the fused kernel

### Prerequisites

- The kernel architecture overview from Chapter 22
- Understanding of softmax and cross-entropy loss
- Basic Triton programming concepts

---

## 23.1 The Problem: O(V×T) Memory

Standard cross-entropy in PyTorch follows this sequence:

```python
# Standard PyTorch cross-entropy (simplified)
logits = lm_head(hidden_states)      # Shape: [batch*seq_len, vocab_size]
loss = F.cross_entropy(logits, labels)
```

The `logits` tensor is enormous:

| Vocabulary | Sequence Length | Logit Tensor Size |
|-----------|----------------|-------------------|
| 32K (Llama 2) | 2048 | 256 MB |
| 128K (Llama 3) | 2048 | 1 GB |
| 128K (Llama 3) | 8192 | 4 GB |
| 152K (Qwen 2.5) | 2048 | 1.2 GB |

This tensor exists only to compute a single scalar loss value — massive memory waste.

---

## 23.2 The Solution: Chunked Computation

Unsloth's kernel processes the vocabulary dimension in chunks (typically 4,096–8,192 tokens). For each chunk:

1. **Compute partial logits** — matrix multiply hidden states with a slice of the LM head weights
2. **Online softmax** — update running max and sum statistics incrementally
3. **Accumulate loss** — add the log-softmax value at the target token position

```
Standard:  [hidden] × [full LM head]  →  [full logits]  →  loss
           Materializes V × T tensor

Chunked:   [hidden] × [LM head chunk 1]  →  partial logits  →  update stats
           [hidden] × [LM head chunk 2]  →  partial logits  →  update stats
           ...
           [hidden] × [LM head chunk N]  →  partial logits  →  final loss
           Peak memory: chunk_size × T  (not V × T)
```

### Online Softmax Algorithm

The key mathematical trick is the **online softmax** — computing `log_sum_exp` incrementally without needing all logits at once:

```python
# Numerically stable online log-sum-exp
running_max = -float('inf')
running_sum = 0.0

for chunk in logit_chunks:
    chunk_max = max(chunk)
    if chunk_max > running_max:
        running_sum = running_sum * exp(running_max - chunk_max)
        running_max = chunk_max
    running_sum += sum(exp(chunk - running_max))

log_sum_exp = running_max + log(running_sum)
```

---

## 23.3 Memory Savings

| Model | Vocab | Standard Peak | Unsloth Peak | Reduction |
|-------|-------|--------------|-------------|-----------|
| Llama 2 (32K) | 32,000 | 256 MB | 32 MB | **8×** |
| Llama 3 (128K) | 128,256 | 1 GB | 32 MB | **32×** |
| Qwen 2.5 (152K) | 152,064 | 1.2 GB | 32 MB | **38×** |

The savings grow proportionally with vocabulary size, making this kernel increasingly valuable as models adopt larger vocabularies.

---

## 23.4 Gradient Computation

The fused kernel also computes gradients without materializing the full Jacobian. During the backward pass:

```python
# Standard: gradient of cross-entropy w.r.t. logits
# d_logits = softmax(logits) - one_hot(target)
# This requires the full softmax output (V × T)

# Fused: compute gradient chunk-by-chunk
# Only needs: running_max, running_sum, and the target token info
```

---

## 23.5 Configuration

The kernel is enabled automatically by `Fast*Model` patching. It replaces `nn.CrossEntropyLoss` in the model's forward pass:

```python
# During model patching (simplified)
model.loss_fn = fast_cross_entropy_loss  # Replace standard loss
```

The chunk size is auto-tuned by Triton's `@autotune` decorator based on the specific GPU's characteristics.

## 23.6 Label Smoothing Support

The fused kernel also supports label smoothing — a regularization technique that softens the one-hot target distribution:

```python
# Without smoothing: target = [0, 0, 1, 0, 0]  (one-hot)
# With smoothing (ε=0.1): target = [0.02, 0.02, 0.92, 0.02, 0.02]
```

Label smoothing is computed inside the kernel without allocating the full smoothed distribution, maintaining the memory savings.

### Numerical Stability

The kernel uses the log-sum-exp trick for numerical stability:
- `max(logits)` is subtracted before exponentiation to prevent overflow
- The online algorithm updates `running_max` as new chunks are processed
- All operations use float32 accumulation regardless of input dtype

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Cross-entropy kernel | `unsloth/kernels/cross_entropy_loss.py` |
| Kernel injection | `unsloth/models/llama.py` |
| Loss utilities | `unsloth/kernels/utils.py` |
