# Chapter 32: Qwen 2, Qwen 3, and Qwen 3 MoE

> *"From Llama clone to thinking machine."*

---

## Introduction

Alibaba's Qwen model family has evolved rapidly. Qwen 2 is architecturally nearly identical to Llama, making it a thin wrapper in Unsloth. Qwen 3 introduces QK normalization and a "thinking" mode with `<think>` tags. Qwen 3 MoE adds sparse expert routing, leveraging the grouped GEMM kernels from Chapter 29. This chapter traces the progression and covers the implementation details of each variant.

### What You'll Learn

- Qwen 2: why it's essentially a Llama fork and what differs
- Qwen 3: QK normalization, thinking mode, version requirements
- Qwen 3 MoE: expert routing and grouped GEMM integration
- Qwen 2.5 VL: vision-language multimodal support

### Prerequisites

- The FastLlamaModel reference from Chapter 30
- MoE concepts from Chapter 29
- LoRA concepts from Chapter 12

---

## 32.1 Qwen 2 — The Llama Twin

**File:** `qwen2.py` (3.7K — one of the smallest model files)

Qwen 2's architecture is so similar to Llama that `FastQwen2Model` is essentially a class alias:

```python
class FastQwen2Model(FastLlamaModel):
    @staticmethod
    def from_pretrained(model_name, max_seq_length, load_in_4bit, ...):
        # Delegates almost entirely to FastLlamaModel.from_pretrained()
        # Differences:
        #   - Model architecture name: "Qwen2ForCausalLM"
        #   - Vocabulary size: 152,064 (vs. Llama's 128,256)
        #   - Default RoPE base: 1,000,000 (vs. Llama's 10,000)
```

### Key Differences from Llama

| Feature | Llama 3 | Qwen 2.5 |
|---------|---------|----------|
| Vocabulary | 128,256 | 152,064 |
| RoPE base | 500,000 | 1,000,000 |
| Activation | SwiGLU | SwiGLU |
| Normalization | RMSNorm | RMSNorm |
| GQA | Yes | Yes |
| Tie word embeddings | Some variants | Some variants |

Because the architectures are so similar, Qwen 2 benefits from all Llama optimizations with zero additional work.

---

## 32.2 Qwen 3 — Thinking Mode

**File:** `qwen3.py` (17K)

Qwen 3 introduces two significant architectural changes:

### QK Normalization

Qwen 3 applies RMSNorm to query and key vectors before computing attention scores:

```python
# Standard attention (Llama):
Q = q_proj(x)
K = k_proj(x)
scores = (Q @ K.T) / sqrt(head_dim)

# Qwen 3 (with QK-Norms):
Q = q_norm(q_proj(x))    # RMSNorm on queries
K = k_norm(k_proj(x))    # RMSNorm on keys
scores = (Q @ K.T) / sqrt(head_dim)
```

This normalization stabilizes training at higher learning rates and helps with longer context lengths. Unsloth patches these additional norms with the fused RMSNorm kernel from Chapter 27.

### Thinking Mode

Qwen 3 supports chain-of-thought reasoning with explicit `<think>` and `</think>` tags:

```
User: What is 15 × 23?

<think>
I need to multiply 15 by 23.
15 × 20 = 300
15 × 3 = 45
300 + 45 = 345
</think>

The answer is 345.
```

The thinking mode is handled at the chat template level (Chapter 11) — the model architecture itself doesn't change. The template includes the `<think>` tokens in the vocabulary and wraps reasoning content appropriately.

### Version Requirements

Qwen 3 requires `transformers ≥ 4.50.3` for correct `QK-Norm` support.

---

## 32.3 Qwen 3 MoE — Sparse Expert Routing

**File:** `qwen3_moe.py` (9.5K)

Qwen 3 MoE replaces the dense MLP with a sparse Mixture of Experts:

| Parameter | Qwen 3 Dense (30B) | Qwen 3 MoE (30B) |
|-----------|--------------------|--------------------|
| Total parameters | 30B | 30B |
| Active params/token | 30B | ~3.5B |
| Number of experts | 1 | 128 |
| Top-K | N/A | 8 |
| Expert hidden dim | 13,824 | 1,536 |

### Integration with Grouped GEMM

The MoE forward pass uses the grouped GEMM kernel from `kernels/moe/`:

```python
# MoE forward (simplified):
router_logits = self.gate(x)                    # [batch*seq, num_experts]
top_k_scores, top_k_indices = topk(router_logits, k=8)
expert_outputs = grouped_gemm(                   # Single fused kernel
    expert_weights=self.experts,
    inputs=x,
    indices=top_k_indices,
    scores=top_k_scores,
)
output = reduce(expert_outputs)
```

---

## 32.4 Qwen 2.5 VL — Vision-Language

Qwen 2.5 VL (Vision-Language) is a multimodal variant that processes both images and text. It uses a ViT vision encoder with a Resampler connector that maps visual features to the language model's dimension space. See Chapter 16 for vision fine-tuning details.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| FastQwen2Model | `unsloth/models/qwen2.py` (3.7K) |
| FastQwen3Model | `unsloth/models/qwen3.py` (17K) |
| FastQwen3MoeModel | `unsloth/models/qwen3_moe.py` (9.5K) |
| MoE kernels | `unsloth/kernels/moe/` |
| Qwen chat templates | `unsloth/chat_templates.py` |
