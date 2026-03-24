# Chapter 13: Full Fine-Tuning and FP8 Training

> *"When LoRA isn't enough — training every parameter."*

---

## Introduction

While LoRA and QLoRA are the most common training modes in Unsloth, some scenarios demand updating every parameter in the model. Research experiments, domain-specific pretraining, and situations where LoRA's low-rank approximation falls short all call for full fine-tuning. Unsloth supports this through its `full_finetuning=True` mode, which loads the model in full precision and trains all parameters with the same kernel optimizations that make LoRA training fast.

FP8 (8-bit floating point) training offers a middle ground — full parameter updates with reduced memory footprint. This chapter covers both approaches.

### What You'll Learn

- When and why to use full fine-tuning over LoRA
- How `full_finetuning=True` configures the model
- FP8 quantization: E4M3 vs. E5M2 formats
- Memory comparison across training modes
- Combining FP8 with LoRA and with RL training

### Prerequisites

- The LoRA and QLoRA concepts from Chapter 12
- Understanding of floating-point number formats
- The model loading pipeline from Chapter 7

---

## 13.1 Full Fine-Tuning Mode

Full fine-tuning is enabled by passing `full_finetuning=True` (or `load_in_16bit=True`) to `from_pretrained()`:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B",
    max_seq_length = 4096,
    full_finetuning = True,  # No quantization, all params trainable
)
# No get_peft_model() call needed — all parameters are already trainable
```

In this mode:
- All model weights are loaded in BF16 or FP16 (no 4-bit quantization)
- All parameters have `requires_grad=True`
- Unsloth's Triton kernel patches are still applied for speedup
- Gradient checkpointing is strongly recommended to manage memory

---

## 13.2 Memory Requirements

Full fine-tuning requires significantly more memory than QLoRA:

| Mode | Weights | Gradients | Optimizer | Total (7B) |
|------|---------|-----------|-----------|------------|
| QLoRA (4-bit + LoRA r=16) | 3.5 GB | ~0.1 GB | ~0.2 GB | **~4 GB** |
| LoRA (FP16 + LoRA r=16) | 14 GB | ~0.1 GB | ~0.2 GB | **~14.5 GB** |
| Full FP16 | 14 GB | 14 GB | 28 GB | **~56 GB** |
| Full FP8 | 7 GB | 7 GB | 14 GB | **~28 GB** |

Gradient checkpointing trades compute for memory by recomputing activations during the backward pass, reducing the effective memory requirement by 3-5×.

---

## 13.3 FP8 Training

FP8 (8-bit floating point) uses two formats:

| Format | Exponent Bits | Mantissa Bits | Range | Use Case |
|--------|---------------|---------------|-------|----------|
| E4M3 | 4 | 3 | ±448 | Forward pass (weights, activations) |
| E5M2 | 5 | 2 | ±57344 | Backward pass (gradients) |

Unsloth supports FP8 through `load_in_fp8=True`:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B",
    load_in_fp8 = True,
)
```

### Offline FP8 Quantization

When the user requests FP8 loading, the loader checks for pre-quantized FP8 weights. If none exist, it calls `_offline_quantize_to_fp8()` in `loader_utils.py` to quantize the FP16/BF16 weights to FP8 on-the-fly:

```python
# loader_utils.py — simplified
def _offline_quantize_to_fp8(model_name, save_directory):
    # 1. Load model in FP16/BF16
    # 2. Quantize each linear layer to E4M3
    # 3. Save quantized weights to local directory
    # 4. Return path to quantized model
```

---

## 13.4 FP8 + LoRA

FP8 can be combined with LoRA for even greater memory efficiency:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B",
    load_in_fp8 = True,
)
model = FastLanguageModel.get_peft_model(model, r=16, ...)
```

This gives you 8-bit base weights (half the memory of FP16) with trainable LoRA adapters — a good compromise between QLoRA's aggressive 4-bit quantization and full FP16 fidelity.

---

## 13.5 FP8 + Reinforcement Learning

Combining FP8 with RL training (GRPO) is one of Unsloth's highlighted use cases, enabling RL training on consumer GPUs:

```
Memory comparison for RL (7B model):
  Full FP16 + GRPO:    ~80 GB (requires A100)
  FP8 + LoRA + GRPO:   ~16 GB (fits on RTX 4090)
  QLoRA + GRPO:         ~8 GB  (fits on RTX 3090)
```

---

## 13.6 When to Choose Each Mode

| Scenario | Recommended Mode |
|----------|-----------------|
| Quick task adaptation | QLoRA (default) |
| High-quality fine-tuning | LoRA FP16 |
| Domain pretraining | Full fine-tuning |
| Memory-constrained + quality | FP8 + LoRA |
| RL on consumer GPU | QLoRA or FP8 + LoRA |
| Research experiments | Full fine-tuning |

## 13.7 Hardware Requirements

| Training Mode | Minimum GPU | Recommended GPU |
|--------------|-------------|-----------------|
| QLoRA 4-bit (7B) | 6 GB (RTX 3060) | 12 GB (RTX 4070) |
| FP8 + LoRA (7B) | 12 GB (RTX 4070) | 16 GB (RTX 4080) |
| Full FP16 (7B) | 48 GB (A40) | 80 GB (A100) |
| QLoRA 4-bit (70B) | 40 GB (A6000) | 80 GB (A100) |
| Full FP16 (70B) | 4×80 GB (4×A100) | 8×80 GB |

Gradient checkpointing can reduce memory requirements by approximately 3-5×, making full fine-tuning of 7B models feasible on 16 GB GPUs at the cost of ~30% slower training.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Full fine-tuning path | `unsloth/models/loader.py` |
| FP8 quantization | `unsloth/models/loader_utils.py` |
| BitsAndBytes config | `unsloth/models/loader_utils.py` |
| Gradient checkpointing | `unsloth/models/llama.py` |
