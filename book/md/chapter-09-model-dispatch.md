# Chapter 9: Model Dispatch — Architecture-Specific Fast Paths

> *"Every architecture gets its own optimized treatment."*

---

## Introduction

Once `FastLanguageModel.from_pretrained()` identifies a model's architecture (Chapter 7), it dispatches to a specialized `Fast*Model` class that applies architecture-specific optimizations. These classes are the workhorses of Unsloth — they're where generic Hugging Face models become "fast" by replacing standard PyTorch operations with custom Triton kernels and fused operations.

This chapter explains the `Fast*Model` class hierarchy, the common patching pattern they share, and the architecture-specific differences that make each one unique.

### What You'll Learn

- The `Fast*Model` class hierarchy and its inherited pattern
- What "fast" means: which operations are replaced and why
- Architecture-specific optimizations for each supported model family
- The fallback path for unrecognized architectures
- How `get_peft_model()` adds LoRA adapters after patching

### Prerequisites

- The loading pipeline from Chapter 7
- Basic understanding of transformer architecture components (attention, MLP, LayerNorm)

---

## 9.1 The Fast*Model Pattern

All `Fast*Model` classes share a common pattern established by `FastLlamaModel` — the reference implementation. Each class provides three static methods:

```python
class FastLlamaModel:
    @staticmethod
    def from_pretrained(model_name, max_seq_length, load_in_4bit, ...):
        """Load model + tokenizer, apply all optimizations."""
        # 1. Load model via AutoModelForCausalLM
        # 2. Load tokenizer via load_correct_tokenizer()
        # 3. Patch attention, MLP, and norm layers
        # 4. Configure gradient checkpointing
        return model, tokenizer

    @staticmethod
    def get_peft_model(model, r, target_modules, lora_alpha, ...):
        """Wrap model with LoRA adapters, then re-patch."""
        # 1. Apply LoRA via PEFT's get_peft_model()
        # 2. Re-patch model (PEFT wrapping removes some patches)
        return model

    @staticmethod
    def patch_peft_model(model, use_gradient_checkpointing):
        """Re-apply patches that PEFT wrapping removed."""
        # Called after PEFT wrapping to restore Unsloth patches
```

Simpler architectures (Qwen2, Mistral) are thin wrappers that call into `FastLlamaModel` with architecture-specific adjustments.

---

## 9.2 What "Fast" Means

The `Fast*Model` classes replace standard PyTorch operations with optimized alternatives at four levels:

### Kernel Replacements

| Original Operation | Unsloth Replacement | Speedup Source |
|-------------------|---------------------|----------------|
| `F.cross_entropy()` | `fast_cross_entropy_loss()` | Avoids materializing V×T logit tensor |
| `apply_rotary_pos_emb()` | `fast_rope_embedding()` | Fused cos/sin in one kernel launch |
| `RMSNorm.forward()` | `fast_rms_layernorm()` | Single kernel instead of three ops |
| `SwiGLU (gate·silu·up)` | `fast_swiglu_fg()` | Three matmuls fused into one kernel |
| `torch.matmul` (LoRA) | `get_lora_parameters_bias()` | Fused dequant + matmul + LoRA |

### Attention Dispatch

Attention is replaced based on what's available on the system:

```
Priority order:
  1. Flash Attention 2   (if flash-attn installed + Ampere+)
  2. FlexAttention        (if PyTorch 2.5+ + attention_dispatch enabled)
  3. Scaled Dot Product   (PyTorch's built-in sdpa)
  4. Manual attention     (fallback for compatibility)
```

### Gradient Checkpointing

Unsloth's `"unsloth"` gradient checkpointing strategy is more memory-efficient than the standard HF implementation. It selectively checkpoints layers based on their memory footprint rather than applying uniform checkpointing to all layers.

### Memory Optimizations

- **In-place operations** where safe (avoiding unnecessary tensor copies)
- **Tiled MLP** for reducing peak memory during MLP forward passes
- **Embedding learning rate separation** — the embedding layer can use a different learning rate than the rest of the model

---

## 9.3 Architecture-Specific Differences

### Llama (`llama.py`, 143K — ~3,500 lines)

The reference implementation. All other architectures inherit from or delegate to this:

- **RoPE scaling** — supports standard, dynamic, linear, and LongRoPE scaling
- **Llama 3.1+ fixes** — special handling for attention mask edge cases
- **Tiled MLP** — optional chunked MLP computation to reduce peak VRAM
- **Grouped Query Attention (GQA)** — optimized for different head count configurations

### Gemma (`gemma.py`, 19K) and Gemma 2 (`gemma2.py`, 25K)

- **Fixed embeddings** — Gemma uses a fixed scaling factor on embeddings (`hidden_size ** 0.5`)
- **Attention softcapping** — Gemma 2 applies `tanh(logits / softcap_value) * softcap_value` to attention scores, requiring Flash Attention ≥ 2.6.3 for efficient support
- **Sliding window attention** — alternating layers use full vs. sliding window attention

### Qwen 2 (`qwen2.py`, 4K)

A thin wrapper around `FastLlamaModel` — Qwen 2's architecture is nearly identical to Llama:

```python
class FastQwen2Model(FastLlamaModel):
    # Overrides only the model-specific loading configs
    pass
```

### Qwen 3 (`qwen3.py`, 17K)

More substantial differences from Llama:

- **QK-Norms** — applies layer normalization to queries and keys before attention
- **Thinking mode** — supports `<think>` / `</think>` tags for chain-of-thought

### Qwen 3 MoE (`qwen3_moe.py`, 10K)

- **Mixture of Experts** — routes tokens to a subset of expert MLPs
- **Grouped GEMM** — uses Triton kernels from `kernels/moe/` for efficient batched matrix multiplication across experts

### Mistral (`mistral.py`, 18K)

- **Sliding window attention** — attention window limited to a fixed size (4096 by default)
- **Attention mask differences** — 2D vs 4D mask handling

### Other Architectures

| Architecture | File | Key Difference |
|-------------|------|----------------|
| Cohere | `cohere.py` (19K) | Custom tokenizer handling, layerwise scaling |
| Granite | `granite.py` (23K) | IBM's architecture with custom initialization |
| Falcon H1 | `falcon_h1.py` (29K) | Hybrid attention + state-space model |

---

## 9.4 The Fallback: FastModel

When the model architecture is not in the dispatch map, Unsloth falls back to `FastModel` (defined in `_utils.py`):

```python
class FastModel:
    @staticmethod
    def from_pretrained(...):
        # Uses torch.compile for optimization instead of hand-written kernels
        # Supports full fine-tuning, 8-bit, and 16-bit modes
        # No architecture-specific kernel patches
```

This path uses `torch.compile` for optimization, which provides decent speedups (~1.5×) without hand-tuned kernels. It ensures that new model architectures work with Unsloth immediately, even before architecture-specific optimizations are added.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Llama (reference impl) | `unsloth/models/llama.py` |
| Gemma dispatch | `unsloth/models/gemma.py`, `gemma2.py` |
| Qwen dispatch | `unsloth/models/qwen2.py`, `qwen3.py`, `qwen3_moe.py` |
| Mistral dispatch | `unsloth/models/mistral.py` |
| Cohere dispatch | `unsloth/models/cohere.py` |
| Granite dispatch | `unsloth/models/granite.py` |
| Falcon H1 dispatch | `unsloth/models/falcon_h1.py` |
| Shared utilities | `unsloth/models/_utils.py` (107K) |
| Fallback model | `unsloth/models/_utils.py` → `FastModel` |
