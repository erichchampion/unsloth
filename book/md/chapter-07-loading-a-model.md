# Chapter 7: Loading a Model — FastLanguageModel.from_pretrained

> *"One function to load them all."*

---

## Introduction

`FastLanguageModel.from_pretrained()` is the single most important function in Unsloth. Every user interaction — training, inference, export — begins with this call. Behind its simple API lies a sophisticated pipeline that resolves model names through the registry, detects the architecture from the model's configuration, selects the right quantization strategy, dispatches to an architecture-specific `Fast*Model` class, applies patches, loads LoRA adapters, and optionally initializes vLLM for fast inference.

This chapter traces the complete execution path from the user's call to a ready-to-use model and tokenizer.

### What You'll Learn

- The full parameter surface of `from_pretrained()`
- The four-stage pipeline: resolve → configure → load → patch
- How model names are resolved through the registry and HF mapper
- Architecture detection via `AutoConfig` and the dispatch map
- Post-loading steps: LoRA adapters, gradient checkpointing, RoPE fixes

### Prerequisites

- The repository layout from Chapter 2
- The dependency stack from Chapter 3
- Understanding of LoRA and quantization concepts (or read Chapter 12 first)

---

## 7.1 The from_pretrained Signature

The function accepts a large parameter surface to cover the full range of use cases:

```python
FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Llama-3.2-1B-Instruct",  # HF ID or local path
    max_seq_length = 2048,         # Context window size
    load_in_4bit   = True,         # QLoRA 4-bit quantization (default)
    load_in_8bit   = False,        # 8-bit quantization
    load_in_16bit  = False,        # No quantization, 16-bit weights
    load_in_fp8    = False,        # FP8 quantization
    full_finetuning = False,       # Disable all quantization
    fast_inference  = False,       # Enable vLLM backend
    dtype           = None,        # Override dtype (bf16/fp16/fp32)
    token           = None,        # Hugging Face auth token
    trust_remote_code = False,     # Allow custom model code
    use_gradient_checkpointing = "unsloth",  # Gradient checkpointing strategy
    gpu_memory_utilization = 0.5,  # vLLM GPU memory fraction
    float8_kv_cache = False,       # FP8 KV cache for vLLM
)
# Returns: (model, tokenizer)
```

The most common call is just:

```python
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
```

This loads the model in 4-bit QLoRA mode with a 2048-token context.

---

## 7.2 Stage 1: Model Name Resolution

The first thing `from_pretrained()` does is resolve the model name to a concrete Hugging Face repository path. This involves several transformations:

```
User input: "unsloth/Llama-3.2-1B-Instruct"
    │
    ├─ 1. get_model_name()
    │      Checks the registry mapper for pre-quantized variants
    │      If load_in_4bit: may append "-bnb-4bit" suffix
    │      If load_in_fp8: tries _offline_quantize_to_fp8()
    │
    ├─ 2. Suffix analysis
    │      Strips "-bnb-4bit" → sets load_in_4bit = True
    │      Strips "-bf16" → sets load_in_16bit = True  
    │      Detects GGUFs → uses GGUF loading path
    │
    ├─ 3. AMD compatibility check
    │      If ALLOW_PREQUANTIZED_MODELS is False:
    │        strips "-bnb-4bit" suffix, downloads full weights
    │
    └─ 4. ModelScope fallback
           For Chinese users: checks ModelScope mirror
```

The mapper (`models/mapper.py`, 49K) contains a large dictionary that maps model names to their quantized and full-precision variants, ensuring the user always gets the best available version for their configuration.

---

## 7.3 Stage 2: Architecture Detection

Once the name is resolved, Unsloth must determine the model's architecture to select the correct `Fast*Model` class:

```python
# loader.py — architecture detection
model_config = AutoConfig.from_pretrained(model_name, token=token)
model_type = model_config.model_type  # e.g., "llama", "gemma2", "qwen3"
```

For LoRA adapters, the process is more complex. The function tries `PeftConfig.from_pretrained()` to read the adapter's `base_model_name_or_path`, then loads the base model's config to determine the architecture. This handles the case where the user passes a LoRA checkpoint path instead of a base model.

---

## 7.4 Stage 3: The Dispatch Map

With the architecture known, `from_pretrained()` dispatches to the correct `Fast*Model`:

```python
# loader.py — simplified dispatch logic
DISPATCH_MAP = {
    "llama":       FastLlamaModel,
    "mistral":     FastMistralModel,
    "gemma":       FastGemmaModel,
    "gemma2":      FastGemma2Model,
    "gemma3":      FastGemma3Model,
    "qwen2":       FastQwen2Model,
    "qwen3":       FastQwen3Model,
    "qwen3_moe":   FastQwen3MoeModel,
    "cohere":      FastCohereModel,
    "cohere2":     FastCohere2Model,
    "granite":     FastGraniteModel,
    "falcon_h1":   FastFalconH1Model,
    # ... more architectures
}

FastModel = DISPATCH_MAP.get(model_type)
if FastModel is None:
    # Fallback: use generic FastModel with torch.compile
    FastModel = _FastModel
```

Each version-gated architecture checks the `SUPPORTS_*` constants from `loader.py` before enabling itself. If the installed transformers is too old for a given architecture, the user gets a clear error message:

```python
if model_type == "qwen3" and not SUPPORTS_QWEN3:
    raise ImportError(
        f"Unsloth: Qwen3 requires transformers >= 4.50.3, "
        f"but you have {transformers_version}"
    )
```

---

## 7.5 Stage 4: Loading and Patching

The selected `Fast*Model.from_pretrained()` then:

1. **Loads the model** — calls `AutoModelForCausalLM.from_pretrained()` with the appropriate quantization config (`BitsAndBytesConfig` for 4-bit, custom config for FP8)
2. **Loads the tokenizer** — calls `load_correct_tokenizer()` which handles slow→fast conversion, special token fixes, and chat template validation
3. **Applies patches** — replaces standard PyTorch operations with optimized Triton kernels:
   - Attention layers → FlexAttention or Flash Attention
   - MLP layers → fused SwiGLU kernel
   - LayerNorm → fused RMSNorm kernel
   - RoPE → fused rotary embedding kernel
   - Cross-entropy → chunked cross-entropy kernel
4. **Configures gradient checkpointing** — Unsloth's custom strategy saves more memory than the standard HF implementation

---

## 7.6 Post-Loading Steps

### LoRA Adapter Loading

If a LoRA adapter is detected (via `adapter_config.json` in the model path), Unsloth applies it:

```python
model = PeftModel.from_pretrained(model, model_name, token=token)
```

### RoPE Fix for Transformers v5

Transformers v5.0+ changed how models are initialized on the "meta" device, which corrupted RoPE positional embedding buffers. Unsloth detects this and recomputes `inv_freq`:

```python
if _NEEDS_ROPE_FIX:
    _fix_rope_inv_freq(model)
```

### Fast Inference Setup

If `fast_inference=True`, the function calls `fast_inference_setup()` to prepare the model config for vLLM loading (see Chapter 10).

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| FastLanguageModel.from_pretrained | `unsloth/models/loader.py` |
| Model name mapper | `unsloth/models/mapper.py` |
| Loader utilities (FP8, device map) | `unsloth/models/loader_utils.py` |
| Tokenizer loading + fixing | `unsloth/tokenizer_utils.py` |
| Architecture-specific patches | `unsloth/models/llama.py`, `gemma.py`, etc. |
| Version gate constants | `unsloth/models/loader.py` (lines 70-83) |
