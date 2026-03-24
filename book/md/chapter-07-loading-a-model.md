# Chapter 7: Loading a Model — FastLanguageModel.from_pretrained

---

## Introduction

`FastLanguageModel.from_pretrained()` is the primary entry point for loading any model in Unsloth. This single method handles model name resolution, quantization, architecture dispatch, tokenizer fixing, and LoRA adapter loading.

### What You'll Learn

- The full parameter surface of `from_pretrained()`
- How model names are resolved through the registry and HF mapper
- The architecture detection → dispatch flow
- LoRA adapter detection and merging
- Integration with vLLM for fast inference

---

## Notes & Key Points

### 7.1 The from_pretrained Signature

Key parameters:
- `model_name` — HF model ID or local path (default: `"unsloth/Llama-3.2-1B-Instruct"`)
- `max_seq_length` — context window (default: 2048)
- `load_in_4bit` — QLoRA mode (default: `True`)
- `load_in_8bit`, `load_in_16bit`, `load_in_fp8` — alternative precisions
- `full_finetuning` — disable quantization for full fine-tuning
- `fast_inference` — enable vLLM backend
- `trust_remote_code` — allow custom model code
- `use_gradient_checkpointing` — `"unsloth"` for optimized checkpointing

### 7.2 Model Name Resolution

1. `get_model_name()` checks the registry mapper for pre-quantized variants
2. If load_in_fp8, attempts `_offline_quantize_to_fp8()`
3. Strips `-bnb-4bit` or `-bf16` suffixes to adjust load flags
4. ModelScope fallback for Chinese mirror

### 7.3 Architecture Detection

- Uses `AutoConfig.from_pretrained()` to get `model_type`
- Also checks `PeftConfig.from_pretrained()` for LoRA adapters
- If both exist (and old transformers), raises an error
- `get_transformers_model_type()` returns the architecture string(s)

### 7.4 Dispatch Map

```python
dispatch_map = {
    "llama":   FastLlamaModel,
    "mistral": FastMistralModel,
    "gemma":   FastGemmaModel,
    "gemma2":  FastGemma2Model,
    "qwen2":   FastQwen2Model,
    "qwen3":   FastQwen3Model,
    # Fallback → FastModel (unoptimized but functional)
}
```

### 7.5 Post-Loading Steps

- LoRA adapter applied via `PeftModel.from_pretrained()` if detected
- Gradient checkpointing configured
- Quantization config attached to `model.config`
- `_fix_rope_inv_freq()` for transformers v5 compatibility
- Tiled MLP patching if enabled

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| FastLanguageModel | `unsloth/models/loader.py` |
| Model name mapper | `unsloth/models/mapper.py` |
| Loader utilities | `unsloth/models/loader_utils.py` |
| AutoConfig | `transformers` (external) |
