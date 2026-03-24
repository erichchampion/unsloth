# Chapter 20: GGUF Export and Quantization

---

## Introduction

GGUF is the format used by llama.cpp for local inference. Unsloth integrates with llama.cpp to convert and quantize models into GGUF format with various quantization levels.

### What You'll Learn

- The GGUF quantization zoo: q4_k_m, q5_k_m, q8_0, etc.
- How `save_to_gguf()` orchestrates the conversion
- llama.cpp installation and building
- SentencePiece tokenizer fixes for GGUF

---

## Notes & Key Points

### 20.1 GGUF Quantization Methods

```python
ALLOWED_QUANTS = {
    "not_quantized": "Fast conversion. Slow inference, big files.",
    "fast_quantized": "Fast conversion. OK inference, OK file size.",
    "quantized":      "Slow conversion. Fast inference, small files.",
    "q4_k_m":         "Recommended. Q6_K for half of attn + FFN, else Q4_K",
    "q5_k_m":         "Recommended. Q6_K for half of attn + FFN, else Q5_K",
    "q8_0":           "Fast conversion. High resource use.",
    "bf16":           "Fastest + 100% accuracy. Slow inference.",
    # ... 15+ methods total
}
```

### 20.2 Conversion Pipeline

1. Save model as merged 16-bit safetensors
2. Install/find llama.cpp (uses `install_llama_cpp()` or `use_local_gguf()`)
3. Convert HF format → GGUF via `convert_to_gguf()` / `_download_convert_hf_to_gguf()`
4. Quantize GGUF via `quantize_gguf()` using `llama-quantize`

### 20.3 llama.cpp Integration

- Builds specific targets: `llama-quantize`, `llama-cli`, `llama-server`
- Checks for cURL support: `-DLLAMA_CURL=ON/OFF`
- Default build directory: `llama.cpp/`
- Windows compatibility via `IS_WINDOWS` flag

### 20.4 Ollama Integration

- Converted models get Ollama-compatible Modelfile templates
- `OLLAMA_TEMPLATES` and `MODEL_TO_OLLAMA_TEMPLATE_MAPPER` map models to templates
- Enables `ollama run <model>` workflow

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| GGUF export | `unsloth/save.py` → `save_to_gguf()` |
| Quant method list | `unsloth/save.py` → `ALLOWED_QUANTS` |
| llama.cpp tools | `unsloth_zoo/llama_cpp.py` (external) |
| Ollama templates | `unsloth/ollama_template_mappers.py` |
| SP fixes | `unsloth/tokenizer_utils.py` → `fix_sentencepiece_gguf()` |
