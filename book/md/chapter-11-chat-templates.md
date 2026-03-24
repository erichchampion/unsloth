# Chapter 11: Chat Templates and Tokenizer Utilities

---

## Introduction

Unsloth provides an extensive library of chat templates (~120K lines) and tokenizer utilities to ensure models generate correctly formatted output. This chapter explains the template system and tokenizer fixes.

### What You'll Learn

- The `chat_templates.py` module and its scope
- How Ollama template mappers work
- Tokenizer fixing and SentencePiece compatibility
- How chat formatting affects training and inference

---

## Notes & Key Points

### 11.1 Chat Template Library

- `chat_templates.py` (120K) — Defines chat template strings for dozens of model families
- Templates follow the Jinja2 format used by HF `tokenizer.apply_chat_template()`
- Covers system prompts, tool calling, multi-turn formatting
- Includes templates for: ChatML, Alpaca, Vicuna, Llama, Mistral, Gemma, Qwen, and more

### 11.2 Ollama Template Mapping

- `ollama_template_mappers.py` (83K) — Maps HF model names → Ollama template formats
- `MODEL_TO_OLLAMA_TEMPLATE_MAPPER` — dict mapping model IDs to Ollama Modelfile template strings
- Used when exporting models for Ollama serving

### 11.3 Tokenizer Utilities

- `tokenizer_utils.py` (43K) — Fixes for tokenizer edge cases:
  - Padding token configuration
  - Special token mapping
  - SentencePiece GGUF compatibility (`fix_sentencepiece_gguf`)
  - BOS/EOS token handling

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Chat templates | `unsloth/chat_templates.py` |
| Ollama mappers | `unsloth/ollama_template_mappers.py` |
| Tokenizer fixes | `unsloth/tokenizer_utils.py` |
