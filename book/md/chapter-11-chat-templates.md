# Chapter 11: Chat Templates and Tokenizer Utilities

> *"The difference between a working model and a broken one is often just a missing newline in the chat template."*

---

## Introduction

Every instruction-tuned model has its own way of formatting conversations — the delimiters between user and assistant turns, where special tokens go, whether the BOS token is included, and how system prompts are handled. Get the formatting wrong, and your model produces gibberish. Get it right, and multi-turn conversations flow naturally.

Unsloth's `chat_templates.py` (120K, 2,749 lines) and `tokenizer_utils.py` (43K, 1,127 lines) together form a comprehensive system for handling these details. The chat template library covers over 30 template families. The tokenizer utilities fix bugs in upstream tokenizers, handle slow→fast conversion, and ensure special tokens are correctly mapped.

### What You'll Learn

- The `CHAT_TEMPLATES` dictionary and how templates are structured
- Major template families: ChatML, Llama 3, Gemma, Qwen, Phi
- How Ollama template mapping works for GGUF export
- Tokenizer loading, fixing, and validation
- The `train_on_responses_only` feature for response-only training

### Prerequisites

- Basic understanding of tokenization and special tokens
- Familiarity with Jinja2 template syntax
- The model loading pipeline from Chapter 7

---

## 11.1 The Chat Template Library

Chat templates are stored in the global `CHAT_TEMPLATES` dictionary. Each entry is a tuple of four elements:

```python
CHAT_TEMPLATES["llama-3"] = (
    llama3_template,         # Jinja2 template string
    llama3_template_eos_token,  # EOS token: string or tuple
    False,                   # Whether to add special tokens
    llama3_ollama,           # Ollama Go-template equivalent
)
```

### Template Structure

Templates use Jinja2 syntax to format conversations. Here's a simplified example of the ChatML template:

```python
chatml_template = \
    "{% for message in messages %}" \
        "{% if message['role'] == 'user' %}" \
            "{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n'}}" \
        "{% elif message['role'] == 'assistant' %}" \
            "{{'<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n'}}" \
        "{% else %}" \
            "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}" \
        "{% endif %}" \
    "{% endfor %}" \
    "{% if add_generation_prompt %}" \
        "{{ '<|im_start|>assistant\n' }}" \
    "{% endif %}"
```

### Supported Template Families

| Template | Models | Key Feature |
|----------|--------|-------------|
| `chatml` | Qwen, Yi, many others | `<\|im_start\|>` / `<\|im_end\|>` delimiters |
| `llama-3` | Llama 3 | `<\|start_header_id\|>` / `<\|eot_id\|>` |
| `llama-3.1` | Llama 3.1, 3.2, 3.3 | Tool calling, `ipython` role support |
| `gemma` | Gemma 1, 2 | `<start_of_turn>` / `<end_of_turn>` |
| `gemma-3` | Gemma 3 | Multimodal (image) support, system prompt in first user message |
| `qwen-2.5` | Qwen 2.5 | Tool calling with XML tags |
| `qwen-3` | Qwen 3 | `<think>` / `</think>` reasoning blocks |
| `mistral` | Mistral v1 | `[INST]` / `[/INST]` (no system prompt) |
| `phi-3` | Phi-3, Phi-3.5 | `<\|user\|>` / `<\|end\|>` |
| `phi-4` | Phi-4 | `<\|im_sep\|>` separator |
| `llama` | Llama 2 | `<<SYS>>` system message format |
| `alpaca` | Alpaca family | `### Instruction:` / `### Response:` |
| `vicuna` | Vicuna | `USER:` / `ASSISTANT:` format |
| `zephyr` | Zephyr | `<\|user\|>` / `<\|assistant\|>` |
| `unsloth` | Unsloth default | `>>> User:` / `>>> Assistant:` |

Many templates also have aliases — for example, `"llama3"` and `"llama-3"` point to the same template.

---

## 11.2 Default System Messages

Each template family can optionally define a default system message:

```python
DEFAULT_SYSTEM_MESSAGE["vicuna"] = \
    "A chat between a curious user and an artificial intelligence assistant. " \
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
DEFAULT_SYSTEM_MESSAGE["llama-3"] = None  # No default system message
DEFAULT_SYSTEM_MESSAGE["llama-3.1"] = ""  # Empty string (different from None)
```

---

## 11.3 Ollama Template Mapping

For GGUF export, Unsloth needs to convert Jinja2 templates to Ollama's Go template format. The `ollama_template_mappers.py` file (83K, ~2,000 lines) contains a massive `MODEL_TO_OLLAMA_TEMPLATE_MAPPER` dictionary that maps model IDs to their Ollama-compatible template strings.

Each chat template entry includes an Ollama equivalent as its fourth element:

```python
# chat_templates.py
llama3_ollama = _ollama_template("llama-3")
CHAT_TEMPLATES["llama-3"] = (llama3_template, ..., llama3_ollama)
```

The templates are stored in `OLLAMA_TEMPLATES` in the mapper file and use Go's `{{ }}` template syntax instead of Jinja2's `{% %}`:

```go
{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}
{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}
<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>
```

---

## 11.4 Tokenizer Loading and Fixing

The `tokenizer_utils.py` module handles a surprising number of edge cases in tokenizer loading. The main entry point is `load_correct_tokenizer()`:

```python
def load_correct_tokenizer(tokenizer_name, ...):
    # 1. Try loading slow tokenizer (SentencePiece)
    # 2. Load fast tokenizer (Rust-based)
    # 3. Verify they produce the same tokenization
    # 4. If not, convert slow → fast with fix_tokenizer()
    # 5. Fix chat template (add_generation_prompt check)
    return tokenizer
```

### Key Fixes

| Problem | Fix |
|---------|-----|
| Slow/fast tokenizer mismatch | `assert_same_tokenization()` detects differences; `convert_to_fast_tokenizer()` converts |
| Missing `add_generation_prompt` | `fix_chat_template()` detects and patches Jinja2 template |
| Wrong special token IDs | `try_fix_tokenizer()` edits the tokenizer's JSON representation directly |
| SentencePiece prepend bug | Removes incorrect `▁` (space) prepending for non-Llama models |
| GGUF SentencePiece extension | `fix_sentencepiece_gguf()` extends vocab with `added_tokens.json` |
| BOS/EOS token sync | Copies `add_bos_token` / `add_eos_token` flags from slow to fast tokenizer |
| Out-of-bounds token IDs | `check_tokenizer()` detects tokens with IDs beyond the embedding matrix size |

### Environment Detection

The tokenizer utilities also detect the runtime environment for caching:

```python
IS_COLAB_ENVIRONMENT  = "\nCOLAB_" in keynames
IS_KAGGLE_ENVIRONMENT = "\nKAGGLE_" in keynames
```

On Kaggle, tokenizers are cached in `/tmp` to use the 80GB temp storage allocation.

---

## 11.5 The get_chat_template Function

`get_chat_template()` is the user-facing API for applying templates:

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",           # Template name
    mapping = {"role": "from", "content": "value"},  # Field mapping
    system_message = "You are a helpful AI.",
)
```

This function:
1. Looks up the template in `CHAT_TEMPLATES`
2. Sets `tokenizer.chat_template` to the Jinja2 string
3. Handles EOS token configuration (string replacement or new token addition)
4. Applies system message substitution
5. Patches the tokenizer's saving functions to preserve the template

---

## 11.6 Training on Responses Only

The `train_on_responses_only` function (delegated to `unsloth_zoo`) creates training labels that mask out the user/system turn tokens, so the model only learns from assistant responses:

```python
from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

This sets the loss labels to -100 (ignored) for all tokens in user turns, ensuring the model's training signal comes exclusively from learning to generate assistant responses.

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Chat template library | `unsloth/chat_templates.py` |
| Ollama template mappers | `unsloth/ollama_template_mappers.py` |
| Tokenizer loading + fixing | `unsloth/tokenizer_utils.py` |
| Response-only training | `unsloth_zoo/dataset_utils.py` → `train_on_responses_only` |
| Data format standardization | `unsloth_zoo/dataset_utils.py` → `standardize_data_formats` |
