# Chapter 21: Pushing to Hugging Face Hub

> *"Share your model with the world in one line."*

---

## Introduction

The Hugging Face Hub is the standard distribution platform for language models. Unsloth integrates hub pushing into every save operation — you can upload LoRA adapters, merged models, and GGUF files with a single `push_to_hub=True` parameter. Behind the scenes, Unsloth handles authentication, repository creation, model card generation, tagging, and the special considerations needed for Kaggle and Colab environments.

### What You'll Learn

- How `push_to_hub` works in `unsloth_save_model()`
- Token management and authentication flow
- Model card generation and tagging
- Organization vs. personal repository handling
- Kaggle/Colab disk space management during upload

### Prerequisites

- The save methods from Chapter 19
- A Hugging Face account and API token
- Basic understanding of git-based model repositories

---

## 21.1 Enabling Hub Push

Push to hub is activated by a single parameter in any save call:

```python
# Push LoRA adapters
model.save_pretrained(
    "username/my-model-lora",
    push_to_hub = True,
    token = "hf_...",
)

# Push merged 16-bit model
model.save_pretrained_merged(
    "username/my-model",
    save_method = "merged_16bit",
    push_to_hub = True,
    token = "hf_...",
)

# Push GGUF
model.save_pretrained_gguf(
    "username/my-model-gguf",
    tokenizer,
    quantization_method = "q4_k_m",
    push_to_hub = True,
    token = "hf_...",
)
```

---

## 21.2 Authentication Flow

Unsloth retrieves the HF token through multiple fallback paths (handling different `huggingface_hub` versions):

```python
# save.py (lines 60-67)
try:
    from huggingface_hub import get_token
except:
    try:
        from huggingface_hub.utils import get_token
    except:
        from huggingface_hub.utils._token import get_token
```

Before any upload, Unsloth validates the token:

```python
from huggingface_hub import whoami
try:
    username = whoami(token=token)["name"]
except:
    raise RuntimeError("Unsloth: Please supply a token! "
                       "Go to https://huggingface.co/settings/tokens")
```

---

## 21.3 Repository Management

### Organization Support

When the save directory contains a slash (e.g., `"my-org/my-model"`), Unsloth detects the organization prefix and handles repo creation accordingly:

```python
if "/" in save_directory:
    username = save_directory[:save_directory.find("/")]
    new_save_directory = save_directory[save_directory.find("/") + 1:]
```

The `_determine_username()` helper resolves whether the prefix is a personal username or an organization name, adjusting the upload path appropriately.

### Repository Creation

`create_huggingface_repo()` handles repo creation using the HF API:

```python
api = HfApi()
api.create_repo(
    repo_id = save_directory,
    token = token,
    private = private,
    exist_ok = True,  # Don't fail if repo already exists
)
```

---

## 21.4 Model Card and Tagging

Every uploaded model gets automatic metadata:

### Tags

```python
tags = ["unsloth"]  # Always added
if user_tags:
    tags = list(user_tags) + ["unsloth"]
```

### Commit Messages

```python
commit_message = "Trained with Unsloth"
commit_description = "Upload model trained with Unsloth 2x faster"
```

If the user provides custom messages, "Unsloth" is appended if not already present.

### Model Card Generation

The `upload_to_huggingface()` function creates a model card with:
- Training framework: `"trl"`
- Model type: `"finetuned"`
- Dataset references (if provided via `datasets=` parameter)
- The `"unsloth"` tag for discoverability

---

## 21.5 Environment-Specific Handling

### Kaggle

Kaggle has limited disk space. When pushing to hub, Unsloth moves the save directory to `/tmp` to use Kaggle's larger temp storage:

```python
if IS_KAGGLE_ENVIRONMENT:
    new_save_directory = os.path.join(KAGGLE_TMP, new_save_directory)
    logger.warning(f"To save memory, we shall move to {new_save_directory}")
```

### Colab

On Colab, the HF model cache is deleted before saving to free disk space. This is safe because the model is already in GPU/CPU memory:

```python
if IS_COLAB_ENVIRONMENT:
    _free_cached_model(internal_model)  # Frees 4-16GB of disk
```

---

## 21.6 Complete Upload Example

```python
# Full workflow: train → save → push
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct", load_in_4bit=True
)
model = FastLanguageModel.get_peft_model(model, r=16, ...)

# ... training happens here ...

# Push LoRA adapter (fast, ~100MB)
model.save_pretrained(
    "myuser/llama-3.2-1b-finetuned-lora",
    push_to_hub = True,
    token = "hf_...",
    private = False,
    tags = ["llama", "instruction-tuning"],
    datasets = ["myuser/my-dataset"],
)

# Push merged model + GGUF (slower, ~4GB + ~4GB)
model.save_pretrained_merged(
    "myuser/llama-3.2-1b-finetuned",
    save_method = "merged_16bit",
    push_to_hub = True,
    token = "hf_...",
)
model.save_pretrained_gguf(
    "myuser/llama-3.2-1b-finetuned-gguf",
    tokenizer,
    quantization_method = "q4_k_m",
    push_to_hub = True,
    token = "hf_...",
)
```

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Push to hub integration | `unsloth/save.py` → `unsloth_save_model()` |
| Repository creation | `unsloth/save.py` → `create_huggingface_repo()` |
| Model card upload | `unsloth/save.py` → `upload_to_huggingface()` |
| Token management | `huggingface_hub` → `get_token()`, `whoami()` |
