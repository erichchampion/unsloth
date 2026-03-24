# Chapter 21: Pushing to Hugging Face Hub

---

## Introduction

Unsloth integrates with the Hugging Face Hub for model uploading, repository creation, and model card generation.

### What You'll Learn

- How `push_to_hub` works in `unsloth_save_model()`
- HF token management and authentication
- Model tagging and card generation
- Organization/personal repo handling

---

## Notes & Key Points

### 21.1 Push to Hub Flow

- Authenticate via `huggingface_hub.whoami(token=...)`
- `upload_to_huggingface()` creates/updates the repo with model cards
- Adds `"unsloth"` tag to model and tokenizer
- Commit messages default to `"Trained with Unsloth"`

### 21.2 Token Management

```python
from huggingface_hub import get_token
# Falls back through multiple import paths for older HF Hub versions
```

### 21.3 Organization Support

- Detects `username/model-name` format
- Creates repos under personal or organization accounts
- Handles Kaggle/Colab disk space constraints by cleaning cached models

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Push to hub | `unsloth/save.py` → `unsloth_save_model()` |
| Repo creation | `unsloth/save.py` → `create_huggingface_repo()` |
| HF Hub utils | `unsloth/utils/hf_hub.py` |
