# Chapter 17: Embedding and Sentence-Transformer Training

---

## Introduction

Unsloth supports fine-tuning embedding models (e.g., EmbeddingGemma) and sentence transformers with 1.8x–3.3x speed improvements.

### What You'll Learn

- How embedding models differ from generative models
- The `sentence_transformer.py` implementation
- Training with the `sentence-transformers` library integration
- Embedding-specific learning rate in `UnslothTrainer`

---

## Notes & Key Points

### 17.1 Embedding Fine-Tuning

- `sentence_transformer.py` (85K) — Major module for embedding model support
- Supports ~300M–2B parameter embedding models
- Uses separate learning rates for embedding layers via `embedding_learning_rate`

### 17.2 Training Flow

- Load embedding model with `FastLanguageModel.from_pretrained()`
- Use `UnslothTrainer` with `embedding_learning_rate` set
- The optimizer creates separate parameter groups for embeddings vs. other weights

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Sentence transformer | `unsloth/models/sentence_transformer.py` |
| Embedding optimizer | `unsloth/trainer.py` → `_create_unsloth_optimizer()` |
