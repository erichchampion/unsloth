# Chapter 17: Embedding and Sentence-Transformer Training

> *"Turning language models into semantic search engines."*

---

## Introduction

Embedding models transform text into dense vector representations that capture semantic meaning. These vectors power search, retrieval-augmented generation (RAG), clustering, and classification systems. Unsloth supports fine-tuning embedding models — including Google's EmbeddingGemma and sentence-transformers — with 1.8×–3.3× speed improvements over standard training. The implementation lives primarily in the large `sentence_transformer.py` module (85K) and integrates with both the `sentence-transformers` library and Unsloth's own training infrastructure.

### What You'll Learn

- How embedding models differ from generative models
- The `sentence_transformer.py` implementation
- Separate embedding learning rates with `UnslothTrainer`
- Contrastive and triplet loss training
- Integration with the `sentence-transformers` library

### Prerequisites

- The LoRA/QLoRA concepts from Chapter 12
- The trainer infrastructure from Chapter 14
- Basic understanding of vector representations and similarity search

---

## 17.1 Embedding vs. Generative Models

Generative language models produce text token-by-token. Embedding models produce a single fixed-length vector for an entire input text:

```
Generative:
  Input: "What is machine learning?"
  Output: "Machine learning is a subset of AI..."  (text)

Embedding:
  Input: "What is machine learning?"
  Output: [0.023, -0.145, 0.891, ...]  (768-dim vector)
```

The key training difference: embedding models use **contrastive losses** (e.g., InfoNCE, triplet loss) instead of next-token prediction loss. The model learns to place semantically similar texts close together in vector space and dissimilar texts far apart.

---

## 17.2 The sentence_transformer.py Module

At 85K, `sentence_transformer.py` is one of the largest files in the Unsloth codebase. It provides:

| Component | Purpose |
|-----------|---------|
| Model patching | Apply Unsloth's Triton kernels to embedding models |
| Loss functions | Contrastive, triplet, cosine similarity losses |
| Pooling strategies | Mean pooling, CLS token, last token |
| Evaluation metrics | Cosine similarity, Spearman correlation |
| Data formatting | Pair and triplet dataset handling |

### Supported Models

Embedding fine-tuning works with any model that Unsloth supports for generative tasks:

| Model | Embedding Dimensions | Use Case |
|-------|---------------------|----------|
| Gemma 2B | 2048 | General embedded search |
| Llama 3.2 1B | 2048 | Lightweight RAG |
| Qwen 2.5 0.5B | 896 | Ultra-compact embeddings |
| Mistral 7B | 4096 | High-quality retrieval |

---

## 17.3 Separate Embedding Learning Rates

When fine-tuning embedding models with LoRA, the input embeddings and output head often need a lower learning rate than the LoRA adapters. This is handled by `UnslothTrainer`:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens",  # Include input embeddings
        "lm_head",       # Include output head
    ],
)

trainer = UnslothTrainer(
    model = model,
    args = UnslothTrainingArguments(
        learning_rate = 2e-4,            # LoRA adapter LR
        embedding_learning_rate = 5e-5,  # Embedding layer LR (lower)
        output_dir = "./embedding_output",
    ),
    train_dataset = pairs_dataset,
)
```

The optimizer creates two parameter groups (see Chapter 14, Section 14.2):
- LoRA adapters: `lr=2e-4`
- Embedding layers: `lr=5e-5`

This prevents the embedding layers from being updated too aggressively, which can destabilize training.

---

## 17.4 Training with sentence-transformers

Unsloth integrates with the `sentence-transformers` library for standardized embedding training:

```python
from sentence_transformers import SentenceTransformer, losses

# Load with Unsloth optimizations
model = SentenceTransformer("unsloth/Gemma-2-2B")

# Define contrastive loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
)
```

### Common Loss Functions

| Loss | Training Data | Description |
|------|--------------|-------------|
| `MultipleNegativesRankingLoss` | (anchor, positive) pairs | In-batch negatives contrastive |
| `TripletLoss` | (anchor, positive, negative) triplets | Explicit negative mining |
| `CosineSimilarityLoss` | (text_a, text_b, score) | Regression on similarity |
| `ContrastiveLoss` | (text_a, text_b, label) | Binary similar/dissimilar |

## 17.5 Evaluating Embedding Models

After fine-tuning, embedding quality is measured with retrieval and similarity benchmarks:

| Metric | Measures | Range |
|--------|----------|-------|
| **NDCG@10** | Retrieval ranking quality | 0–1 (higher is better) |
| **MRR** | Mean reciprocal rank of first relevant result | 0–1 |
| **Cosine Similarity** | Agreement between predicted and gold similarity | -1 to 1 |
| **Spearman Correlation** | Rank correlation of similarity scores | -1 to 1 |

A typical evaluation workflow:

```python
from sentence_transformers import evaluation
evaluator = evaluation.InformationRetrievalEvaluator(
    queries=eval_queries,
    corpus=eval_corpus,
    relevant_docs=eval_relevant,
)
results = evaluator(model)
print(f"NDCG@10: {results['ndcg@10']:.4f}")
```

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Sentence transformer patches | `unsloth/models/sentence_transformer.py` |
| Embedding optimizer (separate LR) | `unsloth/trainer.py` → `_create_unsloth_optimizer()` |
| UnslothTrainingArguments | `unsloth/trainer.py` |
