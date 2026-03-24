# Chapter 37: Data Recipes — Visual Dataset Creation

> *"From raw documents to training data in minutes."*

---

## Introduction

Creating high-quality training datasets is often the hardest part of fine-tuning. Data Recipes is Studio's visual node-based workflow editor that automates the transformation of raw documents (PDFs, CSVs, DOCX files, web pages) into structured instruction-following datasets. It integrates with the `dataprep` module (Chapter 18) to generate synthetic question-answer pairs, conversation threads, and classification examples using a running LLM.

### What You'll Learn

- The visual node editor interface and workflow concepts
- Document ingestion and processing pipeline
- LLM-powered synthetic data generation
- The OXC JavaScript validator for data transforms
- Dataset export and integration with training

### Prerequisites

- Data preparation from Chapter 18
- SyntheticDataKit from Chapter 18
- The studio architecture from Chapter 34

---

## 37.1 The Node Editor

Data Recipes uses a visual node-graph editor where each node represents a processing step:

```
[Document Upload] → [Chunking] → [Prompt Template] → [LLM Generation] → [Validation] → [Export]
     PDF, CSV          Split by       Define Q&A         Generate with        Validate        HF datasets
     DOCX, JSON       paragraphs     format template    loaded model         JSON schema       format
```

### Node Types

| Node | Purpose | Input | Output |
|------|---------|-------|--------|
| **Source** | Upload or connect to documents | Files / URLs | Raw text |
| **Chunker** | Split documents into segments | Raw text | Text chunks |
| **Prompt** | Define generation template | Template + chunks | Prompts |
| **Generator** | Run LLM on prompts | Prompts | Generated text |
| **Validator** | Check output format/quality | Generated text | Valid examples |
| **Filter** | Remove low-quality examples | Examples | Filtered examples |
| **Export** | Save as HF dataset | Examples | Dataset file |

---

## 37.2 Document Ingestion

Data Recipes supports multiple input formats:

| Format | Processing Method | Chunking Strategy |
|--------|------------------|-------------------|
| PDF | Text extraction (PyPDF2/pdfplumber) | By page or paragraph |
| DOCX | python-docx text extraction | By section heading |
| CSV | Pandas DataFrame loading | By row or row group |
| JSON/JSONL | Direct parsing | By record |
| Plain text | Raw loading | By paragraph or sentence |
| Web URL | HTML scraping and cleaning | By section |

### Chunking Strategies

The chunker node offers multiple strategies:

```python
# Fixed-size chunking (most common)
chunks = split_by_tokens(text, chunk_size=512, overlap=64)

# Semantic chunking (smarter, slower)
chunks = split_by_semantic_boundaries(text)  # Paragraph/section breaks

# Sliding window
chunks = sliding_window(text, window=1024, stride=256)
```

---

## 37.3 LLM-Powered Generation

The Generator node uses the loaded model (or a separate vLLM instance via SyntheticDataKit) to create training examples:

```python
# Example prompt template for Q&A generation:
template = """Based on the following context, generate a question and answer pair.

Context: {chunk}

Generate exactly one Q&A pair in this JSON format:
{"question": "...", "answer": "..."}"""

# The node iterates over all chunks:
for chunk in chunks:
    prompt = template.format(chunk=chunk)
    response = llm.generate(prompt)
    examples.append(parse_json(response))
```

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Model | Currently loaded | Which model generates data |
| Examples per chunk | 3 | Number of examples per document chunk |
| Temperature | 0.7 | Creativity of generation |
| Max retries | 2 | Retries for malformed output |

---

## 37.4 OXC JavaScript Validator

Data Recipes includes an OXC-based JavaScript validator for custom data transformation rules:

```
studio/backend/core/data_recipe/oxc-validator/
```

This validator allows users to write JavaScript expressions that filter, transform, or validate generated examples:

```javascript
// Example validation rule:
// Ensure answer is at least 50 characters
example.answer.length >= 50

// Example transform:
// Normalize whitespace in questions
example.question = example.question.replace(/\s+/g, ' ').trim()
```

OXC (Oxidation Compiler) provides fast JavaScript parsing and validation without a full Node.js runtime.

---

## 37.5 Dataset Export

Validated examples are exported in Hugging Face `datasets`-compatible format:

```python
# Output format (JSONL):
{"messages": [
    {"role": "user", "content": "What is photosynthesis?"},
    {"role": "assistant", "content": "Photosynthesis is the process by which..."}
]}
```

The exported dataset can be:
1. **Used directly** — loaded into the training pipeline without leaving Studio
2. **Saved locally** — downloaded as a JSONL file
3. **Pushed to Hub** — uploaded to Hugging Face for sharing

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Data recipe core | `studio/backend/core/data_recipe/` |
| OXC validator | `studio/backend/core/data_recipe/oxc-validator/` |
| SyntheticDataKit | `unsloth/dataprep/synthetic.py` |
| Dataset routes | `studio/backend/routes/datasets.py` |
