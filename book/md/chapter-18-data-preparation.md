# Chapter 18: Data Preparation — Raw Text, Synthetic Data, and Data Recipes

> *"The quality of your model is bounded by the quality of your data."*

---

## Introduction

No matter how fast your trainer or how clever your LoRA configuration, the final model quality depends on the training data. Unsloth provides three data preparation pathways: loading raw text for continued pretraining, generating synthetic instruction-response pairs using an LLM, and visually constructing data workflows through Studio's Data Recipe editor. This chapter covers all three, with particular focus on the `SyntheticDataKit` — a self-contained pipeline that spins up a vLLM server, chunks documents, and generates Q&A pairs from any source material.

### What You'll Learn

- `RawTextDataLoader` and `TextPreprocessor` for plain text data
- The `SyntheticDataKit` pipeline: vLLM server, chunking, Q&A generation
- Data Recipe: Studio's visual workflow editor
- Supported input formats: PDF, HTML, YouTube, DOCX, TXT

### Prerequisites

- The vLLM concepts from Chapter 10
- Basic understanding of instruction-response dataset formats
- The trainer infrastructure from Chapter 14

---

## 18.1 Raw Text Data

For continued pretraining (teaching the model new domain knowledge without instruction formatting), Unsloth provides `RawTextDataLoader` in `dataprep/raw_text.py` (13K):

```python
from unsloth.dataprep.raw_text import RawTextDataLoader, TextPreprocessor

# Load and preprocess raw text
loader = RawTextDataLoader(
    file_path = "medical_textbook.txt",
    chunk_size = 2048,
    overlap = 64,
)
dataset = loader.load()
```

### TextPreprocessor

`TextPreprocessor` handles cleaning and normalization:
- Unicode normalization (NFC)
- Whitespace collapsing
- Sentence boundary detection
- Paragraph splitting with configurable overlap

---

## 18.2 Synthetic Data Generation

The `SyntheticDataKit` class (`dataprep/synthetic.py`, 474 lines) automates the creation of instruction-response training pairs from source documents. It works by spinning up a dedicated vLLM server, chunking input documents, and using the loaded model to generate Q&A pairs for each chunk.

### Architecture

```
SyntheticDataKit
    ├─ __init__() → starts vLLM server as subprocess
    │     ├─ Load model config and tokenizer
    │     ├─ Configure vLLM engine args
    │     ├─ Launch `vllm serve` subprocess
    │     ├─ PipeCapture (stdout/stderr monitoring)
    │     └─ Wait for readiness signal
    │
    ├─ chunk_data(filename) → split document into chunks
    │     ├─ Read file, tokenize
    │     ├─ Calculate optimal chunk boundaries
    │     └─ Write numbered chunk files
    │
    ├─ prepare_qa_generation(config) → configure generation
    │     ├─ Create output folder structure
    │     └─ Generate YAML config from parameters
    │
    └─ cleanup() → terminate vLLM server
```

### Usage Pattern

```python
from unsloth.dataprep.synthetic import SyntheticDataKit

# Start the kit (launches vLLM in background)
kit = SyntheticDataKit.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
    max_seq_length = 2048,
    gpu_memory_utilization = 0.9,
)

# Chunk a source document
chunks = kit.chunk_data(filename="textbook.txt")

# Configure Q&A generation
kit.prepare_qa_generation(
    output_folder = "data",
    max_generation_tokens = 512,
    temperature = 0.7,
    default_num_pairs = 25,    # Q&A pairs per chunk
)

# Cleanup when done
kit.cleanup()
```

### The vLLM Server

The kit launches vLLM as a subprocess rather than using it as a library. This provides process isolation — if vLLM crashes, it doesn't take down the Python process. The readiness detection uses regex matching on stdout:

```python
ready_re = re.compile(r"Starting vLLM API server(?:\s+\d+)?\s+on\b")
```

The `PipeCapture` class captures stdout and stderr in background threads, maintaining a rolling buffer of the last 2,000 lines for debugging.

### Document Chunking

The `chunk_data()` method splits documents into overlapping chunks:

```python
max_tokens = max_seq_length - max_generation_tokens * 2 - 128
# Subtract generation tokens (question + answer) and safety margin

n_chunks = ceil(length / (max_tokens - overlap))
boundaries = linspace(0, length - overlap, n_chunks)
```

The overlap (default: 64 tokens) ensures that context at chunk boundaries isn't lost.

---

## 18.3 Output Folder Structure

`prepare_qa_generation()` creates a standardized folder structure:

```
data/
  ├─ pdf/           # Source PDFs
  ├─ html/          # Source HTML files
  ├─ youtube/       # YouTube transcripts
  ├─ docx/          # Word documents
  ├─ ppt/           # PowerPoint files
  ├─ txt/           # Plain text sources
  ├─ output/        # Raw generation output
  ├─ generated/     # Parsed Q&A pairs
  ├─ cleaned/       # Quality-filtered pairs
  └─ final/         # Final training dataset
```

---

## 18.4 Data Recipes (Studio)

For users who prefer a visual interface, Unsloth Studio provides a Data Recipe editor — a node-based workflow builder for data preparation:

```
Node types:
  ├─ Source nodes       (PDF, CSV, DOCX, URL, YouTube)
  ├─ Transform nodes    (Chunk, Filter, Extract, Merge)
  ├─ Generation nodes   (Q&A pairs, Summaries, Instructions)
  ├─ Quality nodes      (Dedup, Score, Clean)
  └─ Output nodes       (HuggingFace Dataset, JSONL, CSV)
```

The backend lives in `studio/backend/core/data_recipe/` and includes an OXC validator for JavaScript-based data transformations.

---

## 18.5 Configuration Parameters

The `prepare_qa_generation()` method accepts fine-grained control:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_generation_tokens` | 512 | Max tokens per generated answer |
| `temperature` | 0.7 | Sampling temperature |
| `top_p` | 0.95 | Nucleus sampling threshold |
| `overlap` | 64 | Token overlap between chunks |
| `default_num_pairs` | 25 | Q&A pairs to generate per chunk |
| `cleanup_threshold` | 1.0 | Quality filtering threshold |
| `cleanup_batch_size` | 4 | Batch size for quality filtering |
| `cleanup_temperature` | 0.3 | Temperature for cleanup pass |

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Raw text loading | `unsloth/dataprep/raw_text.py` |
| SyntheticDataKit | `unsloth/dataprep/synthetic.py` |
| Synthetic configs | `unsloth/dataprep/synthetic_configs.py` |
| Data recipe backend | `studio/backend/core/data_recipe/` |
| vLLM utilities | `unsloth_zoo/vllm_utils.py` |
