# Chapter 18: Data Preparation — Raw Text, Synthetic Data, and Data Recipes

---

## Introduction

Unsloth provides tools for preparing training data from multiple sources: raw text files, synthetic data generation, and the visual Data Recipe builder in Studio.

### What You'll Learn

- `RawTextDataLoader` and `TextPreprocessor` for text data
- Synthetic dataset generation from PDFs, CSVs, and documents
- Data Recipe node-based editing in Studio

---

## Notes & Key Points

### 18.1 Raw Text Data

- `dataprep/raw_text.py` (13K) — `RawTextDataLoader`, `TextPreprocessor`
- Loads and preprocesses plain text files for continued pretraining
- Handles chunking, tokenization, and formatting

### 18.2 Synthetic Data Generation

- `dataprep/synthetic.py` (16K) — LLM-driven synthetic dataset creation
- `dataprep/synthetic_configs.py` (4K) — Configuration presets
- Generate instruction-response pairs from source documents
- Uses the loaded model to synthesize training data

### 18.3 Data Recipes (Studio)

- Visual node-based workflow editor in Studio
- Auto-create datasets from PDFs, CSVs, DOCX, and other formats
- Backend in `studio/backend/core/data_recipe/`
- Includes an OXC validator for JavaScript-based data transforms

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Raw text loading | `unsloth/dataprep/raw_text.py` |
| Synthetic generation | `unsloth/dataprep/synthetic.py` |
| Synthetic configs | `unsloth/dataprep/synthetic_configs.py` |
| Data recipe backend | `studio/backend/core/data_recipe/` |
