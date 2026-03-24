# Chapter 37: Data Recipes — Visual Dataset Creation

---

## Introduction

Data Recipes is Studio's visual node-based workflow for creating training datasets from raw documents. It automates the process of turning PDFs, CSVs, and other files into instruction-following datasets.

### What You'll Learn

- The visual node editor interface
- How documents are processed into training pairs
- OXC JavaScript validator for data transforms
- Integration with the core `dataprep` module

---

## Notes & Key Points

### 37.1 Data Recipe Workflow

1. Upload source documents (PDF, CSV, DOCX, JSON, etc.)
2. Configure processing nodes in the visual editor
3. Set generation parameters (model to use, number of examples)
4. Execute the recipe to generate a training dataset
5. Review and edit generated data before training

### 37.2 Backend Implementation

- `studio/backend/core/data_recipe/` — Core logic
- Includes OXC validator (`oxc-validator/`) for JavaScript-based data validation
- Uses the `dataprep/synthetic.py` module for LLM-driven generation

### 37.3 Source Connection

- Data Recipes feed directly into the training pipeline
- Generated datasets are compatible with HF `datasets` format
- Can be exported and reused across training runs

---

## Source File Map

| Concept | Primary File(s) |
|---------|-----------------|
| Data recipe core | `studio/backend/core/data_recipe/` |
| OXC validator | `studio/backend/core/data_recipe/oxc-validator/` |
| Synthetic data | `unsloth/dataprep/synthetic.py` |
