# NewsSumm Dataset (Project Usage)

This directory contains the dataset files used for benchmarking multi-document news summarization models on the **NewsSumm** dataset.

The data is derived from the official NewsSumm test split and organized into fixed subsets to support reproducible experimentation under limited computational resources.

---

## Dataset Source
The original dataset is introduced in:

Motghare et al., *“NewsSumm: The World’s Largest Human-Annotated Multi-Document News Summarization Dataset for Indian English”*, Computers, 2025.

NewsSumm is a large-scale, human-annotated dataset designed for **multi-document abstractive news summarization** in Indian English.

---

## Files in This Directory

    data/
    ├── newssumm_test_subset_1000.csv
    ├── newssumm_test_subset_100.csv
    └── README.md

---

## File Descriptions

### `newssumm_test_subset_1000.csv`
- Subset of **1,000 clusters** sampled from the official NewsSumm test split
- Used for:
  - LongT5-base
  - LED-base
  - PRIMERA
  - Flan-T5-XL
  - HGP-Lite-LongT5 (Proposed)
- Suitable for long-context encoder–decoder models

### `newssumm_test_subset_100.csv`
- Subset of **100 clusters** sampled from the official NewsSumm test split
- Used for:
  - Qwen2-7B-Instruct
  - LLaMA-3-8B-Instruct
- Designed for instruction-tuned, decoder-only LLMs under GPU memory constraints

---

## CSV Schema
Both CSV files follow the same schema:

- `cluster_id` – Unique identifier for each news event
- `input_text` – Concatenated source articles with explicit document markers
- `reference_summary` – Human-written abstractive summary

Example document formatting inside `input_text`:

    [DOC 1] Article text ...
    [DOC 2] Article text ...
    [DOC 3] Article text ...

---

## Preprocessing Notes
- All clusters originate **exclusively from the official test split**
- No training or validation data is used
- Document order is preserved
- No aggressive text normalization is applied
- Truncation is applied only when required by model context limits

---

## Reproducibility
- Subset indices are fixed and deterministic
- CSV files are version-controlled
- All models are evaluated using the same preprocessing and evaluation pipeline

Using these files ensures **fair and reproducible comparison** across all evaluated models.

---

## Notes
These subsets are intended for **benchmarking and analysis only** and not for training new models. Future work may extend evaluation to larger portions of the test split or include human evaluation.
