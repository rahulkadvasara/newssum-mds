# NewsSumm Dataset Pipeline

## 1. Dataset Source
The NewsSumm dataset is obtained from the publicly available Zenodo repository as described in:

Motghare et al., *“NewsSumm: The World’s Largest Human-Annotated Multi-Document News Summarization Dataset for Indian English”*, Computers, 2025.

NewsSumm is a large-scale, human-annotated dataset designed for **multi-document abstractive news summarization (MDS)** in Indian English.

---

## 2. Dataset Overview
Each data sample corresponds to a **news event cluster**, consisting of multiple news articles reporting the same real-world event from different sources.

Each cluster contains:
- **Cluster ID**: Unique identifier for the news event
- **Source Articles**: Multiple news articles from different newspapers
- **Human Summary**: A professionally written abstractive summary capturing the core information of the event

The dataset is divided into official **train**, **validation**, and **test** splits.
Due to computational constraints, this project focuses on subsets of the official test split.

---

## 3. File Structure Used in This Project

    data/
    ├── newssumm_test_subset_1000.csv
    ├── newssumm_test_subset_100.csv

---

## 4. CSV Schema
Each CSV file contains the following columns:
- `cluster_id`
- `input_text`
- `reference_summary`

The `input_text` field contains concatenated articles belonging to a cluster,
separated using special document markers.

---

## 5. Preprocessing Pipeline

### 5.1 Document Concatenation
All articles belonging to the same cluster are concatenated into a single input
sequence using explicit document markers:

    [DOC 1] Article text ...
    [DOC 2] Article text ...
    [DOC 3] Article text ...

This allows long-context encoder–decoder models and decoder-only LLMs to distinguish
between documents while preserving cross-document relationships.

---

### 5.2 Text Cleaning
The following preprocessing steps are applied:
- Removal of HTML tags and formatting artifacts (if present)
- Preservation of punctuation, capitalization, and named entities
- No aggressive normalization is applied to avoid loss of factual information

---

### 5.3 Subset Creation
To enable efficient experimentation within limited computational resources:

- A **1,000-cluster subset** of the official test split is used for long-context
  encoder–decoder models (LongT5, LED, PRIMERA, Flan-T5-XL) and the proposed
  **HGP-Lite-LongT5** model.
- A **100-cluster subset** is used for instruction-tuned, decoder-only large
  language models (Qwen2-7B-Instruct, LLaMA-3-8B-Instruct) due to GPU memory
  constraints.

All subsets are extracted exclusively from the official test split, ensuring
that no training or validation data leakage occurs in any experiment.

---

## 6. Tokenization Strategy
Tokenization is performed using Hugging Face tokenizers corresponding to each model.

- Encoder–decoder models use their native tokenizers.
- Decoder-only instruction-tuned LLMs use model-specific tokenizers and prompt-based
  input formatting.

For dataset statistics, token counts are computed using the LongT5 tokenizer, as it
supports long-context inputs and provides a consistent upper-bound estimate across
encoder–decoder models.

Tokenization settings:
- Truncation is applied only when required by model constraints
- No padding is applied during statistics computation

---

## 7. Dataset Statistics
The following statistics are computed for analysis and reporting:
- Number of clusters
- Average number of tokens per cluster
- Maximum token length per cluster
- Average number of documents per cluster (approximated using `[DOC]` markers)

These statistics help assess long-context requirements and guide model selection.

---

## 8. Dataset Loader Implementation
A reusable Python module (`newssumm_dataset.py`) is implemented to provide
model-agnostic access to the dataset. Each sample exposes:
- Cluster ID
- Concatenated source documents
- Human reference summary

This design allows seamless integration with:
- Long-context encoder–decoder models
- Instruction-tuned decoder-only LLMs
- The proposed **HGP-Lite-LongT5** model
- Unified evaluation pipelines

---

## 9. Limitations
- Full train/validation splits are not used in the current phase due to
  computational constraints.
- Document counts are approximated using document markers rather than explicit
  metadata.
- Instruction-tuned LLMs are evaluated on reduced cluster subsets.

These limitations are documented transparently and do not affect reproducibility.

---

## 10. Reproducibility Notes
- All dataset subsets are stored as CSV files under version control.
- Random seeds, preprocessing rules, and subset indices are fixed and documented,
  ensuring deterministic regeneration of all experimental inputs.
- Future users can reproduce results by reusing the same CSV files and scripts.

This pipeline ensures consistency across experiments and supports fair comparison
between all evaluated models.
