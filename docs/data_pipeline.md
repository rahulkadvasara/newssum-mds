# NewsSumm Dataset Pipeline

## 1. Dataset Source
The NewsSumm dataset is obtained from the publicly available Zenodo repository as
described in:

Motghare et al., *“NewsSumm: The World’s Largest Human-Annotated Multi-Document
News Summarization Dataset for Indian English”*, Computers, 2025.

NewsSumm is a large-scale, human-annotated dataset designed for **multi-document
abstractive news summarization (MDS)** in Indian English.

---

## 2. Dataset Overview
Each data sample corresponds to a **news event cluster**, consisting of multiple
news articles reporting the same real-world event from different sources.

Each cluster contains:
- **Cluster ID**: Unique identifier for the news event
- **Source Articles**: Multiple news articles from different newspapers
- **Human Summary**: A professionally written abstractive summary capturing the
  core information of the event

The dataset is divided into official **train**, **validation**, and **test** splits.
Due to computational constraints, this project focuses on subsets of the official
test split.

---

## 3. File Structure Used in This Project
```data/
├── newssumm_test_subset_1000.csv
├── newssumm_test_subset_100.csv
```

## 
Each CSV file contains the following columns:
- `cluster_id`
- `input_text`
- `reference_summary`

The `input_text` field contains concatenated articles belonging to a cluster,
separated using special document markers.

---

## 4. Preprocessing Pipeline

### 4.1 Document Concatenation
All articles belonging to the same cluster are concatenated into a single input
sequence using explicit document markers:

```
[DOC 1] Article text ...
[DOC 2] Article text ...
[DOC 3] Article text ...
```

This allows long-context and large language models to distinguish between documents
while preserving cross-document relationships.

---

### 4.2 Text Cleaning
The following preprocessing steps are applied:
- Removal of HTML tags and formatting artifacts (if present)
- Preservation of punctuation, capitalization, and named entities
- No aggressive normalization is applied to avoid loss of factual information

---

### 4.3 Subset Creation
To enable efficient experimentation within limited computational resources:
- A **1,000-cluster subset** of the official test split is used for encoder–decoder
  baselines (e.g., LongT5, LED).
- A **100-cluster subset** is used for large language models due to GPU memory
  constraints.

All subsets are **fixed and deterministic**, ensuring reproducibility.

---

## 5. Tokenization Strategy
Tokenization is performed using Hugging Face tokenizers corresponding to each model.
For dataset statistics, token counts are computed using the LongT5 tokenizer.

Tokenization settings:
- Truncation is applied only when required by model constraints
- No padding is applied during statistics computation

---

## 6. Dataset Statistics
The following statistics are computed for analysis and reporting:
- Number of clusters
- Average number of tokens per cluster
- Maximum token length per cluster
- Average number of documents per cluster (approximated using `[DOC]` markers)

These statistics help assess long-context requirements and guide model selection.

---

## 7. Dataset Loader Implementation
A reusable Python module (`newssumm_dataset.py`) is implemented to provide
model-agnostic access to the dataset. Each sample exposes:
- Cluster ID
- Concatenated source documents
- Human reference summary

This design allows seamless integration with encoder–decoder models, decoder-only
LLMs, and evaluation pipelines.

---

## 8. Limitations
- Full train/validation splits are not used in the current phase due to
  computational constraints.
- Document counts are approximated using document markers rather than explicit
  metadata.
- Large language models are evaluated on reduced cluster subsets.

These limitations are documented transparently and do not affect reproducibility.

---

## 9. Reproducibility Notes
- All dataset subsets are stored as CSV files under version control.
- Random seeds and preprocessing rules are fixed.
- Future users can reproduce results by reusing the same CSV files and scripts.

This pipeline ensures consistency across experiments and supports fair comparison
between models.
