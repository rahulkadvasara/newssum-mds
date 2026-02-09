# HGP-Lite-LongT5: Hierarchical Planner-Enhanced Model for NewsSumm

## 1. Motivation
Multi-document news summarization requires aggregating overlapping and complementary information from multiple articles while avoiding redundancy and hallucination. Although long-context encoder–decoder models can process large inputs, they rely primarily on implicit attention mechanisms, which often struggle to enforce global structure and content planning across documents.

**HGP-Lite-LongT5** introduces a **lightweight hierarchical planning mechanism** on top of a LongT5 backbone to explicitly guide summary generation, improving coherence and reducing redundancy under limited computational budgets. The design intentionally avoids heavy graph neural networks or complex entity pipelines.

---

## 2. Model Overview
HGP-Lite-LongT5 extends **LongT5-base** with a minimal planning component operating prior to decoding.

The model consists of:
1. A shared LongT5 encoder for long-context document encoding
2. A lightweight hierarchical planner head for salience estimation

No additional pretraining data or external knowledge sources are used.

---

## 3. Architecture

### 3.1 Encoder Backbone
All documents within a cluster are concatenated using explicit document delimiters and encoded using the standard LongT5 encoder.

Let:
- \( D = \{d_1, d_2, ..., d_n\} \) denote documents in a cluster
- The encoder produces contextual token representations \( H \)

No document-specific encoders are introduced, ensuring architectural simplicity and compatibility with existing checkpoints.

---

### 3.2 Hierarchical Planning Representation
To approximate hierarchical structure without explicit graphs, encoder outputs are aggregated at coarse granularity:

- Token representations are grouped using document delimiters
- Mean-pooled representations are computed to obtain **document-level salience vectors**
- Sentence-level representations are approximated using sliding-window pooling

This design provides hierarchical signals while remaining computationally lightweight.

---

### 3.3 Planner Head
A small feed-forward planner head predicts salience scores over aggregated representations:

\[
p_i = \sigma(W h_i)
\]

where \( h_i \) represents pooled encoder features.

Top-K salient representations are selected and used to **reweight encoder outputs** prior to decoding. No explicit graph construction or message passing is performed.

---

### 3.4 Decoder
The LongT5 decoder generates the final summary conditioned on:
- Original encoder representations
- Planner-reweighted encoder states

This allows the decoder to prioritize globally salient content during generation.

---

## 4. Training Objective

### 4.1 Primary Objective
Standard cross-entropy loss for abstractive summarization:

\[
\mathcal{L}_{summary}
\]

### 4.2 Planner Regularization (Lightweight)
An auxiliary sparsity regularization term encourages focused planning:

\[
\mathcal{L}_{planner}
\]

The final objective is:

\[
\mathcal{L} = \mathcal{L}_{summary} + \lambda \mathcal{L}_{planner}
\]

No additional salience annotations are required.

---

## 5. Design Choices and Constraints
- No heavy graph neural networks
- No entity-level supervision
- No increase in encoder depth
- Minimal parameter overhead compared to LongT5-base

These choices ensure feasibility under limited compute while preserving reproducibility.

---

## 6. Expected Benefits
- Improved discourse-level coherence
- Reduced redundancy across documents
- Better global content organization
- Compatibility with long-context summarization benchmarks

---

## 7. Implementation Status
✔ LongT5-base backbone  
✔ Lightweight planner head  
✔ End-to-end trainable  
✔ Evaluated under identical pipeline as baselines  

Future extensions may explore explicit entity modeling or richer hierarchical supervision without compromising efficiency.
