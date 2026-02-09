# Hierarchical Graph-Planner (HGP) for NewsSumm

## 1. Motivation
Multi-document news summarization requires aggregating overlapping and complementary information from multiple articles while avoiding redundancy and hallucination. Existing long-context models rely primarily on implicit attention mechanisms, which struggle with explicit cross-document reasoning.

The Hierarchical Graph-Planner (HGP) model introduces structured aggregation and planning to improve factual coverage and coherence for Indian English news summarization.

---

## 2. Model Overview
The HGP model extends a LongT5 backbone with two lightweight modules:
1. A hierarchical sentence/document encoder
2. A salience-aware planning head

---

## 3. Architecture

### 3.1 Hierarchical Encoder
Each document in a cluster is encoded independently using a shared LongT5 encoder.

Let:
- \( D = \{d_1, d_2, ..., d_n\} \) be the set of documents
- \( s_{ij} \) be the j-th sentence in document i

The encoder produces:
- Sentence embeddings: \( h_{ij} \)
- Document embeddings: \( H_i = \text{mean}(h_{ij}) \)

---

### 3.2 Graph-Based Aggregation
Sentence embeddings across documents are connected using a similarity-based graph:

- Nodes: sentence embeddings
- Edges: cosine similarity > threshold or shared named entities

A lightweight graph attention layer propagates salience information:
\[
\tilde{h}_{ij} = \sum_{k \in \mathcal{N}(ij)} \alpha_{ik} h_k
\]

---

### 3.3 Planner Head
A planner predicts a salience score for each sentence:
\[
p_{ij} = \sigma(W \tilde{h}_{ij})
\]

Top-K salient sentences are used to condition the decoder input.

---

### 3.4 Decoder
The LongT5 decoder generates the final summary conditioned on:
- Original encoder outputs
- Planner-weighted representations

---

## 4. Training Objective

### 4.1 Main Loss
Cross-entropy loss for summary generation:
\[
\mathcal{L}_{summary}
\]

### 4.2 Auxiliary Loss (Optional)
Salience prediction loss:
\[
\mathcal{L}_{salience}
\]

Total loss:
\[
\mathcal{L} = \mathcal{L}_{summary} + \lambda \mathcal{L}_{salience}
\]

---

## 5. Indian-News-Aware Adaptation
- Special tokens for states, ministries, and political entities
- Entity coverage constraints (future work)

---

## 6. Expected Benefits
- Improved factual coverage
- Reduced redundancy
- Better cross-document coherence
- Domain sensitivity to Indian news

---

## 7. Implementation Status
✔ Hierarchical encoding  
✔ Planner head  
✔ Trainable end-to-end  

Graph aggregation and entity supervision are implemented in simplified form and can be extended in future work.
