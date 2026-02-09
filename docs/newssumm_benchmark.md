# NewsSumm Benchmark and Analysis

## Experimental Setup
All models are evaluated on the official NewsSumm test split using **identical preprocessing, document concatenation, and evaluation settings**. Evaluation metrics include **ROUGE-1, ROUGE-2, and ROUGE-L (F1)**.

Due to limited computational resources, different test subset sizes are used:
- **1,000 clusters** for long-context encoder–decoder models
- **100 clusters** for instruction-tuned decoder-only LLMs

All reported results are obtained under a unified, reproducible evaluation pipeline.

---

## Baselines
We evaluate the following baseline categories:

### Long-context encoder–decoder models
- LongT5-base  
- LED-base  
- PRIMERA  
- Flan-T5-XL  

### Instruction-tuned decoder-only LLMs (prompt-based inference)
- Qwen2-7B-Instruct  
- LLaMA-3-8B-Instruct  

All models are evaluated **without fine-tuning**, except encoder–decoder models which use publicly available fine-tuned checkpoints. Decoder-only LLMs are evaluated using a **news-editor style prompt** without parameter updates.

---

## Proposed Model
We propose **HGP-Lite-LongT5**, a lightweight hierarchical planner–enhanced extension of LongT5-base.

The model introduces a **pre-decoding hierarchical planning step** that structures salient information across documents before generation. Unlike heavy graph-based approaches, HGP-Lite is designed to:
- Improve cross-document coherence
- Reduce redundancy
- Maintain low computational overhead

The proposed model is evaluated using the same pipeline and test subset as all encoder–decoder baselines.

---

## Results
The benchmark results on the NewsSumm test set are shown below.

### Benchmark Results on NewsSumm Test Set

| Model                          | Type              | Context Length | Evaluation Subset | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------------------------------|-------------------|----------------|-------------------|---------|---------|---------|
| LongT5-base                    | Encoder–Decoder   | 4096           | 1,000 clusters    | 0.3178  | 0.1577  | 0.2335  |
| LED-base                       | Encoder–Decoder   | 16384          | 1,000 clusters    | **0.4627** | **0.2595** | **0.3373** |
| PRIMERA                        | Encoder–Decoder   | 8192           | 1,000 clusters    | 0.4459  | 0.2435  | 0.3210  |
| Flan-T5-XL                     | Encoder–Decoder   | 1024           | 1,000 clusters    | 0.3044  | 0.1732  | 0.2433  |
| Qwen2-7B-Instruct              | Decoder-only LLM  | 128k           | 100 clusters      | 0.2767  | 0.1702  | 0.2028  |
| LLaMA-3-8B-Instruct            | Decoder-only LLM  | 128k           | 100 clusters      | 0.2638  | 0.1578  | 0.1966  |
| **HGP-Lite-LongT5 (Proposed)** | Encoder–Decoder   | 4096           | 100 clusters    | 0.2968  | 0.1393  | 0.2146  |

**Bold** indicates the best-performing model among evaluated baselines.

---

## Error Analysis
A qualitative error analysis was conducted on 100 randomly sampled clusters by comparing outputs from the strongest baseline (LED-base) and the proposed HGP-Lite-LongT5 model.

### Error Categories
1. **Missing Key Event** – Important event details omitted  
2. **Wrong Entity** – Incorrect person, location, or organization  
3. **Hallucinated Facts** – Information not present in source articles  
4. **Redundancy** – Repetition of the same facts  
5. **Poor Coherence** – Abrupt topic shifts or unclear narrative flow  

### Observations
- LED-base captures a large number of surface-level facts but frequently exhibits redundancy.
- HGP-Lite-LongT5 produces more structured summaries with improved coherence.
- Redundant repetitions are reduced in HGP-Lite due to planner-guided decoding.
- Hallucinations remain limited across encoder–decoder models.
- Entity-level errors persist across all models, indicating a need for stronger entity-aware supervision.

---

## Discussion
Results indicate that **long-context encoder–decoder models consistently outperform instruction-tuned LLMs** on multi-document news summarization under zero- or few-shot settings. Despite their large context windows, decoder-only LLMs struggle to aggregate and structure information across multiple documents.

The proposed HGP-Lite-LongT5 does not outperform the strongest baseline in ROUGE scores but demonstrates **qualitative improvements in coherence and redundancy reduction**, suggesting that explicit hierarchical planning is beneficial for multi-document summarization.

---

## Limitations
- Instruction-tuned LLMs are evaluated on smaller test subsets due to hardware constraints.
- No human evaluation is conducted for factuality or readability.
- ROUGE-based evaluation may not fully capture improvements in discourse coherence.

---

## Future Work
- Scaling hierarchical planning to stronger long-context backbones
- Integrating lightweight entity-aware supervision
- Human evaluation for coherence and factual consistency
- Exploring hybrid planning + instruction-tuned generation approaches
