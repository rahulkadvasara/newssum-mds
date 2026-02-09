# Results

This folder contains **model predictions and evaluation results** for all NewsSumm experiments reported in this project.

All results are produced using a **single, consistent preprocessing and evaluation pipeline** to ensure fair comparison and reproducibility.

---

## Files

- `predictions/`  
  Contains model-generated summaries, organized by model name.

- `newssumm_benchmark_scores.csv`  
  Final ROUGE-based benchmark results for all evaluated models, including:
  - ROUGE-1 (F1)
  - ROUGE-2 (F1)
  - ROUGE-L (F1)

---

## Models Covered

Results are reported for the following models:

### Long-context encoder–decoder models
- LongT5-base  
- LED-base  
- PRIMERA  
- Flan-T5-XL  

### Instruction-tuned decoder-only LLMs
- Qwen2-7B-Instruct  
- LLaMA-3-8B-Instruct  

### Proposed model
- HGP-Lite-LongT5  

Encoder–decoder models and the proposed model are evaluated on **1,000 test clusters**.  
Instruction-tuned LLMs are evaluated on **100 test clusters** due to GPU memory constraints.

---

## Notes
- All experiments use identical document concatenation, truncation, and evaluation settings.
- All data originates exclusively from the official NewsSumm test split.
- No training or validation data is used.
- ROUGE metrics are reported consistently across all models.
- BERTScore is excluded due to computational overhead.

This folder represents the **authoritative source** for all quantitative results reported in the paper.
