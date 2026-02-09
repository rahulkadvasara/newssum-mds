# NewsSumm-MDS

Multi-Document Abstractive Summarization on the NewsSumm dataset using long-context transformers and large language models.

---

## Repository Structure

data/
- raw/
- processed/

models/
- baseline_longt5.py
- baseline_led.py
- baseline_primerA.py
- baseline_flan_t5.py
- newssumm_dataset.py
- novel_hgp_model.py

scripts/
- preprocess_newssumm.py
- train_primeraled_longt5.py
- train_llm_lora.py
- generate_predictions.py
- evaluate_newssumm.py

configs/
- experiment_configs.yaml

results/
- predictions/
- newssumm_benchmark_scores.csv

docs/
- data_pipeline.md
- novel_model_spec.md

---

## Setup

Install dependencies:

pip install transformers datasets evaluate peft bitsandbytes accelerate rouge-score bert-score

---

## Dataset

Download the NewsSumm dataset from Zenodo and place files in:

data/raw/

Preprocess the dataset:

python scripts/preprocess_newssumm.py --input_dir data/raw --output_dir data/processed

---

## Training Baselines

### Encoder–Decoder Models (PRIMERA, LED, LongT5)

python scripts/train_primeraled_longt5.py \
  --model_name google/long-t5-tglobal-base \
  --train_csv data/processed/train.csv \
  --val_csv data/processed/dev.csv \
  --output_dir models/longt5

---

### Instruction-Tuned LLMs (LoRA)

python scripts/train_llm_lora.py \
  --base_model mistralai/Mistral-7B-Instruct-v0.3 \
  --train_csv data/processed/train.csv \
  --output_dir models/mistral_lora

---

## Generating Predictions

python scripts/generate_predictions.py \
  --model_type longt5 \
  --test_csv data/processed/test.csv \
  --output_path results/predictions/longt5.json

---

## Evaluation

python scripts/evaluate_newssumm.py \
  --predictions_dir results/predictions \
  --output_csv results/newssumm_benchmark_scores.csv

Metrics:
- ROUGE-1
- ROUGE-2
- ROUGE-L
- BERTScore (optional)

---

## Proposed Model

A Hierarchical Graph-Planner (HGP) model is proposed for cross-document aggregation and planning-based decoding.

See:
docs/novel_model_spec.md

---

## Reproducibility

Each experiment logs:
- Model configuration
- Random seed
- Hardware used
- Runtime

---

## Author

Rahul Kumar  
Machine Learning Intern  
Suvidha Foundation (Suvidha Mahila Mandal)

---

## License

For academic and research use only. Dataset usage follows the NewsSumm Zenodo license.

## Reproducibility
All experiments are fully reproducible. Configuration files in `configs/` specify
random seeds, hyperparameters, and hardware details. Any future intern can re-run
at least one baseline model and the proposed HGP model using the provided scripts
and recover comparable results.
