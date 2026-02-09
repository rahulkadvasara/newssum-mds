import argparse
import json
from tqdm import tqdm

# Encoder–decoder baselines
from models.baseline_longt5 import LongT5Summarizer
from models.baseline_led import LEDSummarizer
from models.baseline_primerA import PRIMERASummarizer
from models.baseline_flan_t5 import FlanT5Summarizer

# Instruction-tuned LLMs (prompt-based)
from models.llm_qwen2 import Qwen2Summarizer
from models.llm_llama3 import LLaMA3Summarizer

# Proposed model
from models.novel_hgp_model import HGPLiteLongT5Summarizer

from models.newssumm_dataset import NewsSummDataset


MODEL_REGISTRY = {
    # Encoder–decoder models
    "longt5": LongT5Summarizer,
    "led": LEDSummarizer,
    "primera": PRIMERASummarizer,
    "flan_t5": FlanT5Summarizer,

    # Instruction-tuned LLMs
    "qwen2": Qwen2Summarizer,
    "llama3": LLaMA3Summarizer,

    # Proposed model
    "hgp_lite_longt5": HGPLiteLongT5Summarizer,
}


def main(args):
    dataset = NewsSummDataset(args.test_csv)

    model_class = MODEL_REGISTRY[args.model_type]
    model = model_class()

    results = []

    for i in tqdm(range(len(dataset)), desc=f"Generating summaries ({args.model_type})"):
        sample = dataset[i]

        summary = model.generate(sample["input_text"])

        results.append({
            "cluster_id": sample["cluster_id"],
            "generated_summary": summary,
            "reference_summary": sample["reference_summary"]
        })

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NewsSumm predictions")

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model identifier"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="Path to NewsSumm test CSV"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save generated predictions (JSON)"
    )

    args = parser.parse_args()
    main(args)


"""
Example usage:

# Encoder–decoder baseline
python scripts/generate_predictions.py \
  --model_type longt5 \
  --test_csv data/newssumm_test_subset_1000.csv \
  --output_path results/predictions/longt5.json

# Instruction-tuned LLM
python scripts/generate_predictions.py \
  --model_type qwen2 \
  --test_csv data/newssumm_test_subset_100.csv \
  --output_path results/predictions/qwen2.json

# Proposed model
python scripts/generate_predictions.py \
  --model_type hgp_lite_longt5 \
  --test_csv data/newssumm_test_subset_1000.csv \
  --output_path results/predictions/hgp_lite_longt5.json
"""
