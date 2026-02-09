import argparse
import json
from tqdm import tqdm

from models.baseline_longt5 import LongT5Summarizer
from models.baseline_led import LEDSummarizer
from models.baseline_primerA import PRIMERASummarizer
from models.baseline_flan_t5 import FlanT5Summarizer
from models.newssumm_dataset import NewsSummDataset


MODEL_REGISTRY = {
    "longt5": LongT5Summarizer,
    "led": LEDSummarizer,
    "primera": PRIMERASummarizer,
    "flan_t5": FlanT5Summarizer
}


def main(args):
    dataset = NewsSummDataset(args.test_csv)

    model_class = MODEL_REGISTRY[args.model_type]
    model = model_class()

    results = []

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]

        summary = model.generate(sample["input_text"])

        results.append({
            "cluster_id": sample["cluster_id"],
            "generated_summary": summary,
            "reference_summary": sample["reference_summary"]
        })

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, required=True,
                        choices=["longt5", "led", "primera", "flan_t5"])
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()
    main(args)



# python scripts/generate_predictions.py \
#   --model_type longt5 \
#   --test_csv data/processed/test.csv \
#   --output_path results/predictions/longt5.json