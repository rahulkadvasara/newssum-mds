import argparse
import json
import os
import pandas as pd
import evaluate
from tqdm import tqdm


rouge = evaluate.load("rouge")


def compute_rouge(preds, refs):
    scores = rouge.compute(
        predictions=preds,
        references=refs
    )
    return scores["rouge1"], scores["rouge2"], scores["rougeL"]


def main(args):
    rows = []

    for file in os.listdir(args.predictions_dir):
        if not file.endswith(".json"):
            continue

        model_name = file.replace(".json", "")
        path = os.path.join(args.predictions_dir, file)

        with open(path) as f:
            data = json.load(f)

        preds = [d["generated_summary"] for d in data]
        refs = [d["reference_summary"] for d in data]

        r1, r2, rl = compute_rouge(preds, refs)

        rows.append({
            "model_name": model_name,
            "ROUGE-1": r1,
            "ROUGE-2": r2,
            "ROUGE-L": rl,
            "BERTScore": "NA"
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved evaluation results to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions_dir", required=True)
    parser.add_argument("--output_csv", required=True)

    main(parser.parse_args())



# From repo root:

# python scripts/evaluate_newssumm.py \
#   --predictions_dir results/predictions \
#   --output_csv results/newssumm_baselines_scores.csv

# Optional (only if GPU/CPU allows):
# python scripts/evaluate_newssumm.py \
#   --predictions_dir results/predictions \
#   --output_csv results/newssumm_baselines_scores.csv \
#   --compute_bertscore
