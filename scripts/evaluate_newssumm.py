# scripts/evaluate_newssumm.py

import json
import argparse
import pandas as pd
import evaluate

rouge = evaluate.load("rouge")

def evaluate_predictions(pred_path):
    with open(pred_path) as f:
        data = json.load(f)

    preds = [x["prediction"] for x in data]
    refs = [x["reference"] for x in data]

    scores = rouge.compute(predictions=preds, references=refs)

    return {
        "ROUGE-1": scores["rouge1"],
        "ROUGE-2": scores["rouge2"],
        "ROUGE-L": scores["rougeL"]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--csv_path", default="results/newssumm_benchmark_scores.csv")
    args = parser.parse_args()

    scores = evaluate_predictions(args.predictions)

    row = {
        "Model": args.model_name,
        "ROUGE-1": scores["ROUGE-1"],
        "ROUGE-2": scores["ROUGE-2"],
        "ROUGE-L": scores["ROUGE-L"]
    }

    if os.path.exists(args.csv_path):
        df = pd.read_csv(args.csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(args.csv_path, index=False)
    print("✅ Updated benchmark table saved to:", args.csv_path)

if __name__ == "__main__":
    import os
    main()
