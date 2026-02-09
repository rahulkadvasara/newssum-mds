# scripts/run_hgp.py

import json
import pandas as pd
from tqdm import tqdm
from models.novel_hgp_lite import HGPLiteSummarizer

DATA_PATH = "data/processed/newssumm_test_subset_100.csv"
SAVE_PATH = "results/predictions/hgp_lite_longt5_predictions_100.json"

df = pd.read_csv(DATA_PATH)

model = HGPLiteSummarizer()

results = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    docs = row["input_text"].split("[DOC")
    summary = model.generate(docs)

    results.append({
        "cluster_id": row["cluster_id"],
        "prediction": summary,
        "reference": row["reference_summary"]
    })

with open(SAVE_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("✅ HGP-Lite predictions saved to:", SAVE_PATH)
