# scripts/data_stats.py

import pandas as pd
from transformers import AutoTokenizer
import numpy as np

CSV_PATH = "data/newssumm_test_subset_1000.csv"
MODEL_NAME = "google/long-t5-tglobal-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
df = pd.read_csv(CSV_PATH)

token_counts = []
doc_counts = []

for text in df["input_text"]:
    tokens = tokenizer.encode(text, truncation=False)
    token_counts.append(len(tokens))
    doc_counts.append(text.count("[DOC"))

print("Number of clusters:", len(df))
print("Average tokens per cluster:", int(np.mean(token_counts)))
print("Max tokens per cluster:", int(np.max(token_counts)))
print("Average documents per cluster:", round(np.mean(doc_counts), 2))
