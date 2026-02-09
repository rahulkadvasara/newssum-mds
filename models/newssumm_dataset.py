# models/newssumm_dataset.py

import pandas as pd
from typing import List, Dict

class NewsSummDataset:
    def __init__(self, csv_path: str):
        """
        csv_path: path to CSV with columns
        [cluster_id, input_text, reference_summary]
        """
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def get_sample(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        return {
            "cluster_id": row["cluster_id"],
            "documents": row["input_text"],
            "reference_summary": row["reference_summary"]
        }

    def get_all(self) -> List[Dict]:
        return [self.get_sample(i) for i in range(len(self.df))]
