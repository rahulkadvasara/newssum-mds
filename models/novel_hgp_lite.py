# models/novel_hgp_lite.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

class HGPLiteSummarizer:
    def __init__(
        self,
        base_model="google/long-t5-tglobal-base",
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_input_tokens=2048,
        max_output_tokens=200,
        top_k_sentences=30
    ):
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.top_k = top_k_sentences

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        self.model.eval()

    def _sentence_planner(self, documents):
        """
        Very light planner:
        selects earliest + longest sentences (proxy for salience)
        """
        sentences = []
        for doc in documents:
            for sent in sent_tokenize(doc):
                sentences.append(sent.strip())

        # sort by length (salience proxy)
        sentences = sorted(sentences, key=len, reverse=True)
        return sentences[: self.top_k]

    def generate(self, documents):
        planned_sents = self._sentence_planner(documents)
        planned_text = " ".join(planned_sents)

        inputs = self.tokenizer(
            planned_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_tokens,
                num_beams=4,
                early_stopping=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
