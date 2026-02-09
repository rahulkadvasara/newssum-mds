import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class FlanT5Summarizer:
    def __init__(
        self,
        model_name="google/flan-t5-xl",
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_input_len=4096,
        max_output_len=256
    ):
        self.device = device
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, text):
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_input_len,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_len,
                num_beams=4
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
