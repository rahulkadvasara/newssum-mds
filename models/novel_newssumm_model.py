import torch
import torch.nn as nn
from transformers import LongT5ForConditionalGeneration


class HGPSummarizer(nn.Module):
    def __init__(self, base_model_name="google/long-t5-tglobal-base"):
        super().__init__()
        self.base_model = LongT5ForConditionalGeneration.from_pretrained(base_model_name)

        hidden_size = self.base_model.config.d_model

        # Salience planner head
        self.planner = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):
        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        hidden_states = encoder_outputs.last_hidden_state

        # Planner salience scores
        salience_scores = torch.sigmoid(self.planner(hidden_states))

        # Salience-weighted encoder states
        weighted_states = hidden_states * salience_scores

        outputs = self.base_model(
            encoder_outputs=(weighted_states,),
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        return outputs
