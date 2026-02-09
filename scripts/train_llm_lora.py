import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("csv", data_files={"train": args.train_csv})

    def preprocess(batch):
        prompts = [
            f"You are a news editor. Write a concise and factual summary of the following news articles:\n{doc}"
            for doc in batch["input_text"]
        ]
        inputs = tokenizer(
            prompts,
            truncation=True,
            max_length=args.max_input_len
        )
        labels = tokenizer(
            batch["reference_summary"],
            truncation=True,
            max_length=args.max_output_len
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--max_input_len", type=int, default=4096)
    parser.add_argument("--max_output_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    main(parser.parse_args())


# Commands
# Qwen2-7B
# python scripts/train_llm_lora.py \
#   --base_model Qwen/Qwen2-7B-Instruct \
#   --train_csv data/newssumm_train.csv \
#   --output_dir models/qwen2_lora

# Mistral-7B
# python scripts/train_llm_lora.py \
#   --base_model mistralai/Mistral-7B-Instruct-v0.3 \
#   --train_csv data/newssumm_train.csv \
#   --output_dir models/mistral_lora