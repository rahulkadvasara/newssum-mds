import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments
)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    dataset = load_dataset(
        "csv",
        data_files={
            "train": args.train_csv,
            "validation": args.val_csv
        }
    )

    def preprocess(batch):
        inputs = tokenizer(
            batch["input_text"],
            truncation=True,
            max_length=args.max_input_len
        )
        targets = tokenizer(
            batch["reference_summary"],
            truncation=True,
            max_length=args.max_output_len
        )

        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_input_len", type=int, default=4096)
    parser.add_argument("--max_output_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)

    args = parser.parse_args()
    main(args)


# Commands
# LongT5
# python scripts/train_primeraled_longt5.py \
#   --model_name google/long-t5-tglobal-base \
#   --train_csv data/newssumm_train.csv \
#   --val_csv data/newssumm_dev.csv \
#   --output_dir models/longt5 \
#   --max_input_len 4096

# LED
# python scripts/train_primeraled_longt5.py \
#   --model_name allenai/led-base-16384 \
#   --train_csv data/newssumm_train.csv \
#   --val_csv data/newssumm_dev.csv \
#   --output_dir models/led \
#   --max_input_len 8192

# PRIMERA
# python scripts/train_primeraled_longt5.py \
#   --model_name allenai/PRIMERA \
#   --train_csv data/newssumm_train.csv \
#   --val_csv data/newssumm_dev.csv \
#   --output_dir models/primera \
#   --max_input_len 8192