import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from models.novel_newssumm_model import HGPSummarizer


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = HGPSummarizer(args.base_model)

    dataset = load_dataset("csv", data_files={
        "train": args.train_csv,
        "validation": args.val_csv
    })

    def preprocess(batch):
        inputs = tokenizer(
            batch["input_text"],
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
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
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

    parser.add_argument("--base_model", default="google/long-t5-tglobal-base")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--max_input_len", type=int, default=4096)
    parser.add_argument("--max_output_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)

    main(parser.parse_args())
