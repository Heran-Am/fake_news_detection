import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score, average_precision_score


def load_split(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]]


def build_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, preserve_index=False)


def compute_metrics(eval_pred):
    """
    eval_pred: (logits, labels)
    We'll compute:
      - F1 for fake class (label=1)
      - Macro F1
      - PR-AUC for fake class
    """
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)

    f1_fake = f1_score(labels, preds, pos_label=1)
    f1_macro = f1_score(labels, preds, average="macro")
    pr_auc = average_precision_score(labels, probs)

    return {
        "f1_fake": f1_fake,
        "f1_macro": f1_macro,
        "pr_auc": pr_auc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed_time")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--out_dir", type=str, default="models/bert_time_distilbert")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    train_df = load_split(data_dir, "train")
    val_df = load_split(data_dir, "val")
    test_df = load_split(data_dir, "test")

    print("Train/Val/Test shapes:", train_df.shape, val_df.shape, test_df.shape)
    print("Label counts (train):\n", train_df["label"].value_counts())
    print("Label counts (test):\n", test_df["label"].value_counts())

    # HuggingFace datasets
    train_ds = build_hf_dataset(train_df)
    val_ds = build_hf_dataset(val_df)
    test_ds = build_hf_dataset(test_df)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    # Training args (CPU-friendly)
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="f1_fake",
        greater_is_better=True,
        fp16=False,  # CPU
        report_to="none",
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

    print("\nTraining...")
    trainer.train()

    print("\nEvaluating on VAL...")
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    print(val_metrics)

    print("\nEvaluating on TEST...")
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    print(test_metrics)

    # Save final model + tokenizer
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Save metrics JSON
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "model_name": args.model_name,
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val": {
            "f1_fake": float(val_metrics.get("eval_f1_fake", np.nan)),
            "f1_macro": float(val_metrics.get("eval_f1_macro", np.nan)),
            "pr_auc": float(val_metrics.get("eval_pr_auc", np.nan)),
        },
        "test": {
            "f1_fake": float(test_metrics.get("eval_f1_fake", np.nan)),
            "f1_macro": float(test_metrics.get("eval_f1_macro", np.nan)),
            "pr_auc": float(test_metrics.get("eval_pr_auc", np.nan)),
        },
    }

    out_json = report_dir / f"metrics_bert_time_distilbert_e{args.epochs}_len{args.max_length}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    print("\nSaved model to:", out_dir.resolve())
    print("Saved metrics to:", out_json.resolve())


if __name__ == "__main__":
    main()