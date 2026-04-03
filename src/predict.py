import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


BASELINE_DIR = Path("models/baseline")
BERT_E2_DIR = Path("models/bert_time_distilbert_e2")  # your trained model folder


def predict_baseline(text: str, run_name: str):
    tfidf_path = BASELINE_DIR / f"tfidf_{run_name}.joblib"
    model_path = BASELINE_DIR / f"logreg_{run_name}.joblib"

    if not tfidf_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Missing baseline artifacts:\n  {tfidf_path}\n  {model_path}\n"
            f"Did you train with --run_name {run_name} ?"
        )

    vectorizer = joblib.load(tfidf_path)
    clf = joblib.load(model_path)

    X = vectorizer.transform([text])
    proba_fake = float(clf.predict_proba(X)[0, 1])
    pred = 1 if proba_fake >= 0.5 else 0
    return pred, proba_fake


@torch.inference_mode()
def predict_bert(text: str, model_dir: Path, max_length: int = 128):
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing BERT model directory: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    proba_fake = float(probs[1])
    pred = int(np.argmax(probs))
    return pred, proba_fake


def label_to_str(label: int) -> str:
    return "FAKE" if label == 1 else "TRUE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline_random", "baseline_time", "bert_e2"], required=True)
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--max_length", type=int, default=128, help="Max tokens for BERT")
    args = parser.parse_args()

    text = args.text
    if not text:
        text = input("Paste news text: ").strip()

    if args.model == "baseline_random":
        pred, proba_fake = predict_baseline(text, run_name="baseline_random")
    elif args.model == "baseline_time":
        pred, proba_fake = predict_baseline(text, run_name="baseline_time")
    else:
        pred, proba_fake = predict_bert(text, model_dir=BERT_E2_DIR, max_length=args.max_length)

    print("\nModel:", args.model)
    print("Prediction:", label_to_str(pred))
    print("P(fake):", round(proba_fake, 4))


if __name__ == "__main__":
    main()