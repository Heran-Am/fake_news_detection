import json
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
)

MODEL_DIR = Path("models/baseline")
REPORT_DIR = Path("reports")


def load_split(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def find_best_threshold(y_true: np.ndarray, proba: np.ndarray, metric: str = "f1_fake"):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_score = -1.0

    for t in thresholds:
        pred = (proba >= t).astype(int)
        if metric == "f1_fake":
            score = f1_score(y_true, pred, pos_label=1)
        elif metric == "precision_fake":
            score = precision_score(y_true, pred, pos_label=1, zero_division=0)
        elif metric == "recall_fake":
            score = recall_score(y_true, pred, pos_label=1, zero_division=0)
        else:
            raise ValueError("metric must be one of: f1_fake, precision_fake, recall_fake")

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, float(best_score)


def evaluate_at_threshold(name: str, y_true: np.ndarray, proba: np.ndarray, t: float):
    pred = (proba >= t).astype(int)
    f1_fake = f1_score(y_true, pred, pos_label=1)
    f1_macro = f1_score(y_true, pred, average="macro")
    prec = precision_score(y_true, pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, pred, pos_label=1, zero_division=0)
    pr_auc = average_precision_score(y_true, proba)

    print(f"\n=== {name} @ threshold={t:.2f} ===")
    print("F1(fake=1):", round(f1_fake, 4))
    print("F1 macro:", round(f1_macro, 4))
    print("Precision(fake):", round(prec, 4))
    print("Recall(fake):", round(rec, 4))
    print("PR-AUC(fake):", round(pr_auc, 4))
    print(classification_report(y_true, pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, pred))

    return {
        "threshold": float(t),
        "f1_fake": float(f1_fake),
        "f1_macro": float(f1_macro),
        "precision_fake": float(prec),
        "recall_fake": float(rec),
        "pr_auc": float(pr_auc),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)  # e.g. baseline_time
    parser.add_argument("--optimize", type=str, default="f1_fake")  # f1_fake / precision_fake / recall_fake
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    run_name = args.run_name

    # Load saved model artifacts from train_baseline.py
    tfidf_path = MODEL_DIR / f"tfidf_{run_name}.joblib"
    model_path = MODEL_DIR / f"logreg_{run_name}.joblib"

    if not tfidf_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Missing model files. Expected:\n  {tfidf_path}\n  {model_path}\n"
            f"Run: python src/train_baseline.py --data_dir {data_dir} --run_name {run_name}"
        )

    vectorizer = joblib.load(tfidf_path)
    clf = joblib.load(model_path)

    val_df = load_split(data_dir, "val")
    test_df = load_split(data_dir, "test")

    X_val = vectorizer.transform(val_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    val_proba = clf.predict_proba(X_val)[:, 1]
    test_proba = clf.predict_proba(X_test)[:, 1]

    best_t, best_score = find_best_threshold(val_df["label"].values, val_proba, metric=args.optimize)
    print(f"Best threshold on VAL optimizing {args.optimize}: t={best_t:.2f}, score={best_score:.4f}")

    val_metrics = evaluate_at_threshold("Validation", val_df["label"].values, val_proba, best_t)
    test_metrics = evaluate_at_threshold("Test", test_df["label"].values, test_proba, best_t)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"threshold_tuning_{run_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "data_dir": str(data_dir),
                "optimized_for": args.optimize,
                "best_threshold": best_t,
                "val": val_metrics,
                "test": test_metrics,
            },
            f,
            indent=2,
        )

    print("\nSaved:", out_path.as_posix())


if __name__ == "__main__":
    main()