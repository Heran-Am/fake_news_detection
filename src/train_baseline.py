import json
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    average_precision_score,
)


MODEL_DIR = Path("models/baseline")
REPORT_DIR = Path("reports")
SEED = 42


def load_split(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run data preparation first.")
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain 'text' and 'label'.")
    df["text"] = df["text"].astype(str)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--run_name", type=str, default="baseline")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    run_name = args.run_name

    train_df = load_split(data_dir, "train")
    val_df = load_split(data_dir, "val")
    test_df = load_split(data_dir, "test")

    X_train, y_train = train_df["text"], train_df["label"].astype(int)
    X_val, y_val = val_df["text"], val_df["label"].astype(int)
    X_test, y_test = test_df["text"], test_df["label"].astype(int)

    # TF-IDF baseline
    vectorizer = TfidfVectorizer(
        max_features=80_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    Xtr = vectorizer.fit_transform(X_train)
    Xva = vectorizer.transform(X_val)
    Xte = vectorizer.transform(X_test)

    # Logistic Regression baseline
    clf = LogisticRegression(
        max_iter=2000,
        random_state=SEED,
        class_weight="balanced",
    )
    clf.fit(Xtr, y_train)

    # Predictions
    val_pred = clf.predict(Xva)
    test_pred = clf.predict(Xte)

    # Probabilities for PR-AUC (needs predict_proba)
    val_proba = clf.predict_proba(Xva)[:, 1]
    test_proba = clf.predict_proba(Xte)[:, 1]

    # Metrics
    val_f1_pos = f1_score(y_val, val_pred, pos_label=1)
    test_f1_pos = f1_score(y_test, test_pred, pos_label=1)

    val_f1_macro = f1_score(y_val, val_pred, average="macro")
    test_f1_macro = f1_score(y_test, test_pred, average="macro")

    val_pr_auc = average_precision_score(y_val, val_proba)
    test_pr_auc = average_precision_score(y_test, test_proba)

    print("\n=== Validation ===")
    print("F1 (fake=1):", round(val_f1_pos, 4))
    print("F1 macro:", round(val_f1_macro, 4))
    print("PR-AUC (fake=1):", round(val_pr_auc, 4))
    print(classification_report(y_val, val_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_val, val_pred))

    print("\n=== Test ===")
    print("F1 (fake=1):", round(test_f1_pos, 4))
    print("F1 macro:", round(test_f1_macro, 4))
    print("PR-AUC (fake=1):", round(test_pr_auc, 4))
    print(classification_report(y_test, test_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, test_pred))

    # Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, MODEL_DIR / f"tfidf_{run_name}.joblib")
    joblib.dump(clf, MODEL_DIR / f"logreg_{run_name}.joblib")

    metrics = {
        "run_name": run_name,
        "data_dir": str(data_dir),
        "label_mapping": {"0": "true", "1": "fake"},
        "val": {
            "f1_fake": float(val_f1_pos),
            "f1_macro": float(val_f1_macro),
            "pr_auc": float(val_pr_auc),
        },
        "test": {
            "f1_fake": float(test_f1_pos),
            "f1_macro": float(test_f1_macro),
            "pr_auc": float(test_pr_auc),
        },
        "tfidf": {
            "max_features": 80000,
            "ngram_range": [1, 2],
            "min_df": 2,
            "sublinear_tf": True,
        },
        "model": {"type": "LogisticRegression", "class_weight": "balanced"},
    }

    with open(REPORT_DIR / f"metrics_{run_name}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved:")
    print(" -", (MODEL_DIR / f"tfidf_{run_name}.joblib").as_posix())
    print(" -", (MODEL_DIR / f"logreg_{run_name}.joblib").as_posix())
    print(" -", (REPORT_DIR / f"metrics_{run_name}.json").as_posix())


if __name__ == "__main__":
    main()