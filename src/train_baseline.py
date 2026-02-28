import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/baseline")
REPORT_DIR = Path("reports")

SEED = 42


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run data_preparation.py first.")
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    return df


def main():
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    X_train, y_train = train_df["text"], train_df["label"]
    X_val, y_val = val_df["text"], val_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]

    # TF-IDF (unigrams + bigrams usually works well for fake news)
    vectorizer = TfidfVectorizer(
        max_features=80_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    Xtr = vectorizer.fit_transform(X_train)
    Xva = vectorizer.transform(X_val)
    Xte = vectorizer.transform(X_test)

    # Logistic Regression baseline (balanced to compensate for class skew)
    clf = LogisticRegression(
        max_iter=2000,
        random_state=SEED,
        class_weight="balanced",
    )
    clf.fit(Xtr, y_train)

    # Evaluate
    val_pred = clf.predict(Xva)
    test_pred = clf.predict(Xte)

    val_f1 = f1_score(y_val, val_pred)
    test_f1 = f1_score(y_test, test_pred)

    print("\n=== Validation ===")
    print("F1:", round(val_f1, 4))
    print(classification_report(y_val, val_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_val, val_pred))

    print("\n=== Test ===")
    print("F1:", round(test_f1, 4))
    print(classification_report(y_test, test_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, test_pred))

    # Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, MODEL_DIR / "tfidf.joblib")
    joblib.dump(clf, MODEL_DIR / "logreg.joblib")

    metrics = {
        "val": {"f1": float(val_f1)},
        "test": {"f1": float(test_f1)},
        "label_mapping": {"0": "true", "1": "fake"},
        "tfidf": {
            "max_features": 80000,
            "ngram_range": [1, 2],
            "min_df": 2,
            "sublinear_tf": True,
        },
        "model": {"type": "LogisticRegression", "class_weight": "balanced"},
    }

    with open(REPORT_DIR / "metrics_baseline.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved:")
    print(" -", (MODEL_DIR / "tfidf.joblib").as_posix())
    print(" -", (MODEL_DIR / "logreg.joblib").as_posix())
    print(" -", (REPORT_DIR / "metrics_baseline.json").as_posix())


if __name__ == "__main__":
    main()