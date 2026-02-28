import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

SEED = 42
MIN_CHARS = 50  # change to 30/80 if you want to experiment


def _read_csv_case_insensitive(path_a: Path, path_b: Path) -> pd.DataFrame:
    """Helps when Kaggle files are Fake.csv/True.csv but code uses fake.csv/true.csv."""
    if path_a.exists():
        return pd.read_csv(path_a)
    if path_b.exists():
        return pd.read_csv(path_b)
    raise FileNotFoundError(f"Could not find {path_a} or {path_b}")


def load_data() -> pd.DataFrame:
    fake = _read_csv_case_insensitive(RAW_DIR / "Fake.csv", RAW_DIR / "fake.csv")
    true = _read_csv_case_insensitive(RAW_DIR / "True.csv", RAW_DIR / "true.csv")

    fake["label"] = 1
    true["label"] = 0

    df = pd.concat([fake, true], axis=0, ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop leakage columns if they exist
    leakage_cols = ["subject", "date"]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors="ignore")

    if "text" not in df.columns:
        raise ValueError(f"'text' column not found. Columns are: {list(df.columns)}")

    # Ensure string type
    df["text"] = df["text"].astype(str)

    # Remove missing / empty-ish text
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() > 0]

    # Remove extremely short texts
    before = len(df)
    df = df[df["text"].str.strip().str.len() >= MIN_CHARS]
    removed_short = before - len(df)

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    removed_dupes = before - len(df)

    df = df.reset_index(drop=True)

    print(f"Removed short texts (<{MIN_CHARS} chars): {removed_short}")
    print(f"Removed duplicate texts: {removed_dupes}")

    return df


def split_data(df: pd.DataFrame):
    # First split: train+val vs test
    trainval_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"],
    )

    # Second split: train vs val (val is 10% of total here)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=0.125,  # 0.125 * 0.8 = 0.1 of total
        random_state=SEED,
        stratify=trainval_df["label"],
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)


def main():
    print("Starting data preparation...")

    df = load_data()
    print("Loaded data:", df.shape)
    print("Label counts:\n", df["label"].value_counts())

    df = clean_data(df)
    print("After cleaning:", df.shape)
    print("Label counts after cleaning:\n", df["label"].value_counts())

    train_df, val_df, test_df = split_data(df)
    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    save_data(train_df, val_df, test_df)
    print("Saved to:", OUT_DIR.resolve())
    print("Data preparation completed successfully.")


if __name__ == "__main__":
    main()