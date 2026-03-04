import re
import html
import unicodedata
from pathlib import Path
from hashlib import blake2b

import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

SEED = 42
MIN_CHARS = 20  # keep configurable

# Reuters leakage patterns
REUTERS_TOKEN_RE = re.compile(r"\breuters\b", flags=re.IGNORECASE)
REUTERS_PREFIX_RE = re.compile(r"^\s*\(?reuters\)?\s*[-–—:]\s*", flags=re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Light normalization for model input (removes Reuters artifact + fixes encoding)."""
    if text is None:
        return ""
    text = str(text)

    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)

    # remove Reuters artifacts
    text = REUTERS_PREFIX_RE.sub("", text)
    text = REUTERS_TOKEN_RE.sub(" ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_dedupe(text: str) -> str:
    """
    Stronger normalization ONLY for detecting duplicates/near-duplicates.
    This is not necessarily what we feed the model.
    """
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)     # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)              # drop punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_dedupe_key(text: str) -> str:
    norm = normalize_for_dedupe(text)
    return blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()


def _read_csv_case_insensitive(path_a: Path, path_b: Path) -> pd.DataFrame:
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
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    leakage_cols = ["subject", "date"]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors="ignore")

    if "text" not in df.columns:
        raise ValueError(f"'text' column not found. Columns are: {list(df.columns)}")

    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)

    # Normalize for model input (Reuters removal etc.)
    df["text"] = df["text"].map(normalize_text)

    # Remove empty-ish
    df = df[df["text"].str.strip().str.len() > 0]

    # Remove extremely short texts
    before = len(df)
    df = df[df["text"].str.len() >= MIN_CHARS]
    removed_short = before - len(df)

    # Create dedupe_key for near-duplicate grouping (important)
    df["dedupe_key"] = df["text"].map(make_dedupe_key)

    # Optional: drop exact duplicates based on dedupe_key (keeps 1 per group)
    # This is stricter than drop_duplicates(subset=["text"])
    before = len(df)
    df = df.drop_duplicates(subset=["dedupe_key"])
    removed_dupes = before - len(df)

    df = df.reset_index(drop=True)

    print(f"Removed short texts (<{MIN_CHARS} chars): {removed_short}")
    print(f"Removed duplicate/near-duplicate groups (by dedupe_key): {removed_dupes}")

    return df


def split_data_grouped(df: pd.DataFrame):
    """
    Group-aware split: same dedupe_key can never appear in multiple splits.
    """
    groups = df["dedupe_key"].values
    y = df["label"].values

    # 1) train+val vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    trainval_idx, test_idx = next(gss1.split(df, y, groups))

    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # 2) train vs val
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED)
    train_idx, val_idx = next(
        gss2.split(trainval_df, trainval_df["label"].values, trainval_df["dedupe_key"].values)
    )

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


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

    train_df, val_df, test_df = split_data_grouped(df)
    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    save_data(train_df, val_df, test_df)
    print("Saved to:", OUT_DIR.resolve())
    print("Data preparation completed successfully.")


if __name__ == "__main__":
    main()