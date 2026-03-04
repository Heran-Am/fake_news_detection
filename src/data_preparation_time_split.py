import re
import html
import unicodedata
from pathlib import Path
from hashlib import blake2b

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed_time")

SEED = 42
MIN_CHARS = 20

# Reuters leakage patterns
REUTERS_TOKEN_RE = re.compile(r"\breuters\b", flags=re.IGNORECASE)
REUTERS_PREFIX_RE = re.compile(r"^\s*\(?reuters\)?\s*[-–—:]\s*", flags=re.IGNORECASE)


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = REUTERS_PREFIX_RE.sub("", text)
    text = REUTERS_TOKEN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_dedupe(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_dedupe_key(text: str) -> str:
    norm = normalize_for_dedupe(text)
    return blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()


def read_raw(name_a: str, name_b: str) -> pd.DataFrame:
    a = RAW_DIR / name_a
    b = RAW_DIR / name_b
    if a.exists():
        return pd.read_csv(a)
    if b.exists():
        return pd.read_csv(b)
    raise FileNotFoundError(f"Could not find {a} or {b}")


def parse_date_series(s: pd.Series) -> pd.Series:
    """
    Parse dates robustly across pandas versions.
    Unparseable rows become NaT.
    """
    # First attempt: generic parse
    dt = pd.to_datetime(s, errors="coerce")

    # If many are NaT, try a second pass with common cleanup
    if dt.isna().mean() > 0.2:
        cleaned = (
            s.astype(str)
             .str.replace("GMT", "", regex=False)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip()
        )
        dt2 = pd.to_datetime(cleaned, errors="coerce")
        # keep whichever parses better
        if dt2.isna().mean() < dt.isna().mean():
            dt = dt2

    return dt

def build_df() -> pd.DataFrame:
    fake = read_raw("Fake.csv", "fake.csv")
    true = read_raw("True.csv", "true.csv")

    fake["label"] = 1
    true["label"] = 0

    df = pd.concat([fake, true], ignore_index=True)

    # Keep date for splitting, drop subject (leakage-ish)
    if "subject" in df.columns:
        df = df.drop(columns=["subject"])

    if "date" not in df.columns:
        raise ValueError("No 'date' column found, cannot time-split.")

    df["date_parsed"] = parse_date_series(df["date"])
    df = df.dropna(subset=["text", "date_parsed"]).copy()

    df["text"] = df["text"].astype(str).map(normalize_text)
    df = df[df["text"].str.len() >= MIN_CHARS].copy()

    # Near-duplicate key
    df["dedupe_key"] = df["text"].map(make_dedupe_key)

    # Drop near-duplicate groups (keep 1 per group)
    df = df.drop_duplicates(subset=["dedupe_key"]).reset_index(drop=True)

    return df


def time_split(df: pd.DataFrame):
    """
    Train on older data, test on newer data.
    We take the newest 20% (by date) as test.
    Then split the older 80% into train/val with group-safety.
    """
    df = df.sort_values("date_parsed").reset_index(drop=True)

    n = len(df)
    cut = int(n * 0.8)

    trainval_df = df.iloc[:cut].reset_index(drop=True)   # older 80%
    test_df = df.iloc[cut:].reset_index(drop=True)       # newest 20%

    # Group-safe split train/val within trainval
    gss = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED)  # 10% of total approx
    train_idx, val_idx = next(gss.split(trainval_df, trainval_df["label"], trainval_df["dedupe_key"]))

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Don’t feed date columns to models; keep dedupe_key only if you want debugging
    drop_cols = [c for c in ["date", "date_parsed"] if c in train_df.columns]

    train_df.drop(columns=drop_cols, errors="ignore").to_csv(OUT_DIR / "train.csv", index=False)
    val_df.drop(columns=drop_cols, errors="ignore").to_csv(OUT_DIR / "val.csv", index=False)
    test_df.drop(columns=drop_cols, errors="ignore").to_csv(OUT_DIR / "test.csv", index=False)

    # Save a small metadata file for README/debugging
    meta = {
        "min_chars": MIN_CHARS,
        "n_total": int(len(train_df) + len(val_df) + len(test_df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "train_date_range": [str(train_df["date_parsed"].min()), str(train_df["date_parsed"].max())],
        "test_date_range": [str(test_df["date_parsed"].min()), str(test_df["date_parsed"].max())],
        "label_counts_train": train_df["label"].value_counts().to_dict(),
        "label_counts_test": test_df["label"].value_counts().to_dict(),
    }
    pd.Series(meta).to_json(OUT_DIR / "split_meta.json", indent=2)


def main():
    print("Building dataset for TIME split...")
    df = build_df()
    print("After cleaning:", df.shape)
    print("Label counts:\n", df["label"].value_counts())

    train_df, val_df, test_df = time_split(df)

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)
    print("Train date range:", train_df["date_parsed"].min(), "->", train_df["date_parsed"].max())
    print("Test date range:", test_df["date_parsed"].min(), "->", test_df["date_parsed"].max())

    save_splits(train_df, val_df, test_df)
    print("Saved splits to:", OUT_DIR.resolve())
    print("Done.")


if __name__ == "__main__":
    main()