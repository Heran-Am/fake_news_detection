import pandas as pd
import re

print("EDA started ✅")

fake_df = pd.read_csv("data/raw/fake.csv")
true_df = pd.read_csv("data/raw/true.csv")

text_col = "text" if "text" in fake_df.columns else fake_df.columns[0]


def uppercase_ratio(s: str) -> float:
    s = str(s)
    letters = sum(ch.isalpha() for ch in s)
    if letters == 0:
        return 0.0
    upper = sum(ch.isupper() for ch in s)
    return upper / letters

def allcaps_word_ratio(s: str) -> float:
    words = re.findall(r"[A-Za-z]+", str(s))
    if not words:
        return 0.0
    allcaps = sum(1 for w in words if len(w) >= 2 and w.isupper())
    return allcaps / len(words)

def punct_count(s: str, chars="!?") -> int:
    s = str(s)
    return sum(s.count(c) for c in chars)

for name, df in [("FAKE", fake_df), ("TRUE", true_df)]:
    sample = df[text_col].astype(str)
    print(f"\n=== {name} ===")
    print("Rows:", len(df))
    print("Avg text length:", sample.str.len().mean())
    print("Avg uppercase ratio:", sample.map(uppercase_ratio).mean())
    print("Avg ALLCAPS word ratio:", sample.map(allcaps_word_ratio).mean())
    print("Avg !/? count:", sample.map(lambda x: punct_count(x, '!?')).mean())
    print("Columns:", fake_df.columns.tolist())
    print("Using column for features:", text_col)