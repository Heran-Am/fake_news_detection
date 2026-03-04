import pandas as pd

df = pd.read_csv("data/processed/train.csv")
contains_reuters = df["text"].str.contains(r"\breuters\b", case=False, na=False)

print("Reuters rate overall:", contains_reuters.mean())
print(
    "Reuters rate by label:\n",
    df.groupby("label")["text"].apply(
        lambda s: s.str.contains(r"\breuters\b", case=False, na=False).mean()
    ),
)