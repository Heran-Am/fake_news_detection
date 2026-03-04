import re
import pandas as pd
from hashlib import blake2b

def norm_for_hash(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)   # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def hash_text(s: str) -> str:
    return blake2b(norm_for_hash(s).encode("utf-8"), digest_size=16).hexdigest()

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

train_hashes = set(train["text"].map(hash_text))
test_hashes = set(test["text"].map(hash_text))

overlap = len(train_hashes & test_hashes)
print("Train unique hashes:", len(train_hashes))
print("Test unique hashes:", len(test_hashes))
print("Overlap hashes:", overlap)
print("Overlap rate (test):", overlap / max(len(test_hashes), 1))