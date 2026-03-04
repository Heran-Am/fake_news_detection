import re
import numpy as np
import pandas as pd

PUNCT_RE = re.compile(r"[^\w\s]")
ALLCAPS_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")
DIGIT_RE = re.compile(r"\d")
EXCL_RE = re.compile(r"!")
Q_RE = re.compile(r"\?")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

def compute_style_features(text_series: pd.Series) -> np.ndarray:
    feats = []
    for t in text_series.fillna("").astype(str):
        if len(t) == 0:
            feats.append([0]*10)
            continue

        n_chars = len(t)
        n_upper = sum(1 for c in t if c.isupper())
        uppercase_ratio = n_upper / n_chars

        words = t.split()
        n_words = len(words)
        allcaps_words = len(ALLCAPS_WORD_RE.findall(t))
        allcaps_ratio = allcaps_words / max(n_words, 1)

        punct_count = len(PUNCT_RE.findall(t))
        punct_ratio = punct_count / n_chars

        excl = len(EXCL_RE.findall(t))
        ques = len(Q_RE.findall(t))
        digits = len(DIGIT_RE.findall(t))
        digit_ratio = digits / n_chars

        urls = len(URL_RE.findall(t))

        avg_word_len = (sum(len(w) for w in words) / max(n_words, 1))

        feats.append([
            n_words,
            n_chars,
            uppercase_ratio,
            allcaps_ratio,
            punct_count,
            punct_ratio,
            excl,
            ques,
            digit_ratio,
            avg_word_len,
        ])

    return np.array(feats, dtype=np.float32)