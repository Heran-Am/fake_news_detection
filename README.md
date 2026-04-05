# 📰 Fake News Detection (NLP) — TF-IDF Baseline → DistilBERT (Time-Split Benchmark)

A portfolio-grade NLP project that builds a **fake news detection system** in two phases:

1) **Classical baseline**: **TF-IDF + Logistic Regression**  
2) **Transformer upgrade**: Fine-tune **DistilBERT** and compare results

This repo goes beyond “just training a model” by focusing on **evaluation integrity**:
- detecting & removing dataset shortcuts (**Reuters leakage**),
- removing duplicate / near-duplicate overlap,
- and evaluating on a **time-based split** to test generalization into the future.

---

## what this project demonstrates

- Clean and reproducible ML project structure (scripts, metrics artifacts, inference)
- Text preprocessing + leakage mitigation in real datasets
- Baseline building with TF-IDF and Logistic Regression
- Transformer fine-tuning (DistilBERT) with Hugging Face
- Proper evaluation under class imbalance using **F1(fake)**, **Macro F1**, and **PR-AUC**
- Understanding why “too-perfect” scores can happen and how to stress test them

---

## Dataset

Kaggle “Fake and True News” dataset:
- `Fake.csv`
- `True.csv`

**Label mapping**
- `1 = fake`
- `0 = true`

> ⚠️ Important: this dataset contains strong source/style artifacts. This project *explicitly detects and mitigates them* instead of blindly reporting inflated results.

---

## 🔍 Leakage & Data Quality Fixes 

### 1) Reuters leakage (major shortcut)
We measured how often the token **“Reuters”** appears by class and found it was almost perfectly aligned with TRUE news.

Fix:
- removed Reuters wire-prefix/token during preprocessing.

Script:
- `src/check_reuters.py`

### 2) Duplicate / near-duplicate overlap across splits
Even after dropping exact duplicates, we detected near-identical texts appearing in both train and test.

Fix:
- created a strong normalized hash key (`dedupe_key`)
- ensured **zero overlap** after cleaning/splitting

Script:
- `src/check_leakage_overlap.py`

### 3) Harder evaluation: time-based split
Random splits can be overly optimistic. We created a **time split**:

- Train/Val: **2015-05-01 → 2017-10-25**
- Test: **2017-10-25 → 2017-12-31**

Script:
- `src/data_preparation_time_split.py`

---

## 📈 Metrics used (why these matter)

Because the time-split test set can be **highly imbalanced** (few fake examples), we report:

- **F1(fake)**: F1 score on FAKE class (`label=1`)  
- **Macro F1**: average F1 across both classes (fair under imbalance)  
- **PR-AUC**: Precision–Recall AUC for FAKE class (very informative when positives are rare)

---

## 🏁 Results

### Random split (cleaned + overlap-safe)
| Split | Model | F1(fake) | Macro F1 | PR-AUC |
|---|---|---:|---:|---:|
| Random | TF-IDF + Logistic Regression | 0.9837 | 0.9852 | 0.9982 |

### Time split (recommended benchmark)
| Split | Model | F1(fake) | Macro F1 | PR-AUC |
|---|---|---:|---:|---:|
| Time | TF-IDF + Logistic Regression | 0.9463 | 0.9721 | 0.9957 |
| Time | DistilBERT (2 epochs, max_len=128) | **0.9893** | **0.9945** | **0.9996** |

**Key takeaway:** On the harder time split, DistilBERT improved **F1(fake)** from **0.9463 → 0.9893**.

---

##  Threshold tuning (important lesson)
We also tested threshold tuning on validation (`tune_threshold.py`).

- Best threshold on **VAL** did **not** improve performance on the **future TEST** window.
- This is a real-world lesson: **thresholds may not transfer under distribution shift**.

Script:
- `src/tune_threshold.py`

---
## Project Structure

- `src/data_preparation.py`  
  Preprocessing for random split (cleaning, Reuters removal, dedupe-safe)

- `src/data_preparation_time_split.py`  
  Preprocessing for time-based split (train older → test newer)

- `src/check_reuters.py`  
  Measures Reuters leakage rate by label

- `src/check_leakage_overlap.py`  
  Measures train/test overlap after normalized hashing

- `src/train_baseline.py`  
  TF-IDF + Logistic Regression training + metrics saving (`--run_name` supported)

- `src/tune_threshold.py`  
  Threshold tuning experiment (shows drift: best val threshold may not transfer to future test)

- `src/train_bert.py`  
  DistilBERT fine-tuning (Hugging Face Trainer)

- `src/predict.py`  
  Inference script for baseline or BERT model

---

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
Run: Data Preparation


