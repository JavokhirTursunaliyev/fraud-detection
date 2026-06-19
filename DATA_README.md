# Dataset README

## Source

This project expects the Kaggle Credit Card Fraud Detection dataset:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download the dataset from Kaggle and place the CSV file in the repository root:

```text
creditcard.csv
```

The CSV is intentionally ignored by git through `.gitignore`.

## Shape

The notebook uses the full dataset:

| Item | Value |
| --- | ---: |
| Transactions | 284,807 |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Fraud rate | 0.17% |
| Class imbalance | 577:1 |

## Columns

| Column | Description |
| --- | --- |
| `Time` | Seconds elapsed between each transaction and the first transaction in the dataset |
| `Amount` | Transaction amount |
| `V1` to `V28` | PCA-transformed anonymized features |
| `Class` | Target label, where `0` is legitimate and `1` is fraud |

## Local Setup

Expected file layout:

```text
fraud-det/
+-- creditcard.csv
+-- main.ipynb
+-- requirements.txt
+-- README.md
```

Quick validation in Python:

```python
import pandas as pd

df = pd.read_csv("creditcard.csv")
print(df.shape)
print(df["Class"].value_counts())
```

Expected output:

```text
(284807, 31)
Class
0    284315
1       492
Name: count, dtype: int64
```

## Handling Notes

- Do not commit `creditcard.csv` to git.
- Check Kaggle for the dataset license and usage terms before redistributing.
- Keep train/test splitting stratified because the fraud class is very rare.
- Apply oversampling techniques such as SMOTE only to the training set.
