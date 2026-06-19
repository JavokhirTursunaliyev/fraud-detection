# Credit Card Fraud Detection

Machine learning project for detecting fraudulent credit card transactions with
classical models, SMOTE oversampling, and decision-threshold tuning.

The main workflow lives in `main.ipynb`. It loads the Kaggle credit card fraud
dataset, performs exploratory analysis, preprocesses `Amount` and `Time`, trains
multiple classifiers, tunes the Random Forest threshold, and saves the final
model artifacts.

## Project Contents

| Path | Purpose |
| --- | --- |
| `main.ipynb` | End-to-end notebook for EDA, preprocessing, training, evaluation, and export |
| `requirements.txt` | Python dependencies needed to run the notebook |
| `fraud_detection_rf_model.pkl` | Saved Random Forest classifier |
| `scaler.pkl` | Saved preprocessing scaler artifact from the notebook |
| `DATA_README.md` | Dataset source, expected schema, and local setup notes |
| `MODEL_README.md` | Model artifact details, input schema, threshold, and inference notes |
| `creditcard.csv` | Local dataset file, ignored by git because of size |

## Dataset

The project uses the Kaggle Credit Card Fraud Detection dataset:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Expected local file:

```text
creditcard.csv
```

The dataset contains 284,807 transactions, with 492 fraudulent transactions
and 284,315 legitimate transactions. See `DATA_README.md` for schema and setup
details.

## Pipeline

1. Load `creditcard.csv`.
2. Explore class imbalance, transaction amount/time distributions, feature
   correlations, and fraud-related feature patterns.
3. Scale `Amount` and `Time` using `RobustScaler`, then drop the original
   columns.
4. Split data into stratified train/test sets with an 80/20 split.
5. Apply SMOTE to the training set only.
6. Train and compare Logistic Regression, Random Forest, and Gradient Boosting.
7. Tune the Random Forest decision threshold with the precision-recall curve.
8. Save the Random Forest model and scaler artifacts.

## Results

Model comparison from the notebook:

| Model | ROC-AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9712 | 0.0591 | 0.9184 | 0.1110 |
| Random Forest | 0.9626 | 0.8791 | 0.8163 | 0.8466 |
| Gradient Boosting | 0.9807 | 0.1085 | 0.9082 | 0.1939 |

Final Random Forest threshold tuning:

| Threshold | Precision | Recall | F1 | False Positives | False Negatives |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.8791 | 0.8163 | 0.8466 | 12 | 17 |
| 0.80 | 0.9740 | 0.7653 | 0.8571 | 2 | 23 |

At threshold `0.80`, the model caught 75 of 98 fraud cases in the test set and
blocked 2 legitimate transactions out of 56,864.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download `creditcard.csv` from Kaggle and place it in the repository root.

Run the notebook:

```bash
jupyter notebook main.ipynb
```

## Using the Saved Model

See `MODEL_README.md` for the model input schema and caveats. The model expects
preprocessed columns in this order:

```text
V1, V2, ..., V28, scaled_Amount, scaled_Time
```

Use the tuned threshold:

```text
0.80
```

Minimal prediction example with an already-preprocessed transaction:

```python
import joblib
import pandas as pd

MODEL_THRESHOLD = 0.80
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + [
    "scaled_Amount",
    "scaled_Time",
]

model = joblib.load("fraud_detection_rf_model.pkl")

transaction = pd.DataFrame(
    [[...]],  # V1..V28, scaled_Amount, scaled_Time
    columns=FEATURE_COLUMNS,
)

fraud_probability = model.predict_proba(transaction)[0, 1]
prediction = int(fraud_probability >= MODEL_THRESHOLD)

print("FRAUD" if prediction else "LEGITIMATE")
```

## Notes

This is a learning and experimentation project, not a production fraud system.
For production use, wrap preprocessing and modeling in a single versioned
pipeline, validate on time-based holdout data, monitor drift, and choose the
decision threshold based on business costs for false positives and false
negatives.
