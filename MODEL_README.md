# Model README

## Artifacts

| File | Description |
| --- | --- |
| `fraud_detection_rf_model.pkl` | Saved `RandomForestClassifier` trained in `main.ipynb` |
| `scaler.pkl` | Saved `RobustScaler` object from the notebook preprocessing step |

The Random Forest was trained with:

```text
n_estimators=100
random_state=42
n_jobs=-1
```

Training data was balanced with SMOTE after an 80/20 stratified train/test
split.

## Final Threshold

Use this tuned decision threshold for the saved model:

```text
0.80
```

At this threshold, the notebook reports:

| Metric | Value |
| --- | ---: |
| Precision | 0.9740 |
| Recall | 0.7653 |
| F1 | 0.8571 |
| False positives | 2 |
| False negatives | 23 |

## Input Schema

The model expects already-preprocessed numeric features in this exact order:

```text
V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28,
scaled_Amount, scaled_Time
```

`Amount` and `Time` are not passed directly to the model. The notebook creates
`scaled_Amount` and `scaled_Time`, then drops the original columns.

## Prediction Example

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
is_fraud = fraud_probability >= MODEL_THRESHOLD

print(f"Fraud probability: {fraud_probability:.4f}")
print("FRAUD" if is_fraud else "LEGITIMATE")
```

## Important Preprocessing Caveat

In `main.ipynb`, the same `RobustScaler` variable is fit once for `Amount` and
then fit again for `Time`. The saved `scaler.pkl` therefore reflects the final
fit from the notebook, not a complete reusable preprocessing pipeline for both
raw columns.

For reliable raw-input inference, prefer one of these approaches:

- Save separate scalers for `Amount` and `Time`.
- Save a single `ColumnTransformer` or `Pipeline` that includes preprocessing
  and the classifier.
- Pass only already-preprocessed values to the current saved model.

## Production Notes

This artifact is suitable for reproducing notebook results and lightweight
experiments. Before production use, add model versioning, preprocessing
versioning, threshold governance, drift monitoring, and validation on data that
matches the deployment environment.
