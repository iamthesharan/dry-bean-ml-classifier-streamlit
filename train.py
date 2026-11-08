"""
Train 5 classical ML classifiers on the UCI Dry Bean Dataset and persist
everything a Streamlit app needs to score new data without retraining:
  - model/*.joblib          trained sklearn Pipelines (scaler + classifier)
  - model/label_encoder.joblib
  - model/feature_columns.json
  - test_data.csv           held-out stratified test split (raw features + target)
  - metrics_summary.csv     reference metrics computed once during training
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix,
)

RANDOM_STATE = 42
SAMPLE_SIZE = 3000   # stratified subsample of the full ~13.6k-row dataset (keeps SVM probability-calibration fast)
TARGET_COL = "Class"

print("Fetching Dry Bean dataset from UCI repository (id=602)...")
from ucimlrepo import fetch_ucirepo
dry_bean = fetch_ucirepo(id=602)
X_full = dry_bean.data.features.copy()
y_full = dry_bean.data.targets.copy().iloc[:, 0]
print("Full dataset shape:", X_full.shape, "| classes:", sorted(y_full.unique()))

df = X_full.copy()
df[TARGET_COL] = y_full.values

# Stratified subsample to keep training fast while comfortably exceeding
# the assignment's minimum of 500 instances / 12 features.
df_sample, _ = train_test_split(
    df, train_size=SAMPLE_SIZE, stratify=df[TARGET_COL], random_state=RANDOM_STATE,
)
df_sample = df_sample.reset_index(drop=True)
print("Working sample shape:", df_sample.shape)

feature_cols = [c for c in df_sample.columns if c != TARGET_COL]

X = df_sample[feature_cols]
y_raw = df_sample[TARGET_COL]

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
    X, y, y_raw, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
)
print("Train/test sizes:", X_train.shape[0], X_test.shape[0])

# Persist the held-out test split in its ORIGINAL (unscaled) feature space —
# this is exactly what a user re-uploads to the deployed Streamlit app.
test_df = X_test.copy()
test_df[TARGET_COL] = y_test_raw.values
test_df.to_csv("test_data.csv", index=False)
print("Saved test_data.csv ->", test_df.shape)

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]),
    "decision_tree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)),
    ]),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7)),
    ]),
    "naive_bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB()),
    ]),
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ]),
}

display_names = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
    "svm": "Support Vector Machine",
}

rows = []
reports = {}
for key, pipe in models.items():
    print(f"Training {key} ...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    rows.append({
        "ML Model Name": display_names[key],
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
    })

    reports[key] = {
        "classification_report": classification_report(
            y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    joblib.dump(pipe, f"model/{key}.joblib")
    print(f"  acc={acc:.4f} auc={auc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} mcc={mcc:.4f}")

joblib.dump(le, "model/label_encoder.joblib")

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv("metrics_summary.csv", index=False)
print("\n=== Metrics summary ===")
print(metrics_df.to_string(index=False))

with open("model/reports.json", "w") as f:
    json.dump(reports, f, indent=2)

with open("model/feature_columns.json", "w") as f:
    json.dump({
        "feature_columns": feature_cols,
        "target_column": TARGET_COL,
        "classes": le.classes_.tolist(),
    }, f, indent=2)

print("\nDone. Artifacts written: model/*.joblib, model/label_encoder.joblib,")
print("model/feature_columns.json, model/reports.json, test_data.csv, metrics_summary.csv")
