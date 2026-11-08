"""
Streamlit dashboard for the Dry Bean multi-class classification assignment.

Loads five PRE-TRAINED classifiers (Logistic Regression, Decision Tree, KNN,
Naive Bayes, SVM) and scores whatever test CSV the user uploads. Nothing is
retrained here -- the models were fit once in train.py and persisted to
model/*.joblib.
"""
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix,
)

st.set_page_config(page_title="Dry Bean Classifier Dashboard", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}


@st.cache_resource
def load_artifacts():
    models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
    label_encoder = joblib.load("model/label_encoder.joblib")
    with open("model/feature_columns.json") as f:
        meta = json.load(f)
    return models, label_encoder, meta


models, label_encoder, meta = load_artifacts()
FEATURE_COLS = meta["feature_columns"]
TARGET_COL = meta["target_column"]
CLASS_NAMES = meta["classes"]


def score_model(pipe, X, y_true_enc):
    y_pred = pipe.predict(X)
    y_proba = pipe.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y_true_enc, y_pred),
        "AUC": roc_auc_score(y_true_enc, y_proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_true_enc, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true_enc, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true_enc, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true_enc, y_pred),
    }, y_pred


st.title("🫘 Dry Bean Classifier — Model Comparison Dashboard")
st.caption(
    "Upload held-out test data to evaluate five pre-trained classifiers "
    "(Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) on the UCI Dry Bean dataset. "
    "No model is retrained here — all five are loaded exactly as they were saved after training."
)

with st.sidebar:
    st.header("1. Provide test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    use_sample = st.checkbox("Use bundled test_data.csv instead", value=not uploaded)
    st.header("2. Pick a model to inspect")
    chosen_model = st.selectbox("Model", list(models.keys()))
    st.divider()
    st.markdown(
        "**Expected columns:**\n\n"
        f"- {len(FEATURE_COLS)} feature columns: `{', '.join(FEATURE_COLS[:4])}, ...`\n"
        f"- target column: `{TARGET_COL}` with values in {CLASS_NAMES}"
    )

if uploaded is not None and not use_sample:
    df = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    df = pd.read_csv("test_data.csv")
    source = "bundled test_data.csv"
else:
    df = None
    source = None

if df is None:
    st.info("Upload a CSV from the sidebar (or tick 'Use bundled test_data.csv') to see results.")
    st.stop()

missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing:
    st.error(f"Uploaded CSV is missing required column(s): {missing}")
    st.stop()

st.success(f"Loaded **{len(df)}** rows from **{source}**.")

X = df[FEATURE_COLS]
y_raw = df[TARGET_COL].astype(str)
try:
    y_true_enc = label_encoder.transform(y_raw)
except ValueError as e:
    st.error(f"Target column contains labels the models were not trained on: {e}")
    st.stop()

# ---- Metrics for ALL five models on this uploaded data ----
st.subheader("Results of all 5 models on your uploaded test data")
all_rows = []
all_preds = {}
for name, pipe in models.items():
    metrics, y_pred = score_model(pipe, X, y_true_enc)
    all_preds[name] = y_pred
    all_rows.append({"ML Model Name": name, **{k: round(v, 4) for k, v in metrics.items()}})

comparison_df = pd.DataFrame(all_rows).set_index("ML Model Name")
st.dataframe(comparison_df.style.highlight_max(axis=0, color="#c6f6c6"), use_container_width=True)

best_model = comparison_df["F1"].idxmax()
st.caption(f"Highest macro-F1 on this data: **{best_model}**")

# ---- Detailed view for the model selected in the sidebar ----
st.subheader(f"Detailed view — {chosen_model}")
y_pred = all_preds[chosen_model]

col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("**Evaluation metrics**")
    metric_vals = comparison_df.loc[chosen_model]
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{metric_vals['Accuracy']:.4f}")
    m2.metric("AUC", f"{metric_vals['AUC']:.4f}")
    m3.metric("MCC", f"{metric_vals['MCC']:.4f}")
    m4, m5, m6 = st.columns(3)
    m4.metric("Precision", f"{metric_vals['Precision']:.4f}")
    m5.metric("Recall", f"{metric_vals['Recall']:.4f}")
    m6.metric("F1", f"{metric_vals['F1']:.4f}")

    st.markdown("**Classification report**")
    report = classification_report(
        y_true_enc, y_pred, target_names=label_encoder.classes_,
        output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

with col2:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_true_enc, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

with st.expander("Preview of uploaded data"):
    st.dataframe(df.head(20), use_container_width=True)
