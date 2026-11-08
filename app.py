"""
Streamlit dashboard for the Dry Bean multi-class classification assignment.

Loads five PRE-TRAINED classifiers (Logistic Regression, Decision Tree, KNN,
Naive Bayes, SVM) and scores whatever test CSV the user uploads. Nothing is
retrained here -- the models were fit once in train.py and persisted to
model/*.joblib.
"""
import json

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix,
)

st.set_page_config(page_title="Dry Bean Classifier Dashboard", layout="wide")

MODEL_FILE_PATHS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}


@st.cache_resource
def load_models_and_metadata():
    trained_models = {name: joblib.load(path) for name, path in MODEL_FILE_PATHS.items()}
    label_encoder = joblib.load("model/label_encoder.joblib")
    with open("model/feature_columns.json") as metadata_file:
        model_metadata = json.load(metadata_file)
    return trained_models, label_encoder, model_metadata


trained_models, label_encoder, model_metadata = load_models_and_metadata()
FEATURE_COLUMNS = model_metadata["feature_columns"]
TARGET_COLUMN = model_metadata["target_column"]
CLASS_NAMES = model_metadata["classes"]


def evaluate_model(model_pipeline, feature_matrix, true_labels_encoded):
    """Run inference only (no .fit) and return the 6 required metrics + predictions."""
    predicted_labels = model_pipeline.predict(feature_matrix)
    predicted_probabilities = model_pipeline.predict_proba(feature_matrix)
    metrics = {
        "Accuracy": accuracy_score(true_labels_encoded, predicted_labels),
        "AUC": roc_auc_score(true_labels_encoded, predicted_probabilities, multi_class="ovr", average="macro"),
        "Precision": precision_score(true_labels_encoded, predicted_labels, average="macro", zero_division=0),
        "Recall": recall_score(true_labels_encoded, predicted_labels, average="macro", zero_division=0),
        "F1": f1_score(true_labels_encoded, predicted_labels, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(true_labels_encoded, predicted_labels),
    }
    return metrics, predicted_labels


st.title("🫘 Dry Bean Classifier — Model Comparison Dashboard")
st.caption(
    "Upload held-out test data to evaluate five pre-trained classifiers "
    "(Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) on the UCI Dry Bean dataset. "
    "No model is retrained here — all five are loaded exactly as they were saved after training."
)

with st.sidebar:
    st.header("1. Provide test data")
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
    use_bundled_sample = st.checkbox("Use bundled test_data.csv instead", value=not uploaded_file)
    st.header("2. Pick a model to inspect")
    selected_model_name = st.selectbox("Model", list(trained_models.keys()))
    st.divider()
    st.markdown(
        "**Expected columns:**\n\n"
        f"- {len(FEATURE_COLUMNS)} feature columns: `{', '.join(FEATURE_COLUMNS[:4])}, ...`\n"
        f"- target column: `{TARGET_COLUMN}` with values in {CLASS_NAMES}"
    )

if uploaded_file is not None and not use_bundled_sample:
    test_dataframe = pd.read_csv(uploaded_file)
    data_source_label = uploaded_file.name
elif use_bundled_sample:
    test_dataframe = pd.read_csv("test_data.csv")
    data_source_label = "bundled test_data.csv"
else:
    test_dataframe = None
    data_source_label = None

if test_dataframe is None:
    st.info("Upload a CSV from the sidebar (or tick 'Use bundled test_data.csv') to see results.")
    st.stop()

missing_columns = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in test_dataframe.columns]
if missing_columns:
    st.error(f"Uploaded CSV is missing required column(s): {missing_columns}")
    st.stop()

st.success(f"Loaded **{len(test_dataframe)}** rows from **{data_source_label}**.")

feature_matrix = test_dataframe[FEATURE_COLUMNS]
true_class_names = test_dataframe[TARGET_COLUMN].astype(str)
try:
    true_labels_encoded = label_encoder.transform(true_class_names)
except ValueError as error:
    st.error(f"Target column contains labels the models were not trained on: {error}")
    st.stop()

# ---- Metrics for ALL five models on this uploaded data ----
st.subheader("Results of all 5 models on your uploaded test data")
comparison_rows = []
predictions_by_model = {}
for model_name, model_pipeline in trained_models.items():
    model_metrics, predicted_labels = evaluate_model(model_pipeline, feature_matrix, true_labels_encoded)
    predictions_by_model[model_name] = predicted_labels
    comparison_rows.append({
        "ML Model Name": model_name,
        **{metric_name: round(metric_value, 4) for metric_name, metric_value in model_metrics.items()},
    })

comparison_table = pd.DataFrame(comparison_rows).set_index("ML Model Name")
st.dataframe(comparison_table.style.highlight_max(axis=0, color="#c6f6c6"), use_container_width=True)

best_model_name = comparison_table["F1"].idxmax()
st.caption(f"Highest macro-F1 on this data: **{best_model_name}**")

# ---- Detailed view for the model selected in the sidebar ----
st.subheader(f"Detailed view — {selected_model_name}")
predicted_labels = predictions_by_model[selected_model_name]

metrics_column, plot_column = st.columns([1, 1.2])

with metrics_column:
    st.markdown("**Evaluation metrics**")
    selected_model_metrics = comparison_table.loc[selected_model_name]
    accuracy_box, auc_box, mcc_box = st.columns(3)
    accuracy_box.metric("Accuracy", f"{selected_model_metrics['Accuracy']:.4f}")
    auc_box.metric("AUC", f"{selected_model_metrics['AUC']:.4f}")
    mcc_box.metric("MCC", f"{selected_model_metrics['MCC']:.4f}")
    precision_box, recall_box, f1_box = st.columns(3)
    precision_box.metric("Precision", f"{selected_model_metrics['Precision']:.4f}")
    recall_box.metric("Recall", f"{selected_model_metrics['Recall']:.4f}")
    f1_box.metric("F1", f"{selected_model_metrics['F1']:.4f}")

    st.markdown("**Classification report**")
    classification_report_dict = classification_report(
        true_labels_encoded, predicted_labels, target_names=label_encoder.classes_,
        output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(classification_report_dict).transpose().round(3), use_container_width=True)

with plot_column:
    st.markdown("**Confusion matrix**")
    confusion_matrix_values = confusion_matrix(true_labels_encoded, predicted_labels)
    confusion_matrix_figure, confusion_matrix_axis = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        confusion_matrix_values, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=confusion_matrix_axis,
    )
    confusion_matrix_axis.set_xlabel("Predicted")
    confusion_matrix_axis.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(confusion_matrix_figure)

with st.expander("Preview of uploaded data"):
    st.dataframe(test_dataframe.head(20), use_container_width=True)
