"""Streamlit frontend for the fake news detection API."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"
APP_TITLE = "Fake News Detector"
APP_ICON = "📰"
LAYOUT = "wide"
SIDEBAR_TITLE = "About This App"
SIDEBAR_DESCRIPTION = (
    "A production-style fake-news demo that scores articles with a trained Logistic Regression model."
)
TECH_STACK = [
    "Python",
    "pandas",
    "numpy",
    "scikit-learn",
    "nltk",
    "joblib",
    "pydantic v2",
    "FastAPI",
    "uvicorn",
    "Streamlit",
    "seaborn",
    "matplotlib",
    "requests",
]
CONFIDENCE_EXPLANATION = (
    "Confidence is the model's probability assigned to the predicted class, so higher values mean stronger model certainty."
)
PREDICT_ENDPOINT = "/predict"
BATCH_ENDPOINT = "/predict/batch"
METRICS_PATH = Path("reports/evaluation_metrics.json")
VAL_CONFUSION_MATRIX_PATH = Path("reports/confusion_matrix_val.png")
TEST_CONFUSION_MATRIX_PATH = Path("reports/confusion_matrix_test.png")
ROC_CURVE_PATH = Path("reports/roc_curve.png")
SINGLE_PAGE = "Single Prediction"
BATCH_PAGE = "Batch Prediction"
INFO_PAGE = "Model Info"
MAX_BATCH_ROWS = 50
REQUEST_TIMEOUT_SECONDS = 30
REAL_LABEL = 0
FAKE_LABEL = 1
REAL_NAME = "REAL"
FAKE_NAME = "FAKE"
GREEN_BACKGROUND = "#e8f5e9"
RED_BACKGROUND = "#ffebee"
GREEN_TEXT = "#1b5e20"
RED_TEXT = "#b71c1c"
RESULT_CARD_STYLE = "padding: 1.25rem; border-radius: 1rem; margin-bottom: 1rem;"
CSV_DOWNLOAD_NAME = "fake_news_predictions.csv"
TITLE_FIELD = "title"
TEXT_FIELD = "text"
PREDICTION_FIELD = "prediction"
LABEL_FIELD = "label"
CONFIDENCE_FIELD = "confidence"
REAL_PROBABILITY_FIELD = "real_probability"
FAKE_PROBABILITY_FIELD = "fake_probability"
SUMMARY_FIELD = "summary"
PREDICTIONS_FIELD = "predictions"
TOTAL_COUNT_FIELD = "total_count"
FAKE_COUNT_FIELD = "fake_count"
REAL_COUNT_FIELD = "real_count"
ACCURACY_FIELD = "accuracy"
PRECISION_FIELD = "precision_weighted"
RECALL_FIELD = "recall_weighted"
F1_FIELD = "f1_weighted"
ROC_AUC_FIELD = "roc_auc"
VAL_FIELD = "val"
TEST_FIELD = "test"


def load_metrics() -> dict[str, Any]:
    """Load saved evaluation metrics if available.

    Returns:
        Metrics dictionary or an empty dictionary if the file is unavailable.
    """

    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def style_batch_dataframe(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply color styling to batch prediction rows.

    Args:
        frame: Prediction dataframe.

    Returns:
        A styled dataframe.
    """

    def highlight_row(row: pd.Series) -> list[str]:
        background = GREEN_BACKGROUND if row[LABEL_FIELD] == REAL_NAME else RED_BACKGROUND
        return [f"background-color: {background}"] * len(row)

    return frame.style.apply(highlight_row, axis=1)


def render_sidebar() -> None:
    """Render the application sidebar."""

    st.sidebar.title(SIDEBAR_TITLE)
    st.sidebar.markdown(SIDEBAR_DESCRIPTION)
    st.sidebar.subheader("Tech Stack")
    for item in TECH_STACK:
        st.sidebar.markdown(f"- {item}")
    st.sidebar.subheader("Confidence Score")
    st.sidebar.write(CONFIDENCE_EXPLANATION)



def post_json(endpoint: str, payload: Any) -> requests.Response:
    """Send a JSON request to the backend.

    Args:
        endpoint: API endpoint path.
        payload: JSON-serializable request body.

    Returns:
        A requests response.
    """

    return requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)



def render_single_prediction_page() -> None:
    """Render the single-article prediction page."""

    st.subheader("Single Prediction")
    title = st.text_area("Article Title", height=120)
    article_text = st.text_area("Article Text", height=260)
    if st.button("Analyze", type="primary"):
        try:
            with st.spinner("Analyzing article..."):
                response = post_json(PREDICT_ENDPOINT, {TITLE_FIELD: title, TEXT_FIELD: article_text})
                response.raise_for_status()
                result = response.json()
        except requests.RequestException as exc:
            st.error(f"Prediction request failed: {exc}")
            return
        except ValueError as exc:
            st.error(f"Invalid JSON response from API: {exc}")
            return

        label = int(result[LABEL_FIELD])
        prediction = str(result[PREDICTION_FIELD])
        confidence = float(result[CONFIDENCE_FIELD])
        real_probability = float(result[REAL_PROBABILITY_FIELD])
        fake_probability = float(result[FAKE_PROBABILITY_FIELD])
        background = GREEN_BACKGROUND if label == REAL_LABEL else RED_BACKGROUND
        text_color = GREEN_TEXT if label == REAL_LABEL else RED_TEXT
        display_label = REAL_NAME if label == REAL_LABEL else FAKE_NAME
        st.markdown(
            f"""
            <div style="{RESULT_CARD_STYLE} background: {background}; color: {text_color};">
                <h2 style="margin: 0;">{display_label}</h2>
                <p style="margin: 0.25rem 0 0 0; font-size: 1.05rem;">Prediction: {prediction}</p>
                <p style="margin: 0.25rem 0 0 0; font-size: 1.05rem;">Confidence: {confidence:.3f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(confidence)
        chart_frame = pd.DataFrame(
            {
                "Class": [REAL_NAME, FAKE_NAME],
                "Probability": [real_probability, fake_probability],
            }
        )
        chart_figure, chart_axis = plt.subplots(figsize=(7, 2.8))
        chart_axis.barh(chart_frame["Class"], chart_frame["Probability"], color=["#2e7d32", "#c62828"])
        chart_axis.set_xlim(0, 1)
        chart_axis.set_xlabel("Probability")
        chart_axis.set_title("Prediction Probabilities")
        st.pyplot(chart_figure, clear_figure=True)



def render_batch_prediction_page() -> None:
    """Render the batch prediction page."""

    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV with title and text columns", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to score a batch of articles.")
        return

    try:
        input_frame = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to read the uploaded CSV: {exc}")
        return

    missing_columns = sorted({TITLE_FIELD, TEXT_FIELD} - set(input_frame.columns))
    if missing_columns:
        st.error(f"The uploaded file is missing required columns: {', '.join(missing_columns)}")
        return

    preview_frame = input_frame[[TITLE_FIELD, TEXT_FIELD]].head(MAX_BATCH_ROWS).copy()
    st.write("Preview of rows to be sent to the API:")
    st.dataframe(preview_frame, use_container_width=True)

    if st.button("Run Batch Prediction", type="primary"):
        records = preview_frame.to_dict(orient="records")
        try:
            with st.spinner("Scoring batch..."):
                response = post_json(BATCH_ENDPOINT, records)
                response.raise_for_status()
                result = response.json()
        except requests.RequestException as exc:
            st.error(f"Batch prediction request failed: {exc}")
            return
        except ValueError as exc:
            st.error(f"Invalid JSON response from API: {exc}")
            return

        predictions = pd.DataFrame(result[PREDICTIONS_FIELD])
        display_frame = pd.concat([preview_frame.reset_index(drop=True), predictions], axis=1)
        display_frame[LABEL_FIELD] = display_frame[LABEL_FIELD].map({REAL_LABEL: REAL_NAME, FAKE_LABEL: FAKE_NAME})
        st.write("Batch Results")
        st.dataframe(style_batch_dataframe(display_frame), use_container_width=True)
        csv_bytes = display_frame.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results as CSV",
            data=csv_bytes,
            file_name=CSV_DOWNLOAD_NAME,
            mime="text/csv",
        )
        summary = result[SUMMARY_FIELD]
        st.write(
            f"Total: {summary[TOTAL_COUNT_FIELD]} | Fake: {summary[FAKE_COUNT_FIELD]} | Real: {summary[REAL_COUNT_FIELD]}"
        )



def render_metrics_columns(split_name: str, metrics: dict[str, Any]) -> None:
    """Render a metric block for a single dataset split.

    Args:
        split_name: Split label.
        metrics: Metrics dictionary.
    """

    st.markdown(f"### {split_name.title()}")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Accuracy", f"{metrics[ACCURACY_FIELD]:.3f}")
    metric_columns[1].metric("Precision", f"{metrics[PRECISION_FIELD]:.3f}")
    metric_columns[2].metric("Recall", f"{metrics[RECALL_FIELD]:.3f}")
    metric_columns[3].metric("F1", f"{metrics[F1_FIELD]:.3f}")
    st.metric("ROC-AUC", f"{metrics[ROC_AUC_FIELD]:.3f}")



def render_model_info_page() -> None:
    """Render the model information page."""

    st.subheader("Model Info")
    metrics = load_metrics()
    if not metrics:
        st.warning("Evaluation metrics are not available yet. Run the training pipeline first.")
    else:
        left_column, right_column = st.columns(2)
        with left_column:
            render_metrics_columns(VAL_FIELD, metrics[VAL_FIELD])
        with right_column:
            render_metrics_columns(TEST_FIELD, metrics[TEST_FIELD])

    st.markdown("### Artifacts")
    if VAL_CONFUSION_MATRIX_PATH.exists():
        st.image(str(VAL_CONFUSION_MATRIX_PATH), caption="Validation confusion matrix", use_column_width=True)
    if TEST_CONFUSION_MATRIX_PATH.exists():
        st.image(str(TEST_CONFUSION_MATRIX_PATH), caption="Test confusion matrix", use_column_width=True)
    if ROC_CURVE_PATH.exists():
        st.image(str(ROC_CURVE_PATH), caption="ROC curve", use_column_width=True)



def main() -> None:
    """Run the Streamlit application."""

    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=LAYOUT)
    render_sidebar()
    page = st.sidebar.radio("Navigation", [SINGLE_PAGE, BATCH_PAGE, INFO_PAGE])
    if page == SINGLE_PAGE:
        render_single_prediction_page()
    elif page == BATCH_PAGE:
        render_batch_prediction_page()
    else:
        render_model_info_page()


if __name__ == "__main__":
    main()
