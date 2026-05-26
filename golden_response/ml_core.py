"""Core training, evaluation, and orchestration logic for fake news detection."""

from __future__ import annotations

import json
import logging
import os
import smtplib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from data_preprocessing import (
    CLEAN_CONTENT_COLUMN,
    CONTENT_COLUMN,
    TEXT_COLUMN,
    TITLE_COLUMN,
    combine_title_and_text,
    preprocess_content_frame,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
TRUE_CSV_PATH = RAW_DIR / "True.csv"
FAKE_CSV_PATH = RAW_DIR / "Fake.csv"
MERGED_NEWS_PATH = PROCESSED_DIR / "merged_news.csv"
CLEANED_NEWS_PATH = PROCESSED_DIR / "cleaned_news.csv"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODELS_DIR / "logistic_regression_model.joblib"
EVALUATION_METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_VAL_PATH = REPORTS_DIR / "confusion_matrix_val.png"
CONFUSION_MATRIX_TEST_PATH = REPORTS_DIR / "confusion_matrix_test.png"
ROC_CURVE_PATH = REPORTS_DIR / "roc_curve.png"
API_NAME = "Fake News Detection API"
API_VERSION = "1.0.0"
SUMMARY_HEADER = ("Stage", "Status", "Seconds")
TRAIN_STAGE = "train"
INGEST_STAGE = "ingest"
PREPROCESS_STAGE = "preprocess"
EVALUATE_STAGE = "evaluate"
EMAIL_STAGE = "email"
PASSED = "PASSED"
FAILED = "FAILED"
SKIPPED = "SKIPPED"
REAL_LABEL = 0
FAKE_LABEL = 1
RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
TEMP_FRACTION = 0.30
SPLIT_FRACTION = 0.50
MAX_FEATURES = 50000
NGRAM_RANGE: tuple[int, int] = (1, 2)
SUBLINEAR_TF = True
MIN_DF = 2
STOP_WORDS = "english"
LOGISTIC_SOLVER = "lbfgs"
MAX_ITER = 1000
CLASS_WEIGHT = "balanced"
C_VALUE = 1.0
AVERAGE = "weighted"
ZERO_DIVISION = 0
PLOT_DPI = 180
FIGURE_SIZE: tuple[float, float] = (10.0, 7.0)
CM_CMAP = "Greens"
ROC_CMAP = {"validation": "#1f77b4", "test": "#d62728"}
TOP_FEATURE_COUNT = 20
BATCH_MAX_SIZE = 50
SMTP_HOST_ENV = "SMTP_HOST"
SMTP_PORT_ENV = "SMTP_PORT"
SMTP_USERNAME_ENV = "SMTP_USERNAME"
SMTP_PASSWORD_ENV = "SMTP_PASSWORD"
SMTP_FROM_ENV = "SMTP_FROM"
SMTP_TO_ENV = "SMTP_TO"
EMAIL_SUBJECT = "Fake News Pipeline Summary"
EMAIL_BODY_TEMPLATE = "Fake news pipeline finished with status: {status}\n\n{summary}"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
MISSING_COLUMNS_TEMPLATE = "{source_name} is missing required columns: {missing_columns}"
RAW_COLUMNS = (TITLE_COLUMN, TEXT_COLUMN, "subject", "date")
LABEL_COLUMN = "label"
SOURCE_LABELS = ((TRUE_CSV_PATH, REAL_LABEL), (FAKE_CSV_PATH, FAKE_LABEL))
SUMMARY_COLUMNS = ["stage", "status", "seconds"]
SERIALIZED_INDENT = 2
SCORE_LABEL = "score"
REAL_NAME = "REAL"
FAKE_NAME = "FAKE"
ROOT_STAGE = "root"
HEALTH_STAGE = "health"
BATCH_STAGE = "batch"
PREDICT_STAGE = "predict"
MISSING_FILE_MESSAGE = "Required model artifact missing"
EMPTY_STRING = ""
SPACE = " "
COLON = ":"
NEWLINE = "\n"


@dataclass
class StageResult:
    """Summary information for a pipeline stage."""

    name: str
    status: str
    seconds: float


@dataclass
class PipelineArtifacts:
    """Container for the trained model and vectorizer."""

    model: LogisticRegression
    vectorizer: TfidfVectorizer


def configure_logging() -> None:
    """Configure application logging with timestamps."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def ensure_directories() -> None:
    """Create all required directories for pipeline artifacts."""

    for directory in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def validate_raw_columns(frame: pd.DataFrame, source_name: str) -> None:
    """Validate the required Kaggle news CSV columns.

    Args:
        frame: Input dataframe.
        source_name: Human-readable source name for error messages.

    Raises:
        ValueError: If any required columns are missing.
    """

    missing_columns = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(MISSING_COLUMNS_TEMPLATE.format(source_name=source_name, missing_columns=", ".join(missing_columns)))


def load_and_merge_raw_news() -> pd.DataFrame:
    """Load True.csv and Fake.csv, validate them, and build a merged dataframe.

    Returns:
        A shuffled dataframe with labels and combined content.

    Raises:
        FileNotFoundError: If a raw CSV file is missing.
        ValueError: If a raw CSV file is missing required columns.
    """

    frames: list[pd.DataFrame] = []
    for csv_path, label in SOURCE_LABELS:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing raw news file: {csv_path}")
        frame = pd.read_csv(csv_path)
        validate_raw_columns(frame, csv_path.name)
        frame = frame.copy()
        frame[LABEL_COLUMN] = label
        frame[CONTENT_COLUMN] = frame.apply(
            lambda row: combine_title_and_text(row[TITLE_COLUMN], row[TEXT_COLUMN]), axis=1
        )
        frames.append(frame)

    merged_frame = pd.concat(frames, ignore_index=True)
    merged_frame = merged_frame.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    merged_frame.to_csv(MERGED_NEWS_PATH, index=False)
    LOGGER.info("Saved merged news dataset to %s with %d rows", MERGED_NEWS_PATH, len(merged_frame))
    return merged_frame


def preprocess_news_data(merged_frame: pd.DataFrame) -> pd.DataFrame:
    """Clean merged news data and save the cleaned dataset.

    Args:
        merged_frame: Raw merged news dataframe.

    Returns:
        The cleaned dataframe.
    """

    cleaned_frame = preprocess_content_frame(merged_frame, content_column=CONTENT_COLUMN, cleaned_column=CLEAN_CONTENT_COLUMN)
    cleaned_frame.to_csv(CLEANED_NEWS_PATH, index=False)
    LOGGER.info("Saved cleaned news dataset to %s with %d rows", CLEANED_NEWS_PATH, len(cleaned_frame))
    return cleaned_frame


def split_train_validation_test(
    frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split cleaned data into train, validation, and test subsets.

    Args:
        frame: Cleaned dataframe containing labels and cleaned content.

    Returns:
        A tuple containing train, validation, and test features and labels.
    """

    features = frame[CLEAN_CONTENT_COLUMN]
    labels = frame[LABEL_COLUMN]
    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=TEMP_FRACTION,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=SPLIT_FRACTION,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def create_vectorizer() -> TfidfVectorizer:
    """Create the configured TF-IDF vectorizer.

    Returns:
        A configured TfidfVectorizer instance.
    """

    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=SUBLINEAR_TF,
        min_df=MIN_DF,
        stop_words=STOP_WORDS,
    )


def fit_vectorizer(train_texts: pd.Series) -> tuple[TfidfVectorizer, Any]:
    """Fit the TF-IDF vectorizer on the training text only.

    Args:
        train_texts: Training text series.

    Returns:
        The fitted vectorizer and the transformed training matrix.
    """

    vectorizer = create_vectorizer()
    train_matrix = vectorizer.fit_transform(train_texts)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    LOGGER.info("Saved vectorizer to %s", VECTORIZER_PATH)
    LOGGER.info("Vocabulary size: %d", len(vectorizer.vocabulary_))
    log_top_features(vectorizer, train_matrix)
    return vectorizer, train_matrix


def transform_texts(vectorizer: TfidfVectorizer, texts: pd.Series) -> Any:
    """Transform text with a fitted TF-IDF vectorizer.

    Args:
        vectorizer: Fitted TF-IDF vectorizer.
        texts: Text series to transform.

    Returns:
        Sparse feature matrix.
    """

    return vectorizer.transform(texts)


def log_top_features(vectorizer: TfidfVectorizer, train_matrix: Any) -> None:
    """Log the top features by average TF-IDF weight.

    Args:
        vectorizer: Fitted TF-IDF vectorizer.
        train_matrix: Training TF-IDF matrix.
    """

    feature_names = vectorizer.get_feature_names_out()
    average_weights = np.asarray(train_matrix.mean(axis=0)).ravel()
    top_indices = np.argsort(average_weights)[::-1][:TOP_FEATURE_COUNT]
    top_features = [
        {"feature": str(feature_names[index]), "average_weight": float(average_weights[index])}
        for index in top_indices
    ]
    LOGGER.info("Top TF-IDF features by average weight: %s", top_features)


def train_classifier(x_train_matrix: Any, y_train: pd.Series) -> LogisticRegression:
    """Train the logistic regression classifier.

    Args:
        x_train_matrix: Training feature matrix.
        y_train: Training labels.

    Returns:
        A fitted logistic regression model.
    """

    start_time = datetime.now(timezone.utc)
    monotonic_start = time.perf_counter()
    LOGGER.info("Training started at %s", start_time.isoformat())
    model = LogisticRegression(
        solver=LOGISTIC_SOLVER,
        max_iter=MAX_ITER,
        C=C_VALUE,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train_matrix, y_train)
    elapsed = time.perf_counter() - monotonic_start
    LOGGER.info("Training finished at %s after %.2f seconds", datetime.now(timezone.utc).isoformat(), elapsed)
    joblib.dump(model, MODEL_PATH)
    LOGGER.info("Saved model to %s", MODEL_PATH)
    return model


def calculate_split_metrics(
    model: LogisticRegression,
    x_matrix: Any,
    y_true: pd.Series,
) -> dict[str, Any]:
    """Calculate numeric evaluation metrics for a split.

    Args:
        model: Fitted classifier.
        x_matrix: Feature matrix for evaluation.
        y_true: Ground-truth labels.

    Returns:
        A dictionary of metrics and predictions.
    """

    y_pred = model.predict(x_matrix)
    y_prob = model.predict_proba(x_matrix)
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = [REAL_LABEL, FAKE_LABEL]
    try:
        real_index = classes.index(REAL_LABEL)
        fake_index = classes.index(FAKE_LABEL)
    except ValueError as exc:
        raise RuntimeError("The trained model does not expose the expected label classes 0 and 1") from exc

    if len(set(y_true)) < 2:
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob[:, fake_index]))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average=AVERAGE, zero_division=ZERO_DIVISION)),
        "recall_weighted": float(recall_score(y_true, y_pred, average=AVERAGE, zero_division=ZERO_DIVISION)),
        "f1_weighted": float(f1_score(y_true, y_pred, average=AVERAGE, zero_division=ZERO_DIVISION)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[REAL_LABEL, FAKE_LABEL]).tolist(),
        "predictions": y_pred.tolist(),
        "probabilities": y_prob.tolist(),
        "real_index": real_index,
        "fake_index": fake_index,
    }


def save_confusion_matrix_plot(matrix: list[list[int]], file_path: Path, title: str) -> None:
    """Save a confusion matrix heatmap.

    Args:
        matrix: 2x2 confusion matrix.
        file_path: Output image path.
        title: Figure title.
    """

    plt.figure(figsize=FIGURE_SIZE)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=CM_CMAP,
        xticklabels=[REAL_NAME, FAKE_NAME],
        yticklabels=[REAL_NAME, FAKE_NAME],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_path, dpi=PLOT_DPI)
    plt.close()
    LOGGER.info("Saved confusion matrix plot to %s", file_path)


def save_roc_curve_plot(
    y_true_val: pd.Series,
    val_probabilities: Sequence[Sequence[float]],
    y_true_test: pd.Series,
    test_probabilities: Sequence[Sequence[float]],
    fake_index: int,
) -> None:
    """Save a combined ROC curve for validation and test sets.

    Args:
        y_true_val: Validation labels.
        val_probabilities: Validation probabilities.
        y_true_test: Test labels.
        test_probabilities: Test probabilities.
        fake_index: Index of the fake-news class in the probability arrays.
    """

    plt.figure(figsize=FIGURE_SIZE)
    if len(set(y_true_val)) >= 2:
        fpr_val, tpr_val, _ = roc_curve(y_true_val, np.asarray(val_probabilities)[:, fake_index])
        auc_val = roc_auc_score(y_true_val, np.asarray(val_probabilities)[:, fake_index])
        plt.plot(fpr_val, tpr_val, color=ROC_CMAP["validation"], label=f"Validation ROC (AUC = {auc_val:.3f})")
    if len(set(y_true_test)) >= 2:
        fpr_test, tpr_test, _ = roc_curve(y_true_test, np.asarray(test_probabilities)[:, fake_index])
        auc_test = roc_auc_score(y_true_test, np.asarray(test_probabilities)[:, fake_index])
        plt.plot(fpr_test, tpr_test, color=ROC_CMAP["test"], label=f"Test ROC (AUC = {auc_test:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=PLOT_DPI)
    plt.close()
    LOGGER.info("Saved ROC curve plot to %s", ROC_CURVE_PATH)


def serialize_metrics(metrics: dict[str, dict[str, Any]]) -> None:
    """Persist evaluation metrics to JSON.

    Args:
        metrics: Metrics dictionary keyed by split name.
    """

    with EVALUATION_METRICS_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=SERIALIZED_INDENT)
    LOGGER.info("Saved evaluation metrics to %s", EVALUATION_METRICS_PATH)


def evaluate_and_save_artifacts(
    model: LogisticRegression,
    x_val_matrix: Any,
    y_val: pd.Series,
    x_test_matrix: Any,
    y_test: pd.Series,
) -> dict[str, dict[str, Any]]:
    """Evaluate the model and persist all assessment artifacts.

    Args:
        model: Fitted classifier.
        x_val_matrix: Validation matrix.
        y_val: Validation labels.
        x_test_matrix: Test matrix.
        y_test: Test labels.

    Returns:
        A dictionary of metrics keyed by split name.
    """

    val_metrics = calculate_split_metrics(model, x_val_matrix, y_val)
    test_metrics = calculate_split_metrics(model, x_test_matrix, y_test)
    save_confusion_matrix_plot(val_metrics["confusion_matrix"], CONFUSION_MATRIX_VAL_PATH, "Validation Confusion Matrix")
    save_confusion_matrix_plot(test_metrics["confusion_matrix"], CONFUSION_MATRIX_TEST_PATH, "Test Confusion Matrix")
    save_roc_curve_plot(y_val, val_metrics["probabilities"], y_test, test_metrics["probabilities"], val_metrics["fake_index"])

    metrics = {
        "val": {
            "accuracy": val_metrics["accuracy"],
            "precision_weighted": val_metrics["precision_weighted"],
            "recall_weighted": val_metrics["recall_weighted"],
            "f1_weighted": val_metrics["f1_weighted"],
            "roc_auc": val_metrics["roc_auc"],
            "confusion_matrix": val_metrics["confusion_matrix"],
        },
        "test": {
            "accuracy": test_metrics["accuracy"],
            "precision_weighted": test_metrics["precision_weighted"],
            "recall_weighted": test_metrics["recall_weighted"],
            "f1_weighted": test_metrics["f1_weighted"],
            "roc_auc": test_metrics["roc_auc"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
    }
    serialize_metrics(metrics)
    return metrics


def load_artifacts() -> PipelineArtifacts:
    """Load the trained model and vectorizer from disk.

    Returns:
        A PipelineArtifacts container.

    Raises:
        FileNotFoundError: If either artifact is missing.
    """

    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        missing_paths = [str(path) for path in (MODEL_PATH, VECTORIZER_PATH) if not path.exists()]
        raise FileNotFoundError(f"Missing artifact files: {', '.join(missing_paths)}")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return PipelineArtifacts(model=model, vectorizer=vectorizer)


def build_stage_result(name: str, status: str, seconds: float) -> StageResult:
    """Build a stage result record.

    Args:
        name: Stage name.
        status: Stage status.
        seconds: Wall-clock time in seconds.

    Returns:
        A populated StageResult.
    """

    return StageResult(name=name, status=status, seconds=seconds)


def format_stage_table(stage_results: Sequence[StageResult]) -> str:
    """Format stage results as a printable ASCII table.

    Args:
        stage_results: Stage execution summary records.

    Returns:
        A formatted table string.
    """

    rows = [[result.name, result.status, f"{result.seconds:.2f}"] for result in stage_results]
    widths = [len(SUMMARY_HEADER[index]) for index in range(len(SUMMARY_HEADER))]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render_row(values: Sequence[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    table_lines = [render_row(SUMMARY_HEADER), separator]
    table_lines.extend(render_row(row) for row in rows)
    return NEWLINE.join(table_lines)


def build_summary_text(stage_results: Sequence[StageResult]) -> str:
    """Create a compact text summary for optional email notification.

    Args:
        stage_results: Stage execution summary.

    Returns:
        A readable summary string.
    """

    lines = [f"{result.name}: {result.status} ({result.seconds:.2f}s)" for result in stage_results]
    return NEWLINE.join(lines)


def maybe_send_email_summary(stage_results: Sequence[StageResult]) -> None:
    """Optionally email the pipeline summary when SMTP settings are present.

    Args:
        stage_results: Stage execution summary.
    """

    smtp_host = os.getenv(SMTP_HOST_ENV, EMPTY_STRING).strip()
    smtp_to = os.getenv(SMTP_TO_ENV, EMPTY_STRING).strip()
    if not smtp_host or not smtp_to:
        LOGGER.info("Skipping email notification because SMTP configuration is incomplete")
        return

    smtp_port = int(os.getenv(SMTP_PORT_ENV, "587"))
    smtp_username = os.getenv(SMTP_USERNAME_ENV, EMPTY_STRING).strip()
    smtp_password = os.getenv(SMTP_PASSWORD_ENV, EMPTY_STRING).strip()
    smtp_from = os.getenv(SMTP_FROM_ENV, smtp_username or "noreply@example.com").strip()

    message = EmailMessage()
    message["Subject"] = EMAIL_SUBJECT
    message["From"] = smtp_from
    message["To"] = smtp_to
    summary_text = build_summary_text(stage_results)
    overall_status = FAILED if any(result.status == FAILED for result in stage_results) else PASSED
    message.set_content(EMAIL_BODY_TEMPLATE.format(status=overall_status, summary=summary_text))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.starttls()
            if smtp_username and smtp_password:
                server.login(smtp_username, smtp_password)
            server.send_message(message)
        LOGGER.info("Sent pipeline summary email to %s", smtp_to)
    except (OSError, smtplib.SMTPException, ValueError) as exc:
        LOGGER.exception("Email stage failed: unable to send summary email")
        raise RuntimeError(f"Email stage failed: unable to send summary email ({exc})") from exc


def run_pipeline() -> list[StageResult]:
    """Run the full training and evaluation pipeline.

    Returns:
        A list of stage execution summaries.
    """

    configure_logging()
    ensure_directories()
    stage_results: list[StageResult] = []

    def execute_stage(name: str, action: Callable[[], Any]) -> Any:
        stage_start = time.perf_counter()
        try:
            result = action()
            stage_results.append(build_stage_result(name, PASSED, time.perf_counter() - stage_start))
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - stage_start
            stage_results.append(build_stage_result(name, FAILED, elapsed))
            LOGGER.exception("Stage '%s' failed", name)
            raise

    try:
        merged_frame = execute_stage(INGEST_STAGE, load_and_merge_raw_news)
    except Exception:
        return stage_results

    try:
        cleaned_frame = execute_stage(PREPROCESS_STAGE, lambda: preprocess_news_data(merged_frame))
    except Exception:
        return stage_results

    def training_action() -> tuple[LogisticRegression, Any, pd.Series, Any, pd.Series]:
        x_train, x_val, x_test, y_train, y_val, y_test = split_train_validation_test(cleaned_frame)
        vectorizer, x_train_matrix = fit_vectorizer(x_train)
        x_val_matrix = transform_texts(vectorizer, x_val)
        x_test_matrix = transform_texts(vectorizer, x_test)
        model = train_classifier(x_train_matrix, y_train)
        return model, x_val_matrix, y_val, x_test_matrix, y_test

    try:
        model, x_val_matrix, y_val, x_test_matrix, y_test = execute_stage(TRAIN_STAGE, training_action)
    except Exception:
        return stage_results

    try:
        execute_stage(EVALUATE_STAGE, lambda: evaluate_and_save_artifacts(model, x_val_matrix, y_val, x_test_matrix, y_test))
    except Exception:
        return stage_results

    try:
        execute_stage(EMAIL_STAGE, lambda: maybe_send_email_summary(stage_results))
    except Exception:
        LOGGER.warning("Continuing despite optional email stage failure")

    return stage_results


if __name__ == "__main__":
    try:
        results = run_pipeline()
        print(format_stage_table(results))
    except Exception:
        raise
