"""# Golden Benchmark Solution

## Purpose

This file is the executable benchmark entry point for the fake-news detection
prompt. It validates the required project files and Kaggle raw data, runs the
complete automated training pipeline, checks that all expected artifacts were
created, and exits with a clear success or failure code.

## Run Instructions

1. Place the Kaggle fake-news files at `data/raw/True.csv` and
   `data/raw/Fake.csv`.
2. Install the exact dependencies from `requirements.txt`.
3. Run `python golden_response.py`.
4. Review the printed stage summary table and timestamped logs.

## Expected Outputs

- `data/processed/merged_news.csv`
- `data/processed/cleaned_news.csv`
- `models/tfidf_vectorizer.joblib`
- `models/logistic_regression_model.joblib`
- `reports/evaluation_metrics.json`
- `reports/confusion_matrix_val.png`
- `reports/confusion_matrix_test.png`
- `reports/roc_curve.png`
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from ml_core import (
    CLEANED_NEWS_PATH,
    CONFUSION_MATRIX_TEST_PATH,
    CONFUSION_MATRIX_VAL_PATH,
    EVALUATION_METRICS_PATH,
    FAILED,
    FAKE_CSV_PATH,
    LOG_FORMAT,
    MERGED_NEWS_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    ROC_CURVE_PATH,
    RAW_COLUMNS,
    TRUE_CSV_PATH,
    VECTORIZER_PATH,
    StageResult,
    format_stage_table,
    run_pipeline,
)

LOGGER = logging.getLogger(__name__)

# Exit codes are explicit so CI, graders, and shell scripts can evaluate the
# benchmark run without parsing log text.
SUCCESS_EXIT_CODE = 0
FAILURE_EXIT_CODE = 1

# These files make up the production reference implementation. The benchmark
# entry point fails early if any are missing, which gives a clearer error than
# allowing a later import or runtime stage to fail indirectly.
REQUIRED_PROJECT_FILES: tuple[Path, ...] = (
    Path("data_preprocessing.py"),
    Path("ml_core.py"),
    Path("schemas.py"),
    Path("api.py"),
    Path("streamlit_app.py"),
    Path("run_pipeline.py"),
    Path("requirements.txt"),
    Path("README.md"),
)

RAW_DATA_PATHS: tuple[Path, ...] = (TRUE_CSV_PATH, FAKE_CSV_PATH)

# Artifact validation happens after the pipeline succeeds so silent partial
# runs are caught before the process exits with success.
EXPECTED_OUTPUT_PATHS: tuple[Path, ...] = (
    MERGED_NEWS_PATH,
    CLEANED_NEWS_PATH,
    VECTORIZER_PATH,
    MODEL_PATH,
    EVALUATION_METRICS_PATH,
    CONFUSION_MATRIX_VAL_PATH,
    CONFUSION_MATRIX_TEST_PATH,
    ROC_CURVE_PATH,
)

PIPELINE_STAGE_NAME = "golden_response"
PREFLIGHT_STAGE_NAME = "preflight"
SUCCESS_MESSAGE = "Golden benchmark pipeline completed successfully."
FAILURE_MESSAGE = "Golden benchmark pipeline failed."
PREFLIGHT_SUCCESS_MESSAGE = "Preflight checks passed."
STAGE_START_TEMPLATE = "Starting %s stage."
STAGE_FAILURE_DETAIL = "%s One or more stages failed."
MISSING_FILE_TEMPLATE = "Missing required project file: {path}"
MISSING_RAW_FILE_TEMPLATE = "Missing required raw Kaggle CSV: {path}"
EMPTY_RAW_FILE_TEMPLATE = "Raw CSV exists but is empty: {path}"
RAW_COLUMN_HINT_TEMPLATE = "Each raw CSV must contain these columns: {columns}"
RAW_FILE_ERROR_TEMPLATE = "{missing_message}. {column_hint}"
MISSING_OUTPUT_TEMPLATE = "Pipeline finished but did not create expected artifact: {path}"


def configure_entrypoint_logging() -> None:
    """Configure timestamped logging for the benchmark entry point."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def validate_project_files(required_files: Sequence[Path]) -> None:
    """Validate that all source files required by the reference solution exist.

    Args:
        required_files: Paths that must exist before the pipeline can run.

    Raises:
        FileNotFoundError: If any required source file is missing.
    """

    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        missing_text = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(MISSING_FILE_TEMPLATE.format(path=missing_text))


def validate_raw_data_files(raw_paths: Sequence[Path]) -> None:
    """Validate that raw Kaggle CSV files are present and non-empty.

    Args:
        raw_paths: Raw CSV paths expected by the training pipeline.

    Raises:
        FileNotFoundError: If a raw CSV is missing.
        ValueError: If a raw CSV is empty.
    """

    for raw_path in raw_paths:
        if not raw_path.is_file():
            column_hint = RAW_COLUMN_HINT_TEMPLATE.format(columns=", ".join(RAW_COLUMNS))
            missing_message = MISSING_RAW_FILE_TEMPLATE.format(path=raw_path)
            raise FileNotFoundError(RAW_FILE_ERROR_TEMPLATE.format(missing_message=missing_message, column_hint=column_hint))
        if raw_path.stat().st_size == 0:
            raise ValueError(EMPTY_RAW_FILE_TEMPLATE.format(path=raw_path))


def validate_expected_outputs(output_paths: Sequence[Path]) -> None:
    """Validate that the pipeline produced every required artifact.

    Args:
        output_paths: Files expected after a successful pipeline run.

    Raises:
        FileNotFoundError: If any expected artifact is missing.
    """

    missing_outputs = [path for path in output_paths if not path.is_file()]
    if missing_outputs:
        missing_text = ", ".join(str(path) for path in missing_outputs)
        raise FileNotFoundError(MISSING_OUTPUT_TEMPLATE.format(path=missing_text))


def run_preflight_checks() -> None:
    """Run all preflight checks before starting the expensive ML pipeline."""

    validate_project_files(REQUIRED_PROJECT_FILES)
    validate_raw_data_files(RAW_DATA_PATHS)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.info(PREFLIGHT_SUCCESS_MESSAGE)


def pipeline_failed(stage_results: Sequence[StageResult]) -> bool:
    """Return whether any pipeline stage failed.

    Args:
        stage_results: Stage summaries returned by the pipeline.

    Returns:
        True when at least one stage has failed; otherwise False.
    """

    return any(result.status == FAILED for result in stage_results)


def execute_golden_solution() -> int:
    """Execute the complete benchmark solution.

    Returns:
        Process exit code: 0 on success and 1 on failure.
    """

    configure_entrypoint_logging()
    try:
        LOGGER.info(STAGE_START_TEMPLATE, PREFLIGHT_STAGE_NAME)
        run_preflight_checks()
        LOGGER.info(STAGE_START_TEMPLATE, PIPELINE_STAGE_NAME)
        stage_results = run_pipeline()
        print(format_stage_table(stage_results))
        if pipeline_failed(stage_results):
            LOGGER.error(STAGE_FAILURE_DETAIL, FAILURE_MESSAGE)
            return FAILURE_EXIT_CODE
        validate_expected_outputs(EXPECTED_OUTPUT_PATHS)
        LOGGER.info(SUCCESS_MESSAGE)
        return SUCCESS_EXIT_CODE
    except Exception as exc:
        LOGGER.exception("%s %s", FAILURE_MESSAGE, exc)
        return FAILURE_EXIT_CODE


def main() -> None:
    """Run the benchmark solution and exit with an explicit status code."""

    raise SystemExit(execute_golden_solution())


if __name__ == "__main__":
    main()
