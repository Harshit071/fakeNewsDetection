# Fake News Detection Benchmark

## Project Overview
This repository contains a complete fake-news detection benchmark and a runnable reference implementation.

The project is organized around a single master script, `run_pipeline.py`, which:

- ingests `True.csv` and `Fake.csv` from `data/raw/`
- cleans and preprocesses article text
- trains a TF-IDF + Logistic Regression classifier
- evaluates validation and test splits
- saves model artifacts and reports
- exposes a FastAPI backend and a Streamlit frontend

## Repository Structure
- [prompt.md](prompt.md): the benchmark prompt that defines the task.
- [justification.md](justification.md): the side-by-side evaluation framework.
- [run_pipeline.py](run_pipeline.py): the one-command master pipeline entrypoint.
- [data_preprocessing.py](data_preprocessing.py): shared text preprocessing logic for training and inference.
- [ml_core.py](ml_core.py): ingestion, training, evaluation, artifact management, and pipeline orchestration.
- [api.py](api.py): FastAPI serving layer.
- [schemas.py](schemas.py): Pydantic request and response models.
- [streamlit_app.py](streamlit_app.py): Streamlit frontend.
- [golden_response.py](golden_response.py): compatibility wrapper that invokes the master pipeline.
- [requirements.txt](requirements.txt): exact dependency versions.
- `data/`: generated datasets and raw Kaggle inputs.
- `models/`: saved model and vectorizer artifacts.
- `reports/`: metrics and plotted evaluation outputs.

## Setup
1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the Kaggle fake-news dataset files and place them here:

```text
data/raw/True.csv
data/raw/Fake.csv
```

## Environment Variables
| Variable | Purpose | Required |
| --- | --- | --- |
| `SMTP_HOST` | Optional SMTP server host for pipeline email summaries | No |
| `SMTP_PORT` | Optional SMTP port, defaults to `587` | No |
| `SMTP_USERNAME` | Optional SMTP username | No |
| `SMTP_PASSWORD` | Optional SMTP password | No |
| `SMTP_FROM` | Optional sender address | No |
| `SMTP_TO` | Optional recipient address | No |
| `API_BASE_URL` | Base URL used by the Streamlit app when calling the API | No |

## Run Locally
Run the full pipeline in one command:

```bash
python run_pipeline.py
```

That command will generate:

- `data/processed/merged_news.csv`
- `data/processed/cleaned_news.csv`
- `models/tfidf_vectorizer.joblib`
- `models/logistic_regression_model.joblib`
- `reports/evaluation_metrics.json`
- `reports/confusion_matrix_val.png`
- `reports/confusion_matrix_test.png`
- `reports/roc_curve.png`

To start the API after training:

```bash
uvicorn api:app --reload
```

To start the Streamlit frontend:

```bash
streamlit run streamlit_app.py
```

## Testing and Validation
A quick syntax check for the pipeline is:

```bash
python -m py_compile run_pipeline.py api.py streamlit_app.py ml_core.py data_preprocessing.py schemas.py
```

For a functional check, place the Kaggle CSVs in `data/raw/` and run `python run_pipeline.py`.

## Evaluation Methodology
The benchmark scores candidate responses on four main areas:

1. Requirement coverage: every explicit prompt constraint should be met.
2. Correctness: preprocessing, split strategy, vectorization, training, evaluation, and serving should behave as specified.
3. Robustness: missing files, malformed inputs, and artifact-loading errors should be handled clearly.
4. Quality: code should remain modular, typed, documented, and easy to maintain.

The reference implementation is intended to be a strong baseline for comparing multiple model outputs against the same benchmark prompt.

## Deployment Notes
The API can be deployed behind Uvicorn, a process manager, or a reverse proxy.

A minimal production-style flow is:

1. Run `python run_pipeline.py` during build or release preparation.
2. Deploy the generated `models/` and `reports/` artifacts with the service image or host.
3. Start `uvicorn api:app --host 0.0.0.0 --port 8000`.
4. Point the Streamlit frontend at the API base URL and run `streamlit run streamlit_app.py`.

If SMTP settings are provided, the pipeline can send an optional summary email when it finishes.
