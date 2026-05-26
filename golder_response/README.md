# Golden Response Runnable Solution

This folder contains the complete executable reference implementation for the
fake-news detection benchmark. It is self-contained at the source-code level:
after placing the Kaggle CSV files in `data/raw/`, the full training and
evaluation pipeline can be run from this folder in one command.

## Folder Contents

- `run_pipeline.py`: one-command pipeline entry point.
- `data_preprocessing.py`: shared training and inference preprocessing.
- `ml_core.py`: ingestion, training, evaluation, artifact saving, and pipeline orchestration.
- `schemas.py`: Pydantic v2 request and response schemas.
- `api.py`: FastAPI prediction backend.
- `streamlit_app.py`: Streamlit user interface.
- `requirements.txt`: exact pinned dependencies.
- `data/raw/`: place `True.csv` and `Fake.csv` here before running.
- `data/processed/`: generated merged and cleaned datasets.
- `models/`: generated model and vectorizer artifacts.
- `reports/`: generated metrics and plots.

## Run The Pipeline

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Expected raw input files:

```text
data/raw/True.csv
data/raw/Fake.csv
```

Expected generated outputs:

```text
data/processed/merged_news.csv
data/processed/cleaned_news.csv
models/tfidf_vectorizer.joblib
models/logistic_regression_model.joblib
reports/evaluation_metrics.json
reports/confusion_matrix_val.png
reports/confusion_matrix_test.png
reports/roc_curve.png
```

## Run The API

Run this after the model artifacts have been generated:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Run The Frontend

Run this while the API is available:

```bash
streamlit run streamlit_app.py
```
