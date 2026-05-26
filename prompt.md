# Fake News Detection System

## Overview

Build a production-quality, end-to-end fake news detection system that can be trained, evaluated, served, and used by non-technical users. The solution must be fully automated from a single master script and must include a machine learning pipeline, reusable preprocessing logic, a FastAPI prediction service, a Streamlit user interface, evaluation reports, and complete documentation.

The goal is not only to train a classifier, but to deliver a maintainable reference implementation that demonstrates correct data handling, reproducible model training, clean API design, usable frontend workflows, robust error handling, and deployment-ready project structure.

## Objective

Create a binary classifier that predicts whether a news article is real or fake.

- Real news must use label `0`.
- Fake news must use label `1`.
- Predictions must be returned as both numeric labels and human-readable strings: `REAL` or `FAKE`.
- The complete system must run after placing the raw Kaggle CSV files in `data/raw/` and executing one command.

## Inputs

### Required Raw Data Files

The user will manually place the Kaggle fake news dataset files in the following paths before running the pipeline:

- `data/raw/True.csv`
- `data/raw/Fake.csv`

### Required Raw Data Columns

Both CSV files must contain these columns:

- `title`
- `text`
- `subject`
- `date`

The implementation must validate these columns before processing. If any column is missing, raise a descriptive `ValueError` that names the file and the missing columns.

### Prediction Inputs

The API and frontend must accept news articles with:

- `title`: non-empty string
- `text`: non-empty string

Batch prediction must accept a CSV containing `title` and `text` columns. The batch API must reject empty batches and batches larger than 50 records.

## Expected Outputs

The completed implementation must generate these artifacts:

- `data/processed/merged_news.csv`
- `data/processed/cleaned_news.csv`
- `models/tfidf_vectorizer.joblib`
- `models/logistic_regression_model.joblib`
- `reports/evaluation_metrics.json`
- `reports/confusion_matrix_val.png`
- `reports/confusion_matrix_test.png`
- `reports/roc_curve.png`

The API must expose prediction responses with:

- `label`
- `prediction`
- `confidence`
- `real_probability`
- `fake_probability`

The Streamlit app must allow single prediction, batch prediction, CSV export, and model-metrics inspection.

## Required Project Structure

Use a clean multi-file structure. Every file must be fully implemented with no placeholders, no TODOs, and no unfinished functions.

Required files:

- `run_pipeline.py`: master orchestration script
- `data_preprocessing.py`: shared preprocessing used by both training and inference
- `ml_core.py`: data ingestion, feature extraction, training, evaluation, and artifact helpers
- `schemas.py`: Pydantic v2 request and response models
- `api.py`: FastAPI backend
- `streamlit_app.py`: Streamlit frontend
- `requirements.txt`: exact dependency versions
- `README.md`: setup, usage, folder structure, environment variables, and deployment instructions

## Single-Command Pipeline Requirement

Running the following command after placing `True.csv` and `Fake.csv` in `data/raw/` must execute the complete pipeline with zero manual intervention:

```bash
python run_pipeline.py
```

The master script must:

- create required folders if they do not exist
- ingest raw data
- validate required columns
- merge and shuffle records
- preprocess text
- split data into train, validation, and test sets
- fit the vectorizer only on training data
- train the model
- evaluate validation and test performance
- save all model and report artifacts
- catch and log errors per stage
- print a final summary table with stage name, status, and wall-clock seconds

## Data Ingestion Requirements

Load `True.csv` and `Fake.csv` from `data/raw/`.

Implementation details:

- Validate that both files exist.
- Validate that `title`, `text`, `subject`, and `date` exist in both files.
- Label rows from `True.csv` as `0`.
- Label rows from `Fake.csv` as `1`.
- Merge `title` and `text` into a single `content` column.
- Concatenate real and fake records.
- Shuffle with `random_state=42`.
- Save the merged dataset to `data/processed/merged_news.csv`.

## Data Preprocessing Requirements

Read `data/processed/merged_news.csv`. Drop null values and duplicates in the `content` column.

Clean text in exactly this order:

1. convert to lowercase
2. remove URLs with regex
3. remove HTML tags
4. remove punctuation
5. remove digits
6. strip extra whitespace
7. tokenize with `nltk.word_tokenize`
8. remove English stopwords
9. apply `PorterStemmer` to each token
10. apply `WordNetLemmatizer` to each stemmed token
11. rejoin tokens into one cleaned string

The same preprocessing function must be reused during inference. Do not approximate or duplicate a different inference-time cleaning pipeline.

Log a before-and-after sample of three rows so the cleaning can be visually checked. Save the cleaned output to `data/processed/cleaned_news.csv`.

## Feature Extraction Requirements

Load `data/processed/cleaned_news.csv` and create the TF-IDF feature matrix.

Use exactly:

```python
TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    stop_words="english",
)
```

Fit the vectorizer only on the training split to avoid data leakage. Save the fitted vectorizer to `models/tfidf_vectorizer.joblib`.

Log:

- vocabulary size
- top 20 features by average TF-IDF weight

## Model Training Requirements

Split the cleaned dataset into 70 percent training, 15 percent validation, and 15 percent test.

Use two successive `train_test_split` calls:

1. split 30 percent into a temporary set
2. split the temporary set equally into validation and test sets

Both splits must use:

- `stratify=y`
- `random_state=42`

Train exactly:

```python
LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    C=1.0,
    class_weight="balanced",
)
```

Log the start and end time of training. Save the trained model to `models/logistic_regression_model.joblib`.

## Model Assessment Requirements

Evaluate validation and test sets separately.

For both splits, calculate:

- accuracy
- weighted precision
- weighted recall
- weighted F1 score
- ROC-AUC
- per-class confusion matrix

Save:

- validation confusion matrix heatmap to `reports/confusion_matrix_val.png`
- test confusion matrix heatmap to `reports/confusion_matrix_test.png`
- ROC curve to `reports/roc_curve.png`
- numeric metrics to `reports/evaluation_metrics.json`

The metrics JSON must be a clean dictionary with `val` and `test` as top-level keys.

## Backend Requirements

Build a production-ready FastAPI backend in `api.py`.

### Startup Behavior

- Load the trained model and vectorizer at startup.
- Use a FastAPI lifespan context manager.
- Do not use the deprecated `on_event` decorator.
- Store loaded artifacts on `app.state`.
- Return HTTP 503 with a clear message if model files are missing or unavailable.

### Endpoints

Expose exactly these endpoints:

1. `GET /`
   - returns API name
   - returns version string
   - returns status `ok`

2. `GET /health`
   - returns whether the model is loaded
   - returns whether the vectorizer is loaded
   - returns a timestamp

3. `POST /predict`
   - accepts `NewsInput` with `title` and `text`
   - combines and preprocesses the article using `data_preprocessing.py`
   - transforms with the loaded vectorizer
   - returns label, prediction string, confidence, real probability, and fake probability

4. `POST /predict/batch`
   - accepts a list of up to 50 `NewsInput` records
   - preprocesses all records in memory
   - performs one vectorizer transform call for the full batch
   - returns a list of predictions
   - returns a summary with total count, fake count, and real count

### API Quality Requirements

- Add CORS middleware allowing all origins for Streamlit compatibility.
- Return HTTP 422 with descriptive messages for validation failures.
- Never expose stack traces in API response bodies.
- Use the Python logging module throughout.
- Do not use `print` statements in API code.
- Do not re-fit the vectorizer during API requests.
- Do not perform disk I/O per prediction.

## Frontend Requirements

Build a clean Streamlit app in `streamlit_app.py` for non-technical users.

### Global UI

- Set page config with a custom title, emoji favicon, and wide layout.
- Add a sidebar containing:
  - app description
  - tech stack list
  - explanation of the confidence score
- Use sidebar radio navigation with three pages:
  - Single Prediction
  - Batch Prediction
  - Model Info

### Single Prediction Page

Must include:

- text area for title
- text area for article text
- Analyze button
- `requests.post` call wrapped in `try/except`
- `st.spinner` while waiting
- styled result card using `st.markdown`
- green background with large `REAL` text when label is `0`
- red background with large `FAKE` text when label is `1`
- confidence shown with `st.progress`
- real vs fake probabilities shown as a horizontal bar chart

### Batch Prediction Page

Must include:

- CSV file uploader
- validation that uploaded CSV has `title` and `text` columns
- request to `/predict/batch`
- color-coded `st.dataframe`
- green rows for real predictions
- red rows for fake predictions
- `st.download_button` to export predictions as CSV

### Model Info Page

Must include:

- load `reports/evaluation_metrics.json`
- display validation and test metrics side by side with `st.metric`
- show confusion matrix images with captions
- show ROC curve image with a caption

## Tech Stack and Rationale

Use only Python standard library modules plus the libraries listed below. Do not add any other third-party libraries.

Each dependency must be pinned exactly in `requirements.txt` to make grading, deployment, and reproducibility deterministic.

- `fastapi==0.111.0`
  - Use FastAPI because it provides typed request validation, automatic OpenAPI documentation, clean dependency structure, and high-performance async-friendly serving for the prediction API. Version `0.111.0` is required so the implementation can use modern lifespan startup patterns and avoid deprecated `on_event` usage.

- `uvicorn[standard]==0.29.0`
  - Use Uvicorn as the ASGI server for running FastAPI locally and in deployment. The `standard` extra provides production-friendly server dependencies. Version `0.29.0` keeps runtime behavior consistent across environments.

- `streamlit==1.35.0`
  - Use Streamlit because it allows a clean interactive frontend for non-technical users without building a separate JavaScript application. Version `1.35.0` fixes the expected UI API behavior for page config, file uploads, metrics, progress bars, and download buttons.

- `scikit-learn==1.4.2`
  - Use scikit-learn for TF-IDF vectorization, train/validation/test splitting, Logistic Regression, and evaluation metrics. Version `1.4.2` ensures consistent model behavior, serialization compatibility, and metric outputs.

- `pandas==2.2.2`
  - Use pandas for CSV ingestion, validation, merging, cleaning, batch upload handling, and report-friendly tabular operations. Version `2.2.2` gives stable dataframe and CSV behavior.

- `numpy==1.26.4`
  - Use numpy for efficient numeric operations, probability handling, metric support, and compatibility with scikit-learn. Version `1.26.4` is pinned to avoid binary compatibility drift.

- `nltk==3.8.1`
  - Use NLTK for tokenization, English stopwords, stemming, and lemmatization as required by the preprocessing specification. Version `3.8.1` provides the expected `word_tokenize`, `PorterStemmer`, and `WordNetLemmatizer` APIs.

- `joblib==1.4.2`
  - Use joblib to persist and reload the trained model and vectorizer efficiently. Version `1.4.2` is required for reliable artifact serialization with the pinned scikit-learn version.

- `pydantic==2.7.1`
  - Use Pydantic v2 for request and response schemas. Version `2.7.1` is required because validators must use `field_validator`; deprecated v1 `validator` syntax is not allowed.

- `seaborn==0.13.2`
  - Use seaborn for readable confusion matrix heatmaps. Version `0.13.2` gives stable plotting behavior with the pinned matplotlib version.

- `matplotlib==3.9.0`
  - Use matplotlib for saving confusion matrix and ROC curve image artifacts. Version `3.9.0` fixes rendering behavior for generated reports.

- `requests==2.32.2`
  - Use requests in the Streamlit frontend to call the FastAPI backend. Version `2.32.2` provides stable timeout, exception, and JSON response handling.

- `python-multipart==0.0.9`
  - Include python-multipart for compatibility with file upload workflows used by Streamlit and web form tooling around the application. Version `0.0.9` keeps upload parsing support reproducible.

## Code Quality Requirements

The implementation must be readable, maintainable, and production-oriented.

Required standards:

- Use type hints on every function signature.
- Use Google-style docstrings on every function and class.
- Use uppercase constants at the top of each file for paths, hyperparameters, endpoint names, labels, and repeated configuration values.
- Do not hardcode paths, strings, or numeric hyperparameters inside function bodies when they belong in constants.
- Keep training, preprocessing, API, schemas, and frontend concerns separated into appropriate modules.
- Use concise comments only where they explain non-obvious behavior.
- Avoid duplicated business logic, especially preprocessing.
- Use deterministic randomness with `random_state=42` everywhere randomness is involved.

## Error Handling and Logging Requirements

Handle failure modes explicitly:

- missing raw CSV files
- missing required columns
- empty text inputs
- empty or oversized batch requests
- malformed uploaded CSV files
- malformed JSON bodies
- missing model or vectorizer files
- startup artifact loading failures
- optional email service failure if email notification is implemented
- unexpected API errors

Every caught exception must log a descriptive message that includes:

- stage name or component name
- what failed
- enough context to debug the issue

The API must return sanitized error responses and must never expose raw stack traces to clients.

## Documentation Requirements

The `README.md` must document:

- project overview
- folder structure
- setup steps
- dependency installation
- raw data placement
- local pipeline execution
- FastAPI launch instructions
- Streamlit launch instructions
- endpoint summary
- environment variable table
- deployment steps
- expected generated artifacts
- troubleshooting notes

`run_pipeline.py` must include full run instructions as a block comment at the top of the file before any imports.

## Performance and Scalability Requirements

- Fit the vectorizer once during training.
- Load the fitted vectorizer once at API startup.
- Never re-fit the vectorizer per request.
- Run inference preprocessing in memory.
- Avoid disk I/O inside prediction request handlers.
- In batch prediction, process the entire batch with one vectorizer transform call.
- Keep logging asynchronous-friendly by avoiding blocking file writes inside request handlers.
- Rate limiting may be handled at the Uvicorn or reverse proxy layer without changing application code.

## Security and Reliability Requirements

- Reject invalid inputs early with clear validation errors.
- Use Pydantic models for request and response schemas.
- Forbid unexpected fields where appropriate.
- Avoid exposing implementation details in API errors.
- Keep model artifacts outside source code.
- Keep generated datasets and model artifacts out of Git if they are too large for normal repository usage.

## Acceptance Criteria

A solution is complete only if all of the following are true:

- `python run_pipeline.py` completes the full pipeline after raw CSV placement.
- All required artifacts are generated in the expected paths.
- The vectorizer is fit only on the training split.
- Validation and test metrics are saved separately.
- FastAPI uses lifespan startup and exposes all required endpoints.
- The API returns correct response schemas and sanitized errors.
- The batch endpoint transforms the full batch at once.
- Streamlit provides all three required pages and CSV export.
- The same preprocessing implementation is used for training and inference.
- All files contain executable, finished code with no placeholders or TODOs.
- The README explains setup, local run, deployment, environment variables, and artifacts.

## Constraints

- Do not skip any required file.
- Do not leave any function unimplemented.
- Do not add placeholder text such as "you can add this yourself".
- Do not add extra third-party libraries.
- Do not use deprecated FastAPI `on_event`.
- Do not use deprecated Pydantic v1 `validator`.
- Do not use `print` statements in API code.
- Do not leak data by fitting the vectorizer on validation or test data.
- Do not approximate preprocessing during inference.
- Do not require more than one command to run the training and evaluation pipeline.
