# Fake News Detection System

## Overview

Build a production-quality, end-to-end fake news detection system that can be trained, evaluated, served, and used by non-technical users. The solution must be fully automated from a single master script and must include a machine learning pipeline, reusable preprocessing logic, a FastAPI prediction service, a Streamlit user interface, evaluation reports, and complete documentation.

The goal is not only to train a classifier, but to deliver a maintainable reference implementation that demonstrates correct data handling, reproducible model training, clean API design, usable frontend workflows, robust error handling, and deployment-ready project structure.

## Objective

Create a binary classifier that predicts whether a news article is real or fake.

1. Real news must use label `0`.
2. Fake news must use label `1`.
3. Predictions must be returned as both numeric labels and human-readable strings: `REAL` or `FAKE`.
4. The complete system must run after placing the raw Kaggle CSV files in `data/raw/` and executing one command.

## Inputs

### Required Raw Data Files

The user will manually place the Kaggle fake news dataset files in the following paths before running the pipeline:

1. `data/raw/True.csv`
2. `data/raw/Fake.csv`

### Required Raw Data Columns

Both CSV files must contain these columns:

1. `title`
2. `text`
3. `subject`
4. `date`

The implementation must validate these columns before processing. If any column is missing, raise a descriptive `ValueError` that names the file and the missing columns.

### Prediction Inputs

The API and frontend must accept news articles with:

1. `title`: non-empty string
2. `text`: non-empty string

Batch prediction must accept a CSV containing `title` and `text` columns. The batch API must reject empty batches and batches larger than 50 records.

## Expected Outputs

The completed implementation must generate these artifacts:

1. `data/processed/merged_news.csv`
2. `data/processed/cleaned_news.csv`
3. `models/tfidf_vectorizer.joblib`
4. `models/logistic_regression_model.joblib`
5. `reports/evaluation_metrics.json`
6. `reports/confusion_matrix_val.png`
7. `reports/confusion_matrix_test.png`
8. `reports/roc_curve.png`

The API must expose prediction responses with:

1. `label`
2. `prediction`
3. `confidence`
4. `real_probability`
5. `fake_probability`

The Streamlit app must allow single prediction, batch prediction, CSV export, and model-metrics inspection.

## Required Project Structure

Use a clean multi-file structure. Every file must be fully implemented with no placeholders, no TODOs, and no unfinished functions.

Required files:

1. `run_pipeline.py`: master orchestration script
2. `data_preprocessing.py`: shared preprocessing used by both training and inference
3. `ml_core.py`: data ingestion, feature extraction, training, evaluation, and artifact helpers
4. `schemas.py`: Pydantic v2 request and response models
5. `api.py`: FastAPI backend
6. `streamlit_app.py`: Streamlit frontend
7. `requirements.txt`: exact dependency versions
8. `README.md`: setup, usage, folder structure, environment variables, and deployment instructions

## Single-Command Pipeline Requirement

Running the following command after placing `True.csv` and `Fake.csv` in `data/raw/` must execute the complete pipeline with zero manual intervention:

```bash
python run_pipeline.py
```

The master script must:

1. create required folders if they do not exist
2. ingest raw data
3. validate required columns
4. merge and shuffle records
5. preprocess text
6. split data into train, validation, and test sets
7. fit the vectorizer only on training data
8. train the model
9. evaluate validation and test performance
10. save all model and report artifacts
11. catch and log errors per stage
12. print a final summary table with stage name, status, and wall-clock seconds

## Data Ingestion Requirements

Load `True.csv` and `Fake.csv` from `data/raw/`.

Implementation details:

1. Validate that both files exist.
2. Validate that `title`, `text`, `subject`, and `date` exist in both files.
3. Label rows from `True.csv` as `0`.
4. Label rows from `Fake.csv` as `1`.
5. Merge `title` and `text` into a single `content` column.
6. Concatenate real and fake records.
7. Shuffle with `random_state=42`.
8. Save the merged dataset to `data/processed/merged_news.csv`.

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

1. vocabulary size
2. top 20 features by average TF-IDF weight

## Model Training Requirements

Split the cleaned dataset into 70 percent training, 15 percent validation, and 15 percent test.

Use two successive `train_test_split` calls:

1. split 30 percent into a temporary set
2. split the temporary set equally into validation and test sets

Both splits must use:

1. `stratify=y`
2. `random_state=42`

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

1. accuracy
2. weighted precision
3. weighted recall
4. weighted F1 score
5. ROC-AUC
6. per-class confusion matrix

Save:

1. validation confusion matrix heatmap to `reports/confusion_matrix_val.png`
2. test confusion matrix heatmap to `reports/confusion_matrix_test.png`
3. ROC curve to `reports/roc_curve.png`
4. numeric metrics to `reports/evaluation_metrics.json`

The metrics JSON must be a clean dictionary with `val` and `test` as top-level keys.

## Backend Requirements

Build a production-ready FastAPI backend in `api.py`.

### Startup Behavior

1. Load the trained model and vectorizer at startup.
2. Use a FastAPI lifespan context manager.
3. Do not use the deprecated `on_event` decorator.
4. Store loaded artifacts on `app.state`.
5. Return HTTP 503 with a clear message if model files are missing or unavailable.

### Endpoints

Expose exactly these endpoints:

1. `GET /`
   1. returns API name
   2. returns version string
   3. returns status `ok`

2. `GET /health`
   1. returns whether the model is loaded
   2. returns whether the vectorizer is loaded
   3. returns a timestamp

3. `POST /predict`
   1. accepts `NewsInput` with `title` and `text`
   2. combines and preprocesses the article using `data_preprocessing.py`
   3. transforms with the loaded vectorizer
   4. returns label, prediction string, confidence, real probability, and fake probability

4. `POST /predict/batch`
   1. accepts a list of up to 50 `NewsInput` records
   2. preprocesses all records in memory
   3. performs one vectorizer transform call for the full batch
   4. returns a list of predictions
   5. returns a summary with total count, fake count, and real count

### API Quality Requirements

1. Add CORS middleware allowing all origins for Streamlit compatibility.
2. Return HTTP 422 with descriptive messages for validation failures.
3. Never expose stack traces in API response bodies.
4. Use the Python logging module throughout.
5. Do not use `print` statements in API code.
6. Do not re-fit the vectorizer during API requests.
7. Do not perform disk I/O per prediction.

## Frontend Requirements

Build a clean Streamlit app in `streamlit_app.py` for non-technical users.

### Global UI

1. Set page config with a custom title, emoji favicon, and wide layout.
2. Add a sidebar with three parts: app description, tech stack list, and confidence-score explanation.
3. Use sidebar radio navigation with three pages: Single Prediction, Batch Prediction, and Model Info.

### Single Prediction Page

Must include:

1. text area for title
2. text area for article text
3. Analyze button
4. `requests.post` call wrapped in `try/except`
5. `st.spinner` while waiting
6. styled result card using `st.markdown`
7. green background with large `REAL` text when label is `0`
8. red background with large `FAKE` text when label is `1`
9. confidence shown with `st.progress`
10. real vs fake probabilities shown as a horizontal bar chart

### Batch Prediction Page

Must include:

1. CSV file uploader
2. validation that uploaded CSV has `title` and `text` columns
3. request to `/predict/batch`
4. color-coded `st.dataframe`
5. green rows for real predictions
6. red rows for fake predictions
7. `st.download_button` to export predictions as CSV

### Model Info Page

Must include:

1. load `reports/evaluation_metrics.json`
2. display validation and test metrics side by side with `st.metric`
3. show confusion matrix images with captions
4. show ROC curve image with a caption

## Tech Stack and Rationale

Use only Python standard library modules plus the libraries listed below. Do not add any other third-party libraries.

Each dependency must be pinned exactly in `requirements.txt` to make grading, deployment, and reproducibility deterministic.

1. `fastapi==0.111.0`: Use FastAPI because it provides typed request validation, automatic OpenAPI documentation, clean dependency structure, and high-performance async-friendly serving for the prediction API. Version `0.111.0` is required so the implementation can use modern lifespan startup patterns and avoid deprecated `on_event` usage.
2. `uvicorn[standard]==0.29.0`: Use Uvicorn as the ASGI server for running FastAPI locally and in deployment. The `standard` extra provides production-friendly server dependencies. Version `0.29.0` keeps runtime behavior consistent across environments.
3. `streamlit==1.35.0`: Use Streamlit because it allows a clean interactive frontend for non-technical users without building a separate JavaScript application. Version `1.35.0` fixes the expected UI API behavior for page config, file uploads, metrics, progress bars, and download buttons.
4. `scikit-learn==1.4.2`: Use scikit-learn for TF-IDF vectorization, train/validation/test splitting, Logistic Regression, and evaluation metrics. Version `1.4.2` ensures consistent model behavior, serialization compatibility, and metric outputs.
5. `pandas==2.2.2`: Use pandas for CSV ingestion, validation, merging, cleaning, batch upload handling, and report-friendly tabular operations. Version `2.2.2` gives stable dataframe and CSV behavior.
6. `numpy==1.26.4`: Use numpy for efficient numeric operations, probability handling, metric support, and compatibility with scikit-learn. Version `1.26.4` is pinned to avoid binary compatibility drift.
7. `nltk==3.8.1`: Use NLTK for tokenization, English stopwords, stemming, and lemmatization as required by the preprocessing specification. Version `3.8.1` provides the expected `word_tokenize`, `PorterStemmer`, and `WordNetLemmatizer` APIs.
8. `joblib==1.4.2`: Use joblib to persist and reload the trained model and vectorizer efficiently. Version `1.4.2` is required for reliable artifact serialization with the pinned scikit-learn version.
9. `pydantic==2.7.1`: Use Pydantic v2 for request and response schemas. Version `2.7.1` is required because validators must use `field_validator`; deprecated v1 `validator` syntax is not allowed.
10. `seaborn==0.13.2`: Use seaborn for readable confusion matrix heatmaps. Version `0.13.2` gives stable plotting behavior with the pinned matplotlib version.
11. `matplotlib==3.9.0`: Use matplotlib for saving confusion matrix and ROC curve image artifacts. Version `3.9.0` fixes rendering behavior for generated reports.
12. `requests==2.32.2`: Use requests in the Streamlit frontend to call the FastAPI backend. Version `2.32.2` provides stable timeout, exception, and JSON response handling.
13. `python-multipart==0.0.9`: Include python-multipart for compatibility with file upload workflows used by Streamlit and web form tooling around the application. Version `0.0.9` keeps upload parsing support reproducible.

## Code Quality Requirements

The implementation must be readable, maintainable, and production-oriented.

Required standards:

1. Use type hints on every function signature.
2. Use Google-style docstrings on every function and class.
3. Use uppercase constants at the top of each file for paths, hyperparameters, endpoint names, labels, and repeated configuration values.
4. Do not hardcode paths, strings, or numeric hyperparameters inside function bodies when they belong in constants.
5. Keep training, preprocessing, API, schemas, and frontend concerns separated into appropriate modules.
6. Use concise comments only where they explain non-obvious behavior.
7. Avoid duplicated business logic, especially preprocessing.
8. Use deterministic randomness with `random_state=42` everywhere randomness is involved.

## Error Handling and Logging Requirements

Handle failure modes explicitly:

1. missing raw CSV files
2. missing required columns
3. empty text inputs
4. empty or oversized batch requests
5. malformed uploaded CSV files
6. malformed JSON bodies
7. missing model or vectorizer files
8. startup artifact loading failures
9. optional email service failure if email notification is implemented
10. unexpected API errors

Every caught exception must log a descriptive message that includes:

1. stage name or component name
2. what failed
3. enough context to debug the issue

The API must return sanitized error responses and must never expose raw stack traces to clients.

## Documentation Requirements

The `README.md` must document:

1. project overview
2. folder structure
3. setup steps
4. dependency installation
5. raw data placement
6. local pipeline execution
7. FastAPI launch instructions
8. Streamlit launch instructions
9. endpoint summary
10. environment variable table
11. deployment steps
12. expected generated artifacts
13. troubleshooting notes

`run_pipeline.py` must include full run instructions as a block comment at the top of the file before any imports.

## Performance and Scalability Requirements

1. Fit the vectorizer once during training.
2. Load the fitted vectorizer once at API startup.
3. Never re-fit the vectorizer per request.
4. Run inference preprocessing in memory.
5. Avoid disk I/O inside prediction request handlers.
6. In batch prediction, process the entire batch with one vectorizer transform call.
7. Keep logging asynchronous-friendly by avoiding blocking file writes inside request handlers.
8. Rate limiting may be handled at the Uvicorn or reverse proxy layer without changing application code.

## Security and Reliability Requirements

1. Reject invalid inputs early with clear validation errors.
2. Use Pydantic models for request and response schemas.
3. Forbid unexpected fields where appropriate.
4. Avoid exposing implementation details in API errors.
5. Keep model artifacts outside source code.
6. Keep generated datasets and model artifacts out of Git if they are too large for normal repository usage.

## Acceptance Criteria

A solution is complete only if all of the following are true:

1. `python run_pipeline.py` completes the full pipeline after raw CSV placement.
2. All required artifacts are generated in the expected paths.
3. The vectorizer is fit only on the training split.
4. Validation and test metrics are saved separately.
5. FastAPI uses lifespan startup and exposes all required endpoints.
6. The API returns correct response schemas and sanitized errors.
7. The batch endpoint transforms the full batch at once.
8. Streamlit provides all three required pages and CSV export.
9. The same preprocessing implementation is used for training and inference.
10. All files contain executable, finished code with no placeholders or TODOs.
11. The README explains setup, local run, deployment, environment variables, and artifacts.

## Constraints

1. Do not skip any required file.
2. Do not leave any function unimplemented.
3. Do not add placeholder text such as "you can add this yourself".
4. Do not add extra third-party libraries.
5. Do not use deprecated FastAPI `on_event`.
6. Do not use deprecated Pydantic v1 `validator`.
7. Do not use `print` statements in API code.
8. Do not leak data by fitting the vectorizer on validation or test data.
9. Do not approximate preprocessing during inference.
10. Do not require more than one command to run the training and evaluation pipeline.
