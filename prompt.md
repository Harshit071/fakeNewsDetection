# Fake News Detection System

Build an end-to-end fake news detection system that is fully driven from a single master script.

## Pipeline Requirement

### Data Ingestion

Build an end-to-end fake news detection system that:

- Ingests and aggregates raw news CSV data from Kaggle.
- Cleans and preprocesses text using NLP techniques.
- Performs feature extraction with TF-IDF vectorization.
- Trains and evaluates a Logistic Regression binary classifier.
- Serves predictions with a production-ready FastAPI backend.
- Provides a clean Streamlit frontend for non-technical users.
- Is completely driven from a single master script, with no manual intervention.

Load `True.csv` and `Fake.csv` from `data/raw/`. Label real news as `0` and false news as `1`. Concatenate and shuffle with `random_state=42`. First, ensure that the columns `title`, `text`, `subject`, and `date` all exist, raising a descriptive `ValueError` if any are missing. Merge `title` and `text` into one column.

### Data Preprocessing

Read `merged_news.csv`. Drop null values and duplicates in the `content` column. Clean in this order: lower case, regex strip URLs, remove HTML tags, remove punctuation, remove digits, strip extra whitespace, tokenize with `nltk.word_tokenize`, remove English stopwords, apply `PorterStemmer` to each token, apply `WordNetLemmatizer` to each stemmed token, and rejoin into one string. Log a before-and-after sample of three rows so the cleaning can be visually checked. Save to `data/processed/cleaned_news.csv`.

### Feature Extraction

Load `cleaned_news.csv`. Use `TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=2, stop_words='english')`. Fit the vectorizer only on the training data to avoid leakage. Save the fitted vectorizer to `models/tfidf_vectorizer.joblib`. Log the vocabulary size and the top 20 features by average TF-IDF weight.

### Training the Model

Split the data 70/15/15 with two successive `train_test_split` calls. First split off 30% as temp, then split temp into equal validation and test halves. Use `stratify=y` and `random_state=42` in both calls. Train `LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0, class_weight='balanced')`. Log the start and end time of training. Save the trained model to `models/logistic_regression_model.joblib`.

### Model Assessment

Evaluate separately on validation and test sets. Calculate accuracy, precision, recall, F1 weighted, and per-class confusion matrix. Save the confusion matrices as seaborn heatmaps to `reports/confusion_matrix_val.png` and `reports/confusion_matrix_test.png`. Compute ROC-AUC and save the full ROC curve to `reports/roc_curve.png`. Dump all numeric metrics to `reports/evaluation_metrics.json` as a clean dict with `val` and `test` as top-level keys.

### Backend Prerequisites

Load the model and vectorizer at startup using a FastAPI lifespan context manager, not the deprecated `on_event` decorator. Store them on `app.state` so they are available in all route handlers. Always use the same preprocessing pipeline from `data_preprocessing.py` before every inference call. Do not approximate it.

Expose four endpoints:

1. `GET /` returns API name, version string, and status `ok`.
2. `GET /health` returns model loaded boolean, vectorizer loaded boolean, and timestamp.
3. `POST /predict` accepts `NewsInput` with `title` and `text`, processes the combined content, transforms it using the vectorizer, and returns the label as an int, the prediction as a string (`REAL` or `FAKE`), the confidence as a float, the `real_probability` as a float, and the `fake_probability` as a float.
4. `POST /predict/batch` takes a list of up to 50 `NewsInput` objects, returns a list of `PredictionOutput` plus a summary with total count, fake count, and real count.

Add CORS middleware allowing all origins for Streamlit compatibility. Return HTTP 503 with a clear message if model files are not found at startup. Return HTTP 422 with a descriptive message for any validation failure. Use logging throughout, with no print statements in the API code.

### Frontend Requirements

Set page config with a custom title, emoji favicon, and wide layout. Add a sidebar with an app description, tech stack list, and an explanation of what the confidence score means. Use three pages via sidebar radio navigation:

- Single Prediction page: two text areas for title and article text, an Analyze button, a `requests.post` call wrapped in `try/except` with `st.spinner` while waiting. Display the result as a styled `st.markdown` card, green background with `REAL` in large text if label is `0`, red background with `FAKE` in large text if label is `1`. Show confidence as `st.progress` and show real vs fake probabilities as a horizontal bar chart.
- Batch Prediction page: CSV file uploader, validate that `title` and `text` columns exist before sending, post to the `/predict/batch` endpoint, display results in a color-coded `st.dataframe` with green rows for real and red rows for fake, and provide a `st.download_button` to export results as CSV.
- Model Info page: load `reports/evaluation_metrics.json` and display validation and test metrics in side-by-side `st.metric` columns, then show confusion matrix and ROC curve images using `st.image` with captions.

### Data

Use the Kaggle fake news dataset with `True.csv` and `Fake.csv` in `data/raw/`. Both files contain `title`, `text`, `subject`, and `date`. Label real news as `0` and fake news as `1`. The files are placed manually in `data/raw/` before running the pipeline. Everything downstream is fully automated.

### Tech Stack

Use Python, pandas, numpy, scikit-learn, nltk, joblib, pydantic v2, FastAPI, uvicorn, Streamlit, seaborn, matplotlib, and requests. Do not use libraries outside this list except the Python standard library. Target these versions exactly:

- `fastapi==0.111.0`
- `uvicorn[standard]==0.29.0`
- `streamlit==1.35.0`
- `scikit-learn==1.4.2`
- `pandas==2.2.2`
- `numpy==1.26.4`
- `nltk==3.8.1`
- `joblib==1.4.2`
- `pydantic==2.7.1`
- `seaborn==0.13.2`
- `matplotlib==3.9.0`
- `requests==2.32.2`
- `python-multipart==0.0.9`

### Output

Every file must be fully implemented with zero placeholders. All file paths and hyperparameters must be uppercase constants at the top of each file, never hardcoded inside function bodies. Use type hints on every function signature. Use Google-style docstrings on every function and class. Use the Python logging module with timestamps throughout. Use `random_state=42` everywhere randomness is involved. Use Pydantic v2 `field_validator` syntax only, with no deprecated `validator` decorator. Running `python run_pipeline.py` after placing the two CSV files in `data/raw/` must execute the entire pipeline and produce trained model artifacts, the saved vectorizer, and evaluation reports with zero manual intervention. Put full run instructions as a block comment at the top of `run_pipeline.py` before any imports.

### Error Handling and Documentation

Handle every failure mode explicitly: missing CSV files, missing columns, empty text inputs, model files not found, email service down, malformed JSON. Every exception caught must have a descriptive log message that includes the stage name and what failed. The API must never expose stack traces in response bodies. `run_pipeline.py` must catch errors per stage, log them, and print a final summary table showing stage name, status as `PASSED` or `FAILED`, and wall-clock time in seconds. The README must document folder structure, setup steps, environment variable table, how to run locally, and deployment steps.

### Performance and Scalability

The vectorizer must be fitted once at training time and loaded from disk at API startup, never re-fitted per request. Preprocessing must run in memory with no disk I/O per prediction. The batch endpoint must process all articles in a single vectorizer transform call, not a loop of single transforms. Logging must be asynchronous-friendly, with no blocking I/O in request handlers. Rate limiting may be added at the Uvicorn or reverse proxy layer without touching application code.

### Constraints

Do not leave any function unimplemented. Do not add TODOs. Do not use placeholders such as "you can add this yourself". Do not skip files. Do not add extra libraries. Do not use print statements in API code. Do not use deprecated FastAPI or Pydantic syntax. Do not leak data by fitting the vectorizer on anything other than the training split. Do not hardcode strings or numbers inside function bodies when the value belongs in a constant. Do not approximate preprocessing; inference preprocessing must be identical to training preprocessing. One command must run everything.
