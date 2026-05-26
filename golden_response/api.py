"""FastAPI application for fake news predictions."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data_preprocessing import preprocess_single_news
from ml_core import API_NAME, API_VERSION, MODEL_PATH, VECTORIZER_PATH, load_artifacts
from schemas import (
    BatchPredictionResponse,
    BatchSummary,
    HealthResponse,
    NewsInput,
    PredictionOutput,
    RootResponse,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
APP_TITLE = "Fake News Detection API"
APP_DESCRIPTION = "Production-ready prediction API for a Kaggle fake-news classifier."
APP_VERSION = "1.0.0"
MAX_BATCH_SIZE = 50
REAL_LABEL = 0
FAKE_LABEL = 1
REAL_LABEL_NAME = "REAL"
FAKE_LABEL_NAME = "FAKE"
STARTUP_ERROR_FIELD = "startup_error"
MODEL_LOADED_FIELD = "model_loaded"
VECTORIZER_LOADED_FIELD = "vectorizer_loaded"
TIMESTAMP_FIELD = "timestamp"
PREDICTIONS_FIELD = "predictions"
SUMMARY_FIELD = "summary"
TOTAL_COUNT_FIELD = "total_count"
FAKE_COUNT_FIELD = "fake_count"
REAL_COUNT_FIELD = "real_count"
STATUS_OK = "ok"
MODEL_NOT_READY_MESSAGE = "Model or vectorizer artifacts are unavailable. Train the pipeline first."
INVALID_BATCH_MESSAGE = f"Batch size must be between 1 and {MAX_BATCH_SIZE}."
INVALID_DOCUMENT_MESSAGE = "Preprocessing produced an empty document."
VALIDATION_FAILED_MESSAGE = "Validation failed"
INTERNAL_ERROR_MESSAGE = "Internal server error"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts before the application starts serving requests.

    Args:
        app: FastAPI application instance.

    Yields:
        Control back to FastAPI after startup.
    """

    app.state.model = None
    app.state.vectorizer = None
    app.state.startup_error = None
    try:
        artifacts = load_artifacts()
        app.state.model = artifacts.model
        app.state.vectorizer = artifacts.vectorizer
        LOGGER.info("Loaded model and vectorizer during startup")
    except FileNotFoundError as exc:
        app.state.startup_error = str(exc)
        LOGGER.exception("Startup failed because trained artifacts are missing")
    except Exception as exc:
        app.state.startup_error = str(exc)
        LOGGER.exception("Startup failed while loading artifacts")
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A configured FastAPI app.
    """

    app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        """Return a concise 422 response for validation failures.

        Args:
            _: The request object.
            exc: Validation exception.

        Returns:
            JSON response with validation details.
        """

        LOGGER.warning("Validation failure: %s", exc.errors())
        return JSONResponse(
            status_code=422,
            content={"detail": VALIDATION_FAILED_MESSAGE, "errors": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        """Return a sanitized 500 response for unexpected errors.

        Args:
            _: The request object.
            exc: Unhandled exception.

        Returns:
            JSON response with a generic error message.
        """

        LOGGER.exception("Unhandled API exception")
        return JSONResponse(status_code=500, content={"detail": INTERNAL_ERROR_MESSAGE})

    return app


app = create_app()


def get_artifacts(request: Request) -> tuple[LogisticRegression, TfidfVectorizer]:
    """Read the loaded model and vectorizer from application state.

    Args:
        request: FastAPI request object.

    Returns:
        A tuple of model and vectorizer.

    Raises:
        HTTPException: If artifacts are unavailable.
    """

    model = request.app.state.model
    vectorizer = request.app.state.vectorizer
    if model is None or vectorizer is None:
        message = request.app.state.startup_error or MODEL_NOT_READY_MESSAGE
        raise HTTPException(status_code=503, detail=message)
    return model, vectorizer


def class_probability_map(model: LogisticRegression, probabilities: list[float]) -> dict[int, float]:
    """Map model class labels to probabilities.

    Args:
        model: Trained classifier.
        probabilities: Predicted probability vector.

    Returns:
        A mapping from label to probability.
    """

    class_map = {int(label): index for index, label in enumerate(model.classes_)}
    if REAL_LABEL not in class_map or FAKE_LABEL not in class_map:
        raise HTTPException(status_code=500, detail="Model classes must include 0 and 1")
    return {
        REAL_LABEL: float(probabilities[class_map[REAL_LABEL]]),
        FAKE_LABEL: float(probabilities[class_map[FAKE_LABEL]]),
    }


def build_prediction_output(model: LogisticRegression, probabilities: list[float]) -> PredictionOutput:
    """Build a prediction payload from model probabilities.

    Args:
        model: Trained classifier.
        probabilities: Probability vector for a single record.

    Returns:
        A PredictionOutput instance.
    """

    probability_map = class_probability_map(model, probabilities)
    label = REAL_LABEL if probability_map[REAL_LABEL] >= probability_map[FAKE_LABEL] else FAKE_LABEL
    prediction = REAL_LABEL_NAME if label == REAL_LABEL else FAKE_LABEL_NAME
    confidence = max(probability_map[REAL_LABEL], probability_map[FAKE_LABEL])
    return PredictionOutput(
        label=label,
        prediction=prediction,
        confidence=float(confidence),
        real_probability=float(probability_map[REAL_LABEL]),
        fake_probability=float(probability_map[FAKE_LABEL]),
    )


def preprocess_documents(payload: list[NewsInput]) -> list[str]:
    """Preprocess one or more news payloads for inference.

    Args:
        payload: News articles submitted for prediction.

    Returns:
        A list of cleaned news documents.

    Raises:
        HTTPException: If any cleaned document is empty.
    """

    cleaned_documents: list[str] = []
    for index, item in enumerate(payload):
        cleaned_document = preprocess_single_news(item.title, item.text)
        if not cleaned_document.strip():
            raise HTTPException(status_code=422, detail=f"{INVALID_DOCUMENT_MESSAGE} at index {index}")
        cleaned_documents.append(cleaned_document)
    return cleaned_documents


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    """Return the API metadata.

    Returns:
        A root response object.
    """

    return RootResponse(api_name=API_NAME, version=API_VERSION, status=STATUS_OK)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return the current health state of the API.

    Args:
        request: FastAPI request object.

    Returns:
        A health response object.
    """

    return HealthResponse(
        model_loaded=request.app.state.model is not None,
        vectorizer_loaded=request.app.state.vectorizer is not None,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: NewsInput, request: Request) -> PredictionOutput:
    """Predict whether a single article is real or fake.

    Args:
        payload: News input payload.
        request: FastAPI request object.

    Returns:
        A prediction output object.
    """

    model, vectorizer = get_artifacts(request)
    cleaned_document = preprocess_documents([payload])[0]
    feature_matrix = vectorizer.transform([cleaned_document])
    probabilities = model.predict_proba(feature_matrix)[0].tolist()
    return build_prediction_output(model, probabilities)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: list[NewsInput], request: Request) -> BatchPredictionResponse:
    """Predict labels for a batch of articles.

    Args:
        payload: Batch of news articles.
        request: FastAPI request object.

    Returns:
        Batch prediction response.
    """

    if not payload or len(payload) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=422, detail=INVALID_BATCH_MESSAGE)

    model, vectorizer = get_artifacts(request)
    cleaned_documents = preprocess_documents(payload)
    feature_matrix = vectorizer.transform(cleaned_documents)
    probabilities_matrix = model.predict_proba(feature_matrix)

    predictions = [build_prediction_output(model, probabilities.tolist()) for probabilities in probabilities_matrix]
    summary = BatchSummary(
        total_count=len(predictions),
        fake_count=sum(1 for item in predictions if item.label == FAKE_LABEL),
        real_count=sum(1 for item in predictions if item.label == REAL_LABEL),
    )
    return BatchPredictionResponse(predictions=predictions, summary=summary)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
