"""Pydantic schemas for the fake news API."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

EMPTY_TEXT = ""
REAL_LABEL_NAME = "REAL"
FAKE_LABEL_NAME = "FAKE"
REQUEST_TITLE = "title"
REQUEST_TEXT = "text"
PREDICTION_LABEL = "label"
PREDICTION_NAME = "prediction"
CONFIDENCE_FIELD = "confidence"
REAL_PROBABILITY_FIELD = "real_probability"
FAKE_PROBABILITY_FIELD = "fake_probability"
TOTAL_COUNT_FIELD = "total_count"
FAKE_COUNT_FIELD = "fake_count"
REAL_COUNT_FIELD = "real_count"
MODEL_LOADED_FIELD = "model_loaded"
VECTORIZER_LOADED_FIELD = "vectorizer_loaded"
TIMESTAMP_FIELD = "timestamp"
API_NAME_FIELD = "api_name"
VERSION_FIELD = "version"
STATUS_FIELD = "status"
PREDICTIONS_FIELD = "predictions"
SUMMARY_FIELD = "summary"
STATUS_OK = "ok"


class NewsInput(BaseModel):
    """News article payload used for prediction requests."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    title: str = Field(..., description="News title")
    text: str = Field(..., description="News article body")

    @field_validator("title", "text", mode="before")
    @classmethod
    def strip_and_validate_text(cls, value: object) -> str:
        """Strip whitespace and reject empty text fields.

        Args:
            value: Input value supplied by the client.

        Returns:
            A trimmed string.

        Raises:
            ValueError: If the value is missing or empty.
        """

        if value is None:
            raise ValueError("Field cannot be empty")
        text_value = str(value).strip()
        if not text_value:
            raise ValueError("Field cannot be empty")
        return text_value


class PredictionOutput(BaseModel):
    """Prediction result returned by the API."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    label: int
    prediction: str
    confidence: float
    real_probability: float
    fake_probability: float

    @field_validator("prediction")
    @classmethod
    def uppercase_prediction(cls, value: object) -> str:
        """Normalize the prediction text to uppercase.

        Args:
            value: Prediction string.

        Returns:
            Uppercase prediction string.
        """

        return str(value).upper()


class BatchSummary(BaseModel):
    """Summary statistics for batch predictions."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    total_count: int
    fake_count: int
    real_count: int


class BatchPredictionResponse(BaseModel):
    """Batch prediction response payload."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    predictions: list[PredictionOutput]
    summary: BatchSummary


class RootResponse(BaseModel):
    """Root endpoint response payload."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    api_name: str
    version: str
    status: str


class HealthResponse(BaseModel):
    """Health endpoint response payload."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model_loaded: bool
    vectorizer_loaded: bool
    timestamp: str
