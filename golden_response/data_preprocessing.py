"""Shared news text preprocessing for training and inference."""

from __future__ import annotations

import logging
import re
import string
from functools import lru_cache
from pathlib import Path
from typing import Any

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

NLTK_RESOURCES: tuple[tuple[str, str], ...] = (
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
)
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_REGEX = re.compile(r"<[^>]+>")
DIGIT_REGEX = re.compile(r"\d+")
WHITESPACE_REGEX = re.compile(r"\s+")
PUNCTUATION_TABLE = str.maketrans({character: " " for character in string.punctuation})
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()
CONTENT_COLUMN = "content"
CLEAN_CONTENT_COLUMN = "clean_content"
TITLE_COLUMN = "title"
TEXT_COLUMN = "text"
EMPTY_TEXT = ""
SPACE = " "


@lru_cache(maxsize=1)
def english_stop_words() -> set[str]:
    """Return the cached English stop word set.

    Returns:
        A set of English stop words.
    """

    ensure_nltk_resources()
    return set(stopwords.words("english"))


def ensure_nltk_resources() -> None:
    """Ensure the NLTK resources required by the preprocessing pipeline exist.

    Raises:
        RuntimeError: If a required resource cannot be downloaded.
    """

    for resource_path, download_name in NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            LOGGER.info("Downloading missing NLTK resource: %s", download_name)
            if not nltk.download(download_name, quiet=True):
                raise RuntimeError(f"Unable to download NLTK resource: {download_name}")


def combine_title_and_text(title: Any, text: Any) -> str:
    """Combine the title and text fields into a single content string.

    Args:
        title: Title value from the source row.
        text: Text value from the source row.

    Returns:
        A combined and trimmed string suitable for preprocessing.
    """

    title_value = EMPTY_TEXT if title is None else str(title).strip()
    text_value = EMPTY_TEXT if text is None else str(text).strip()
    if not title_value and not text_value:
        return EMPTY_TEXT
    return SPACE.join(part for part in (title_value, text_value) if part).strip()


def clean_text(raw_text: Any) -> str:
    """Apply the full preprocessing pipeline to a single text string.

    The transformation order is lowercasing, URL stripping, HTML stripping,
    punctuation removal, digit removal, whitespace normalization, tokenization,
    stop word removal, stemming, lemmatization, and rejoining.

    Args:
        raw_text: Raw content to clean.

    Returns:
        The cleaned text string.
    """

    ensure_nltk_resources()
    if raw_text is None:
        return EMPTY_TEXT

    normalized = str(raw_text).lower()
    normalized = URL_REGEX.sub(SPACE, normalized)
    normalized = HTML_REGEX.sub(SPACE, normalized)
    normalized = normalized.translate(PUNCTUATION_TABLE)
    normalized = DIGIT_REGEX.sub(SPACE, normalized)
    normalized = WHITESPACE_REGEX.sub(SPACE, normalized).strip()
    if not normalized:
        return EMPTY_TEXT

    stop_words = english_stop_words()
    cleaned_tokens: list[str] = []
    for token in word_tokenize(normalized):
        if not token or token in stop_words:
            continue
        if not token.isalpha():
            continue
        stemmed_token = STEMMER.stem(token)
        lemmatized_token = LEMMATIZER.lemmatize(stemmed_token)
        if lemmatized_token:
            cleaned_tokens.append(lemmatized_token)

    return SPACE.join(cleaned_tokens)


def preprocess_single_news(title: Any, text: Any) -> str:
    """Preprocess a single news article from separate title and text values.

    Args:
        title: News title.
        text: News body text.

    Returns:
        The cleaned combined content.
    """

    return clean_text(combine_title_and_text(title, text))


def preprocess_content_frame(
    frame: pd.DataFrame,
    content_column: str = CONTENT_COLUMN,
    cleaned_column: str = CLEAN_CONTENT_COLUMN,
) -> pd.DataFrame:
    """Clean a dataframe containing news content.

    Args:
        frame: DataFrame with a content column.
        content_column: Name of the input content column.
        cleaned_column: Name of the output cleaned column.

    Returns:
        A cleaned copy of the input dataframe.

    Raises:
        ValueError: If the required content column is missing.
    """

    if content_column not in frame.columns:
        raise ValueError(f"Missing required content column: {content_column}")

    working_frame = frame.copy()
    working_frame = working_frame.dropna(subset=[content_column])
    working_frame = working_frame.drop_duplicates(subset=[content_column]).reset_index(drop=True)

    before_sample = working_frame[[content_column]].head(3).to_dict(orient="records")
    LOGGER.info("Preprocessing sample before cleaning: %s", before_sample)

    working_frame[cleaned_column] = working_frame[content_column].map(clean_text)
    working_frame = working_frame[working_frame[cleaned_column].str.len() > 0].reset_index(drop=True)

    after_sample = working_frame[[content_column, cleaned_column]].head(3).to_dict(orient="records")
    LOGGER.info("Preprocessing sample after cleaning: %s", after_sample)

    return working_frame
