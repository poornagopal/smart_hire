"""
tests/test_features.py
-------------------------
Basic unit tests for Phase 1 code (no real Kaggle data required — these test pure
functions: text cleaning and the TF-IDF vectorizer builder).

Run:
    pytest tests/
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import clean_text
from src.features.text_features import build_tfidf_vectorizer, fit_transform_texts


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("Python Developer!!! 5+ Years") == "python developer years"


def test_clean_text_removes_urls_and_emails():
    text = "Contact me at jane@example.com or visit http://example.com for more info"
    cleaned = clean_text(text)
    assert "@" not in cleaned
    assert "http" not in cleaned
    assert "example" not in cleaned  # stripped along with the URL/email tokens


def test_clean_text_handles_non_string_input():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_clean_text_drops_short_tokens_and_stopwords():
    cleaned = clean_text("I am a Data Scientist and I love it")
    assert "data" in cleaned
    assert "scientist" in cleaned
    # short/stopword tokens like "i", "am", "a", "and" should be gone
    for stopword in ("i", "am", "a", "and"):
        assert stopword not in cleaned.split()


def test_build_tfidf_vectorizer_returns_configured_instance():
    vectorizer = build_tfidf_vectorizer()
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.max_features == 5000


def test_fit_transform_texts_produces_expected_shape():
    docs = [
        "python developer machine learning",
        "java backend developer spring boot",
        "data scientist python pandas numpy",
    ]
    matrix, vectorizer = fit_transform_texts(docs)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] == len(vectorizer.get_feature_names_out())
