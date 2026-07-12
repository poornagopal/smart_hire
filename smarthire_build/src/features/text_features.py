"""
src/features/text_features.py
-------------------------------
Reusable TF-IDF vectorization helpers. Used by the Phase 1 resume classifier and will be
reused unchanged by the Phase 2 recommender / clustering modules (same vector space model).
"""

import logging

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF, TFIDF_MAX_DF

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_tfidf_vectorizer(**overrides) -> TfidfVectorizer:
    """
    Returns a TfidfVectorizer configured with project defaults (see src/config.py).
    Pass overrides like build_tfidf_vectorizer(max_features=2000) to tweak per-use-case.
    """
    params = dict(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,
    )
    params.update(overrides)
    return TfidfVectorizer(**params)


def fit_transform_texts(texts, vectorizer: TfidfVectorizer = None):
    """
    Fit (or reuse) a TF-IDF vectorizer on a list/Series of already-cleaned text.
    Returns (sparse_matrix, fitted_vectorizer).
    """
    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    logger.info(
        "TF-IDF fit complete: %d docs x %d features", matrix.shape[0], matrix.shape[1]
    )
    return matrix, vectorizer


def save_vectorizer(vectorizer: TfidfVectorizer, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)
    logger.info("Saved TF-IDF vectorizer -> %s", path)


def load_vectorizer(path) -> TfidfVectorizer:
    return joblib.load(path)
