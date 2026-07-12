"""
src/models/recommender.py
--------------------------
Phase 2 CORE ENGINE: Content-based job recommender using TF-IDF + cosine similarity.

Steps:
  1. Vectorize the entire job corpus with a dedicated TF-IDF vectorizer (separate from
     the Phase 1 resume classifier vectorizer so each can be tuned independently).
  2. Pre-compute and save the job vector matrix so repeat queries are fast.
  3. At query time: transform the resume text into the same vector space and compute
     cosine similarity against every job posting, returning the top-N matches.

Run directly (after src/data/preprocess.py has produced data/processed/job_corpus_clean.csv):
    python -m src.models.recommender
"""

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    JOB_CORPUS_CLEAN,
    JOB_TFIDF_VECTORIZER_PATH,
    JOB_VECTORS_PATH,
    TOP_N_JOBS,
    JOB_TFIDF_MAX_FEATURES,
    JOB_TFIDF_NGRAM_RANGE,
    JOB_TFIDF_MIN_DF,
    JOB_TFIDF_MAX_DF,
    MODELS_DIR,
)
from src.data.preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading the job corpus
# ---------------------------------------------------------------------------

def load_job_corpus() -> pd.DataFrame:
    if not JOB_CORPUS_CLEAN.exists():
        raise FileNotFoundError(
            f"{JOB_CORPUS_CLEAN} not found. "
            "Run `python -m src.data.preprocess` first to build the clean job corpus."
        )
    df = pd.read_csv(JOB_CORPUS_CLEAN)
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    logger.info("Loaded job corpus: %d postings", len(df))
    return df


# ---------------------------------------------------------------------------
# Building / saving the job vector matrix
# ---------------------------------------------------------------------------

def build_job_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=JOB_TFIDF_MAX_FEATURES,
        ngram_range=JOB_TFIDF_NGRAM_RANGE,
        min_df=JOB_TFIDF_MIN_DF,
        max_df=JOB_TFIDF_MAX_DF,
        sublinear_tf=True,
    )


def fit_job_vectors(job_corpus: pd.DataFrame):
    """
    Fit a new TF-IDF vectorizer on the job corpus and pre-compute all job vectors.
    Returns (sparse_matrix, fitted_vectorizer).
    Saves both to models/.
    """
    vectorizer = build_job_vectorizer()
    matrix = vectorizer.fit_transform(job_corpus["clean_text"])
    logger.info("Job vectors: %d docs x %d features", matrix.shape[0], matrix.shape[1])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, JOB_TFIDF_VECTORIZER_PATH)
    joblib.dump(matrix, JOB_VECTORS_PATH)
    logger.info("Saved job TF-IDF vectorizer -> %s", JOB_TFIDF_VECTORIZER_PATH)
    logger.info("Saved job vector matrix    -> %s", JOB_VECTORS_PATH)
    return matrix, vectorizer


def load_job_vectors():
    """Load pre-computed job vectors + vectorizer from disk."""
    if not JOB_VECTORS_PATH.exists() or not JOB_TFIDF_VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Pre-computed job vectors not found. "
            "Run `python -m src.models.recommender` first to build them."
        )
    matrix = joblib.load(JOB_VECTORS_PATH)
    vectorizer = joblib.load(JOB_TFIDF_VECTORIZER_PATH)
    logger.info("Loaded job vectors: %s", matrix.shape)
    return matrix, vectorizer


# ---------------------------------------------------------------------------
# Recommendation (query time)
# ---------------------------------------------------------------------------

def recommend_jobs(
    resume_raw_text: str,
    job_corpus: pd.DataFrame,
    job_vectors=None,
    vectorizer: TfidfVectorizer = None,
    top_n: int = TOP_N_JOBS,
) -> pd.DataFrame:
    """
    Given raw resume text (not yet cleaned), returns the top-N most similar jobs.

    Args:
        resume_raw_text: the raw resume text (will be cleaned internally)
        job_corpus: DataFrame produced by load_job_corpus()
        job_vectors: pre-computed sparse matrix; loaded from disk if None
        vectorizer: fitted TfidfVectorizer; loaded from disk if None
        top_n: how many results to return

    Returns:
        DataFrame with columns [title, company, location, skills, experience, source,
        match_score] sorted by match_score descending.
    """
    if job_vectors is None or vectorizer is None:
        job_vectors, vectorizer = load_job_vectors()

    cleaned = clean_text(resume_raw_text)
    if not cleaned.strip():
        raise ValueError("Resume text is empty after cleaning — check the input.")

    resume_vec = vectorizer.transform([cleaned])
    scores = cosine_similarity(resume_vec, job_vectors).flatten()

    top_indices = np.argsort(scores)[::-1][:top_n]
    results = job_corpus.iloc[top_indices].copy()
    results["match_score"] = scores[top_indices]
    results = results.reset_index(drop=True)

    display_cols = [c for c in
        ["title", "company", "location", "skills", "experience", "source", "match_score"]
        if c in results.columns]
    return results[display_cols]


# ---------------------------------------------------------------------------
# Precision@K evaluation (qualitative + quantitative)
# ---------------------------------------------------------------------------

def evaluate_recommendation(resume_text: str, true_category: str,
                            job_corpus: pd.DataFrame, k: int = 10) -> dict:
    """
    Lightweight Precision@K evaluation: checks how many of the top-K recommended jobs
    contain the expected category keyword in their title or description.

    This is a heuristic proxy — a proper evaluation would need human-labelled (resume, job)
    pairs. Use qualitatively in the notebook for the written report.
    """
    results = recommend_jobs(resume_text, job_corpus, top_n=k)
    keyword = true_category.lower()
    hits = results["title"].str.lower().str.contains(keyword, na=False).sum()
    precision_at_k = hits / k
    return {"k": k, "hits": int(hits), "precision_at_k": round(precision_at_k, 3)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    job_corpus = load_job_corpus()
    fit_job_vectors(job_corpus)
    logger.info("Recommender setup complete. Job vectors are ready for query-time use.")


if __name__ == "__main__":
    main()
