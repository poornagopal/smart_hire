"""
src/features/match_features.py
---------------------------------
Phase 3: feature engineering for the (resume, job) pair used by Model B
(fit/shortlisting predictor).

Each row represents one (resume, job posting) pair.
Features:
  - skill_overlap     : Jaccard similarity between resume tokens and job skill tokens
  - text_similarity   : cosine similarity in the shared TF-IDF job vector space
  - has_experience    : binary — does the job experience requirement appear in the resume?
  - resume_token_count: length proxy for resume detail
  - job_token_count   : length proxy for job description detail
  - title_overlap     : Jaccard between resume tokens and job title tokens
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.data.preprocess import clean_text

logger = logging.getLogger(__name__)


def _token_set(text: str) -> set:
    return set(str(text).lower().split()) if text else set()


def jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def build_pair_features(
    resume_clean_text: str,
    job_row: pd.Series,
    resume_vec,          # (1, V) sparse vector — must already be transformed
    job_vec,             # (1, V) sparse vector — same vectorizer
) -> dict:
    """
    Build the feature dict for one (resume, job) pair.
    Used during both training (loop over synthetic pairs) and inference.
    """
    resume_tokens = _token_set(resume_clean_text)
    skill_tokens  = _token_set(job_row.get("skills", ""))
    title_tokens  = _token_set(job_row.get("title", ""))

    text_sim = float(np.clip(cosine_similarity(resume_vec, job_vec)[0, 0], 0.0, 1.0))

    return {
        "skill_overlap"     : jaccard(resume_tokens, skill_tokens),
        "title_overlap"     : jaccard(resume_tokens, title_tokens),
        "text_similarity"   : text_sim,
        "resume_token_count": len(resume_tokens),
        "job_token_count"   : len(_token_set(job_row.get("clean_text", ""))),
    }


def build_training_features(
    resume_df: pd.DataFrame,      # cleaned resume dataset (category, clean_text)
    job_corpus: pd.DataFrame,     # cleaned job corpus (title, skills, clean_text, ...)
    job_vectors,                  # pre-computed sparse matrix for all jobs
    job_vectorizer,               # fitted TfidfVectorizer for jobs
    neg_pos_ratio: int = 3,
    random_state: int = 42,
) -> tuple:
    """
    Build a synthetic labelled dataset for the fit predictor.

    Positive pairs  (label=1): resume → job whose title/skills contain the resume category keyword.
    Negative pairs  (label=0): resume → randomly sampled job from a *different* category.

    Returns (X: np.ndarray, y: np.ndarray, feature_names: list).
    """
    rng = np.random.default_rng(random_state)
    rows, labels = [], []

    for _, resume_row in resume_df.iterrows():
        category_kw = resume_row["category"].lower().split()[0]   # e.g. "data" from "Data Science"
        clean_txt = resume_row["clean_text"]
        resume_vec = job_vectorizer.transform([clean_txt])

        # Positive: a job that matches the resume category
        pos_mask = job_corpus["title"].str.lower().str.contains(category_kw, na=False)
        pos_jobs = job_corpus[pos_mask]
        if pos_jobs.empty:
            continue
        pos_idx = rng.choice(len(pos_jobs))
        pos_row = pos_jobs.iloc[pos_idx]
        pos_job_vec = job_vectors[job_corpus.index.get_loc(pos_jobs.index[pos_idx])]

        feat = build_pair_features(clean_txt, pos_row, resume_vec, pos_job_vec)
        rows.append(feat)
        labels.append(1)

        # Negatives: jobs that clearly don't match
        neg_mask = ~pos_mask
        neg_jobs = job_corpus[neg_mask]
        n_neg = min(neg_pos_ratio, len(neg_jobs))
        if n_neg == 0:
            continue
        neg_indices = rng.choice(len(neg_jobs), size=n_neg, replace=False)
        for ni in neg_indices:
            neg_row = neg_jobs.iloc[ni]
            neg_job_vec = job_vectors[job_corpus.index.get_loc(neg_jobs.index[ni])]
            feat = build_pair_features(clean_txt, neg_row, resume_vec, neg_job_vec)
            rows.append(feat)
            labels.append(0)

    feature_names = list(rows[0].keys()) if rows else []
    X = np.array([[r[f] for f in feature_names] for r in rows], dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    pos_count = int(y.sum())
    logger.info(
        "Built %d training pairs (%d positive, %d negative), %d features",
        len(y), pos_count, len(y) - pos_count, len(feature_names),
    )
    return X, y, feature_names
