"""
src/data/preprocess.py
------------------------
Phase 1: text cleaning + merging the job datasets into ONE clean job corpus with a
common schema (title, company, location, skills, description, experience, source).

Also cleans the resume dataset's text column for the classifier.

Run directly (after load_data.py confirms files exist):
    python -m src.data.preprocess
"""

import logging
import re

import pandas as pd

from src.config import (
    NAUKRI_COLUMN_MAP,
    LINKEDIN_COLUMN_MAP,
    INDEED_COLUMN_MAP,
    RESUME_COLUMN_MAP,
    JOB_CORPUS_COLUMNS,
    MERGED_JOB_CORPUS_RAW,
    JOB_CORPUS_CLEAN,
    RESUME_DATASET_CLEAN,
    MIN_TOKEN_LEN,
)
from src.data.load_data import load_resume_dataset, load_job_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"http\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTISPACE_RE = re.compile(r"\s+")

# Minimal, dependency-free stopword list so this module works even before `nltk.download`
# has been run. For richer cleaning in the notebooks, swap in nltk.corpus.stopwords.
BASIC_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "as", "by", "this", "that",
    "it", "its", "from", "we", "you", "your", "i", "they", "he", "she", "will", "would",
    "can", "could", "should", "has", "have", "had", "not", "no", "do", "does", "did", "so",
    "am", "my", "me", "our", "us",
}


def clean_text(text: str) -> str:
    """
    Lowercase, strip URLs/emails/punctuation/numbers, collapse whitespace,
    drop very short tokens and basic stopwords.
    Safe on None/NaN — returns an empty string.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    tokens = [t for t in text.split() if len(t) >= MIN_TOKEN_LEN and t not in BASIC_STOPWORDS]
    return _MULTISPACE_RE.sub(" ", " ".join(tokens)).strip()


def _rename_and_select(df: pd.DataFrame, column_map: dict, source_name: str) -> pd.DataFrame:
    """Rename known columns to the common schema; fill any missing target columns with NaN."""
    df = df.rename(columns=column_map)
    for col in JOB_CORPUS_COLUMNS:
        if col != "source" and col not in df.columns:
            df[col] = pd.NA
    df["source"] = source_name
    return df[JOB_CORPUS_COLUMNS]


def merge_job_corpus(job_dfs: dict) -> pd.DataFrame:
    """
    job_dfs: dict like {"naukri_jobs": df, "linkedin_jobs": df, "indeed_jobs": df}
    (only the ones the caller actually loaded — see load_data.load_job_datasets)
    Returns one merged DataFrame with the common schema, unfiltered/unclean (raw merge).
    """
    column_maps = {
        "naukri_jobs": NAUKRI_COLUMN_MAP,
        "linkedin_jobs": LINKEDIN_COLUMN_MAP,
        "indeed_jobs": INDEED_COLUMN_MAP,
    }

    frames = []
    for source_name, df in job_dfs.items():
        col_map = column_maps.get(source_name, {})
        if not col_map:
            logger.warning(
                "No column mapping defined for '%s' — check src/config.py and update "
                "the *_COLUMN_MAP if Kaggle's headers differ from what's assumed.",
                source_name,
            )
        mapped = _rename_and_select(df, col_map, source_name)
        frames.append(mapped)
        logger.info("Mapped %s -> %d rows ready to merge", source_name, len(mapped))

    merged = pd.concat(frames, ignore_index=True)
    logger.info("Merged job corpus (raw): %d total rows from %d sources", len(merged), len(frames))
    return merged


def clean_job_corpus(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Clean + dedupe the merged job corpus:
      - drop rows with no title AND no description (unusable for matching)
      - drop near-duplicate postings (same title + company + location)
      - build a single `clean_text` field (title + skills + description) for TF-IDF
    """
    df = merged.copy()

    df = df.dropna(subset=["title", "description"], how="all")

    dedup_cols = [c for c in ["title", "company", "location"] if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    logger.info("Dropped %d duplicate postings (by %s)", before - len(df), dedup_cols)

    for col in ["title", "company", "location", "skills", "description", "experience"]:
        df[col] = df[col].astype(str).replace({"nan": "", "<NA>": ""})

    df["clean_text"] = (
        df["title"] + " " + df["skills"] + " " + df["description"]
    ).apply(clean_text)

    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    logger.info("Job corpus after cleaning: %d rows", len(df))
    return df


def clean_resume_dataset(resumes: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns and clean resume text for Model A (classifier)."""
    df = resumes.rename(columns=RESUME_COLUMN_MAP).copy()

    missing = [c for c in ("category", "resume_text") if c not in df.columns]
    if missing:
        raise KeyError(
            f"Resume dataset is missing expected column(s) {missing} after applying "
            f"RESUME_COLUMN_MAP={RESUME_COLUMN_MAP}. Inspect the raw CSV headers and "
            "update RESUME_COLUMN_MAP in src/config.py."
        )

    df["clean_text"] = df["resume_text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    logger.info(
        "Cleaned resume dataset: %d rows across %d categories",
        len(df),
        df["category"].nunique(),
    )
    return df[["category", "resume_text", "clean_text"]]


def main():
    resumes = load_resume_dataset()
    job_dfs = load_job_datasets()

    merged_jobs = merge_job_corpus(job_dfs)
    MERGED_JOB_CORPUS_RAW.parent.mkdir(parents=True, exist_ok=True)
    merged_jobs.to_csv(MERGED_JOB_CORPUS_RAW, index=False)
    logger.info("Saved raw merged job corpus -> %s", MERGED_JOB_CORPUS_RAW)

    clean_jobs = clean_job_corpus(merged_jobs)
    JOB_CORPUS_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    clean_jobs.to_csv(JOB_CORPUS_CLEAN, index=False)
    logger.info("Saved clean job corpus -> %s", JOB_CORPUS_CLEAN)

    clean_resumes = clean_resume_dataset(resumes)
    RESUME_DATASET_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    clean_resumes.to_csv(RESUME_DATASET_CLEAN, index=False)
    logger.info("Saved clean resume dataset -> %s", RESUME_DATASET_CLEAN)


if __name__ == "__main__":
    main()
