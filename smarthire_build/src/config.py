"""
src/config.py
--------------
Single source of truth for paths, constants, and hyperparameters.
Every other module imports from here instead of hardcoding paths.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent  # repo root (smarthire/)

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Raw dataset filenames — rename downloaded Kaggle files to match these,
# or edit these constants to match whatever Kaggle actually names them.
RESUME_DATASET_FILE = RAW_DIR / "resume_dataset.csv"
NAUKRI_JOBS_FILE = RAW_DIR / "naukri_jobs.csv"
LINKEDIN_JOBS_FILE = RAW_DIR / "linkedin_jobs.csv"
INDEED_JOBS_FILE = RAW_DIR / "indeed_jobs.csv"  # optional

# Interim / processed outputs
MERGED_JOB_CORPUS_RAW = INTERIM_DIR / "job_corpus_merged.csv"        # post-merge, pre-clean
JOB_CORPUS_CLEAN = PROCESSED_DIR / "job_corpus_clean.csv"            # final, model-ready
RESUME_DATASET_CLEAN = PROCESSED_DIR / "resume_dataset_clean.csv"    # cleaned resumes

# Model artifact paths
CLASSIFIER_MODEL_PATH = MODELS_DIR / "classifier.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
FIT_PREDICTOR_MODEL_PATH = MODELS_DIR / "fit_predictor.pkl"  # Phase 3

# ---------------------------------------------------------------------------
# Common schema (target column names after merging/cleaning)
# ---------------------------------------------------------------------------
JOB_CORPUS_COLUMNS = ["title", "company", "location", "skills", "description", "experience", "source"]
RESUME_COLUMNS = ["category", "resume_text"]

# Best-effort mapping from known raw column names -> our common schema.
# Extend these if your downloaded CSVs use different headers.
NAUKRI_COLUMN_MAP = {
    "title": "title",
    "company": "company",
    "location": "location",
    "skills": "skills",
    "job-description": "description",
    "experience": "experience",
}

LINKEDIN_COLUMN_MAP = {
    "title": "title",
    "description": "description",
    "skills": "skills",
}

INDEED_COLUMN_MAP = {
    "Title": "title",
    "Company": "company",
    "Location": "location",
    "Skills": "skills",
    "Description": "description",
    "Experience": "experience",
}

RESUME_COLUMN_MAP = {
    "Category": "category",
    "Resume": "resume_text",
}

# ---------------------------------------------------------------------------
# Random seed / reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Text cleaning / vectorization params
# ---------------------------------------------------------------------------
MIN_TOKEN_LEN = 2
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.9

# ---------------------------------------------------------------------------
# Classifier params (Phase 1)
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
CLASSIFIER_CANDIDATES = ["logistic_regression", "random_forest", "svm"]

# ---------------------------------------------------------------------------
# Phase 2 — Recommender params
# ---------------------------------------------------------------------------
TOP_N_JOBS = 10

JOB_TFIDF_MAX_FEATURES = 8000
JOB_TFIDF_NGRAM_RANGE = (1, 2)
JOB_TFIDF_MIN_DF = 2
JOB_TFIDF_MAX_DF = 0.85

JOB_TFIDF_VECTORIZER_PATH = MODELS_DIR / "job_tfidf_vectorizer.pkl"
JOB_VECTORS_PATH = MODELS_DIR / "job_vectors.pkl"

# Phase 2 — Clustering params
KMEANS_K_RANGE = (4, 15)
DEFAULT_K = 8
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans.pkl"
CLUSTER_LABELS_PATH = MODELS_DIR / "cluster_labels.pkl"

# Phase 2 — Skill-gap params
TOP_SKILLS_PER_CLUSTER = 15

# ---------------------------------------------------------------------------
# Phase 3 — Fit Predictor params
# ---------------------------------------------------------------------------
FIT_PREDICTOR_MODEL_PATH = MODELS_DIR / "fit_predictor.pkl"
FIT_FEATURE_NAMES_PATH  = MODELS_DIR / "fit_feature_names.pkl"

# Ratio of negative (mismatched) pairs to positive (matched) pairs when
# building the synthetic training set from resume + job corpus
NEG_POS_RATIO = 3
