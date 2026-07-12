"""
src/data/load_data.py
----------------------
Phase 1: load the raw Kaggle CSV files.

Defensive by design — since the real files have to be downloaded manually from Kaggle,
this module:
  * never crashes the whole pipeline if an OPTIONAL file is missing (e.g. Indeed dataset)
  * raises a clear, actionable error if a REQUIRED file is missing (Resume Dataset, and
    at least one of Naukri / LinkedIn for the job corpus)
  * tries a couple of common encodings, since Kaggle CSVs are often not strict UTF-8

Run directly:
    python -m src.data.load_data
"""

import logging
import sys

import pandas as pd

from src.config import (
    RESUME_DATASET_FILE,
    NAUKRI_JOBS_FILE,
    LINKEDIN_JOBS_FILE,
    INDEED_JOBS_FILE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_FILES = {
    "resume_dataset": RESUME_DATASET_FILE,
}

# At least ONE of these must exist to build a usable job corpus
JOB_FILES = {
    "naukri_jobs": NAUKRI_JOBS_FILE,
    "linkedin_jobs": LINKEDIN_JOBS_FILE,
    "indeed_jobs": INDEED_JOBS_FILE,  # optional / extra variety
}


def _read_csv_robust(path) -> pd.DataFrame:
    """Try a few encodings commonly seen in Kaggle exports before giving up."""
    last_err = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err


def check_missing_files() -> dict:
    """Returns a report of which required/optional files are present or missing."""
    report = {"required_missing": [], "job_files_found": [], "job_files_missing": []}

    for name, path in REQUIRED_FILES.items():
        if not path.exists():
            report["required_missing"].append((name, str(path)))

    for name, path in JOB_FILES.items():
        if path.exists():
            report["job_files_found"].append((name, str(path)))
        else:
            report["job_files_missing"].append((name, str(path)))

    return report


def load_resume_dataset() -> pd.DataFrame:
    """Load the ~960-resume, 25-category dataset used to train Model A (classifier)."""
    if not RESUME_DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Resume dataset not found at {RESUME_DATASET_FILE}.\n"
            "Download 'Resume Dataset' from Kaggle and save it as "
            f"'{RESUME_DATASET_FILE.name}' under data/raw/."
        )
    df = _read_csv_robust(RESUME_DATASET_FILE)
    logger.info("Loaded resume dataset: %d rows, columns=%s", len(df), list(df.columns))
    return df


def load_job_datasets() -> dict:
    """
    Load whichever job datasets are present (Naukri, LinkedIn, Indeed).
    Returns a dict of {source_name: DataFrame} for only the files that exist.
    Raises if NONE of the job datasets are present (job corpus would be empty).
    """
    loaded = {}
    for name, path in JOB_FILES.items():
        if path.exists():
            df = _read_csv_robust(path)
            logger.info("Loaded %s: %d rows, columns=%s", name, len(df), list(df.columns))
            loaded[name] = df
        else:
            logger.warning("Skipping %s — file not found at %s", name, path)

    if not loaded:
        raise FileNotFoundError(
            "No job datasets found. Download at least one of Naukri Job Listings or "
            "LinkedIn Job Postings from Kaggle and place it under data/raw/ "
            f"(expected one of: {[str(p) for p in JOB_FILES.values()]})."
        )
    return loaded


def main():
    report = check_missing_files()

    if report["required_missing"]:
        for name, path in report["required_missing"]:
            logger.error("Missing required file '%s' -> expected at %s", name, path)
        logger.error("Cannot proceed without the resume dataset. See README.md Section 2.")
        sys.exit(1)

    if not report["job_files_found"]:
        logger.error(
            "No job datasets found (looked for: %s). "
            "Download at least one before continuing.",
            [name for name, _ in report["job_files_missing"]],
        )
        sys.exit(1)

    if report["job_files_missing"]:
        logger.warning(
            "Proceeding without optional job dataset(s): %s",
            [name for name, _ in report["job_files_missing"]],
        )

    resumes = load_resume_dataset()
    jobs = load_job_datasets()

    logger.info(
        "Load complete. Resumes=%d rows. Job sources=%s",
        len(resumes),
        {k: len(v) for k, v in jobs.items()},
    )


if __name__ == "__main__":
    main()
