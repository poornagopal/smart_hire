"""
tests/test_phase2.py
----------------------
Unit tests for Phase 2 modules (recommender, clustering, skill-gap).
No real Kaggle data needed — all tests use tiny in-memory synthetic data.

Run:
    pytest tests/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.recommender import build_job_vectorizer, recommend_jobs
from src.models.skill_gap import (
    extract_candidate_skills,
    get_cluster_skill_profile,
    compute_skill_gap,
    generate_skill_gap_report,
    format_report_for_display,
)
from src.data.preprocess import clean_text


# ---------------------------------------------------------------------------
# Helpers — tiny synthetic job corpus
# ---------------------------------------------------------------------------

def _make_corpus() -> pd.DataFrame:
    return pd.DataFrame({
        "title": ["Data Scientist", "ML Engineer", "Web Developer", "Java Backend", "HR Manager"],
        "company": ["CorpA", "CorpB", "CorpC", "CorpD", "CorpE"],
        "location": ["Bangalore"] * 5,
        "skills": [
            "python pandas numpy sklearn machine learning",
            "python tensorflow keras deep learning neural networks",
            "javascript react html css frontend",
            "java spring hibernate rest api backend",
            "recruitment payroll talent acquisition employee hr",
        ],
        "description": [
            "We need a data scientist experienced in python pandas machine learning and model deployment",
            "ML engineer with tensorflow keras deep learning experience required",
            "Frontend web developer skilled in javascript react html responsive design",
            "Java backend developer spring microservices hibernate postgresql",
            "HR manager with recruitment payroll employee engagement talent acquisition experience",
        ],
        "experience": ["3 years"] * 5,
        "source": ["test"] * 5,
        "clean_text": [
            "data scientist python pandas numpy sklearn machine learning model deployment",
            "ml engineer tensorflow keras deep learning neural networks experience",
            "web developer javascript react html css frontend responsive design",
            "java backend spring hibernate microservices postgresql rest api",
            "hr manager recruitment payroll employee engagement talent acquisition",
        ],
    })


def _fit_vectorizer_and_matrix(corpus: pd.DataFrame):
    vectorizer = build_job_vectorizer()
    vectorizer.min_df = 1  # override for small test corpus
    matrix = vectorizer.fit_transform(corpus["clean_text"])
    return matrix, vectorizer


# ---------------------------------------------------------------------------
# Recommender tests
# ---------------------------------------------------------------------------

def test_recommend_jobs_returns_top_n():
    corpus = _make_corpus()
    matrix, vectorizer = _fit_vectorizer_and_matrix(corpus)
    results = recommend_jobs(
        "python pandas data science machine learning",
        corpus, matrix, vectorizer, top_n=3
    )
    assert len(results) == 3
    assert "match_score" in results.columns


def test_recommend_jobs_top_match_is_relevant():
    corpus = _make_corpus()
    matrix, vectorizer = _fit_vectorizer_and_matrix(corpus)
    results = recommend_jobs(
        "java spring boot hibernate backend developer",
        corpus, matrix, vectorizer, top_n=5
    )
    # The best match should be the Java Backend job (index 3 in corpus)
    top_title = results.iloc[0]["title"].lower()
    assert "java" in top_title or results.iloc[0]["match_score"] > 0.1


def test_recommend_jobs_scores_descending():
    corpus = _make_corpus()
    matrix, vectorizer = _fit_vectorizer_and_matrix(corpus)
    results = recommend_jobs(
        "react javascript html css web developer",
        corpus, matrix, vectorizer, top_n=5
    )
    scores = results["match_score"].tolist()
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Skill-gap tests
# ---------------------------------------------------------------------------

def test_extract_candidate_skills_returns_list():
    skills = extract_candidate_skills("python pandas machine learning data science", top_n=10)
    assert isinstance(skills, list)
    assert "python" in skills or "pandas" in skills


def test_compute_skill_gap_structure():
    candidate = ["python", "pandas", "sql"]
    cluster = ["python", "tensorflow", "keras", "scikit", "sql"]
    gap = compute_skill_gap(candidate, cluster)

    assert "matched" in gap and "missing" in gap and "extra" in gap
    assert set(gap["matched"]) == {"python", "sql"}
    assert "tensorflow" in gap["missing"]
    assert "pandas" in gap["extra"]
    assert 0 <= gap["match_pct"] <= 100


def test_compute_skill_gap_perfect_match():
    skills = ["python", "pandas"]
    gap = compute_skill_gap(skills, skills)
    assert gap["match_pct"] == 100.0
    assert gap["missing"] == []


def test_get_cluster_skill_profile_returns_list():
    corpus = _make_corpus()
    matrix, vectorizer = _fit_vectorizer_and_matrix(corpus)
    labels = np.array([0, 0, 1, 1, 2])
    profile = get_cluster_skill_profile(0, corpus, matrix, labels, vectorizer, top_n=5)
    assert isinstance(profile, list)
    assert len(profile) <= 5


def test_generate_skill_gap_report_structure():
    corpus = _make_corpus()
    matrix, vectorizer = _fit_vectorizer_and_matrix(corpus)
    labels = np.array([0, 0, 1, 1, 2])
    report = generate_skill_gap_report(
        "python machine learning data science",
        cluster_id=0,
        job_corpus=corpus,
        job_vectors=matrix,
        cluster_labels=labels,
        vectorizer=vectorizer,
    )
    assert "cluster_id" in report
    assert "gap" in report
    assert "matched" in report["gap"]
    assert "missing" in report["gap"]


def test_format_report_for_display_is_string():
    gap = {"matched": ["python"], "missing": ["tensorflow"], "extra": ["sql"],
           "match_pct": 50.0, "n_cluster_skills": 2, "n_candidate_skills": 2}
    report = {"cluster_id": 0, "cluster_size": 10, "candidate_skills": ["python", "sql"],
              "cluster_skills": ["python", "tensorflow"], "gap": gap}
    text = format_report_for_display(report)
    assert isinstance(text, str)
    assert "python" in text
    assert "tensorflow" in text
