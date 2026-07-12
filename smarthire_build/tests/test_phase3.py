"""
tests/test_phase3.py
----------------------
Unit tests for Phase 3 modules (resume parser, match features, fit predictor logic).
No real Kaggle data required.

Run:
    pytest tests/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.match_features import jaccard, _token_set, build_pair_features
from src.models.recommender import build_job_vectorizer
from src.parsing.resume_parser import parse_resume


# ---------------------------------------------------------------------------
# Match features
# ---------------------------------------------------------------------------

def test_jaccard_identical_sets():
    s = {"python", "pandas"}
    assert jaccard(s, s) == 1.0


def test_jaccard_disjoint_sets():
    assert jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_partial_overlap():
    a = {"python", "pandas", "numpy"}
    b = {"python", "sql", "numpy"}
    j = jaccard(a, b)
    assert 0 < j < 1
    # intersection=2, union=4 -> 0.5
    assert abs(j - 0.5) < 1e-9


def test_jaccard_empty_sets():
    assert jaccard(set(), set()) == 0.0


def _make_test_pair():
    texts = [
        "data scientist python pandas sklearn machine learning model",
        "web developer react javascript html css frontend",
    ]
    vectorizer = build_job_vectorizer()
    vectorizer.min_df = 1
    matrix = vectorizer.fit_transform(texts)

    job_row = pd.Series({
        "title": "Data Scientist",
        "skills": "python pandas sklearn machine learning",
        "clean_text": texts[0],
    })
    resume_vec = matrix[0]
    job_vec    = matrix[0]
    return "python pandas machine learning data scientist", job_row, resume_vec, job_vec


def test_build_pair_features_returns_expected_keys():
    resume_text, job_row, rv, jv = _make_test_pair()
    feat = build_pair_features(resume_text, job_row, rv, jv)
    for key in ["skill_overlap", "title_overlap", "text_similarity",
                "resume_token_count", "job_token_count"]:
        assert key in feat, f"Missing feature: {key}"


def test_build_pair_features_self_similarity_is_high():
    resume_text, job_row, rv, jv = _make_test_pair()
    feat = build_pair_features(resume_text, job_row, rv, jv)
    # Same vector vs itself -> cosine similarity should be 1.0
    assert feat["text_similarity"] >= 0.99


def test_build_pair_features_scores_in_range():
    resume_text, job_row, rv, jv = _make_test_pair()
    feat = build_pair_features(resume_text, job_row, rv, jv)
    assert 0.0 <= feat["skill_overlap"] <= 1.0
    assert 0.0 <= feat["text_similarity"] <= 1.0
    assert feat["resume_token_count"] > 0


# ---------------------------------------------------------------------------
# Resume parser
# ---------------------------------------------------------------------------

def test_parse_resume_plain_text():
    text = "Python developer with 3 years experience in machine learning and data science."
    result = parse_resume(text.encode("utf-8"), "resume.txt")
    assert "Python" in result


def test_parse_resume_unsupported_extension_raises():
    import pytest
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_resume(b"data", "resume.xyz")
