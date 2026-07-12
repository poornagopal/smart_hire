"""
src/models/skill_gap.py
-------------------------
Phase 2 INSIGHT: Skill-Gap Report.

Given a candidate resume and a target job cluster, this module:
  1. Extracts the candidate's skill tokens from their cleaned resume text.
  2. Extracts the cluster's "skill profile" — the top-TF-IDF terms from all jobs in
     that cluster (using the already-fitted job TF-IDF vectorizer's feature names +
     per-cluster mean TF-IDF scores).
  3. Computes the gap: skills prominent in the cluster but absent/weak in the resume.
  4. Returns a structured gap report dict suitable for display in the notebook and
     later in the Streamlit app (Phase 3).

This module has no external dependencies beyond scikit-learn and numpy.
"""

import logging
from collections import Counter

import numpy as np
import pandas as pd

from src.config import TOP_SKILLS_PER_CLUSTER
from src.data.preprocess import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Tokens that survive text cleaning but are not meaningful skills — filter them out
# of the cluster skill profile.
NON_SKILL_TOKENS = {
    "experience", "work", "knowledge", "ability", "skills", "role", "team", "working",
    "years", "year", "required", "position", "job", "candidate", "company", "strong",
    "good", "like", "looking", "responsibilities", "include", "including", "related",
    "plus", "also", "need", "needs", "must", "preferred", "advantage", "candidate",
    "equal", "opportunity", "employer",
}


# ---------------------------------------------------------------------------
# Step 1: Extract candidate skills from resume
# ---------------------------------------------------------------------------

def extract_candidate_skills(resume_raw_text: str, top_n: int = 30) -> list:
    """
    Returns the top-n most frequent meaningful tokens from the cleaned resume text.
    These proxy the candidate's skill set for comparison against the cluster profile.
    """
    cleaned = clean_text(resume_raw_text)
    tokens = [t for t in cleaned.split() if t not in NON_SKILL_TOKENS and len(t) > 2]
    freq = Counter(tokens)
    return [token for token, _ in freq.most_common(top_n)]


# ---------------------------------------------------------------------------
# Step 2: Compute the cluster skill profile
# ---------------------------------------------------------------------------

def get_cluster_skill_profile(
    cluster_id: int,
    job_corpus: pd.DataFrame,
    job_vectors,          # sparse TF-IDF matrix (rows = job postings)
    cluster_labels: np.ndarray,
    vectorizer,           # fitted TfidfVectorizer (has .get_feature_names_out())
    top_n: int = TOP_SKILLS_PER_CLUSTER,
) -> list:
    """
    Returns the top-n TF-IDF skill terms that characterize a given cluster.
    Computed as the mean TF-IDF score across all jobs in the cluster.
    """
    cluster_mask = cluster_labels == cluster_id
    if cluster_mask.sum() == 0:
        logger.warning("Cluster %d has no jobs — returning empty profile.", cluster_id)
        return []

    cluster_matrix = job_vectors[cluster_mask]
    mean_scores = np.asarray(cluster_matrix.mean(axis=0)).flatten()

    feature_names = vectorizer.get_feature_names_out()
    top_indices = np.argsort(mean_scores)[::-1]

    skills = []
    for idx in top_indices:
        term = feature_names[idx]
        # Skip bigrams that look like noise, and any non-skill stopword tokens
        if term not in NON_SKILL_TOKENS and mean_scores[idx] > 0:
            skills.append(term)
        if len(skills) >= top_n:
            break

    return skills


# ---------------------------------------------------------------------------
# Step 3: Compute the gap
# ---------------------------------------------------------------------------

def compute_skill_gap(
    candidate_skills: list,
    cluster_skills: list,
) -> dict:
    """
    Returns a structured skill-gap dict:
      - matched:  skills the candidate already has (in both candidate & cluster profile)
      - missing:  skills the cluster values that the candidate lacks
      - extra:    skills the candidate has that aren't in the cluster profile (not wrong,
                  just not a strong signal for this role family)
      - match_pct: percentage of cluster skills the candidate covers
    """
    candidate_set = set(candidate_skills)
    cluster_set = set(cluster_skills)

    matched = sorted(candidate_set & cluster_set)
    missing = sorted(cluster_set - candidate_set)
    extra = sorted(candidate_set - cluster_set)

    match_pct = round(100 * len(matched) / max(len(cluster_set), 1), 1)

    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "match_pct": match_pct,
        "n_cluster_skills": len(cluster_set),
        "n_candidate_skills": len(candidate_set),
    }


# ---------------------------------------------------------------------------
# Full pipeline: one call from the app or notebook
# ---------------------------------------------------------------------------

def generate_skill_gap_report(
    resume_raw_text: str,
    cluster_id: int,
    job_corpus: pd.DataFrame,
    job_vectors,
    cluster_labels: np.ndarray,
    vectorizer,
    candidate_top_n: int = 30,
    cluster_top_n: int = TOP_SKILLS_PER_CLUSTER,
) -> dict:
    """
    End-to-end convenience function used by the notebook and the Streamlit app.

    Returns:
        {
          "cluster_id": int,
          "cluster_size": int,
          "candidate_skills": [...],
          "cluster_skills": [...],
          "gap": { "matched": [...], "missing": [...], "extra": [...], "match_pct": float, ... }
        }
    """
    candidate_skills = extract_candidate_skills(resume_raw_text, top_n=candidate_top_n)
    cluster_skills = get_cluster_skill_profile(
        cluster_id, job_corpus, job_vectors, cluster_labels, vectorizer, top_n=cluster_top_n
    )
    gap = compute_skill_gap(candidate_skills, cluster_skills)

    cluster_size = int((cluster_labels == cluster_id).sum())
    report = {
        "cluster_id": cluster_id,
        "cluster_size": cluster_size,
        "candidate_skills": candidate_skills,
        "cluster_skills": cluster_skills,
        "gap": gap,
    }

    logger.info(
        "Skill-gap report: cluster=%d (%d jobs)  matched=%.1f%%  missing=%d skills",
        cluster_id, cluster_size, gap["match_pct"], len(gap["missing"]),
    )
    return report


def format_report_for_display(report: dict) -> str:
    """Human-readable text rendering of the gap report, for the CLI and the notebook."""
    g = report["gap"]
    lines = [
        f"Cluster #{report['cluster_id']}  ({report['cluster_size']} jobs in this role family)",
        f"Skill match: {g['match_pct']}% ({len(g['matched'])} of {g['n_cluster_skills']} cluster skills)",
        "",
        f"✅  Skills you already have ({len(g['matched'])}):",
        "    " + (", ".join(g["matched"]) if g["matched"] else "—"),
        "",
        f"❌  Skills to add for this role family ({len(g['missing'])}):",
        "    " + (", ".join(g["missing"]) if g["missing"] else "—"),
        "",
        f"ℹ️   Your other skills not in this cluster's profile ({len(g['extra'])}):",
        "    " + (", ".join(g["extra"][:10]) if g["extra"] else "—"),
    ]
    return "\n".join(lines)
