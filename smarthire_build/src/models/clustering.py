"""
src/models/clustering.py
--------------------------
Phase 2 DISCOVERY: K-Means clustering on job vectors to uncover natural job families.

Workflow:
  1. Load pre-computed job TF-IDF vectors (built by recommender.py).
  2. Run a k sweep (elbow method + silhouette score) to choose the best k.
  3. Fit the final K-Means model and persist cluster labels alongside the job corpus.
  4. PCA reduction to 2D for the t-SNE / scatter visualization in the notebook.

Run directly (after recommender.py has saved the job vectors):
    python -m src.models.clustering
"""

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score

from src.config import (
    JOB_CORPUS_CLEAN,
    KMEANS_K_RANGE,
    DEFAULT_K,
    RANDOM_STATE,
    KMEANS_MODEL_PATH,
    CLUSTER_LABELS_PATH,
    MODELS_DIR,
    FIGURES_DIR,
)
from src.models.recommender import load_job_vectors, load_job_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Elbow + silhouette sweep
# ---------------------------------------------------------------------------

def sweep_k(job_vectors, k_min: int = None, k_max: int = None) -> pd.DataFrame:
    """
    Runs K-Means for each k in [k_min, k_max] and records inertia + silhouette score.
    Returns a DataFrame with columns [k, inertia, silhouette] for plotting in the notebook.

    Note: silhouette is computed on a random 5,000-sample subset for speed when the
    corpus is large (full pairwise distances are O(n²)).
    """
    k_min = k_min or KMEANS_K_RANGE[0]
    k_max = k_max or KMEANS_K_RANGE[1]

    # Work in dense PCA-reduced space for silhouette (sparse matrices are slow)
    svd = TruncatedSVD(n_components=50, random_state=RANDOM_STATE)
    reduced = svd.fit_transform(job_vectors)

    # Subsample for silhouette if corpus is large
    n = reduced.shape[0]
    sil_sample = min(n, 5000)
    rng = np.random.default_rng(RANDOM_STATE)
    sil_idx = rng.choice(n, size=sil_sample, replace=False)

    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(reduced)
        inertia = km.inertia_
        sil = silhouette_score(reduced[sil_idx], labels[sil_idx]) if n > k else 0.0
        rows.append({"k": k, "inertia": inertia, "silhouette": round(sil, 4)})
        logger.info("k=%2d  inertia=%.1f  silhouette=%.4f", k, inertia, sil)

    return pd.DataFrame(rows)


def pick_best_k(sweep_df: pd.DataFrame) -> int:
    """
    Simple heuristic: pick the k with the highest silhouette score.
    The notebook shows the full elbow + silhouette plots for the written narrative.
    """
    best = sweep_df.loc[sweep_df["silhouette"].idxmax(), "k"]
    logger.info("Best k by silhouette: %d", best)
    return int(best)


# ---------------------------------------------------------------------------
# Fit the final clustering model
# ---------------------------------------------------------------------------

def fit_clustering(job_vectors, k: int) -> tuple:
    """
    Fits K-Means in the TruncatedSVD-reduced space (50 dims) for stability.
    Returns (kmeans_model, svd_transformer, labels_array).
    """
    svd = TruncatedSVD(n_components=50, random_state=RANDOM_STATE)
    reduced = svd.fit_transform(job_vectors)

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(reduced)

    sil = silhouette_score(reduced, labels)
    logger.info("Final K-Means k=%d  inertia=%.1f  silhouette=%.4f", k, km.inertia_, sil)
    return km, svd, labels


def save_clustering(km, svd, labels):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kmeans": km, "svd": svd}, KMEANS_MODEL_PATH)
    joblib.dump(labels, CLUSTER_LABELS_PATH)
    logger.info("Saved K-Means model -> %s", KMEANS_MODEL_PATH)
    logger.info("Saved cluster labels -> %s", CLUSTER_LABELS_PATH)


def load_clustering() -> tuple:
    bundle = joblib.load(KMEANS_MODEL_PATH)
    labels = joblib.load(CLUSTER_LABELS_PATH)
    return bundle["kmeans"], bundle["svd"], labels


# ---------------------------------------------------------------------------
# 2-D projection for visualization (PCA on TruncatedSVD output)
# ---------------------------------------------------------------------------

def get_2d_projection(job_vectors, svd) -> np.ndarray:
    """
    Project job vectors to 2D via TruncatedSVD (already fitted) → PCA(2 components).
    Suitable for a scatter plot coloured by cluster label.
    """
    reduced_50 = svd.transform(job_vectors)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(reduced_50)
    logger.info(
        "2-D PCA variance explained: %.1f%%",
        pca.explained_variance_ratio_.sum() * 100,
    )
    return coords


# ---------------------------------------------------------------------------
# Predict which cluster a new resume belongs to
# ---------------------------------------------------------------------------

def predict_cluster(resume_clean_text: str, job_vectorizer, km, svd) -> int:
    """
    Used by the Streamlit app (Phase 3) and the skill-gap module:
    maps a resume into the job vector space and returns the nearest cluster.
    """
    from src.data.preprocess import clean_text
    vec = job_vectorizer.transform([resume_clean_text])
    reduced = svd.transform(vec)
    return int(km.predict(reduced)[0])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    job_vectors, vectorizer = load_job_vectors()
    job_corpus = load_job_corpus()

    logger.info("Running k sweep from %d to %d ...", *KMEANS_K_RANGE)
    sweep_df = sweep_k(job_vectors)
    best_k = pick_best_k(sweep_df)

    km, svd, labels = fit_clustering(job_vectors, k=best_k)
    save_clustering(km, svd, labels)

    # Attach cluster labels to corpus and log a quick summary
    job_corpus["cluster"] = labels
    summary = job_corpus.groupby("cluster")["title"].count().rename("n_jobs")
    logger.info("Cluster sizes:\n%s", summary.to_string())


if __name__ == "__main__":
    main()
