"""
src/models/fit_predictor.py
-----------------------------
Phase 3 (Optional): Model B — Shortlisting / Fit Predictor.

Input : engineered features from a (resume, job) pair (see src/features/match_features.py)
Output: fit probability (0–1) — how likely is this resume to be shortlisted for this job?

Trains two models and compares:
  - Logistic Regression  (fast baseline, interpretable coefficients)
  - XGBoost              (tree-based, handles non-linear interactions)

Training data is synthetic: positive pairs = (resume, same-category job),
negative pairs = (resume, different-category job). This is a reasonable proxy
because real shortlisting signal is not publicly available in the Kaggle datasets.

Run directly (needs Phase 1+2 artifacts in models/):
    python -m src.models.fit_predictor
"""

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    RESUME_DATASET_CLEAN,
    FIT_PREDICTOR_MODEL_PATH,
    FIT_FEATURE_NAMES_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    NEG_POS_RATIO,
)
from src.features.match_features import build_training_features, build_pair_features
from src.models.recommender import load_job_corpus, load_job_vectors

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_clean_resumes() -> pd.DataFrame:
    if not RESUME_DATASET_CLEAN.exists():
        raise FileNotFoundError(
            f"{RESUME_DATASET_CLEAN} not found. Run `python -m src.data.preprocess` first."
        )
    return pd.read_csv(RESUME_DATASET_CLEAN)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _model_candidates(scale_for_lr: bool = True):
    """Returns a dict of {name: (model, needs_scaling)}."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
            True,
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                verbosity=0,
            ),
            False,   # XGBoost doesn't need feature scaling
        ),
    }


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_and_compare(X: np.ndarray, y: np.ndarray, feature_names: list) -> tuple:
    """
    Trains and compares LogReg + XGBoost on the same stratified split.
    Returns (best_model_name, best_model, scaler_or_None, results_dict).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = {}
    fitted_models = {}
    scalers = {}

    for name, (model, needs_scale) in _model_candidates().items():
        logger.info("Training %s ...", name)

        if needs_scale:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_train)
            Xte = scaler.transform(X_test)
        else:
            scaler = None
            Xtr, Xte = X_train, X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1]

        results[name] = {
            "accuracy" : round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1"       : round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc"  : round(roc_auc_score(y_test, y_proba), 4),
        }
        fitted_models[name] = model
        scalers[name] = scaler

        logger.info("%s -> %s", name, results[name])
        logger.info("\n%s", classification_report(y_test, y_pred, zero_division=0))

    # Feature importance (LR coefficients + XGB)
    _log_feature_importance(fitted_models, feature_names, scalers)

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    logger.info("Best model: %s (ROC-AUC=%.4f)", best_name, results[best_name]["roc_auc"])

    return best_name, fitted_models[best_name], scalers[best_name], results


def _log_feature_importance(fitted_models, feature_names, scalers):
    logger.info("--- Feature importance ---")
    lr = fitted_models.get("logistic_regression")
    if lr is not None:
        coefs = dict(zip(feature_names, lr.coef_[0].round(3)))
        logger.info("LR coefficients: %s", coefs)

    xgb = fitted_models.get("xgboost")
    if xgb is not None:
        imp = dict(zip(feature_names, xgb.feature_importances_.round(3)))
        logger.info("XGB importances: %s", imp)


# ---------------------------------------------------------------------------
# Saving / loading
# ---------------------------------------------------------------------------

def save_fit_predictor(model, scaler, feature_names: list) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, FIT_PREDICTOR_MODEL_PATH)
    joblib.dump(feature_names, FIT_FEATURE_NAMES_PATH)
    logger.info("Saved fit predictor -> %s", FIT_PREDICTOR_MODEL_PATH)


def load_fit_predictor():
    bundle = joblib.load(FIT_PREDICTOR_MODEL_PATH)
    feature_names = joblib.load(FIT_FEATURE_NAMES_PATH)
    return bundle["model"], bundle["scaler"], feature_names


# ---------------------------------------------------------------------------
# Inference helper (used by the Streamlit app)
# ---------------------------------------------------------------------------

def predict_fit_score(
    resume_clean_text: str,
    job_row: pd.Series,
    job_vectorizer,
    job_vectors,
    job_corpus: pd.DataFrame,
) -> float:
    """
    Returns a 0–1 fit probability for a (resume, job) pair.
    Used by the Streamlit app for each recommended job.
    """
    model, scaler, feature_names = load_fit_predictor()

    resume_vec = job_vectorizer.transform([resume_clean_text])
    job_idx = job_corpus.index.get_loc(job_row.name) if job_row.name in job_corpus.index else 0
    job_vec = job_vectors[job_idx]

    feat = build_pair_features(resume_clean_text, job_row, resume_vec, job_vec)
    X = np.array([[feat[f] for f in feature_names]], dtype=np.float32)

    if scaler is not None:
        X = scaler.transform(X)

    return float(model.predict_proba(X)[0, 1])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    resume_df = load_clean_resumes()
    job_corpus = load_job_corpus()
    job_vectors, job_vectorizer = load_job_vectors()

    logger.info("Building synthetic (resume, job) pair features ...")
    X, y, feature_names = build_training_features(
        resume_df, job_corpus, job_vectors, job_vectorizer,
        neg_pos_ratio=NEG_POS_RATIO, random_state=RANDOM_STATE,
    )

    best_name, best_model, scaler, results = train_and_compare(X, y, feature_names)
    save_fit_predictor(best_model, scaler, feature_names)

    logger.info("Phase 3 fit predictor complete. Best model: %s", best_name)
    comparison = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
    logger.info("\n%s", comparison.to_string())


if __name__ == "__main__":
    main()
