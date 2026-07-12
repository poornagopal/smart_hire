"""
src/models/classifier.py
--------------------------
Model A: Resume Category Classifier (supervised).

Pipeline: clean resume text (already done in src/data/preprocess.py) -> TF-IDF ->
compare Logistic Regression / Random Forest / Linear SVM -> save the best model.

Run directly (after preprocess.py has produced data/processed/resume_dataset_clean.csv):
    python -m src.models.classifier
"""

import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from src.config import (
    RESUME_DATASET_CLEAN,
    CLASSIFIER_MODEL_PATH,
    TFIDF_VECTORIZER_PATH,
    LABEL_ENCODER_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.evaluate import classification_metrics, print_classification_report, compare_models
from src.features.text_features import build_tfidf_vectorizer, save_vectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


MODEL_BUILDERS = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
    ),
    # LinearSVC has no predict_proba; fine here since we only need predicted labels
    "svm": lambda: LinearSVC(random_state=RANDOM_STATE, class_weight="balanced"),
}


def load_clean_resumes() -> pd.DataFrame:
    if not RESUME_DATASET_CLEAN.exists():
        raise FileNotFoundError(
            f"{RESUME_DATASET_CLEAN} not found. Run `python -m src.data.preprocess` first."
        )
    return pd.read_csv(RESUME_DATASET_CLEAN)


def train_and_compare(df: pd.DataFrame):
    """
    Trains all candidate models on the same TF-IDF features and split, evaluates each,
    and returns (best_model_name, best_model, vectorizer, label_encoder, results_dict, split).
    """
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["category"])

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = build_tfidf_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    results = {}
    fitted_models = {}

    for name, builder in MODEL_BUILDERS.items():
        logger.info("Training %s ...", name)
        model = builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = classification_metrics(y_test, y_pred)
        results[name] = metrics
        fitted_models[name] = model

        logger.info("%s -> %s", name, metrics)
        print_classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    compare_models(results)

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = fitted_models[best_name]
    logger.info("Best model: %s (F1=%.3f)", best_name, results[best_name]["f1"])

    split = {"X_test": X_test, "y_test": y_test, "y_pred_best": best_model.predict(X_test)}
    return best_name, best_model, vectorizer, label_encoder, results, split


def save_artifacts(model, vectorizer, label_encoder) -> None:
    CLASSIFIER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CLASSIFIER_MODEL_PATH)
    save_vectorizer(vectorizer, TFIDF_VECTORIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    logger.info("Saved classifier -> %s", CLASSIFIER_MODEL_PATH)
    logger.info("Saved label encoder -> %s", LABEL_ENCODER_PATH)


def predict_category(resume_text_clean: str) -> str:
    """Inference helper for later use by the Streamlit app (Phase 3)."""
    model = joblib.load(CLASSIFIER_MODEL_PATH)
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    vec = vectorizer.transform([resume_text_clean])
    pred_idx = model.predict(vec)[0]
    return label_encoder.inverse_transform([pred_idx])[0]


def main():
    df = load_clean_resumes()
    logger.info("Loaded %d cleaned resumes across %d categories", len(df), df["category"].nunique())

    best_name, best_model, vectorizer, label_encoder, results, _ = train_and_compare(df)
    save_artifacts(best_model, vectorizer, label_encoder)

    logger.info("Phase 1 classifier training complete. Best model: %s", best_name)


if __name__ == "__main__":
    main()
