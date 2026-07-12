"""
app/streamlit_app.py
-----------------------
SmartHire — Phase 3: Full Web Portal.

Run from the repo root:
    streamlit run app/streamlit_app.py

What it does on every submission:
  1. Accept resume as file upload (PDF/DOCX/TXT) or text paste
  2. Parse -> clean text
  3. Predict resume category (Phase 1 classifier)
  4. Recommend top-N jobs (Phase 2 recommender)
  5. Identify nearest job cluster (Phase 2 clustering)
  6. Generate skill-gap report (Phase 2 skill_gap)
  7. Show fit probability per job (Phase 3 fit predictor, optional)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.config import (
    CLASSIFIER_MODEL_PATH, TFIDF_VECTORIZER_PATH, LABEL_ENCODER_PATH,
    JOB_TFIDF_VECTORIZER_PATH, JOB_VECTORS_PATH,
    KMEANS_MODEL_PATH, CLUSTER_LABELS_PATH,
    FIT_PREDICTOR_MODEL_PATH, FIT_FEATURE_NAMES_PATH,
    JOB_CORPUS_CLEAN, TOP_N_JOBS,
)
from src.data.preprocess import clean_text
from src.parsing.resume_parser import parse_resume
from src.models.recommender import recommend_jobs
from src.models.clustering import predict_cluster
from src.models.skill_gap import generate_skill_gap_report


st.set_page_config(page_title="SmartHire", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Loading ML models ...")
def load_all_models():
    required = [
        CLASSIFIER_MODEL_PATH, TFIDF_VECTORIZER_PATH, LABEL_ENCODER_PATH,
        JOB_TFIDF_VECTORIZER_PATH, JOB_VECTORS_PATH,
        KMEANS_MODEL_PATH, CLUSTER_LABELS_PATH, JOB_CORPUS_CLEAN,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return None, missing

    models = {
        "classifier"    : joblib.load(CLASSIFIER_MODEL_PATH),
        "clf_vectorizer": joblib.load(TFIDF_VECTORIZER_PATH),
        "label_encoder" : joblib.load(LABEL_ENCODER_PATH),
        "job_vectorizer": joblib.load(JOB_TFIDF_VECTORIZER_PATH),
        "job_vectors"   : joblib.load(JOB_VECTORS_PATH),
        "job_corpus"    : pd.read_csv(JOB_CORPUS_CLEAN),
        "km_bundle"     : joblib.load(KMEANS_MODEL_PATH),
        "cluster_labels": joblib.load(CLUSTER_LABELS_PATH),
        "fit_predictor" : joblib.load(FIT_PREDICTOR_MODEL_PATH)
                          if FIT_PREDICTOR_MODEL_PATH.exists() else None,
        "fit_feat_names": joblib.load(FIT_FEATURE_NAMES_PATH)
                          if FIT_FEATURE_NAMES_PATH.exists() else None,
    }
    return models, []


def render_sidebar(models):
    with st.sidebar:
        st.title("🎯 SmartHire")
        st.caption("Resume-to-Job Matching · Career Guidance Engine")
        st.divider()
        st.markdown("**How to use**")
        st.markdown("1. Upload your resume (PDF / DOCX / TXT) **or** paste the text.\n"
                    "2. Click **Analyse Resume**.\n"
                    "3. Browse matched jobs, fit scores, and your skill-gap report.")
        st.divider()
        st.markdown("**Settings**")
        top_n = st.slider("Top-N job matches", 5, 25, TOP_N_JOBS, step=1)
        has_fit = models is not None and models.get("fit_predictor") is not None
        show_fit = has_fit and st.checkbox("Show fit/shortlisting scores", value=True)
        if not has_fit:
            st.caption("Fit predictor not found. Run `python -m src.models.fit_predictor` to enable.")
        st.divider()
        st.caption("Built with scikit-learn · XGBoost · Streamlit")
    return top_n, show_fit


def get_resume_text():
    st.subheader("📄 Step 1 — Upload or Paste Your Resume")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        uploaded = st.file_uploader("Upload PDF, DOCX, or TXT",
                                    type=["pdf", "docx", "doc", "txt"])
        if uploaded:
            try:
                raw = parse_resume(uploaded.read(), uploaded.name)
                st.success(f"Parsed **{uploaded.name}** — {len(raw):,} characters.")
                return raw
            except Exception as e:
                st.error(f"Could not parse file: {e}")
    with col2:
        pasted = st.text_area("Or paste resume text here", height=250,
                              placeholder="Paste the text of your resume ...")
        if pasted.strip():
            return pasted.strip()
    return None


def compute_fit_scores(results_df, models, resume_clean):
    from src.features.match_features import build_pair_features
    fit_model  = models["fit_predictor"]["model"]
    fit_scaler = models["fit_predictor"]["scaler"]
    feat_names = models["fit_feat_names"]
    jv    = models["job_vectorizer"]
    jvecs = models["job_vectors"]
    jcorp = models["job_corpus"]
    rv = jv.transform([resume_clean])

    scores = []
    for _, row in results_df.iterrows():
        mask = jcorp["title"] == row["title"]
        ji   = jcorp.index[mask][0] if mask.any() else 0
        jvec = jvecs[jcorp.index.get_loc(ji)]
        feat = build_pair_features(resume_clean, row, rv, jvec)
        X = np.array([[feat[f] for f in feat_names]], dtype=np.float32)
        if fit_scaler:
            X = fit_scaler.transform(X)
        scores.append(float(fit_model.predict_proba(X)[0, 1]))
    return scores


def render_job_results(results_df, show_fit, models, resume_clean):
    display = results_df.copy()
    display["match_%"] = (display["match_score"] * 100).round(1).astype(str) + "%"

    if show_fit:
        with st.spinner("Computing fit scores ..."):
            fit_scores = compute_fit_scores(results_df, models, resume_clean)
        display["fit_score"] = [f"{s:.0%}" for s in fit_scores]

    show_cols = [c for c in ["title", "company", "location", "experience",
                              "match_%", "fit_score", "source"]
                 if c in display.columns]
    st.dataframe(display[show_cols], use_container_width=True, hide_index=True)


def render_match_chart(results_df):
    fig, ax = plt.subplots(figsize=(7, max(3, len(results_df) * 0.45)))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(results_df)))
    ax.barh(range(len(results_df))[::-1], results_df["match_score"].values, color=colors)
    ax.set_yticks(range(len(results_df))[::-1])
    ax.set_yticklabels(results_df["title"].values, fontsize=8)
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_title("Job Match Scores")
    ax.set_xlim(0, max(results_df["match_score"].max() * 1.15, 0.05))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_skill_gap(report):
    gap = report["gap"]
    st.metric("Skill Match", f"{gap['match_pct']}%",
              help="% of the target cluster's top skills found in your resume.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**✅ Skills you already have**")
        if gap["matched"]:
            st.success("  ·  ".join(gap["matched"]))
        else:
            st.info("No direct matches in this cluster's profile.")
    with col_b:
        st.markdown("**❌ Skills to add to your CV**")
        if gap["missing"]:
            tags = "".join(
                f"<span style='background:#e74c3c;color:white;padding:3px 10px;"
                f"border-radius:12px;margin:3px;display:inline-block;font-size:0.85rem'>"
                f"{s}</span>" for s in gap["missing"][:12]
            )
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.success("Your resume covers the key skills for this cluster!")

    if gap["extra"]:
        with st.expander("ℹ️ Your other skills (not in this cluster's profile)"):
            st.write("  ·  ".join(gap["extra"][:15]))

    # Bar chart
    items = ([(s, "Present") for s in gap["matched"]] +
             [(s, "Missing") for s in gap["missing"][:12]])
    if items:
        df2 = pd.DataFrame(items, columns=["skill", "status"])
        colors2 = df2["status"].map({"Present": "#2ecc71", "Missing": "#e74c3c"})
        fig2, ax2 = plt.subplots(figsize=(8, max(3, len(df2) * 0.38)))
        ax2.barh(df2["skill"], [1] * len(df2), color=colors2, edgecolor="white")
        ax2.set_xlim(0, 1.3)
        ax2.get_xaxis().set_visible(False)
        ax2.set_title(
            f"Skill-Gap · Cluster #{report['cluster_id']} "
            f"({report['cluster_size']} jobs) · match: {gap['match_pct']}%"
        )
        ax2.legend(handles=[
            mpatches.Patch(color="#2ecc71", label="Present in resume"),
            mpatches.Patch(color="#e74c3c", label="Missing — add to CV"),
        ], loc="lower right")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


def main():
    models, missing = load_all_models()

    if missing:
        st.error("⚠️ Model artifacts missing. Run these commands first:")
        st.code(
            "python -m src.data.load_data\n"
            "python -m src.data.preprocess\n"
            "python -m src.models.recommender\n"
            "python -m src.models.clustering\n"
            "python -m src.models.fit_predictor  # optional\n",
            language="bash",
        )
        st.info("Missing:\n" + "\n".join(f"- `{p}`" for p in missing))
        return

    top_n, show_fit = render_sidebar(models)

    st.title("🎯 SmartHire")
    st.markdown(
        "Upload your resume and get: a **ranked list of matching jobs**, "
        "a **shortlisting probability**, and a **personalised skill-gap report** — "
        "all powered by classical machine learning."
    )
    st.divider()

    raw_text = get_resume_text()
    st.divider()
    go = st.button("🚀 Analyse Resume", type="primary", use_container_width=True)

    if not go:
        st.info("Upload or paste your resume above, then click **Analyse Resume**.")
        return
    if not raw_text or not raw_text.strip():
        st.warning("Please provide resume text before analysing.")
        return

    with st.spinner("Analysing your resume ..."):
        clean = clean_text(raw_text)
        if len(clean.split()) < 20:
            st.error("Extracted text is too short. Please check your upload.")
            return

        clf_vec  = models["clf_vectorizer"].transform([clean])
        pred_idx = models["classifier"].predict(clf_vec)[0]
        category = models["label_encoder"].inverse_transform([pred_idx])[0]

        jcorp  = models["job_corpus"]
        jvecs  = models["job_vectors"]
        jvect  = models["job_vectorizer"]

        results = recommend_jobs(raw_text, jcorp, jvecs, jvect, top_n=top_n)

        km  = models["km_bundle"]["kmeans"]
        svd = models["km_bundle"]["svd"]
        cluster_id = predict_cluster(clean, jvect, km, svd)

        gap_report = generate_skill_gap_report(
            resume_raw_text=raw_text,
            cluster_id=cluster_id,
            job_corpus=jcorp,
            job_vectors=jvecs,
            cluster_labels=models["cluster_labels"],
            vectorizer=jvect,
        )

    st.success("✅ Analysis complete!")
    st.divider()

    # Section A — Category
    st.subheader("🏷️ Predicted Career Domain")
    st.markdown(
        f"<div style='background:#1f77b4;color:white;padding:8px 20px;"
        f"border-radius:20px;display:inline-block;font-weight:bold;font-size:1.05rem'>"
        f"🏷️ {category}</div>",
        unsafe_allow_html=True,
    )
    st.caption("Predicted by a TF-IDF classifier trained on 960 labelled resumes.")
    st.divider()

    # Section B — Recommendations
    st.subheader(f"💼 Top {top_n} Matching Jobs")
    tab_table, tab_chart = st.tabs(["📋 Table view", "📊 Chart view"])
    with tab_table:
        render_job_results(results, show_fit, models, clean)
    with tab_chart:
        render_match_chart(results)
    st.divider()

    # Section C — Skill-gap
    st.subheader("🔍 Skill-Gap Report")
    st.caption(f"Based on the top skills of **Job Cluster #{cluster_id}** — "
               f"the role family your resume maps to ({gap_report['cluster_size']} jobs in this cluster).")
    render_skill_gap(gap_report)


if __name__ == "__main__":
    main()
