# SmartHire — Resume-to-Job Matching & Career Guidance Engine

A complete, end-to-end classical Machine Learning project (no LLMs or generative AI).
A candidate uploads their resume and SmartHire returns:

- 🏷️ **Predicted career domain** — supervised TF-IDF classifier (25 categories)
- 💼 **Top-N matching jobs** — cosine similarity recommender
- 🔍 **Skill-gap report** — K-Means cluster profile vs resume skill tokens
- 🎯 **Fit/shortlisting score** — XGBoost / Logistic Regression predictor (optional)

> **Team:** 2–3 members · **Stack:** Python, scikit-learn, XGBoost, Streamlit

---

## Project Phases

| Phase | Scope | Status |
|---|---|---|
| **1** | Data loading, cleaning, EDA, Resume Category Classifier | ✅ Complete |
| **2** | Job Recommender, Job Clustering, Skill-Gap Report | ✅ Complete |
| **3** | Fit Predictor, Resume Parser, Streamlit Portal, Final Report | ✅ Complete |

---

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Datasets (download from Kaggle, place in `data/raw/`)

| Dataset | Kaggle search | Filename |
|---|---|---|
| Resume Dataset (~960 rows, 25 categories) | "Resume Dataset" snehaanbhawal | `resume_dataset.csv` |
| Naukri Job Listings | "Naukri Job Listings India" | `naukri_jobs.csv` |
| LinkedIn Job Postings 2023–2024 | "LinkedIn Job Postings" arshkon | `linkedin_jobs.csv` |
| Indeed / Job Recommendation (optional) | "Indeed Job Postings" | `indeed_jobs.csv` |

## 3. Run the Full Pipeline

```bash
# Phase 1 — data + classifier
python -m src.data.load_data
python -m src.data.preprocess
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_resume_classifier.ipynb

# Phase 2 — recommender + clustering
python -m src.models.recommender
python -m src.models.clustering
jupyter notebook notebooks/03_recommender.ipynb
jupyter notebook notebooks/04_clustering_topics.ipynb

# Phase 3 — fit predictor + app
python -m src.models.fit_predictor
jupyter notebook notebooks/05_fit_predictor.ipynb

# Launch the Streamlit portal
streamlit run app/streamlit_app.py
```

## 4. Tests

```bash
pytest tests/    # 24 tests, all phases
```

## 5. Deliverables Checklist

### Phase 1
- [x] `src/data/load_data.py` — defensive Kaggle file loader
- [x] `src/data/preprocess.py` — text cleaning + merged job corpus
- [x] `src/features/text_features.py` — reusable TF-IDF builder
- [x] `src/models/classifier.py` — LogReg / RF / SVM comparison + save best
- [x] `src/evaluate.py` — shared metrics, confusion matrix
- [x] `notebooks/01_eda.ipynb`, `02_resume_classifier.ipynb`

### Phase 2
- [x] `src/models/recommender.py` — cosine similarity job ranking
- [x] `src/models/clustering.py` — K-Means, elbow/silhouette sweep, PCA projection
- [x] `src/models/skill_gap.py` — cluster profile → gap report
- [x] `notebooks/03_recommender.ipynb`, `04_clustering_topics.ipynb`

### Phase 3
- [x] `src/parsing/resume_parser.py` — PDF / DOCX / TXT extractor
- [x] `src/features/match_features.py` — (resume, job) pair feature engineering
- [x] `src/models/fit_predictor.py` — LogReg vs XGBoost, ROC-AUC comparison
- [x] `notebooks/05_fit_predictor.ipynb` — ROC curves, feature importance
- [x] `app/streamlit_app.py` — full working portal
- [x] `tests/` — 24 unit tests across all phases

## 6. Stretch Goals (attempt if time allows)

- [ ] Replace TF-IDF with sentence embeddings (`sentence-transformers`)
- [ ] Deploy to Streamlit Community Cloud — add `requirements.txt` to repo, push to GitHub, connect at https://share.streamlit.io
- [ ] Add LDA topic modelling in `04_clustering_topics.ipynb` for richer skill themes
- [ ] Add a rule-based mentor that answers "What skills am I missing for Data Analyst?" from the skill-gap output

## 7. Team

| Name | Role | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| _Member 1_ | Data | Corpus merge, cleaning | Job vectorization | Resume parser, app data flow |
| _Member 2_ | Modeling | Resume classifier | Recommender, K-Means | Fit predictor |
| _Member 3_ | Analysis | EDA, README | Skill-gap, cluster plots | App UI, report, presentation |

_(Fill in actual names before submission.)_
