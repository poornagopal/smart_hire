# Target Project Structure

This matches Section 7 of the project brief. Folders that don't yet have real content (Phase 2/3)
exist as placeholders so the layout never has to be restructured later — only files get added.

```
smarthire/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/            # <- put downloaded Kaggle CSVs here (git-ignored)
│   ├── interim/        # merged-but-not-fully-cleaned data (auto-generated)
│   └── processed/      # final model-ready data (auto-generated)
│
├── notebooks/
│   ├── 01_eda.ipynb                  # ✅ Phase 1
│   ├── 02_resume_classifier.ipynb    # ✅ Phase 1
│   ├── 03_recommender.ipynb          # ⏳ Phase 2
│   ├── 04_clustering_topics.ipynb    # ⏳ Phase 2
│   └── 05_fit_predictor.ipynb        # ⏳ Phase 3 (optional)
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # ✅ Phase 1 — paths/constants used by every phase
│   ├── evaluate.py                   # ✅ Phase 1 — shared metrics helpers
│   ├── data/
│   │   ├── load_data.py              # ✅ Phase 1
│   │   └── preprocess.py             # ✅ Phase 1
│   ├── features/
│   │   ├── text_features.py          # ✅ Phase 1 (TF-IDF), reused in Phase 2
│   │   └── match_features.py         # ⏳ Phase 3 (skill overlap, experience match)
│   ├── models/
│   │   ├── classifier.py             # ✅ Phase 1
│   │   ├── recommender.py            # ⏳ Phase 2
│   │   ├── clustering.py             # ⏳ Phase 2
│   │   └── fit_predictor.py          # ⏳ Phase 3 (optional)
│   └── parsing/
│       └── resume_parser.py          # ⏳ Phase 3 (PDF/DOCX upload parsing for the app)
│
├── models/                            # saved .pkl artifacts (git-ignored, auto-generated)
│
├── app/
│   └── streamlit_app.py              # ⏳ Phase 3
│
├── reports/
│   ├── figures/                       # plots saved by notebooks
│   └── final_report.pdf               # ⏳ Phase 3
│
└── tests/
    └── test_features.py               # ✅ Phase 1 starter tests
```
