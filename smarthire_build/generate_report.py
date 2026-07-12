"""
generate_report.py
--------------------
Generates reports/final_report.pdf — the written project report.

Run from the repo root:
    python generate_report.py

Requires reportlab:
    pip install reportlab
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

OUTPUT = Path(__file__).parent / "reports" / "final_report.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

W, H = A4
MARGIN = 2.2 * cm

styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    "Title", parent=styles["Title"],
    fontSize=22, textColor=colors.HexColor("#1a1a2e"),
    spaceAfter=6,
)
subtitle_style = ParagraphStyle(
    "Subtitle", parent=styles["Normal"],
    fontSize=12, textColor=colors.HexColor("#4a4e69"),
    spaceAfter=16, alignment=TA_CENTER,
)
h1_style = ParagraphStyle(
    "H1", parent=styles["Heading1"],
    fontSize=14, textColor=colors.HexColor("#1a1a2e"),
    spaceBefore=16, spaceAfter=6,
    borderPad=4,
)
h2_style = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontSize=12, textColor=colors.HexColor("#22223b"),
    spaceBefore=12, spaceAfter=4,
)
body_style = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontSize=10, leading=15,
    spaceAfter=8, alignment=TA_JUSTIFY,
    textColor=colors.HexColor("#333333"),
)
code_style = ParagraphStyle(
    "Code", parent=styles["Code"],
    fontSize=9, leading=14,
    backColor=colors.HexColor("#f5f5f5"),
    leftIndent=12, rightIndent=12,
    spaceAfter=8,
)
note_style = ParagraphStyle(
    "Note", parent=styles["Normal"],
    fontSize=9, textColor=colors.HexColor("#555555"),
    leftIndent=12, spaceAfter=6,
)

DIVIDER = HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=6)

def h(text, style=h1_style):
    return Paragraph(text, style)

def p(text):
    return Paragraph(text, body_style)

def note(text):
    return Paragraph(f"<i>{text}</i>", note_style)

def sp(h=8):
    return Spacer(1, h)

def table(data, col_widths=None, header_color="#22223b"):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])
    t.setStyle(style)
    return t


def build_report():
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )
    story = []

    # ── Cover ──
    story += [
        sp(30),
        Paragraph("SmartHire", title_style),
        Paragraph("Resume-to-Job Matching &amp; Career Guidance Engine", subtitle_style),
        DIVIDER,
        Paragraph(
            "Industrial Project Report &nbsp;·&nbsp; Machine Learning &nbsp;·&nbsp; "
            "Supervised + Unsupervised",
            subtitle_style,
        ),
        sp(12),
        table(
            [["Field", "Value"],
             ["Project Type", "Industrial ML Project (Classical ML only — no LLMs)"],
             ["Team Size", "2–3 members"],
             ["Duration", "3 weeks"],
             ["Stack", "Python · scikit-learn · XGBoost · Streamlit"],
             ["Datasets", "Resume Dataset (Kaggle) · Naukri Job Listings · LinkedIn Job Postings"],
             ["Deliverables", "5 notebooks · 8 src modules · Streamlit app · 24 tests · this report"]],
            col_widths=[5 * cm, 12.5 * cm],
        ),
        PageBreak(),
    ]

    # ── 1. Introduction ──
    story += [
        h("1. Introduction"),
        DIVIDER,
        p("SmartHire is a complete, end-to-end machine learning system that allows a candidate "
          "to upload their resume (CV) and immediately receive: (a) a predicted career domain, "
          "(b) a ranked list of matching job postings from a curated corpus, (c) a fit/shortlisting "
          "score for each recommended job, and (d) a personalised skill-gap report that identifies "
          "which skills are in demand for their target role family."),
        p("The entire system is built using <b>classical machine learning</b> — TF-IDF "
          "vectorisation, Logistic Regression, Random Forest, Support Vector Machines, XGBoost, "
          "K-Means clustering, and cosine similarity. No large language models or generative AI "
          "components are used anywhere in the pipeline."),
        sp(),
    ]

    # ── 2. Datasets ──
    story += [
        h("2. Datasets"),
        DIVIDER,
        p("Four publicly available datasets from Kaggle were used:"),
        table(
            [["Dataset", "Rows", "Purpose"],
             ["Resume Dataset (snehaanbhawal)", "~960", "Train the resume category classifier (Model A)"],
             ["Naukri Job Listings", "~3,000+", "Indian job postings for the job corpus"],
             ["LinkedIn Job Postings 2023–24", "~27,000+", "Rich job descriptions for the corpus"],
             ["Indeed / Job Recommendation", "Optional", "Additional job variety"]],
            col_widths=[7 * cm, 2.5 * cm, 8 * cm],
        ),
        sp(),
        p("The three job datasets were merged into a single <b>job corpus</b> with a common schema: "
          "<i>title, company, location, skills, description, experience, source</i>. "
          "Duplicate postings (same title + company + location) were dropped. A single "
          "<code>clean_text</code> field (title + skills + description, cleaned and tokenised) "
          "was built for TF-IDF vectorisation."),
    ]

    # ── 3. System Architecture ──
    story += [
        h("3. System Architecture"),
        DIVIDER,
        p("All ML logic lives in <code>src/</code> and is exposed through a Streamlit web portal "
          "(<code>app/streamlit_app.py</code>). The pipeline on each resume submission is:"),
        table(
            [["Step", "Module", "Type"],
             ["1. Parse uploaded file (PDF/DOCX/TXT)", "src/parsing/resume_parser.py", "Rule-based"],
             ["2. Clean text", "src/data/preprocess.py → clean_text()", "Rule-based"],
             ["3. Predict career domain", "src/models/classifier.py", "Supervised"],
             ["4. Recommend top-N jobs", "src/models/recommender.py", "Unsupervised"],
             ["5. Assign to job cluster", "src/models/clustering.py", "Unsupervised"],
             ["6. Skill-gap report", "src/models/skill_gap.py", "Unsupervised"],
             ["7. Fit/shortlisting score", "src/models/fit_predictor.py", "Supervised (optional)"]],
            col_widths=[6.5 * cm, 6.5 * cm, 4.5 * cm],
        ),
    ]

    # ── 4. Phase 1 — Supervised: Classifier ──
    story += [
        h("4. Phase 1 — Resume Category Classifier (Model A, Supervised)"),
        DIVIDER,
        h("4.1 Text Pre-processing", h2_style),
        p("Resume text was lowercased, URLs/emails/punctuation stripped, short tokens "
          "(< 2 chars) and common stopwords removed. The result is a single "
          "<code>clean_text</code> field per resume, typically 100–400 tokens."),
        h("4.2 Feature Engineering", h2_style),
        p("TF-IDF vectorisation with bigrams (max 5,000 features, min_df=2, max_df=0.9, "
          "sublinear_tf=True). The same vectoriser parameters are reused across all "
          "three phases for consistency."),
        h("4.3 Models Compared", h2_style),
        table(
            [["Model", "Accuracy", "Weighted F1", "Notes"],
             ["Logistic Regression", "[fill in]", "[fill in]", "Fast, interpretable; baseline"],
             ["Random Forest (300 trees)", "[fill in]", "[fill in]", "Handles class imbalance well"],
             ["Linear SVM", "[fill in]", "[fill in]", "Often best on high-dim text data"]],
            col_widths=[4.5 * cm, 2.8 * cm, 2.8 * cm, 7.4 * cm],
        ),
        note("Fill in the actual metric values after running notebooks/02_resume_classifier.ipynb "
             "on the real Kaggle data. The best model (by F1) is saved to models/classifier.pkl."),
        p("All models use <code>class_weight='balanced'</code> to account for the uneven "
          "distribution across the 25 resume categories. The full confusion matrix is saved "
          "to <code>reports/figures/classifier_confusion_matrix.png</code>."),
    ]

    # ── 5. Phase 2 — Unsupervised ──
    story += [
        h("5. Phase 2 — Unsupervised Components"),
        DIVIDER,
        h("5.1 Job Recommender (Core Engine)", h2_style),
        p("A separate TF-IDF vectoriser (8,000 features, bigrams) is fitted on the full job "
          "corpus at setup time and the resulting sparse matrix is persisted to disk "
          "(<code>models/job_vectors.pkl</code>). At query time, the resume's clean text is "
          "transformed into the same vector space and cosine similarity scores are computed "
          "against every job posting. The top-N scores are returned as the ranked recommendation list."),
        p("Evaluation metric: <b>Precision@K</b> — fraction of the top-K recommended jobs "
          "whose title contains the resume's expected category keyword. "
          "Mean Precision@10 on test resumes: <b>[fill in after running notebook 03]</b>."),
        h("5.2 Job Clustering (Discovery)", h2_style),
        p("K-Means is run in a TruncatedSVD-reduced space (50 components) for numerical "
          "stability. A <b>k sweep</b> (k = 4..15) records inertia (elbow method) and "
          "silhouette score for every k. The k with the highest silhouette score is selected "
          "as the final model. PCA (2 components on the SVD-reduced space) provides a "
          "2-D scatter plot coloured by cluster for the written report."),
        table(
            [["Metric", "Value"],
             ["Chosen k", "[fill in after running notebook 04]"],
             ["Silhouette score", "[fill in]"],
             ["Inertia at best k", "[fill in]"]],
            col_widths=[6 * cm, 11.5 * cm],
        ),
        h("5.3 Skill-Gap Report (Insight)", h2_style),
        p("The candidate's resume token set is compared against the <b>cluster skill profile</b> "
          "— the top-15 TF-IDF terms (by mean score across all jobs in the cluster) that "
          "characterise the role family. The output is a structured gap report: "
          "<i>matched skills</i>, <i>missing skills to add</i>, and a <i>match percentage</i>."),
    ]

    # ── 6. Phase 3 ──
    story += [
        h("6. Phase 3 — Fit Predictor &amp; Web Portal"),
        DIVIDER,
        h("6.1 Fit Predictor (Model B, Supervised, Optional)", h2_style),
        p("A binary classifier that predicts shortlisting probability for a (resume, job) pair. "
          "Because no labelled shortlisting outcome data is publicly available, a "
          "<b>synthetic training set</b> is constructed: positive pairs (label=1) are resumes "
          "matched to same-category jobs; negative pairs (label=0) are resumes matched to "
          "random different-category jobs (ratio 1:3)."),
        h("Features:", h2_style),
        table(
            [["Feature", "Description"],
             ["skill_overlap", "Jaccard similarity between resume tokens and job skills column"],
             ["title_overlap", "Jaccard similarity between resume tokens and job title tokens"],
             ["text_similarity", "Cosine similarity in the TF-IDF job vector space"],
             ["resume_token_count", "Number of unique tokens in the resume (length proxy)"],
             ["job_token_count", "Number of unique tokens in the job description"]],
            col_widths=[4.5 * cm, 13 * cm],
        ),
        table(
            [["Model", "ROC-AUC", "F1", "Notes"],
             ["Logistic Regression", "[fill in]", "[fill in]", "Scaled features; class_weight=balanced"],
             ["XGBoost", "[fill in]", "[fill in]", "No scaling needed; tree-based"]],
            col_widths=[4.5 * cm, 3 * cm, 3 * cm, 7 * cm],
        ),
        note("Fill in the actual metric values after running notebooks/05_fit_predictor.ipynb."),
        h("6.2 Streamlit Web Portal", h2_style),
        p("The full portal is implemented in <code>app/streamlit_app.py</code> and ties "
          "together all six ML modules. The user flow is: upload file → parse (PDF/DOCX/TXT) "
          "→ predicted category → top-N job table with match scores → optional fit scores "
          "→ skill-gap bar chart. All artifacts are loaded once per session via "
          "<code>@st.cache_resource</code>. The app detects missing artifacts and shows "
          "setup instructions if any phase has not been run yet."),
        p("To run the portal: <code>streamlit run app/streamlit_app.py</code> from the repo root."),
    ]

    # ── 7. Results summary ──
    story += [
        h("7. Results Summary"),
        DIVIDER,
        table(
            [["Component", "Key Metric", "Value"],
             ["Resume Classifier (best model)", "Weighted F1", "[fill in]"],
             ["Job Recommender", "Mean Precision@10", "[fill in]"],
             ["Job Clustering", "Silhouette score at best k", "[fill in]"],
             ["Fit Predictor (best model)", "ROC-AUC", "[fill in]"]],
            col_widths=[6.5 * cm, 4.5 * cm, 6.5 * cm],
        ),
        note("Fill in this table after running all four notebooks on the real Kaggle data."),
    ]

    # ── 8. Limitations ──
    story += [
        h("8. Limitations &amp; Future Work"),
        DIVIDER,
        p("<b>Limitations:</b>"),
        p("• The fit/shortlisting predictor is trained on synthetic pairs, not real hiring outcomes. "
          "Precision and recall figures are likely optimistic and will degrade on real data."),
        p("• TF-IDF treats all tokens equally and misses semantic relationships — "
          "'ML' and 'machine learning' are treated as different features."),
        p("• The skill-gap report uses single-word tokens; multi-word skill phrases "
          "(e.g. 'deep learning', 'natural language processing') are partially captured "
          "by bigrams but a dedicated skill-phrase extractor would be more precise."),
        p("• The job corpus is a static offline snapshot; job market trends from the past "
          "year are not reflected."),
        p("<b>Future work:</b>"),
        p("• Replace TF-IDF with sentence embeddings (e.g. <code>all-MiniLM-L6-v2</code>) "
          "for semantic matching — this remains classical ML (embedding lookup + cosine "
          "similarity, no generative component)."),
        p("• Add a learning-to-rank stage (LambdaMART / XGBoost Ranker) on top of "
          "cosine similarity scores to incorporate explicit user feedback."),
        p("• Deploy the Streamlit app to Streamlit Community Cloud "
          "(push repo to GitHub, connect at https://share.streamlit.io)."),
    ]

    # ── 9. Repository structure ──
    story += [
        h("9. Repository Structure"),
        DIVIDER,
        Paragraph(
            "smarthire/ ├── README.md · requirements.txt · .gitignore<br/>"
            "├── data/ raw/ · interim/ · processed/<br/>"
            "├── notebooks/ 01_eda · 02_classifier · 03_recommender · "
            "04_clustering · 05_fit_predictor<br/>"
            "├── src/ config · evaluate · data/ · features/ · models/ · parsing/<br/>"
            "├── models/ *.pkl (git-ignored, auto-generated)<br/>"
            "├── app/ streamlit_app.py<br/>"
            "├── reports/ figures/ · final_report.pdf<br/>"
            "└── tests/ test_features · test_phase2 · test_phase3 (24 tests total)",
            code_style,
        ),
    ]

    # ── 10. References ──
    story += [
        h("10. References"),
        DIVIDER,
        p("1. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. <i>JMLR 12</i>."),
        p("2. Chen, T. &amp; Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. "
          "<i>KDD 2016</i>."),
        p("3. Salton, G. &amp; Buckley, C. (1988). Term-weighting approaches in automatic text "
          "retrieval. <i>Information Processing &amp; Management 24(5)</i>."),
        p("4. Resume Dataset — Kaggle: "
          "https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset"),
        p("5. Naukri Job Listings — Kaggle (search: 'Naukri Job Listings India')"),
        p("6. LinkedIn Job Postings 2023–2024 — Kaggle: "
          "https://www.kaggle.com/datasets/arshkon/linkedin-job-postings"),
        p("7. Streamlit — https://streamlit.io"),
    ]

    doc.build(story)
    print(f"Report saved to {OUTPUT}")


if __name__ == "__main__":
    build_report()
