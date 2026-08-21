# Healthcare Risk Analytics

[![Healthcare ML CI](https://github.com/mahaveerreddysamala/healthcare-risk-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/mahaveerreddysamala/healthcare-risk-analytics/actions/workflows/ci.yml)

A reproducible healthcare machine-learning portfolio project that demonstrates data quality checks, exploratory analysis, leakage-safe preprocessing, model comparison, threshold-based risk classification, evaluation artifacts, automated testing, CI, and a Streamlit dashboard.

> **Important:** This repository is an educational portfolio demonstration, not a clinical decision-support system. The included dataset is intentionally tiny and synthetic for reproducibility. Perfect-looking metrics on 10 rows are **not** evidence of clinical performance or production readiness.

## Portfolio highlights

- **Data analysis:** Pandas-based profiling, missing-value checks, target distribution, and numeric summaries.
- **ML pipeline:** Imputation, scaling, categorical encoding, and model training inside scikit-learn pipelines.
- **Model comparison:** Logistic Regression, Random Forest, and Gradient Boosting.
- **Risk evaluation:** Configurable probability threshold with accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and ROC outputs.
- **Testing:** pytest integration/unit tests with coverage reporting.
- **Automation:** GitHub Actions validates dependencies, compilation, tests, EDA, model comparison, and evaluation on every push and pull request.
- **Delivery:** CI publishes generated evaluation reports as a workflow artifact for review.
- **Dashboard:** Streamlit app for quick exploration of risk distribution and patient-level signals.
- **Responsible ML:** Explicit discussion of leakage, class imbalance, false-negative risk, interpretability, validation, privacy, fairness, and governance.

## Project workflow

```text
patients.csv
    ↓
Data validation / EDA
    ↓
Leakage-safe preprocessing
    ├── numeric imputation + scaling
    └── categorical imputation + one-hot encoding
    ↓
Model comparison
    ├── Logistic Regression
    ├── Random Forest
    └── Gradient Boosting
    ↓
Train/test evaluation
    ↓
Threshold-based classification
    ↓
Metrics + confusion matrix + ROC + feature importance
    ↓
Streamlit dashboard
```

## Repository structure

```text
healthcare-risk-analytics/
├── app.py
├── data/
│   └── patients.csv
├── docs/
│   ├── architecture.md
│   ├── INTERVIEW_GUIDE.md
│   └── PORTFOLIO.md
├── notebooks/
├── sql/
├── src/
│   ├── eda.py
│   ├── evaluate.py
│   ├── model.py
│   ├── model_comparison.py
│   └── download_data.py
├── tests/
├── .github/workflows/ci.yml
├── Makefile
├── pytest.ini
└── requirements.txt
```

## Run locally

```bash
cd C:\Users\mahav\GitHubProjects\healthcare-risk-analytics
python -m pip install -r requirements.txt
pytest -q --cov=src --cov-report=term-missing
python -m src.eda
python -m src.model_comparison
python -m src.evaluate
streamlit run app.py
```

The evaluation pipeline writes generated artifacts to `reports/`, including:

- `metrics.json`
- `confusion_matrix.csv`
- `roc_curve.csv`
- `feature_importance.csv`
- model comparison and EDA summaries/charts

Generated reports and test/coverage files are excluded from Git via `.gitignore`. CI uploads the reports as a downloadable workflow artifact.

## Current validation snapshot

The latest local validation completed with **5 tests passing and 89% total source coverage**. The small demonstration dataset produces 1.00 metrics in the current split; this is expected to be treated as a reproducibility check rather than a meaningful estimate of real-world healthcare performance.

## CI / CD / portfolio deployment

**CI:** Every push and pull request runs dependency installation, Python compilation, pytest + coverage, EDA generation, model comparison, and evaluation. Generated reports are retained as a CI artifact.

**CD:** The Streamlit application is deployment-ready from the repository root. For portfolio delivery, connect this GitHub repository to Streamlit Community Cloud and set the entry point to `app.py`; subsequent pushes to the selected branch can trigger an application redeploy. Keep deployment credentials and environment-specific secrets outside the repository.

**Portfolio presentation:** See [`docs/PORTFOLIO.md`](docs/PORTFOLIO.md) for the recommended recruiter-facing story, architecture explanation, validation commands, deployment checklist, and interview talking points.

## Responsible ML

The project explicitly considers class imbalance, false-negative risk, feature leakage, model interpretability, and threshold selection. Any real healthcare deployment would require substantially larger representative data, clinical validation, privacy/security controls, fairness analysis, monitoring, governance, and appropriate regulatory review.
