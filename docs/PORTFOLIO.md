# Portfolio Presentation Guide

## 1. One-minute project story

> I built a reproducible healthcare risk analytics pipeline in Python that takes a small synthetic patient dataset through data-quality validation, EDA, leakage-safe preprocessing, model comparison, threshold-based evaluation, and a Streamlit dashboard. I automated the workflow with pytest and GitHub Actions and generate machine-readable evaluation artifacts. The key engineering focus is reproducibility and avoiding data leakage; the model scores are intentionally not presented as clinical evidence because the demonstration dataset has only 10 rows.

## 2. What to show a recruiter

1. **README** — problem, architecture, limitations, and commands are visible immediately.
2. **GitHub Actions** — show the green CI workflow and the downloadable evaluation artifact.
3. **`src/model.py`** — explain the preprocessing pipeline and why transformations are fitted only on training data.
4. **`src/model_comparison.py`** — show that multiple baseline models are compared consistently.
5. **`src/evaluate.py`** — explain probability thresholding and precision/recall trade-offs.
6. **Streamlit app** — demonstrate the user-facing analytics layer.
7. **Tests** — show that EDA, modeling, comparison, and evaluation behavior is covered by automated tests.

## 3. CI/CD story

### Continuous Integration

Every push and pull request runs:

```text
Install dependencies
      ↓
Compile Python modules
      ↓
pytest + coverage
      ↓
EDA report generation
      ↓
Model comparison
      ↓
Evaluation
      ↓
Upload reports artifact
```

### Continuous Delivery

The application is packaged as a repository-root Streamlit app (`app.py`). For a portfolio deployment, connect the GitHub repository to Streamlit Community Cloud, select the portfolio branch, and use `app.py` as the entry point. Keep secrets out of Git and use the hosting platform's secret management.

The important distinction is:

- **CI** proves that the code still validates after a change.
- **CD** makes the validated application available to users.
- **Portfolio deployment** is the final public demonstration layer.

## 4. Local verification checklist

From PowerShell:

```powershell
cd C:\Users\mahav\GitHubProjects\healthcare-risk-analytics
python -m pip install -r requirements.txt
pytest -q --cov=src --cov-report=term-missing
python -m src.eda
python -m src.model_comparison
python -m src.evaluate
streamlit run app.py
```

Expected current local checks:

- 5 tests passing
- 89% total source coverage
- EDA completes successfully
- model comparison completes successfully
- evaluation writes `reports/metrics.json` and related artifacts

## 5. Git workflow for the portfolio branch

```powershell
git status
git pull --ff-only origin feature/portfolio-evaluation
# make a focused change
pytest -q --cov=src --cov-report=term-missing
git add .
git commit -m "Describe the focused portfolio improvement"
git push origin feature/portfolio-evaluation
```

Avoid rebasing a branch that already has shared remote commits unless you intentionally coordinate the history rewrite. Prefer small, focused commits so the portfolio history clearly shows incremental engineering improvements.

## 6. Interview talking points

### Why threshold 0.40?

A lower threshold can increase sensitivity to high-risk cases, which can be appropriate when false negatives are more costly than false positives. The threshold must be selected from the business/clinical objective and validated on representative data rather than chosen arbitrarily for production use.

### How did you prevent leakage?

The train/test split occurs before fitting the preprocessing pipeline. Imputation, scaling, and categorical encoding are therefore learned from training data rather than from the full dataset.

### Why are the metrics 1.00?

The demonstration dataset has only 10 rows and is intentionally simple. The result is useful as a deterministic pipeline smoke test, but it is not a reliable estimate of generalization, clinical utility, or fairness.

### What would you do next for production?

Use a larger representative dataset; add temporal and external validation; assess calibration and subgroup performance; establish privacy/security controls; monitor drift; define a model governance process; and obtain appropriate clinical/regulatory validation before deployment.
