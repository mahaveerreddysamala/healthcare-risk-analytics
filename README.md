# Healthcare Risk Analytics

A reproducible healthcare machine-learning portfolio project that demonstrates data quality checks, exploratory analysis, model comparison, threshold-based risk classification, evaluation artifacts, and a Streamlit dashboard.

> **Important:** This repository is an educational portfolio demonstration, not a clinical decision-support system. The included dataset is intentionally small for reproducibility, so model metrics should not be interpreted as production or clinical evidence.

## What this project demonstrates

- Python data analysis with Pandas
- Automated EDA and data-quality reporting
- Categorical encoding and numeric imputation/scaling
- Logistic Regression, Random Forest, and Gradient Boosting comparison
- Class-imbalance handling and threshold optimization
- Reproducible train/test evaluation
- Feature-importance reporting
- Unit/integration tests with pytest and coverage
- GitHub Actions CI with Python 3.11
- Streamlit analytics dashboard
- Responsible ML documentation and healthcare-specific risk considerations

## Project workflow

```text
Data → Quality Checks → EDA → Preprocessing → Model Comparison
                                      ↓
                              Risk Model Training
                                      ↓
                         Threshold-based Evaluation
                                      ↓
                     Metrics / ROC / Feature Importance
                                      ↓
                              Streamlit Dashboard
```

## Repository structure

```text
healthcare-risk-analytics/
├── app.py
├── data/
│   └── patients.csv
├── docs/
│   ├── architecture.md
│   └── INTERVIEW_GUIDE.md
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
pip install -r requirements.txt
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
- EDA summaries and charts

Generated reports and test/coverage files are excluded from Git via `.gitignore`.

## Responsible ML

The project explicitly considers class imbalance, false-negative risk, feature leakage, model interpretability, and threshold selection. Any real healthcare deployment would require substantially larger representative data, clinical validation, privacy/security controls, fairness analysis, monitoring, governance, and appropriate regulatory review.
