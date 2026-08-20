import pandas as pd
from src.model import train


def test_risk_pipeline():
    model = train("data/patients.csv")
    sample = pd.DataFrame([{
        "age": 70, "bmi": 31, "systolic_bp": 160, "cholesterol": 230,
        "smoker": 1, "diabetes": 1, "prior_visits": 4, "insurance_type": "Medicare"
    }])
    probability = model.predict_proba(sample)[:, 1]
    assert 0 <= probability[0] <= 1
