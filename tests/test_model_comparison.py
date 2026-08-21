import pandas as pd

from src.model_comparison import run_comparison


def test_model_comparison_returns_ranked_metrics(tmp_path):
    output = tmp_path / "model_comparison.csv"
    result = run_comparison("data/patients.csv", output)

    assert list(result.columns) == [
        "model", "precision", "recall", "f1", "roc_auc"
    ]
    assert len(result) == 3
    assert result["roc_auc"].between(0, 1).all()
    assert output.exists()
    assert pd.read_csv(output).shape[0] == 3
