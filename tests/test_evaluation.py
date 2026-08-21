from pathlib import Path

from src.evaluate import evaluate


def test_evaluate_writes_expected_artifacts(tmp_path):
    metrics = evaluate("data/patients.csv", tmp_path, threshold=0.40)

    assert metrics["threshold"] == 0.40
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    expected = {
        "metrics.json",
        "confusion_matrix.csv",
        "roc_curve.csv",
        "feature_importance.csv",
    }
    assert expected.issubset({p.name for p in Path(tmp_path).iterdir()})
