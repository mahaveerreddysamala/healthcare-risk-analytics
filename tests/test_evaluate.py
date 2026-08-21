import json

from src.evaluate import evaluate


def test_evaluate_creates_reproducible_artifacts(tmp_path):
    metrics = evaluate("data/patients.csv", tmp_path)

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert metrics["threshold"] == 0.40

    for name in [
        "metrics.json",
        "confusion_matrix.csv",
        "roc_curve.csv",
        "feature_importance.csv",
    ]:
        assert (tmp_path / name).exists()

    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["threshold"] == 0.40
