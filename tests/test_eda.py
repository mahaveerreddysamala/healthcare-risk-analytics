from pathlib import Path

from src.eda import generate_report, profile


def test_profile_returns_expected_dataset():
    df = profile("data/patients.csv")
    assert df.shape == (10, 9)
    assert "high_risk" in df.columns


def test_generate_report_creates_artifacts(tmp_path):
    generate_report("data/patients.csv", tmp_path)
    assert (Path(tmp_path) / "risk_distribution.png").exists()
    assert (Path(tmp_path) / "data_summary.csv").exists()
    assert (Path(tmp_path) / "numeric_correlations.csv").exists()
