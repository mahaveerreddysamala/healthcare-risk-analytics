"""Leakage-safe model evaluation utilities for the healthcare risk demonstration."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.model import build_pipeline


def evaluate(path="data/patients.csv", output_dir="reports", threshold=0.40):
    """Fit only on the training split and write reproducible evaluation artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)
    X = df.drop(columns="high_risk")
    y = df["high_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_test, prediction, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(out / "confusion_matrix.csv")

    fpr, tpr, thresholds = roc_curve(y_test, probability)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
        out / "roc_curve.csv", index=False
    )

    prep = model.named_steps["prep"]
    estimator = model.named_steps["model"]
    names = prep.get_feature_names_out()
    importance = pd.DataFrame(
        {"feature": names, "importance": estimator.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(out / "feature_importance.csv", index=False)

    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
