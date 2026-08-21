from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt


def profile(path="data/patients.csv"):
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("\nTarget distribution:\n", df["high_risk"].value_counts(normalize=True).round(3))
    print("\nNumeric summary:\n", df.select_dtypes("number").describe().T)
    return df


def generate_report(path="data/patients.csv", output_dir="reports"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = profile(path)

    df["high_risk"].value_counts().sort_index().plot(kind="bar")
    plt.title("Healthcare Risk Class Distribution")
    plt.xlabel("High Risk")
    plt.ylabel("Patients")
    plt.tight_layout()
    plt.savefig(out / "risk_distribution.png", dpi=160)
    plt.close()

    df.describe(include="all").transpose().to_csv(out / "data_summary.csv")

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        numeric.corr(numeric_only=True).to_csv(out / "numeric_correlations.csv")

    return df


if __name__ == "__main__":
    generate_report()
