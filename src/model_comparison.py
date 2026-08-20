import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def run_comparison(path="data/patients.csv", output="reports/model_comparison.csv"):
    df = pd.read_csv(path)
    X, y = df.drop(columns="high_risk"), df["high_risk"]
    num = X.select_dtypes(include="number").columns
    cat = X.select_dtypes(exclude="number").columns
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    rows = []
    for name, estimator in models.items():
        pipe = Pipeline([("prep", prep), ("model", estimator)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        prob = pipe.predict_proba(Xte)[:, 1]
        rows.append({"model": name, "precision": precision_score(yte, pred), "recall": recall_score(yte, pred), "f1": f1_score(yte, pred), "roc_auc": roc_auc_score(yte, prob)})
    result = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    result.to_csv(output, index=False)
    return result

if __name__ == "__main__":
    print(run_comparison().to_string(index=False))
