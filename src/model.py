import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


def train(path="data/patients.csv"):
    df = pd.read_csv(path)
    target = "high_risk"
    X, y = df.drop(columns=target), df[target]
    numeric = X.select_dtypes(include="number").columns
    categorical = X.select_dtypes(exclude="number").columns
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    pipeline = Pipeline([
        ("prep", prep),
        ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_test)[:, 1]
    prediction = (probability >= .40).astype(int)
    print(classification_report(y_test, prediction))
    print("ROC-AUC:", round(roc_auc_score(y_test, probability), 4))
    return pipeline


if __name__ == "__main__":
    train()
