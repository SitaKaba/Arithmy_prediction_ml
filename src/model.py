import argparse
import joblib
import warnings
import numpy as np
import pandas as pd
from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_validate

try:
    from xgboost import XGBClassifier  # optional
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from .data import load_heart, features_targets, get_feature_types

SCORERS = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall"
}

def make_preprocessor():
    num, cat = get_feature_types()
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat),
        ]
    )
    return pre

def make_model(name: str, random_state: int = 42):
    name = name.lower()
    if name in ["logreg", "logistic", "lr"]:
        clf = LogisticRegression(max_iter=200, random_state=random_state)
    elif name in ["rf", "randomforest"]:
        clf = RandomForestClassifier(n_estimators=400, max_depth=None, random_state=random_state)
    elif name in ["xgb", "xgboost"] and HAS_XGB:
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown or unavailable model: {name}")
    return clf

def evaluate_models(models: List[str], cv: int = 5, random_state: int = 42):
    df = load_heart()
    X, y = features_targets(df)

    pre = make_preprocessor()

    results = []
    for m in models:
        clf = make_model(m, random_state=random_state)
        pipe = Pipeline([("pre", pre), ("clf", clf)])

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        cv_result = cross_validate(
            pipe, X, y, cv=skf, scoring=SCORERS, n_jobs=-1, return_estimator=True
        )
        row = {k: np.mean(v) for k, v in cv_result.items() if k.startswith("test_")}
        row = {k.replace("test_", ""): v for k, v in row.items()}
        row["model"] = m
        row["estimators"] = cv_result["estimator"]
        results.append(row)

    # Pick best by ROC-AUC, then F1 tie-breaker
    results_df = pd.DataFrame(results).sort_values(["roc_auc", "f1"], ascending=False)
    return results_df

def export_best_pipeline(results_df: pd.DataFrame, export_path: Optional[str] = None):
    top = results_df.iloc[0]
    best_estimators = top["estimators"]
    # Refit on full data using best model type
    df = load_heart()
    X, y = features_targets(df)
    pre = make_preprocessor()
    clf = make_model(top["model"])

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)

    if export_path:
        joblib.dump(pipe, export_path)
    return pipe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=int, default=5, help="Num CV folds (StratifiedKFold)")
    parser.add_argument("--models", nargs="+", default=["logreg", "rf", "xgb"], help="Models to try")
    parser.add_argument("--export", type=str, default="", help="Path to save best pipeline (joblib)")
    args = parser.parse_args()

    if "xgb" in args.models and not HAS_XGB:
        warnings.warn("xgboost not installed. Skipping XGB.")
        args.models = [m for m in args.models if m != "xgb"]
        if not args.models:
            args.models = ["logreg", "rf"]

    results_df = evaluate_models(args.models, cv=args.cv)
    print("\nCV Results (mean across folds):")
    print(results_df[["model","accuracy","roc_auc","f1","precision","recall"]].to_string(index=False))

    if args.export:
        export_best_pipeline(results_df, args.export)
        print(f"\nExported best pipeline to: {args.export}")

if __name__ == "__main__":
    main()
