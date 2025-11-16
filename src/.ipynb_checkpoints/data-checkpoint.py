import io
import os
import sys
import pathlib
import pandas as pd
import numpy as np
from urllib.request import urlopen

UCI_CLEVELAND_URLS = [
    # Primary (classic) location
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    # Mirror/newer paths sometimes used by UCI
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data?download=1",
]

COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
    "exang","oldpeak","slope","ca","thal","target"
]

CATEGORY_MAPS = {
    "sex": {0: "female", 1: "male"},
    "cp": {0: "typical_angina", 1: "atypical_angina", 2: "non_anginal", 3: "asymptomatic"},
    "fbs": {0: "false", 1: "true"},
    "restecg": {0: "normal", 1: "stt_abnormality", 2: "lv_hypertrophy"},
    "exang": {0: "no", 1: "yes"},
    "slope": {0: "up", 1: "flat", 2: "down"},
    "thal": {1: "normal", 2: "fixed_defect", 3: "reversible_defect"},
}

def _read_local_or_download(data_dir: pathlib.Path) -> pd.DataFrame:
    data_dir.mkdir(parents=True, exist_ok=True)
    local_paths = [
        data_dir / "processed.cleveland.data",
        data_dir / "heart.csv",  # user-provided clean CSV fallback
    ]

    for p in local_paths:
        if p.exists():
            if p.suffix == ".data":
                df = pd.read_csv(p, header=None, names=COLUMNS)
            else:
                df = pd.read_csv(p)
            return df

    # Try to download from UCI
    last_err = None
    for url in UCI_CLEVELAND_URLS:
        try:
            with urlopen(url, timeout=30) as f:
                content = f.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(content), header=None, names=COLUMNS)
            # Cache the raw file
            (data_dir / "processed.cleveland.data").write_text(content, encoding="utf-8")
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not load dataset locally or from UCI. Last error: {last_err}")

def load_heart(data_dir: str = "data") -> pd.DataFrame:
    """Load Cleveland subset. Returns a cleaned pandas DataFrame.

    Cleaning steps:
    - Replace '?' with NaN
    - Cast types
    - Map 'target' to binary {0,1}: 0 means no disease; 1 means presence (any of 1..4)
    """
    data_path = pathlib.Path(data_dir)
    df = _read_local_or_download(data_path)

    # Standardize missing values
    df = df.replace("?", np.nan)

    # Coerce numeric columns
    numeric_cols = ["age","trestbps","chol","thalach","oldpeak","ca"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # thal sometimes as '3.0' strings; coerce to numeric then int-coded
    for col in ["sex","cp","fbs","restecg","exang","slope","thal","target"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binary target: presence = 1 'if original target > 0
    df["target"] = (df["target"] > 0).astype(int)

    return df

def features_targets(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)
    return X, y

def get_feature_types():
    numeric = ["age","trestbps","chol","thalach","oldpeak","ca"]
    categorical = ["sex","cp","fbs","restecg","exang","slope","thal"]
    return numeric, categorical
