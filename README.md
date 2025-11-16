# Heart Disease – End‑to‑End ML Portfolio Project

This repository implements a robust, production‑style workflow on the **UCI Heart Disease** dataset:
- Data ingestion & cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training with cross‑validation (LogReg, RandomForest, XGBoost if available)
- Model interpretability (permutation importance, SHAP optional)
- Streamlit app for interactive exploration & inference
- Reproducible environment & simple commands

## Quickstart

```bash
# 1) Create environment (conda or venv) then install deps
pip install -r requirements.txt

# 2) (Optional) Run EDA notebook
#   Open notebooks/01_eda.ipynb in Jupyter/VSCode

# 3) Train models and export the best pipeline
python -m src.model --cv 5 --models logreg rf xgb --export models/best_pipeline.joblib

# 4) Launch the app
streamlit run app/streamlit_app.py
```

## Data
Dataset source: UCI Machine Learning Repository – *Heart Disease*. This project uses the **Cleveland** subset by default (303 rows; 13 features + 1 target). The loader will try to fetch from UCI directly if `data/processed.cleveland.data` isn’t present.

> If UCI is slow or changes URLs, place a CSV at `data/heart.csv` with the standard column names described in `src/data.py`.

## Reproducibility
- Deterministic seeds where applicable
- Pip requirements pinned with compatible upper bounds
- Single `Pipeline` for preprocessing + model
- Cross‑validation for fair evaluation

## Structure
```
heart-disease-ml/
├── app/
│   └── streamlit_app.py
├── data/
│   └── (raw & interim files go here)
├── models/
│   └── best_pipeline.joblib (after training)
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── data.py
│   └── model.py
├── requirements.txt
└── README.md
```
