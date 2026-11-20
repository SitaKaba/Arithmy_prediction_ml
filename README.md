# Heart Disease Prediction — End-to-End ML Project

Predict the **presence of heart disease** from clinical measurements (UCI Heart Disease, Cleveland subset), from data loading to a deployed Streamlit app.

**Live demo:** https://arithmypredictionml-ltuvmrjsjinyy8vb8sxngd.streamlit.app

## What it does

- **Data loading and cleaning** (`src/data.py`): reads the Cleveland file locally or downloads it from the UCI repository, maps categorical codes to readable labels, and turns the original 0–4 target into a binary label (0 = no disease, 1 = disease).
- **Exploratory analysis** (`notebooks/01_eda.ipynb`): descriptive statistics, target balance, feature distributions.
- **Model comparison** (`src/model.py`): logistic regression, random forest and XGBoost (if installed), each inside a single scikit-learn `Pipeline` (imputation, scaling, one-hot encoding). Models are compared with **stratified 5-fold cross-validation** on accuracy, ROC-AUC, F1, precision and recall; the best one by ROC-AUC (F1 as tie-breaker) is refit on all the data and exported.
- **Streamlit app** (`app/streamlit_app.py`): dataset preview, interactive prediction with an adjustable decision threshold, and the evaluation protocol.

The exported model in `models/best_pipeline.joblib` is a logistic regression pipeline.

## Data

UCI Machine Learning Repository, *Heart Disease* dataset, **Cleveland** subset: 303 patients, 13 features and 1 target. A copy is in `data/processed.cleveland.data`.

## Run locally

```bash
git clone https://github.com/SitaKaba/Arithmy_prediction_ml.git
cd Arithmy_prediction_ml
pip install -r requirements.txt

# Compare models with cross-validation and export the best pipeline
python -m src.model --cv 5 --models logreg rf xgb --export models/best_pipeline.joblib

# Launch the app
streamlit run app/streamlit_app.py
```

## Structure

```
app/streamlit_app.py    # Streamlit app
src/data.py             # loading, cleaning, feature types
src/model.py            # preprocessing pipeline, cross-validation, export
notebooks/01_eda.ipynb  # exploratory analysis
data/                   # Cleveland data file
models/                 # exported pipeline (joblib)
```

## Next steps

- Permutation feature importance and SHAP values on the fitted pipeline
- Probability calibration and threshold selection based on the cost of false negatives
- Report the cross-validation results in the README
