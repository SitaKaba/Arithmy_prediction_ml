import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
import sys
from pathlib import Path
# Assure que la racine du projet est dans sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_heart, get_feature_types

st.set_page_config(page_title="Heart Disease ML – Demo", layout="wide")

st.title("🫀 Heart Disease – ML Demo")
st.caption("UCI Cleveland subset • Cross‑validated models • End‑to‑end pipeline")

# Sidebar
st.sidebar.header("Model")
model_path = st.sidebar.text_input("Pipeline .joblib path", "models/best_pipeline.joblib")
threshold = st.sidebar.slider("Decision threshold", 0.1, 0.9, 0.5, 0.01)

@st.cache_data
def get_data():
    df = load_heart()
    return df

df = get_data()
num_cols, cat_cols = get_feature_types()

tab1, tab2, tab3 = st.tabs(["🔎 Explore", "🧠 Inference", "📈 Metrics guide"])

with tab1:
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.write("Rows:", len(df))

    st.subheader("Basic stats")
    st.write("Numeric summary:")
    st.dataframe(df[num_cols].describe().T, use_container_width=True)

    st.write("Target balance:")
    st.bar_chart(df["target"].value_counts(normalize=True))

with tab2:
    st.subheader("Try the model")
    pipe = None
    if Path(model_path).exists():
        try:
            pipe = joblib.load(model_path)
            if not isinstance(pipe, Pipeline):
                st.error("Loaded object is not a sklearn Pipeline.")
                pipe = None
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    cols = st.columns(3)
    # Simple input form (can be improved with domain-informed defaults)
    with cols[0]:
        age = st.number_input("age", 18, 100, 54)
        trestbps = st.number_input("trestbps", 80, 220, 130)
        chol = st.number_input("chol", 100, 700, 246)
        thalach = st.number_input("thalach", 60, 230, 150)
    with cols[1]:
        oldpeak = st.number_input("oldpeak", 0.0, 6.5, 1.0, step=0.1)
        ca = st.number_input("ca (0–3)", 0, 3, 0)
        sex = st.selectbox("sex", options=[0,1], format_func=lambda x: "male" if x==1 else "female")
        exang = st.selectbox("exang", options=[0,1], format_func=lambda x: "yes" if x==1 else "no")
    with cols[2]:
        cp = st.selectbox("cp", options=[0,1,2,3], format_func=lambda x: ["typical","atypical","non-anginal","asymptomatic"][x])
        fbs = st.selectbox("fbs", options=[0,1])
        restecg = st.selectbox("restecg", options=[0,1,2])
        slope = st.selectbox("slope", options=[0,1,2])
        thal = st.selectbox("thal", options=[1,2,3], format_func=lambda x: {1:"normal",2:"fixed_defect",3:"reversible_defect"}[x])

    sample = pd.DataFrame([{
        "age":age,"trestbps":trestbps,"chol":chol,"thalach":thalach,"oldpeak":oldpeak,"ca":ca,
        "sex":sex,"exang":exang,"cp":cp,"fbs":fbs,"restecg":restecg,"slope":slope,"thal":thal
    }])

    st.write("Your inputs:")
    st.dataframe(sample, use_container_width=True)

    if pipe is not None and st.button("Predict"):
        prob = float(pipe.predict_proba(sample)[0,1])
        pred = int(prob >= threshold)
        st.metric("Predicted probability of disease", f"{prob:.3f}")
        st.metric("Decision (threshold {:.2f})".format(threshold), "Positive" if pred==1 else "Negative")

with tab3:
    st.markdown("""
**Recommended evaluation protocol**
- Stratified K‑Fold CV (k=5–10)
- Primary metric: ROC‑AUC; report Accuracy, F1, Precision, Recall
- Keep preprocessing + model in a single `Pipeline`
- Perform permutation feature importance on the fitted pipeline; add SHAP if available
- Report class balance and confidence calibration (optional)
""")
