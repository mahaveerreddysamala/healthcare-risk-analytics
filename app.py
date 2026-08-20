import streamlit as st
import pandas as pd
from src.model import train

st.set_page_config(page_title="Healthcare Risk Analytics", layout="wide")
st.title("Healthcare Risk Analytics")
st.warning("Research/portfolio demonstration only; not a clinical decision-support system.")

df = pd.read_csv("data/patients.csv")
c1, c2, c3 = st.columns(3)
c1.metric("Patients", len(df))
c2.metric("High-risk rate", f"{df.high_risk.mean():.1%}")
c3.metric("Average age", f"{df.age.mean():.1f}")
st.subheader("Risk factors")
st.dataframe(df.describe(include="all"), use_container_width=True)

if st.button("Train risk model"):
    with st.spinner("Training model..."):
        train("data/patients.csv")
    st.success("Risk model trained. Evaluation metrics are printed by the training pipeline.")
