import streamlit as st
import pandas as pd

st.set_page_config(page_title='Healthcare Risk Analytics', layout='wide')
st.title('Healthcare Risk Analytics')
st.warning('Educational portfolio demonstration only; not a clinical decision-support system.')
df = pd.read_csv('data/patients.csv')

c1, c2, c3 = st.columns(3)
c1.metric('Patients', len(df))
c2.metric('High-Risk Rate', f'{df.high_risk.mean():.1%}')
c3.metric('Average Age', f'{df.age.mean():.1f}')

st.subheader('Risk Distribution')
st.bar_chart(df['high_risk'].value_counts().sort_index())

st.subheader('Patient Risk Signals')
st.dataframe(df.sort_values('high_risk', ascending=False), use_container_width=True)

st.subheader('Numeric Feature Summary')
st.dataframe(df.select_dtypes('number').describe().T, use_container_width=True)
