import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("TS Anomaly Detector")
st.markdown("Anomaly detection through LSTM autoencoder")

@st.cache_data
def load_data():
    return pd.read_csv('data_series.csv'), pd.read_csv('anomalies.csv')

if st.button("Start detection"):
    import detect

df, an = load_data()

fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
ax[0].plot(df['value'], label='value'); ax[0].legend()
ax[1].plot(an['reconstruction_error'], label='error'); ax[1].legend()
ax[2].scatter(an.index, an['value'], c=an['anomaly'].map({False: 'blue', True: 'red'}), label='anomaly'); ax[2].legend()
st.pyplot(fig)
