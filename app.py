# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ----------------------------------
# 1. Setup path model & scaler
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "data")

model_path = os.path.join(MODEL_DIR, "logreg_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
example_path = os.path.join(DATA_DIR, "example.txt")  # path example.txt

# Load model & scaler
logreg = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ----------------------------------
# 2. Streamlit UI
# ----------------------------------
st.set_page_config(page_title="Ethanol Concentration Classifier", layout="wide")
st.title("Prediksi Konsentrasi Ethanol dari Time Series")

st.markdown("""
Masukkan data time series untuk prediksi kelas Ethanol.  
Data harus memiliki panjang **1751 poin per channel** dan **3 channel**.  
Anda dapat **upload file CSV**, **input manual**, atau **gunakan example.txt**.
""")

# ----------------------------------
# 3. Pilihan Input
# ----------------------------------
input_option = st.radio("Pilih metode input:", ("Upload CSV", "Input Manual", "Gunakan example.txt"))

X_input_scaled = None
df_input = None

if input_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload file CSV (1751 x 3)", type="csv")
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file, header=None)
        if df_input.shape != (1751, 3):
            st.error(f"File harus memiliki shape (1751, 3). File anda: {df_input.shape}")
        else:
            st.success("File berhasil dimuat")
            st.dataframe(df_input.head())
            X_input = df_input.values.reshape(1, -1)
            X_input_scaled = scaler.transform(X_input)

elif input_option == "Input Manual":
    st.write("Masukkan data untuk masing-masing channel (dipisahkan koma).")
    
    ch1 = st.text_area("Channel 1", value=",".join(["0"]*1751), height=150)
    ch2 = st.text_area("Channel 2", value=",".join(["0"]*1751), height=150)
    ch3 = st.text_area("Channel 3", value=",".join(["0"]*1751), height=150)
    
    if st.button("Prediksi Manual"):
        try:
            ch1_vals = np.array([float(x) for x in ch1.split(",")])
            ch2_vals = np.array([float(x) for x in ch2.split(",")])
            ch3_vals = np.array([float(x) for x in ch3.split(",")])
            
            if len(ch1_vals) != 1751 or len(ch2_vals) != 1751 or len(ch3_vals) != 1751:
                st.error("Semua channel harus memiliki 1751 poin")
            else:
                df_input = np.stack([ch1_vals, ch2_vals, ch3_vals], axis=1)
                X_input = df_input.reshape(1, -1)
                X_input_scaled = scaler.transform(X_input)
        except:
            st.error("Pastikan format input benar, pisahkan angka dengan koma.")

elif input_option == "Gunakan example.txt":
    if os.path.exists(example_path):
        st.write(f"Memuat data dari {example_path}")
        df_input = pd.read_csv(example_path, header=None)
        if df_input.shape != (1751, 3):
            st.error(f"File example.txt harus shape (1751, 3). Ditemukan: {df_input.shape}")
        else:
            st.success("Data example.txt berhasil dimuat")
            st.dataframe(df_input.head())
            X_input = df_input.values.reshape(1, -1)
            X_input_scaled = scaler.transform(X_input)
    else:
        st.error("File example.txt tidak ditemukan di folder data/")

# ----------------------------------
# 4. Prediksi dan Statistik jika data tersedia
# ----------------------------------
if X_input_scaled is not None:
    # Prediksi kelas
    pred_class = logreg.predict(X_input_scaled)[0]
    
    # Probabilitas prediksi
    pred_prob = logreg.predict_proba(X_input_scaled)[0]
    
    st.subheader("Hasil Prediksi")
    st.write(f"Prediksi kelas Ethanol: **{pred_class}**")
    
    st.subheader("Probabilitas Prediksi per Kelas")
    prob_df = pd.DataFrame([pred_prob], columns=[f"Kelas {c}" for c in logreg.classes_])
    st.dataframe(prob_df)
    
    # Statistik dasar
    st.subheader("Statistik Dasar Time Series")
    stats_df = pd.DataFrame({
        "Mean": df_input.mean(axis=0),
        "Min": df_input.min(axis=0),
        "Max": df_input.max(axis=0),
        "Std": df_input.std(axis=0)
    }, index=[f"Channel {i+1}" for i in range(df_input.shape[1])])
    
    st.dataframe(stats_df)
