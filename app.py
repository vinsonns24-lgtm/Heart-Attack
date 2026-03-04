import streamlit as st
import pandas as pd
import joblib
import numpy as np

try:
    model = joblib.load("RF_class.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")
except FileNotFoundError as e:
    st.error(f"Error: File tidak ditemukan. Pastikan kamu sudah menjalankan pipeline.py dan file model tersedia. Detail: {e}")
    st.stop()

def main():

    st.set_page_config(page_title="Heart Attack Predictor", page_icon="🫀", layout="centered")
    
    st.title("🫀 Aplikasi Prediksi Risiko Serangan Jantung")
    st.write("Masukkan data medis pasien pada panel di bawah ini untuk melihat prediksi risiko.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Umur (Age)", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Jenis Kelamin (Sex)", options=[1, 0], format_func=lambda x: "Pria" if x == 1 else "Wanita")
        cp = st.selectbox("Tipe Nyeri Dada (Chest Pain Type)", options=[0, 1, 2, 3])
        trestbps = st.number_input("Tekanan Darah Istirahat (Resting BP)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Kolesterol Serum (Cholesterol)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl (Fasting Blood Sugar)", options=[1, 0], format_func=lambda x: "Ya (>120)" if x == 1 else "Tidak (<120)")
        restecg = st.selectbox("Hasil EKG Istirahat (Resting ECG)", options=[0, 1, 2])

    with col2:
        thalach = st.number_input("Detak Jantung Maksimal (Max Heart Rate)", min_value=60, max_value=250, value=150)
        exang = st.selectbox("Angina Terinduksi Olahraga (Exercise Induced Angina)", options=[1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        oldpeak = st.number_input("Depresi ST (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Kemiringan Segmen ST (Slope)", options=[0, 1, 2])
        ca = st.selectbox("Jumlah Pembuluh Darah Utama (CA)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Hasil Tes Thallium (Thal)", options=[0, 1, 2, 3])

    if st.button("Lakukan Prediksi 🔍", use_container_width=True):

        input_data = pd.DataFrame([[
            age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal
        ]], columns=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ])

        features_scaled = scaler.transform(input_data)

        prediction = model.predict(features_scaled)

        st.markdown("---")
        if prediction[0] == 1:
            st.error("🚨 **Hasil Prediksi: Risiko TINGGI terkena Serangan Jantung!** \n\nSilakan segera konsultasikan kondisi ini dengan dokter.")
        else:
            st.success("✅ **Hasil Prediksi: Risiko RENDAH.** \n\nKondisi jantung diprediksi relatif aman. Tetap jaga gaya hidup sehat!")

if __name__ == "__main__":
    main()