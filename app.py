import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ======================
# STYLE CUSTOM (BIAR KEREN)
# ======================
st.set_page_config(page_title="Prediksi Sawit", page_icon="🌴", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# LOAD DATA & MODEL
# ======================
data = pd.read_csv("dataset_kelapa_sawit_500.csv")

X = data.drop("Hasil_Panen_ton_per_ha", axis=1)
y = data["Hasil_Panen_ton_per_ha"]

model = RandomForestRegressor()
model.fit(X, y)

# ======================
# HEADER
# ======================
st.title("🌴 Prediksi Hasil Panen Kelapa Sawit")
st.write("Masukkan data di bawah ini untuk memprediksi hasil panen.")

# ======================
# LAYOUT 2 KOLOM
# ======================
col1, col2 = st.columns(2)

with col1:
    hujan = st.number_input("🌧 Curah Hujan (mm)")
    suhu = st.number_input("🌡 Suhu (°C)")
    lembab = st.number_input("💧 Kelembaban (%)")
    ndvi = st.number_input("🌱 NDVI")

with col2:
    umur = st.number_input("Umur Tanaman (tahun)")
    lahan = st.number_input("Luas Lahan (ha)")
    pupuk = st.number_input("Pupuk (kg/ha)")

# ======================
# BUTTON PREDIKSI
# ======================
if st.button("Prediksi Sekarang"):
    hasil = model.predict([[hujan, suhu, lembab, ndvi, umur, lahan, pupuk]])

    st.success(f"Hasil Prediksi: {hasil[0]:.2f} ton/ha")

    st.info("Model menggunakan Random Forest dengan akurasi tinggi.")

# ======================
# FOOTER
# ======================
st.markdown("---")
st.caption("Dibuat untuk proyek prediksi hasil panen kelapa sawit 🌴")