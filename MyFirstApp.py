
import streamlit as st
import pandas as pd
import joblib

# Memuat model
model = joblib.load("model_penjualan.pkl")
batas = joblib.load("batas_penjualan.pkl")

st.set_page_config(
    page_title="Prediksi Penjualan",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Aplikasi Klasifikasi Penjualan")
st.write(
    "Sistem klasifikasi tingkat penjualan "
    "menggunakan algoritma Decision Tree."
)

st.info(
    f"Batas kategori penjualan: {batas:.2f}"
)

st.subheader("Input Data Transaksi")

branch = st.selectbox("Cabang", ["A", "B", "C"])

city = st.selectbox(
    "Kota",
    ["Yangon", "Mandalay", "Naypyitaw"]
)

customer = st.selectbox(
    "Jenis Pelanggan",
    ["Member", "Normal"]
)

gender = st.selectbox(
    "Jenis Kelamin",
    ["Male", "Female"]
)

product = st.selectbox(
    "Kategori Produk",
    [
        "Health and beauty",
        "Electronic accessories",
        "Home and lifestyle",
        "Sports and travel",
        "Food and beverages",
        "Fashion accessories"
    ]
)

unit_price = st.number_input(
    "Harga Satuan",
    min_value=0.01,
    value=50.0
)

quantity = st.number_input(
    "Jumlah Barang",
    min_value=1,
    max_value=100,
    value=5
)

payment = st.selectbox(
    "Metode Pembayaran",
    ["Cash", "Credit card", "Ewallet"]
)

if st.button("Prediksi Penjualan"):
    data_baru = pd.DataFrame([{
        "Branch": branch,
        "City": city,
        "Customer type": customer,
        "Gender": gender,
        "Product line": product,
        "Unit price": unit_price,
        "Quantity": quantity,
        "Payment": payment
    }])

    hasil = model.predict(data_baru)[0]

    st.subheader("Hasil Prediksi")

    if hasil == "Tinggi":
        st.success("Kategori Penjualan: TINGGI")
    else:
        st.warning("Kategori Penjualan: RENDAH")

    st.write("Hasil ini merupakan klasifikasi "
             "berdasarkan model Decision Tree.")
