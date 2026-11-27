
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Prediksi Penjualan",
    page_icon="📊",
    layout="centered"
)

model = joblib.load("model_penjualan.pkl")
batas = joblib.load("batas_penjualan.pkl")

st.title("📊 Aplikasi Klasifikasi Penjualan")
st.write(
    "Sistem klasifikasi tingkat penjualan "
    "menggunakan algoritma Decision Tree."
)

st.info(f"Batas kategori penjualan: {batas:.2f}")

st.subheader("Input Data Transaksi")

ship_mode = st.selectbox(
    "Metode Pengiriman",
    ["Second Class", "Standard Class", "First Class", "Same Day"]
)

segment = st.selectbox(
    "Segmen Konsumen",
    ["Consumer", "Corporate", "Home Office"]
)

region = st.selectbox(
    "Wilayah",
    ["Central", "East", "South", "West"]
)

category = st.selectbox(
    "Kategori Produk",
    ["Furniture", "Office Supplies", "Technology"]
)

sub_category = st.selectbox(
    "Sub-Kategori Produk",
    [
        "Accessories", "Appliances", "Art", "Binders",
        "Bookcases", "Chairs", "Copiers", "Envelopes",
        "Fasteners", "Furnishings", "Labels", "Machines",
        "Paper", "Phones", "Storage", "Supplies", "Tables"
    ]
)

order_month = st.selectbox(
    "Bulan Transaksi",
    list(range(1, 13))
)

order_day = st.selectbox(
    "Hari Transaksi (0 = Senin, 6 = Minggu)",
    list(range(7))
)

if st.button("Prediksi Penjualan"):

    data_baru = pd.DataFrame([{
        "Ship Mode": ship_mode,
        "Segment": segment,
        "Region": region,
        "Category": category,
        "Sub-Category": sub_category,
        "Order_Month": order_month,
        "Order_Day": order_day
    }])

    st.write("Data yang akan diprediksi:")
    st.dataframe(data_baru)

    hasil = model.predict(data_baru)[0]

    st.subheader("Hasil Prediksi")

    if hasil == "Tinggi":
        st.success("Kategori Penjualan: TINGGI")
    else:
        st.warning("Kategori Penjualan: RENDAH")

    st.write(
        "Hasil ini merupakan klasifikasi berdasarkan "
        "model Decision Tree."
    )
