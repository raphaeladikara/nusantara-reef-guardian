import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Konfigurasi Halaman ---
st.set_page_config(layout="wide", page_title="Nusantara Reef-Guardian")

# --- Parameter Tampilan ---
PREDICTION_FILE = 'latest_prediction_data.csv'

# --- Tampilan Utama ---
st.title("🌊 Nusantara Reef-Guardian")
st.markdown("Dasbor Prediktif untuk Ketahanan dan Konservasi Terumbu Karang Indonesia")

# Cek apakah file prediksi ada
if not os.path.exists(PREDICTION_FILE):
    st.error(f"File prediksi '{PREDICTION_FILE}' tidak ditemukan.")
    st.warning("Harap jalankan script `generate_prediction.py` terlebih dahulu untuk membuat file hasil prediksi.")
else:
    # Muat data yang sudah diproses
    results_df = pd.read_csv(PREDICTION_FILE)
    
    # Ambil tanggal data dari kolom pertama untuk ditampilkan
    data_date = results_df['data_date'].iloc[0]

    st.header(f"Peta Prediksi Risiko Stres Termal (Untuk 4 Minggu ke Depan)")
    st.caption(f"Prediksi dibuat berdasarkan data satelit hingga tanggal: {data_date}")

    # Konfigurasi Peta (sama seperti sebelumnya)
    labels = ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
    color_map = {
        'Sangat Rendah': '#3b4cc0', 'Rendah': '#5ac8c8', 'Sedang': '#f2e750',
        'Tinggi': '#ff993e', 'Sangat Tinggi': '#e60000'
    }

    st.info("Peta ini hanya menampilkan prediksi di lokasi terumbu karang yang teridentifikasi.")

    fig = px.scatter_mapbox(
        results_df, lat="latitude", lon="longitude", color="Level Risiko",
        color_discrete_map=color_map, category_orders={"Level Risiko": labels},
        mapbox_style="carto-positron", zoom=4, center=dict(lat=-2.5, lon=118),
        opacity=0.8, hover_name="risk_score",
        hover_data={"Level Risiko": True, "latitude": ":.4f", "longitude": ":.4f"},
        labels={"Level Risiko": "Level Risiko"}
    )
    fig.update_traces(marker={'size': 5})
    fig.update_layout(legend_title="Level Risiko", margin={"r":0,"t":0,"l":0,"b":0})

    st.plotly_chart(fig, use_container_width=True)