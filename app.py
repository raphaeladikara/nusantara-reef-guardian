import streamlit as st
import numpy as np
import tensorflow as tf
import xarray as xr
import geopandas as gpd
import pandas as pd
import plotly.express as px
import rioxarray
from datetime import datetime, timedelta

# --- Konfigurasi Halaman dan Pemuatan Aset ---
st.set_page_config(layout="wide", page_title="Nusantara Reef-Guardian")

# --- Parameter ---
BBOX = {"min_lon": 95.0, "max_lon": 141.0, "min_lat": -11.0, "max_lat": 6.0}
VARIABLES = ['CRW_SST', 'CRW_SSTANOMALY', 'CRW_HOTSPOT', 'CRW_DHW']
TIME_STEPS = 12
DATA_URL = "https://pae-paha.pacioos.hawaii.edu/thredds/dodsC/dhw_5km"
SHAPEFILE_PATH = "Peta Terumbu Karang Indonesia/Coral_Ind_250K.shp"
MODEL_PATH = 'nusantara_reef_guardian_model.h5'
EXPECTED_SHAPE = (338, 914)

# --- Fungsi-fungsi Inti (di-cache untuk performa) ---
@st.cache_resource
def load_model_and_map():
    print("Memuat model dan peta...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    coral_map = gpd.read_file(SHAPEFILE_PATH)
    return model, coral_map

@st.cache_data(ttl=3600)
def get_latest_data_and_mask(_today_str, _coral_map_obj, days_needed=12):
    print(f"Mengambil data terbaru hingga {_today_str}...")
    end_date = datetime.strptime(_today_str, '%Y-%m-%d')
    start_date = end_date - timedelta(days=days_needed)
    
    dataset = xr.open_dataset(DATA_URL, engine='netcdf4').sel(
        time=slice(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
        longitude=slice(BBOX['min_lon'], BBOX['max_lon']),
        latitude=slice(BBOX['max_lat'], BBOX['min_lat'])
    )
    
    dataset = dataset.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude").rio.write_crs(_coral_map_obj.crs)
    clipped_data = dataset[VARIABLES].rio.clip(_coral_map_obj.geometry, all_touched=True)
    
    clipped_array = clipped_data.to_array().values
    clipped_array = np.moveaxis(clipped_array, 0, -1)
    
    # PERBAIKAN: Buat MASK dari data yang di-clip SEBELUM mengubah NaN
    # Mask akan bernilai True di lokasi karang, dan False di laut dalam
    reef_mask = ~np.isnan(clipped_array).all(axis=(0, 3))
    
    data_for_model = np.nan_to_num(clipped_array)
    data_norm = (data_for_model - data_for_model.min()) / (data_for_model.max() - data_for_model.min())
    
    final_data = data_norm[:, :EXPECTED_SHAPE[0], :EXPECTED_SHAPE[1], :]
    final_mask = reef_mask[:EXPECTED_SHAPE[0], :EXPECTED_SHAPE[1]]
    
    return final_data, final_mask

def make_prediction(input_data):
    input_data = np.expand_dims(input_data, axis=0)
    prediction = model.predict(input_data)
    return prediction[0, :, :, 0]

# --- Alur Utama ---
model, coral_map = load_model_and_map()

st.title("🌊 Nusantara Reef-Guardian")
st.markdown("Dasbor Prediktif untuk Ketahanan dan Konservasi Terumbu Karang Indonesia")

today = datetime.now() - timedelta(days=1)
today_str = today.strftime('%Y-%m-%d')

with st.spinner(f"Mengambil & memproses data satelit terbaru..."):
    latest_data, reef_mask = get_latest_data_and_mask(today_str, coral_map, days_needed=TIME_STEPS)

with st.spinner("Model AI sedang membuat prediksi..."):
    predicted_risk_map = make_prediction(latest_data)

# PERBAIKAN: Terapkan mask ke hasil prediksi
final_prediction = predicted_risk_map * reef_mask

st.header(f"Peta Prediksi Risiko Stres Termal (Untuk 4 Minggu ke Depan)")
st.caption(f"Prediksi dibuat berdasarkan data satelit hingga tanggal: {today_str}")

# Buat grid koordinat yang presisi
lats = np.linspace(BBOX['max_lat'], BBOX['min_lat'], EXPECTED_SHAPE[0])
lons = np.linspace(BBOX['min_lon'], BBOX['max_lon'], EXPECTED_SHAPE[1])
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Buat DataFrame dari prediksi YANG SUDAH DI-MASK
results_df = pd.DataFrame({
    'latitude': lat_grid.flatten(),
    'longitude': lon_grid.flatten(),
    'risk_score': final_prediction.flatten()
})
# Filter hanya piksel dengan risiko > 0 untuk diplot
results_df = results_df[results_df['risk_score'] > 0.05]

# Buat kategori risiko
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
results_df['Level Risiko'] = pd.cut(results_df['risk_score'], bins=bins, labels=labels, include_lowest=True)

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