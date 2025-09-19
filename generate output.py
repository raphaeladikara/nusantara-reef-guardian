import numpy as np
import tensorflow as tf
import xarray as xr
import geopandas as gpd
import pandas as pd
import rioxarray
from datetime import datetime, timedelta
import os

# --- Konfigurasi dan Parameter (Sama seperti sebelumnya) ---
BBOX = {"min_lon": 95.0, "max_lon": 141.0, "min_lat": -11.0, "max_lat": 6.0}
VARIABLES = ['CRW_SST', 'CRW_SSTANOMALY', 'CRW_HOTSPOT', 'CRW_DHW']
TIME_STEPS = 12
DATA_URL = "https://pae-paha.pacioos.hawaii.edu/thredds/dodsC/dhw_5km"
# Ganti dengan path absolut atau relatif yang benar untuk lingkungan Anda
SHAPEFILE_PATH = r"C:\Users\rapha\Documents\VSCode\Kaggle\Iconic IT\Dataset\Peta Terumbu Karang Indonesia\Coral_Ind_250K.shp"
MODEL_PATH = 'nusantara_reef_guardian_model.h5'
EXPECTED_SHAPE = (338, 914)
OUTPUT_CSV_PATH = 'latest_prediction_data.csv' # Nama file output

# --- Fungsi-fungsi Inti (tanpa cache Streamlit) ---
def load_model_and_map():
    """Memuat model dan shapefile."""
    print("Memuat model dan peta...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    coral_map = gpd.read_file(SHAPEFILE_PATH)
    return model, coral_map

def get_latest_data_and_mask(today_str, coral_map_obj, days_needed=12):
    """Mengambil dan memproses data satelit."""
    print(f"Mengambil data terbaru hingga {today_str}...")
    end_date = datetime.strptime(today_str, '%Y-%m-%d')
    start_date = end_date - timedelta(days=days_needed)
    
    dataset = xr.open_dataset(DATA_URL, engine='netcdf4').sel(
        time=slice(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
        longitude=slice(BBOX['min_lon'], BBOX['max_lon']),
        latitude=slice(BBOX['max_lat'], BBOX['min_lat'])
    )
    
    dataset = dataset.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude").rio.write_crs(coral_map_obj.crs)
    clipped_data = dataset[VARIABLES].rio.clip(coral_map_obj.geometry, all_touched=True)
    
    clipped_array = clipped_data.to_array().values
    clipped_array = np.moveaxis(clipped_array, 0, -1)
    
    reef_mask = ~np.isnan(clipped_array).all(axis=(0, 3))
    
    data_for_model = np.nan_to_num(clipped_array)
    data_norm = (data_for_model - data_for_model.min()) / (data_for_model.max() - data_for_model.min())
    
    final_data = data_norm[:, :EXPECTED_SHAPE[0], :EXPECTED_SHAPE[1], :]
    final_mask = reef_mask[:EXPECTED_SHAPE[0], :EXPECTED_SHAPE[1]]
    
    return final_data, final_mask

def make_prediction(model, input_data):
    """Menjalankan prediksi model."""
    print("Model AI sedang membuat prediksi...")
    input_data = np.expand_dims(input_data, axis=0)
    prediction = model.predict(input_data)
    return prediction[0, :, :, 0]

def main():
    """Fungsi utama untuk menjalankan seluruh pipeline dan menyimpan hasilnya."""
    print("--- Memulai Proses Generasi Prediksi ---")
    
    model, coral_map = load_model_and_map()
    
    today = datetime.now() - timedelta(days=2)
    today_str = today.strftime('%Y-%m-%d')
    
    latest_data, reef_mask = get_latest_data_and_mask(today_str, coral_map, days_needed=TIME_STEPS)
    
    predicted_risk_map = make_prediction(model, latest_data)
    
    final_prediction = predicted_risk_map * reef_mask
    
    print("Membuat DataFrame hasil...")
    lats = np.linspace(BBOX['max_lat'], BBOX['min_lat'], EXPECTED_SHAPE[0])
    lons = np.linspace(BBOX['min_lon'], BBOX['max_lon'], EXPECTED_SHAPE[1])
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    results_df = pd.DataFrame({
        'latitude': lat_grid.flatten(),
        'longitude': lon_grid.flatten(),
        'risk_score': final_prediction.flatten()
    })
    
    results_df = results_df[results_df['risk_score'] > 0.05]
    
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
    results_df['Level Risiko'] = pd.cut(results_df['risk_score'], bins=bins, labels=labels, include_lowest=True)
    
    # Menambahkan tanggal data ke dalam file untuk informasi di UI
    results_df['data_date'] = today_str
    
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"✅ Prediksi berhasil disimpan ke: {OUTPUT_CSV_PATH}")
    print("--- Proses Selesai ---")

if __name__ == "__main__":
    main()