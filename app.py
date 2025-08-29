"""
HDB Resale Price Predictor (Streamlit mini-demo)

Entering flat details to get a price estimate based on the LightGBM model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import requests

# 1. Configuration and paths
BASE_PATH = Path(__file__).resolve().parent
MAE_DISCLAIMER_TEXT = "MAE on 2024-25 hold-out data ≈ SGD 54 k. Prediction can be ±10 % off for rare flat types or prime blocks."

# runtime model fetcher (fallback when models missing in repo)
MODELS_DIR = BASE_PATH / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def fetch_if_missing(local_path: Path, remote_url: str):
    if local_path.exists() and local_path.stat().st_size > 1000:
        return True
    try:
        r = requests.get(remote_url, stream=True, timeout=60)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        try:
            st.error(f"Failed to download {local_path.name}: {e}")
        except Exception:
            print(f"Failed to download {local_path.name}: {e}")
        return False

REMOTE_MODEL_URL = "https://github.com/Sheenal30/hdb-price-predictor/releases/download/v1.0/lightgbm_comparison_model.joblib"
REMOTE_FEATURES_URL = "https://github.com/Sheenal30/hdb-price-predictor/releases/download/v1.0/feature_list.joblib"

# download if missing
fetch_if_missing(MODELS_DIR / "lightgbm_comparison_model.joblib", REMOTE_MODEL_URL)
fetch_if_missing(MODELS_DIR / "feature_list.joblib", REMOTE_FEATURES_URL)

# 2. Loading model and metadata
MODEL_PATH = BASE_PATH / "models/lightgbm_comparison_model.joblib"
FEATURE_PATH = BASE_PATH / "models/feature_list.joblib"

try:
    model_loaded = joblib.load(MODEL_PATH)
    feature_list_loaded = joblib.load(FEATURE_PATH)
except Exception as e:
    st.error(f"Error loading model or feature list: {e}")
    st.stop()

# 3. Building simple UI
st.title("🏠 HDB Resale Price Predictor")

st.markdown(
    "Fill in the flat details below. Model trained on 1990-2023 data. "
    "_Prediction is for demo only!_"
)

# UI Input Widgets
town        = st.selectbox(
    "Town",
    sorted([
        "bedok","bishan","bukit_batok","bukit_merah",
        "bukit_panjang","bukit_timah","central_area","choa_chu_kang",
        "clementi","geylang","hougang","jurong_east","jurong_west",
        "kallang_whampoa","lim_chu_kang","marine_parade","pasir_ris",
        "punggol","queenstown","sembawang","sengkang","serangoon",
        "tampines","toa_payoh","woodlands","yishun"
    ])
)

flat_type   = st.selectbox(
    "Flat type",
    ["2_room","3_room","4_room","5_room","executive","multi_generation"]
)

floor_area  = st.number_input(
    "Floor area (sqm)",
    min_value=20.0, max_value=200.0, value=90.0, step=5.0
)

lease_left  = st.slider("Years of lease remaining", 0, 99, 85)

if st.button("Predict resale price"):
    # 4. Building a one-row dataframe that matches the training columns
    # Using the loaded feature_list_loaded to create the template row
    row = pd.Series(0, index=feature_list_loaded, dtype="float64")

    # Filling in numeric features
    row["floor_area_sqm"]        = floor_area
    row["lease_remaining_years"] = float(lease_left)
    row["flat_age"]              = float(99 - lease_left)
    row["sale_year"]             = 2025
    row["sale_month"]            = 7

    # Activating the correct one-hot encoded dummy columns
    town_col = f"town_{town}"
    if town_col in row.index:
        row[town_col] = 1

    flat_col = f"flat_type_{flat_type}"
    if flat_col in row.index:
        row[flat_col] = 1

    X_pred = row.to_frame().T # Converting to 2D DataFrame

    # 5. Making a prediction and showing the result
    price = model_loaded.predict(X_pred)[0]
    adjusted_price = price * 1.20   # +20% uplift preserved as you requested
    st.subheader("Estimated resale price")
    st.success(f"SGD {adjusted_price:,.0f}")
    st.caption(MAE_DISCLAIMER_TEXT)