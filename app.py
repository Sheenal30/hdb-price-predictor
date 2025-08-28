"""
HDB Resale Price Predictor (Streamlit mini-demo)

Entering flat details to get a price estimate based on the LightGBM model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# 1. Configuration
MAE_DISCLAIMER_TEXT = "MAE on 2024-25 hold-out data ≈ SGD 54 k. Prediction can be ±10 % off for rare flat types or prime blocks."


# 2. Loading model and metadata
BASE_PATH = Path(__file__).resolve().parent
MODEL_PATH = BASE_PATH / "models/lightgbm_comparison_model.joblib"
FEATURE_PATH = BASE_PATH / "models/feature_list.joblib"

try:
    model_loaded = joblib.load(MODEL_PATH)
    feature_list_loaded = joblib.load(FEATURE_PATH)
except Exception as e:
    st.error(f"Error loading model or feature list: {e}")
    st.stop()



# 3. Building simple UI
st.title("🏠 HDB Resale Price Estimator")

st.markdown(
    "Fill in the flat details below. Model trained on 1990-2023 data.\n"
    "_Prediction is for demo only_"
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
    min_value=20.0, max_value=200.0, value=90.0, step=1.0
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
    adjusted_price = price * 1.20   # +20% uplift
    st.subheader("Estimated resale price")
    st.success(f"SGD {adjusted_price:,.0f}")
    st.caption(MAE_DISCLAIMER_TEXT)