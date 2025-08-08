"""
Train LightGBM model on cleaned HDB resale data.
The script expects the cleaned dataset at data/processed/clean_hdb.csv
and writes the model weights plus feature list into models/.
"""

from pathlib import Path
import joblib

import pandas as pd
import lightgbm as lgb


if __name__ == "__main__":
    # project paths
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_PATH = PROJECT_ROOT / "data/processed/clean_hdb.csv"
    MODEL_DIR = PROJECT_ROOT / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # load cleaned data
    df = pd.read_csv(DATA_PATH)

    # feature selection: keep numeric columns used in app
    feature_cols = [
        "floor_area_sqm",
        "lease_remaining_years",
        "flat_age",
        "sale_year",
        "sale_month",
    ]
    feature_cols += [c for c in df.columns if c.startswith("town_")]
    feature_cols += [c for c in df.columns if c.startswith("flat_type_")]

    X = df[feature_cols]
    y = df["resale_price"]

    # train LightGBM model
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X, y)

    # persist model and feature list
    joblib.dump(model, MODEL_DIR / "lightgbm_comparison_model.joblib")
    joblib.dump(feature_cols, MODEL_DIR / "feature_list.joblib")

    print(f"Saved model with {len(feature_cols)} features to {MODEL_DIR}")
