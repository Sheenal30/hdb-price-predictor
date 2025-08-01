#!/usr/bin/env python
# coding: utf-8

# In[2]:


# --- 1. Setup and Data Loading ---
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb # Import LightGBM
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings

# Suppress warnings for cleaner output, if desired
warnings.filterwarnings('ignore')

# Project root setup
PROJECT_ROOT = Path('/app')

# Load processed data
try:
    X_train = pd.read_csv(PROJECT_ROOT / "data/processed/X_train_processed.csv")
    y_train = pd.read_csv(PROJECT_ROOT / "data/processed/y_train_processed.csv").squeeze()
    X_test = pd.read_csv(PROJECT_ROOT / "data/processed/X_test_processed.csv")
    y_test = pd.read_csv(PROJECT_ROOT / "data/processed/y_test_processed.csv").squeeze()
    print("Processed data loaded successfully.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
except FileNotFoundError as e:
    print(f"Error loading processed data: {e}. Make sure 04_tree_model.ipynb was run and saved the data.")
    exit() # Exit if data isn't found, as further steps will fail

# Helper Function for Categorical Reconstruction (from 05_error_analysis)
def reconstruct_category(df, prefix):
    """
    Reconstructs a single categorical column from its one-hot encoded dummy columns.
    Example: If df has 'town_bishan', 'town_bedok', etc., and prefix is 'town_',
    it will create a 'town' column with values 'bishan', 'bedok'.
    """
    category_dummy_cols = [col for col in df.columns if col.startswith(prefix)]
    if not category_dummy_cols:
        return pd.Series(np.nan, index=df.index, name=prefix.rstrip('_'))

    reconstructed_series = df[category_dummy_cols].idxmax(axis=1).str.replace(prefix, '')
    return reconstructed_series


# In[3]:


# --- 2. Model 1: Random Forest Regressor ---
print("\n--- Training Random Forest Regressor ---")
rf_r = RandomForestRegressor(
    n_estimators=10, # Using 10 as per previous setup
    max_depth=None,
    n_jobs=-1,
    random_state=42
)
rf_r.fit(X_train, y_train)

# Evaluate Random Forest
rf_preds = rf_r.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

print(f"Random Forest MAE: {rf_mae:,.0f} SGD")
print(f"Random Forest RMSE: {rf_rmse:,.0f} SGD")


# In[4]:


# --- Create Sample Weights for Rare Classes ---
# Initialize all weights to 1
sample_weights = np.ones(len(X_train))

# Identify Executive and Multi-Generation flats in the training set
# These are the one-hot encoded columns in X_train
executive_mask = X_train['flat_type_executive'] == True
multi_gen_mask = X_train['flat_type_multi_generation'] == True

# Assign a higher weight (e.g., 2 times the normal weight)
sample_weights[executive_mask] = 2.0
sample_weights[multi_gen_mask] = 2.0

print(f"Assigned higher weights to {executive_mask.sum() + multi_gen_mask.sum()} rare class flats.")


# In[5]:


# --- 3. Model 2: LightGBM Regressor ---
print("\n--- Training LightGBM Regressor ---")
lgbm_r = lgb.LGBMRegressor(
    n_estimators=100, # Since LightGBM is optimized for speed and performance
    learning_rate=0.05,
    num_leaves=31, # Default value, can be tuned
    n_jobs=-1,
    random_state=42,
    colsample_bytree=0.7, # Feature subsampling
    subsample=0.7,        # Data subsampling
    reg_alpha=0.1,        # L1 regularization
    reg_lambda=0.1,        # L2 regularization
)
lgbm_r.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluate LightGBM
lgbm_preds = lgbm_r.predict(X_test)
lgbm_mae = mean_absolute_error(y_test, lgbm_preds)
lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_preds))

print(f"LightGBM MAE: {lgbm_mae:,.0f} SGD")
print(f"LightGBM RMSE: {lgbm_rmse:,.0f} SGD")


# In[6]:


# --- 4. Compare Models (Initial Metrics) ---
print("\n--- Model Comparison Summary ---")
print(f"Random Forest MAE: {rf_mae:,.0f} SGD")
print(f"LightGBM MAE:    {lgbm_mae:,.0f} SGD")

if lgbm_mae < rf_mae:
    print(f"\nLightGBM performed better by {rf_mae - lgbm_mae:,.0f} SGD MAE!")
    best_model_name = "LightGBM"
    best_model = lgbm_r
    best_preds = lgbm_preds
    best_mae = lgbm_mae
    best_rmse = lgbm_rmse # <--- ADDED THIS LINE
else:
    print(f"\nRandom Forest performed better by {lgbm_mae - rf_mae:,.0f} SGD MAE!")
    best_model_name = "Random Forest"
    best_model = rf_r
    best_preds = rf_preds
    best_mae = rf_mae
    best_rmse = rf_rmse # <--- ADDED THIS LINE

print(f"\nProceeding with error analysis for the {best_model_name} model.")


# In[7]:


# --- 5. Error Analysis (for the best performing model) ---

# Attach predictions & error to test frame for detailed analysis
test_df_analysis = X_test.copy()
test_df_analysis["actual"] = y_test.values
test_df_analysis["predicted"] = best_preds
test_df_analysis["abs_error"] = (test_df_analysis["actual"] - test_df_analysis["predicted"]).abs()

# Reconstruct 'town' and 'flat_type' from one-hot encoded columns in test_df_analysis
test_df_analysis['town'] = reconstruct_category(test_df_analysis, 'town_')
test_df_analysis['flat_type'] = reconstruct_category(test_df_analysis, 'flat_type_')


# 5.1.2 Worst 20 rows
worst20 = test_df_analysis.sort_values("abs_error", ascending=False).head(20)
display_cols = ["actual", "predicted", "abs_error",
                "sale_year", "town", "flat_type", "floor_area_sqm", "lease_remaining_years"]
print(f"\n--- Top-20 worst absolute errors for {best_model_name} ---")
print(worst20[display_cols].to_string())


# 5.1.3 Error slices – town & flat_type
def mae_by(group_col, df_to_analyze):
    return (df_to_analyze
            .assign(error=df_to_analyze["abs_error"])
            .groupby(group_col)["error"]
            .mean()
            .sort_values(ascending=False)
            .head(10))

print(f"\n--- Worst towns by MAE for {best_model_name} ---")
# Ensure 'town' column has no NaNs that might have resulted from reconstruction failures
print(mae_by("town", test_df_analysis.dropna(subset=['town'])))

print(f"\n--- Worst flat_types by MAE for {best_model_name} ---")
# Ensure 'flat_type' column has no NaNs
print(mae_by("flat_type", test_df_analysis.dropna(subset=['flat_type'])))


# In[8]:


# --- 6. Save Best Model and Metrics ---
(PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, PROJECT_ROOT / f"models/{best_model_name.lower().replace(' ', '_')}_comparison_model.joblib")
print(f"\nBest model ({best_model_name}) saved successfully.")

feature_list = X_train.columns.tolist()
joblib.dump(feature_list, PROJECT_ROOT / "models/feature_list.joblib")
print("Feature list saved successfully.")

(PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
metrics_df = pd.DataFrame([{'model': best_model_name, 'MAE': best_mae, 'RMSE': best_rmse}])
metrics_df.to_csv(PROJECT_ROOT / "reports/model_comparison_metrics.csv", index=False)
print("Model comparison metrics saved.")

print("\nNotebook 06_model_comparison.ipynb execution complete.")
