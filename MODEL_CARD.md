# Model Card — LightGBM HDB Resale Price Predictor
**Repo:** https://github.com/Sheenal30/hdb-price-predictor

---

## 1. Model
A simple model that predicts HDB resale prices in Singapore using LightGBM

---

## 2. Quick numbers
- LightGBM MAE (hold-out 2024+): 53,886.21 SGD
- RandomForest MAE: 59,963.25 SGD
- Linear baseline MAE: 83,733.90 SGD (baseline model file not persisted; metrics CSV only)
- Improvement vs baseline: ~35.6% MAE reduction

---

## 3. Data used
- File: `data/processed/clean_hdb.csv`
- Years: sales from 1990 through 2023 used for training, 2024+ held out for testing
- Main inputs: floor_area_sqm, lease_remaining_years, flat_age, sale_year, sale_month, town dummies, flat_type dummies, storey_range/flat_model dummies
- `price_per_sqm` is derived from the label and used for EDA only

---

## 4. Data preprocessing
- Filled missing numeric values with the median using `SimpleImputer`
- Encoded categories with `pd.get_dummies` and saved the column order in `models/feature_list.joblib`

---

## 5. Training and validation
- Chronological hold-out: train `sale_year <= 2023`, test `sale_year >= 2024`
- Cross-validation: `TimeSeriesSplit(n_splits=5)` on the training set to avoid time leakage
- LightGBM basics used: learning_rate=0.05, many trees, subsample/colsample applied
- Saved artifacts: `models/lightgbm_comparison_model.joblib` and `models/feature_list.joblib`

---

## 6. Model
- SHAP TreeExplainer used to inspect feature effects
- SHAP run on a very small sample (n≈20) for demo speed, saved plot at `reports/figures/shap_summary.png`
- RandomForest Gini importances plotted at `reports/figures/rf_importance.png`
- Error slices (MAE by town and flat_type) computed in the notebooks

---

## 7. Limitations
- Model trained on data through 2023, while the resale market grew ~10% in 2024–25, so the model may underpredict current prices
- Location is coarse: only town-level dummies, no block or MRT distances
- Underperforms on rare or high-value flats (executive, multi-generation)
- SHAP analysis here is demonstrative because it used a tiny sample

---

## 8. Run locally (quick)
1. Create env  
   ```bash
   conda env create -f environment.yml
   conda activate hdb-predictor-app-env