
# HDB Price Predictor

Hi, this is my first end-to-end ML project. I built a small pipeline to predict HDB resale prices in Singapore
See full model card: MODEL_CARD.md

---

## What this repo does
- Loads a cleaned HDB dataset and runs simple EDA charts
- Trains a Linear Regression baseline so we have a transparent starting point
- Trains tree models (RandomForest and LightGBM) to capture non linear effects
- Uses SHAP to explain model behavior and to debug edge cases
- Ships a tiny Streamlit demo so you can enter flat details and get a price estimate

---

## Results
- Linear Regression MAE (2024–25 hold-out): **83,733.90 SGD**
- RandomForest MAE: **59,963.25 SGD**
- **LightGBM (best) MAE: 53,886.21 SGD**
- MAE improvement vs baseline: **29,847.69 SGD** (~**35.6%**)
MAE = mean absolute error. On average predictions are ~SGD 54k away from the true price.

See full model card: MODEL_CARD.md

---

## Quickstart
   ```bash
   conda env create -f environment.yml
   conda activate hdb-predictor-app-env



