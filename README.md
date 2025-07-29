# 🏠 Singapore HDB Resale Price Predictor (Work-in-Progress)

Quick prototype to predict HDB resale prices from open data.  
Goal: prove I can scope, learn, and ship an end-to-end ML project in one week for my NTU MSc (AI) application.

---

## ⚡ Quick-start

```bash
# clone + install
git clone https://github.com/Sheenal30/hdb-price-predictor
cd hdb-price-predictor
pip install -r requirements.txt

# launch (currently empty) notebook
jupyter notebook notebooks/01_end_to_end.ipynb
```
---

## Optional Streamlit demo (to be added later):
```
streamlit run app/streamlit_app.py
```
---

## 🔍 Why this project?
* Real impact: HDB prices affect 80 % of SG residents
* Open, trustworthy data: Government publishes monthly resale prices since 1990
* Tight timeline: Shows rapid learning and execution under pressure

---

## 📂 Data 

| Source | Link | Notes |
|--------|------|------|
| HDB Resale Flat Prices | [data.gov.sg](https://data.gov.sg/dataset/resale-flat-prices) | CSVs by month, 1990-present
2025 |

**Planned engineered features**

* `price_per_sqm` — `resale_price / floor_area_sqm`  
* `lease_remaining_years` — years left on 99-year lease  
* `flat_age` — `sale_year - lease_commence_year`  
* One-hot encodes for `town`, `flat_type`

(Exact row count, date range, and feature list will be updated after data load.)

---

## 🧪 Planned Experiments

| Milestone | Deliverable |
|-------|-----------|
| Baseline model | LinearRegression + time-based train/test split |
| Improved model | LightGBM or RandomForest |
| Interpretability | Permutation/SHAP feature importance |
| Mini-app | Streamlit form → price prediction |

---

## Demo (placeholder)
<details>
<summary>Streamlit GIF (click to open)</summary>
</details>  
---

## 🗺 Project layout

<details>
<summary>Click to view tree</summary>

```text
hdb-price-predictor/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_end_to_end.ipynb
├── src/
│   ├── preprocess.py
│   └── train.py
├── app/
│   └── streamlit_app.py
├── reports/
│   ├── figures/
│   ├── model_card.md
│   └── learning_log.md
├── requirements.txt
└── README.md

</details>
```

---

## 🤔 Limitations & Future Work

* **Granularity**: No block-level GPS or MRT distance yet.
* **Class imbalance**: Very few 1-room and jumbo flats.
* **Feature scope**: Interior condition, floor level, amenities proximity missing.
* **Metrics**: MAE / RMSE numbers will be filled in once models run.

---

## 📚 Model Card (placeholder)

Detailed objective, evaluation, and ethics considerations will be documented in reports/model_card.md after initial results.

---
## 🛠 Stack & Workflow

*Python 3.11 · Pandas · scikit-learn · (planned) LightGBM · SHAP · Streamlit*  

I’ll lean on ChatGPT for boilerplate and bug-hunting but will design, validate, and interpret every step myself. designed and verified by me.

---

## 🙏 Acknowledgements

* **data.gov.sg** for making HDB resale data public
* Community tutorials & docs I’ll reference along the way 😅

