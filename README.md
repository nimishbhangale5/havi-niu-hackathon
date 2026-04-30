# 🏆 HAVI-NIU 2026 QSR Demand Forecasting Hackathon — 1st Place

> **Final wMAPE: 0.1308 — Competition Winner**

## 🎯 Problem
Predict daily demand for **50 menu items** across **15 Quick Service Restaurants** 
in the Midwest United States for October–December 2025.

- 92 days × 15 restaurants × 50 items = **69,000 predictions**
- 5 years of historical sales data (Jan 2021 – Sep 2025)
- Evaluation metric: **wMAPE** (weighted by menu item volume)

## 📊 Results

| Model | Validation wMAPE |
|---|---|
| Naive Baseline (same week last year) | 0.2614 |
| LightGBM | 0.1338 |
| XGBoost | 0.1331 |
| CatBoost | 0.1315 |
| **Final Ensemble** | **0.1308 🏆** |

**50% improvement** over the naive baseline.

## 🔑 Top Features
| Rank | Feature | Description |
|---|---|---|
| 1 | lag_364 | Same day last year |
| 2 | avg_temp_f | Temperature |
| 3 | smonth_avg | Same month historical average |
| 4 | is_holiday | Holiday flag |
| 5 | lag_371 | Same we️ Approach

### Feature Engineering (102 features)
- Lag features (1 to 371 days)
- Same day-of-week lags
- Rolling statistics (mean, std, min, max)
- Cyclical calendar encodings (sin/cos)
- Holiday proximity features
- Weather interaction features
- Target encodings (restaurant × item × day-of-week)
- Historical period averages (Q4 2024, Q4 2023)

### Models
Three gradient boosting models trained with MAE loss and 
volume-based sample weights to align with wMAPE metric:
- **LightGBM** — fastest, great for large datasets
- **XGBoost** — industry standard, strong baseline
- **CatBoost** — best performance, handles categories natively

Ensemble weights optimized by grid search on validation set.
CatBoost dominated with 100% weight.

### Iterative Forecasting
Predicted one day at a time across the 92-day horizon. 
Each day's prediction was stored and used as a lag feature 
for subsequent days — recursive multi-step forecasting.

### Validation
Walk-forward validation: trained on Jan 2021 – Sep 20 on Oct–Dec 2024 (mirrors the test period seasonally).

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Download data from Kaggle
kaggle competitions download -c 2026-havi-niu-hackathon -p data/raw/

# Run the winning model
python main_v5.py
```

## 📁 Repository Structure
├── main_v5.py              # Winning solution (wMAPE 0.1308)
├── outputs/
│   ├── submission_v5.csv   # Final 69,000 predictions
│   ├── eda_charts.png      # Exploratory data analysis
│   └── feature_importance.png
├── requirements.txt
└── README.md
## 👥 Team — Pawthon
- **Nimish Bhangale**
- **Ashutosh Mode**
- **Hubert Stasik**
- **Alexander Mills**

## 🙏 Acknowledgements
- **Professor Biagio Palese** — Northern Illinois University
- **Professor Ying Wang** — Northern Illinois University
- **HAVI** — for organizing this competition

## 🛠️ Tech Stack
Python · LightGBM · XGBoost · CatBoost · Pandas · NumPy
