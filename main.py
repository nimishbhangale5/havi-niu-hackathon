# =============================================================================
# HAVI-NIU 2026 Hackathon — QSR Demand Forecasting
# Author: Nimish Bhangale
# Model: LightGBM + XGBoost Ensemble with advanced feature engineering
# Metric: wMAPE (Weighted Mean Absolute Percentage Error)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import warnings
import os

warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================

print("=" * 60)
print("SECTION 1: Loading Data")
print("=" * 60)

df = pd.read_csv('data/raw/qsr_demand_dataset.csv', parse_dates=['date'])

print(f"Total rows: {len(df):,}")
print(f"Date range: {df['date'].min()} → {df['date'].max()}")
print(f"Restaurants: {df['restaurant_id'].nunique()}")
print(f"Menu items: {df['menu_item_id'].nunique()}")
print(f"Columns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# =============================================================================
# SECTION 2: SPLIT TRAIN vs TEST
# Train = Jan 2021 – Sep 2025 (where quantity is known)
# Test  = Oct 2025 – Dec 2025 (what we need to predict)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: Train/Test Split")
print("=" * 60)

TEST_START = '2025-10-01'

train = df[df['date'] < TEST_START].copy()
test  = df[df['date'] >= TEST_START].copy()

print(f"Train rows: {len(train):,} | {train['date'].min()} → {train['date'].max()}")
print(f"Test rows:  {len(test):,}  | {test['date'].min()} → {test['date'].max()}")

# Handle missing quantity — some rows might have NaN
train = train.dropna(subset=['quantity'])
train['quantity'] = train['quantity'].clip(lower=0)

# =============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: EDA")
print("=" * 60)

os.makedirs('outputs', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('HAVI-NIU QSR Demand Forecasting — EDA', fontsize=16, fontweight='bold')

# Plot 1: Sales by day of week
dow_avg = train.groupby('day_of_week_num')['quantity'].mean()
dow_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
axes[0,0].bar(dow_labels, dow_avg.sort_index(), color='steelblue')
axes[0,0].set_title('Avg Sales by Day of Week')
axes[0,0].set_ylabel('Avg Quantity')

# Plot 2: Sales by month
month_avg = train.groupby('month')['quantity'].mean()
axes[0,1].bar(range(1,13), month_avg.values, color='coral')
axes[0,1].set_title('Avg Sales by Month')
axes[0,1].set_xticks(range(1,13))
axes[0,1].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

# Plot 3: Holiday impact
hol_avg = train.groupby('is_holiday')['quantity'].mean()
axes[0,2].bar(['Non-Holiday','Holiday'], hol_avg.values, color=['steelblue','orange'])
axes[0,2].set_title('Holiday vs Non-Holiday Sales')
axes[0,2].set_ylabel('Avg Quantity')

# Plot 4: Promotion impact
promo_avg = train.groupby('is_promotion')['quantity'].mean()
axes[1,0].bar(['No Promo','Promo'], promo_avg.values, color=['steelblue','green'])
axes[1,0].set_title('Promotion Impact on Sales')

# Plot 5: Top 10 menu items
top_items = train.groupby('menu_item_name')['quantity'].mean().nlargest(10)
axes[1,1].barh(top_items.index, top_items.values, color='purple')
axes[1,1].set_title('Top 10 Menu Items by Avg Sales')
axes[1,1].invert_yaxis()

# Plot 6: Sales trend over time
monthly = train.groupby(train['date'].dt.to_period('M'))['quantity'].mean()
axes[1,2].plot(range(len(monthly)), monthly.values, color='steelblue')
axes[1,2].set_title('Monthly Sales Trend')
axes[1,2].set_xlabel('Month Index')

plt.tight_layout()
plt.savefig('outputs/eda_charts.png', dpi=150, bbox_inches='tight')
plt.close()
print("EDA charts saved to outputs/eda_charts.png")

print(f"\nPromotion lift: {(promo_avg[1]/promo_avg[0]-1)*100:.1f}%")
print(f"Holiday effect: {(hol_avg[1]/hol_avg[0]-1)*100:.1f}%")
print(f"Weekend flag is_weekend=1 avg: {train[train['is_weekend']==1]['quantity'].mean():.2f}")
print(f"Weekday avg: {train[train['is_weekend']==0]['quantity'].mean():.2f}")

# =============================================================================
# SECTION 4: FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4: Feature Engineering")
print("=" * 60)

def engineer_features(df_full):
    """
    Full feature engineering pipeline.
    Must be applied to the FULL dataset (train+test combined)
    so that lag features for test rows can reference train rows.
    """
    df = df_full.sort_values(['restaurant_id', 'menu_item_id', 'date']).copy()

    # --- 4.1 Cyclical calendar features ---
    # Cyclical encoding captures periodic patterns better than raw integers
    df['dow_sin']   = np.sin(2 * np.pi * df['day_of_week_num'] / 7)
    df['dow_cos']   = np.cos(2 * np.pi * df['day_of_week_num'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year']  = df['date'].dt.isocalendar().week.astype(int)
    df['quarter']       = df['date'].dt.quarter

    # --- 4.2 Holiday proximity (pre/post holiday surges in QSR) ---
    holiday_dates = pd.to_datetime(df[df['is_holiday'] == 1]['date'].unique())
    def days_to_next(d):
        future = holiday_dates[holiday_dates > d]
        return (future.min() - d).days if len(future) > 0 else 99
    def days_from_last(d):
        past = holiday_dates[holiday_dates < d]
        return (d - past.max()).days if len(past) > 0 else 99

    df['days_to_holiday']   = df['date'].apply(days_to_next).clip(upper=7)
    df['days_from_holiday'] = df['date'].apply(days_from_last).clip(upper=7)

    # --- 4.3 Weather features ---
    df['temp_cold']   = (df['avg_temp_f'] < 32).astype(int)
    df['temp_hot']    = (df['avg_temp_f'] > 85).astype(int)
    df['heavy_precip']= (df['precip_inches'] > 0.5).astype(int)
    df['precip_type_snow'] = (df['precip_type'] == 'Snow').astype(int)
    df['precip_type_rain'] = (df['precip_type'] == 'Rain').astype(int)
    df['cold_weekend']  = df['temp_cold'] * df['is_weekend']
    df['promo_weekend'] = df['is_promotion'] * df['is_weekend']
    df['event_weekend'] = df['is_special_event'] * df['is_weekend']

    # --- 4.4 Lag features (grouped by restaurant + item) ---
    # These are the most predictive features for time-series forecasting
    grp = df.groupby(['restaurant_id', 'menu_item_id'])['quantity']

    for lag in [7, 14, 21, 28, 91, 182, 364]:
        df[f'lag_{lag}'] = grp.shift(lag)

    # Same day of week lags (very powerful for QSR weekly patterns)
    df['lag_7_same_dow']  = grp.shift(7)
    df['lag_14_same_dow'] = grp.shift(14)

    # --- 4.5 Rolling statistics ---
    for window in [7, 14, 28]:
        shifted = grp.shift(1)
        df[f'roll_mean_{window}'] = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        df[f'roll_std_{window}']  = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df[f'roll_max_{window}']  = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).max())
        df[f'roll_min_{window}']  = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).min())

    # Rolling mean for same day of week (e.g., avg of last 4 Fridays)
    df['roll_mean_same_dow_4w'] = (
        df.groupby(['restaurant_id', 'menu_item_id', 'day_of_week_num'])['quantity']
          .shift(1)
          .transform(lambda x: x.rolling(4, min_periods=1).mean())
    )

    return df


# Combine train+test so lag features for test can reference train history
all_data = pd.concat([train, test], ignore_index=True)
all_data = engineer_features(all_data)

# --- 4.6 Target encoding (computed on train only, applied to all) ---
print("Computing target encodings...")

item_mean     = train.groupby('menu_item_id')['quantity'].mean().rename('item_global_mean')
rest_mean     = train.groupby('restaurant_id')['quantity'].mean().rename('rest_global_mean')
item_rest_mean= train.groupby(['restaurant_id','menu_item_id'])['quantity'].mean().rename('item_rest_mean')
cat_mean      = train.groupby('category')['quantity'].mean().rename('category_mean')
dow_item_mean = train.groupby(['day_of_week_num','menu_item_id'])['quantity'].mean().rename('dow_item_mean')

all_data = all_data.join(item_mean,      on='menu_item_id')
all_data = all_data.join(rest_mean,      on='restaurant_id')
all_data = all_data.join(item_rest_mean, on=['restaurant_id','menu_item_id'])
all_data = all_data.join(cat_mean,       on='category')
all_data = all_data.join(dow_item_mean,  on=['day_of_week_num','menu_item_id'])

print(f"Feature engineering complete. Total features: {all_data.shape[1]}")

# Split back into train and test
train_fe = all_data[all_data['date'] < TEST_START].copy()
test_fe  = all_data[all_data['date'] >= TEST_START].copy()

# =============================================================================
# SECTION 5: MODEL TRAINING
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: Model Training")
print("=" * 60)

# Define feature columns (drop non-numeric / identifier columns)
DROP_COLS = [
    'date', 'quantity', 'restaurant_name', 'city', 'state',
    'menu_item_name', 'day_of_week', 'holiday_name',
    'special_event_name', 'precip_type'
]

FEATURE_COLS = [c for c in train_fe.columns if c not in DROP_COLS]
print(f"Using {len(FEATURE_COLS)} features: {FEATURE_COLS}")

# Label encode categorical ID columns
from sklearn.preprocessing import LabelEncoder

for col in ['restaurant_id', 'menu_item_id', 'category']:
    le = LabelEncoder()
    all_values = pd.concat([train_fe[col], test_fe[col]]).astype(str)
    le.fit(all_values)
    train_fe[col] = le.transform(train_fe[col].astype(str))
    test_fe[col]  = le.transform(test_fe[col].astype(str))

# --- 5.1 Validation split (Oct–Dec 2024 mirrors the test period) ---
VAL_START = '2024-10-01'
VAL_END   = '2024-12-31'

df_tr  = train_fe[train_fe['date'] < VAL_START].copy()
df_val = train_fe[(train_fe['date'] >= VAL_START) & (train_fe['date'] <= VAL_END)].copy()

# Drop rows where lag features are NaN (first few weeks of data)
df_tr  = df_tr.dropna(subset=['lag_364'])
df_val = df_val.dropna(subset=['lag_364'])

X_tr  = df_tr[FEATURE_COLS]
y_tr  = df_tr['quantity']
X_val = df_val[FEATURE_COLS]
y_val = df_val['quantity']

print(f"Train rows: {len(X_tr):,} | Val rows: {len(X_val):,}")

# --- wMAPE metric ---
def wmape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 0, None)
    weights = y_true  # weight by actual volume
    mask = weights > 0
    return np.sum(weights[mask] * np.abs(y_true[mask] - y_pred[mask])) / np.sum(weights[mask] * y_true[mask])

# --- Naive baseline: same week last year ---
print("\nBaseline model (lag_364)...")
naive_preds = df_val['lag_364'].fillna(df_val['item_rest_mean'])
print(f"Naive baseline wMAPE: {wmape(y_val, naive_preds):.4f}")

# --- 5.2 LightGBM ---
print("\nTraining LightGBM...")

lgb_params = {
    'objective':        'regression_l1',   # MAE loss — robust to outliers
    'metric':           'mae',
    'learning_rate':    0.05,
    'num_leaves':       127,
    'min_child_samples':50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':     1,
    'n_estimators':     3000,
    'verbose':          -1,
    'random_state':     42,
}

model_lgb = lgb.LGBMRegressor(**lgb_params)
model_lgb.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(200)]
)

val_preds_lgb = model_lgb.predict(X_val).clip(min=0)
print(f"LightGBM val wMAPE: {wmape(y_val, val_preds_lgb):.4f}")

# --- 5.3 XGBoost ---
print("\nTraining XGBoost...")

model_xgb = xgb.XGBRegressor(
    objective        = 'reg:absoluteerror',
    n_estimators     = 3000,
    learning_rate    = 0.05,
    max_depth        = 7,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 50,
    early_stopping_rounds = 150,
    eval_metric      = 'mae',
    verbosity        = 0,
    random_state     = 42,
)

model_xgb.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=200
)

val_preds_xgb = model_xgb.predict(X_val).clip(min=0)
print(f"XGBoost val wMAPE: {wmape(y_val, val_preds_xgb):.4f}")

# --- 5.4 Ensemble blend ---
w_lgb, w_xgb = 0.6, 0.4
val_preds_blend = (w_lgb * val_preds_lgb + w_xgb * val_preds_xgb).clip(min=0)
print(f"\nEnsemble val wMAPE: {wmape(y_val, val_preds_blend):.4f}")

# --- 5.5 Feature importance ---
feat_imp = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': model_lgb.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 features:\n{feat_imp.head(15).to_string(index=False)}")

# Save feature importance chart
plt.figure(figsize=(10, 8))
sns.barplot(data=feat_imp.head(20), x='importance', y='feature', palette='viridis')
plt.title('Top 20 Feature Importances (LightGBM)')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150)
plt.close()
print("Feature importance chart saved.")

# =============================================================================
# SECTION 6: RETRAIN ON FULL DATA + GENERATE PREDICTIONS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 6: Final Model — Retrain on Full Data")
print("=" * 60)

# Use ALL training data (Jan 2021 – Sep 2025) for final model
train_full = train_fe.dropna(subset=['lag_364'])
X_full = train_full[FEATURE_COLS]
y_full = train_full['quantity']

print(f"Retraining on {len(X_full):,} rows...")

# Retrain LightGBM with best iteration
final_lgb = lgb.LGBMRegressor(**{**lgb_params, 'n_estimators': model_lgb.best_iteration_})
final_lgb.fit(X_full, y_full)

# Retrain XGBoost
final_xgb = xgb.XGBRegressor(
    objective='reg:absoluteerror',
    n_estimators=model_xgb.best_iteration,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=50,
    verbosity=0,
    random_state=42,
)
final_xgb.fit(X_full, y_full)

# Generate test predictions
X_test = test_fe[FEATURE_COLS]

# Fill any remaining NaNs in test features with column means from training
for col in FEATURE_COLS:
    if X_test[col].isnull().any():
        X_test[col] = X_test[col].fillna(X_full[col].mean())

preds_lgb = final_lgb.predict(X_test).clip(min=0)
preds_xgb = final_xgb.predict(X_test).clip(min=0)
final_preds = (w_lgb * preds_lgb + w_xgb * preds_xgb).clip(min=0)

# Round to nearest integer (you can't sell 0.7 of a burger)
final_preds_rounded = np.round(final_preds).astype(int)

print(f"Predictions — min: {final_preds_rounded.min()}, max: {final_preds_rounded.max()}, mean: {final_preds_rounded.mean():.1f}")

# =============================================================================
# SECTION 7: BUILD SUBMISSION FILE
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 7: Build Submission CSV")
print("=" * 60)

# Load original test data for correct restaurant_id and menu_item_id strings
test_original = df[df['date'] >= TEST_START].copy()

submission = test_original[['date', 'restaurant_id', 'menu_item_id']].copy()
submission['predicted_quantity'] = final_preds_rounded
submission['date'] = submission['date'].dt.strftime('%Y-%m-%d')

# Validate
print(f"Submission shape: {submission.shape}")
print(f"Expected:         (69000, 4)")
assert len(submission) == 92 * 15 * 50, f"Row count mismatch! Got {len(submission)}"
assert list(submission.columns) == ['date','restaurant_id','menu_item_id','predicted_quantity']
assert submission['predicted_quantity'].isnull().sum() == 0

submission.to_csv('outputs/submission.csv', index=False)
print("✅ Submission saved to outputs/submission.csv")
print(f"\nSample predictions:\n{submission.head(10).to_string(index=False)}")

# =============================================================================
# SECTION 8: ERROR ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 8: Validation Error Analysis")
print("=" * 60)

# Error analysis on validation set
df_val['pred'] = val_preds_blend
df_val['abs_error'] = np.abs(df_val['quantity'] - df_val['pred'])
df_val['pct_error'] = df_val['abs_error'] / df_val['quantity'].clip(lower=1)

# Worst performing restaurant-item combos
error_by_combo = df_val.groupby(['restaurant_id','menu_item_id']).apply(
    lambda x: wmape(x['quantity'], x['pred'])
).reset_index()
error_by_combo.columns = ['restaurant_id','menu_item_id','wmape']

print("Top 5 hardest combos to forecast (highest wMAPE):")
print(error_by_combo.nlargest(5, 'wmape').to_string(index=False))

print("\nTop 5 easiest combos to forecast (lowest wMAPE):")
print(error_by_combo.nsmallest(5, 'wmape').to_string(index=False))

print("\n" + "=" * 60)
print("✅ ALL DONE!")
print(f"   Validation wMAPE (Ensemble): {wmape(y_val, val_preds_blend):.4f}")
print(f"   Submission rows: {len(submission):,}")
print(f"   Output files:")
print(f"     outputs/submission.csv")
print(f"     outputs/eda_charts.png")
print(f"     outputs/feature_importance.png")
print("=" * 60)
