# =============================================================================
# HAVI-NIU 2026 Hackathon — QSR Demand Forecasting v3
# Key change: log1p target transform — single biggest improvement
# Runtime: ~15 minutes
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import optuna
import warnings
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================

print("=" * 60)
print("SECTION 1: Loading Data")
print("=" * 60)

df = pd.read_csv('data/raw/qsr_demand_dataset.csv', parse_dates=['date'])
df['quantity'] = df['quantity'].clip(lower=0)

TEST_START = '2025-10-01'
VAL_START  = '2024-10-01'
VAL_END    = '2024-12-31'

train = df[df['date'] < TEST_START].dropna(subset=['quantity']).copy()
test  = df[df['date'] >= TEST_START].copy()

print(f"Train: {len(train):,} | Test: {len(test):,}")

# =============================================================================
# SECTION 2: FEATURE ENGINEERING
# =============================================================================

print("\nBuilding features...")

def engineer_features(df_full, train_ref):
    df = df_full.sort_values(['restaurant_id', 'menu_item_id', 'date']).copy()

    # Calendar
    df['dow_sin']       = np.sin(2 * np.pi * df['day_of_week_num'] / 7)
    df['dow_cos']       = np.cos(2 * np.pi * df['day_of_week_num'] / 7)
    df['month_sin']     = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']     = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_month']  = df['date'].dt.day
    df['week_of_year']  = df['date'].dt.isocalendar().week.astype(int)
    df['quarter']       = df['date'].dt.quarter
    df['is_friday']     = (df['day_of_week_num'] == 4).astype(int)
    df['is_saturday']   = (df['day_of_week_num'] == 5).astype(int)
    df['is_sunday']     = (df['day_of_week_num'] == 6).astype(int)

    # Holiday proximity
    holiday_dates = pd.to_datetime(df[df['is_holiday'] == 1]['date'].unique())
    def days_to_next(d):
        future = holiday_dates[holiday_dates > d]
        return (future.min() - d).days if len(future) > 0 else 99
    def days_from_last(d):
        past = holiday_dates[holiday_dates < d]
        return (d - past.max()).days if len(past) > 0 else 99

    df['days_to_holiday']   = df['date'].apply(days_to_next).clip(upper=7)
    df['days_from_holiday'] = df['date'].apply(days_from_last).clip(upper=7)
    df['holiday_window']    = ((df['days_to_holiday'] <= 3) | (df['days_from_holiday'] <= 3)).astype(int)

    # Weather
    df['temp_cold']        = (df['avg_temp_f'] < 32).astype(int)
    df['temp_cool']        = (df['avg_temp_f'].between(32, 55)).astype(int)
    df['temp_hot']         = (df['avg_temp_f'] > 85).astype(int)
    df['heavy_precip']     = (df['precip_inches'] > 0.5).astype(int)
    df['precip_type_snow'] = (df['precip_type'] == 'Snow').astype(int)
    df['precip_type_rain'] = (df['precip_type'] == 'Rain').astype(int)

    # Interactions
    df['promo_weekend']  = df['is_promotion'] * df['is_weekend']
    df['promo_holiday']  = df['is_promotion'] * df['is_holiday']
    df['promo_event']    = df['is_promotion'] * df['is_special_event']
    df['cold_weekend']   = df['temp_cold'] * df['is_weekend']
    df['price_x_promo']  = df['unit_price'] * df['is_promotion']

    # *** LOG1P TRANSFORM ON QUANTITY before computing lags ***
    # This is the key change — we model log(quantity) not quantity directly
    df['log_qty'] = np.log1p(df['quantity'])

    grp = df.groupby(['restaurant_id', 'menu_item_id'])['log_qty']

    # Lags on log scale
    for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 91, 182, 364]:
        df[f'lag_{lag}'] = grp.shift(lag)

    # Rolling stats on log scale
    for window in [7, 14, 28, 56]:
        shifted = grp.shift(1)
        df[f'roll_mean_{window}'] = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        df[f'roll_std_{window}']  = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df[f'roll_max_{window}']  = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).max())

    # EWM on log scale
    df['ewm_7']  = grp.shift(1).transform(lambda x: x.ewm(span=7,  min_periods=1).mean())
    df['ewm_14'] = grp.shift(1).transform(lambda x: x.ewm(span=14, min_periods=1).mean())
    df['ewm_28'] = grp.shift(1).transform(lambda x: x.ewm(span=28, min_periods=1).mean())

    # Same DOW rolling (most powerful feature for QSR)
    dow_grp = df.groupby(['restaurant_id', 'menu_item_id', 'day_of_week_num'])['log_qty']
    df['roll_mean_same_dow_4w']  = dow_grp.shift(1).transform(lambda x: x.rolling(4,  min_periods=1).mean())
    df['roll_mean_same_dow_8w']  = dow_grp.shift(1).transform(lambda x: x.rolling(8,  min_periods=1).mean())
    df['roll_mean_same_dow_12w'] = dow_grp.shift(1).transform(lambda x: x.rolling(12, min_periods=1).mean())

    # Target encodings from train_ref (on log scale)
    log_qty_col = np.log1p(train_ref['quantity'])
    tmp = train_ref.copy()
    tmp['log_qty'] = log_qty_col

    encs = {
        'item_global_mean':  tmp.groupby('menu_item_id')['log_qty'].mean(),
        'rest_global_mean':  tmp.groupby('restaurant_id')['log_qty'].mean(),
        'item_rest_mean':    tmp.groupby(['restaurant_id','menu_item_id'])['log_qty'].mean(),
        'category_mean':     tmp.groupby('category')['log_qty'].mean(),
        'dow_item_mean':     tmp.groupby(['day_of_week_num','menu_item_id'])['log_qty'].mean(),
        'dow_rest_mean':     tmp.groupby(['day_of_week_num','restaurant_id'])['log_qty'].mean(),
        'month_item_mean':   tmp.groupby(['month','menu_item_id'])['log_qty'].mean(),
        'promo_item_mean':   tmp.groupby(['is_promotion','menu_item_id'])['log_qty'].mean(),
        'weekend_item_mean': tmp.groupby(['is_weekend','menu_item_id'])['log_qty'].mean(),
    }
    join_keys = {
        'item_global_mean':  'menu_item_id',
        'rest_global_mean':  'restaurant_id',
        'item_rest_mean':    ['restaurant_id','menu_item_id'],
        'category_mean':     'category',
        'dow_item_mean':     ['day_of_week_num','menu_item_id'],
        'dow_rest_mean':     ['day_of_week_num','restaurant_id'],
        'month_item_mean':   ['month','menu_item_id'],
        'promo_item_mean':   ['is_promotion','menu_item_id'],
        'weekend_item_mean': ['is_weekend','menu_item_id'],
    }
    for name, enc in encs.items():
        enc.name = name
        df = df.join(enc, on=join_keys[name])

    df['item_to_rest_ratio'] = df['item_global_mean'] / df['rest_global_mean'].clip(lower=0.01)
    df['dow_vs_mean_ratio']  = df['dow_item_mean'] / df['item_global_mean'].clip(lower=0.01)

    return df


all_data = pd.concat([train, test], ignore_index=True)
all_data = engineer_features(all_data, train)

train_fe = all_data[all_data['date'] < TEST_START].copy()
test_fe  = all_data[all_data['date'] >= TEST_START].copy()

# Label encode
from sklearn.preprocessing import LabelEncoder
for col in ['restaurant_id', 'menu_item_id', 'category']:
    le = LabelEncoder()
    all_vals = pd.concat([train_fe[col], test_fe[col]]).astype(str)
    le.fit(all_vals)
    train_fe[col] = le.transform(train_fe[col].astype(str))
    test_fe[col]  = le.transform(test_fe[col].astype(str))

DROP_COLS = [
    'date', 'quantity', 'log_qty', 'restaurant_name', 'city', 'state',
    'menu_item_name', 'day_of_week', 'holiday_name',
    'special_event_name', 'precip_type'
]
FEATURE_COLS = [c for c in train_fe.columns if c not in DROP_COLS]
print(f"Total features: {len(FEATURE_COLS)}")

# =============================================================================
# SECTION 3: VALIDATION SPLIT
# =============================================================================

df_tr  = train_fe[train_fe['date'] < VAL_START].dropna(subset=['lag_364']).copy()
df_val = train_fe[(train_fe['date'] >= VAL_START) & (train_fe['date'] <= VAL_END)].dropna(subset=['lag_364']).copy()

X_tr   = df_tr[FEATURE_COLS]
y_tr   = df_tr['log_qty']          # ← train on LOG scale
y_tr_raw = df_tr['quantity']

X_val  = df_val[FEATURE_COLS]
y_val  = df_val['log_qty']
y_val_raw = df_val['quantity']

print(f"Train: {len(X_tr):,} | Val: {len(X_val):,}")

def wmape(y_true_raw, y_pred_raw):
    y_true = np.array(y_true_raw)
    y_pred = np.clip(np.array(y_pred_raw), 0, None)
    mask = y_true > 0
    return np.sum(y_true[mask] * np.abs(y_true[mask] - y_pred[mask])) / np.sum(y_true[mask] * y_true[mask]) * np.mean(y_true[mask])

def wmape(y_true_raw, y_pred_raw):
    yt = np.array(y_true_raw)
    yp = np.clip(np.array(y_pred_raw), 0, None)
    mask = yt > 0
    return np.sum(yt[mask] * np.abs(yt[mask] - yp[mask])) / np.sum(yt[mask] * yt[mask] + 1e-8) * np.mean(yt[mask])

def wmape(y_true_raw, y_pred_raw):
    yt = np.array(y_true_raw)
    yp = np.clip(np.array(y_pred_raw), 0, None)
    mask = yt > 0
    weights = yt[mask]
    return np.sum(weights * np.abs(yt[mask] - yp[mask])) / np.sum(weights * yt[mask])

# Naive baseline
naive = np.expm1(df_val['lag_364'].fillna(df_val['item_global_mean']))
print(f"Naive baseline wMAPE: {wmape(y_val_raw, naive):.4f}")

# =============================================================================
# SECTION 4: OPTUNA TUNING (30 trials — fast but effective)
# =============================================================================

print("\nOptuna tuning (30 trials)...")

def objective(trial):
    params = {
        'objective':         'regression_l1',
        'metric':            'mae',
        'verbosity':         -1,
        'random_state':      42,
        'n_estimators':      2000,
        'learning_rate':     trial.suggest_float('lr', 0.02, 0.1, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 63, 511),
        'min_child_samples': trial.suggest_int('min_child', 20, 100),
        'feature_fraction':  trial.suggest_float('ff', 0.6, 1.0),
        'bagging_fraction':  trial.suggest_float('bf', 0.6, 1.0),
        'bagging_freq':      1,
        'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.0, 1.0),
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    # predict on log scale, convert back for wMAPE
    log_preds = m.predict(X_val)
    raw_preds = np.expm1(log_preds).clip(0)
    return wmape(y_val_raw, raw_preds)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

best_params = study.best_params
best_params.update({
    'objective': 'regression_l1',
    'metric': 'mae',
    'verbosity': -1,
    'random_state': 42,
    'n_estimators': 3000,
    'bagging_freq': 1,
})
print(f"Best Optuna wMAPE: {study.best_value:.4f}")

# =============================================================================
# SECTION 5: TRAIN FINAL MODEL
# =============================================================================

print("\nTraining final LightGBM on log scale...")

model = lgb.LGBMRegressor(**best_params)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(200)]
)

# Convert predictions back from log scale
log_val_preds = model.predict(X_val)
val_preds_raw = np.expm1(log_val_preds).clip(0)
print(f"LightGBM (log target) wMAPE: {wmape(y_val_raw, val_preds_raw):.4f}")

# Blend with naive (same week last year) for robustness
naive_val = np.expm1(df_val['lag_364'].fillna(df_val['item_global_mean']))
blend_preds = (0.85 * val_preds_raw + 0.15 * naive_val).clip(0)
print(f"Blend with naive wMAPE:       {wmape(y_val_raw, blend_preds):.4f}")

# Feature importance
feat_imp = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 features:\n{feat_imp.head(15).to_string(index=False)}")

plt.figure(figsize=(10, 10))
sns.barplot(data=feat_imp.head(25), x='importance', y='feature', palette='viridis')
plt.title('Top 25 Features (v3)')
plt.tight_layout()
plt.savefig('outputs/feature_importance_v3.png', dpi=150)
plt.close()

# =============================================================================
# SECTION 6: RETRAIN ON ALL DATA + PREDICT
# =============================================================================

print("\nRetraining on full data...")

train_full = train_fe.dropna(subset=['lag_364']).copy()
X_full = train_full[FEATURE_COLS]
y_full = train_full['log_qty']   # ← log scale

final_model = lgb.LGBMRegressor(**{**best_params, 'n_estimators': model.best_iteration_})
final_model.fit(X_full, y_full)

X_test = test_fe[FEATURE_COLS].copy()
for col in FEATURE_COLS:
    if X_test[col].isnull().any():
        X_test[col] = X_test[col].fillna(X_full[col].mean())

# Predict on log scale → convert back
log_test_preds = final_model.predict(X_test)
raw_test_preds = np.expm1(log_test_preds).clip(0)

# Blend with lag_364 naive
naive_test = np.expm1(test_fe['lag_364'].fillna(test_fe['item_global_mean']))
final_preds = (0.85 * raw_test_preds + 0.15 * naive_test).clip(0)
final_preds = np.round(final_preds).astype(int)

print(f"Predictions — min: {final_preds.min()}, max: {final_preds.max()}, mean: {final_preds.mean():.1f}")

# =============================================================================
# SECTION 7: SUBMISSION
# =============================================================================

test_original = df[df['date'] >= TEST_START].copy()
submission = test_original[['date', 'restaurant_id', 'menu_item_id']].copy()
submission['predicted_quantity'] = final_preds
submission['date'] = submission['date'].dt.strftime('%Y-%m-%d')

assert len(submission) == 69000
submission.to_csv('outputs/submission_v3.csv', index=False)

print("\n" + "=" * 60)
print("✅ ALL DONE!")
print(f"   Val wMAPE (log model):  {wmape(y_val_raw, val_preds_raw):.4f}")
print(f"   Val wMAPE (with blend): {wmape(y_val_raw, blend_preds):.4f}")
print(f"   Submission: outputs/submission_v3.csv")
print("=" * 60)
