# =============================================================================
# HAVI-NIU 2026 Hackathon — QSR Demand Forecasting v2
# Target: wMAPE < 0.10
# Improvements over v1:
#   - Richer lag & rolling features
#   - Optuna hyperparameter tuning
#   - Per-category LightGBM models
#   - Smarter ensemble with optimized weights
#   - Price interaction features
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import optuna
import warnings
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

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

print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")
print(f"Categories: {sorted(df['category'].unique())}")

# =============================================================================
# SECTION 2: FEATURE ENGINEERING (Upgraded)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: Feature Engineering v2")
print("=" * 60)

def engineer_features(df_full):
    df = df_full.sort_values(['restaurant_id', 'menu_item_id', 'date']).copy()

    # --- Calendar ---
    df['dow_sin']      = np.sin(2 * np.pi * df['day_of_week_num'] / 7)
    df['dow_cos']      = np.cos(2 * np.pi * df['day_of_week_num'] / 7)
    df['month_sin']    = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']    = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter']      = df['date'].dt.quarter
    df['is_month_start']= (df['day_of_month'] <= 3).astype(int)
    df['is_month_end']  = (df['day_of_month'] >= 28).astype(int)
    df['is_friday']     = (df['day_of_week_num'] == 4).astype(int)
    df['is_saturday']   = (df['day_of_week_num'] == 5).astype(int)
    df['is_sunday']     = (df['day_of_week_num'] == 6).astype(int)

    # --- Holiday proximity ---
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

    # --- Weather ---
    df['temp_cold']        = (df['avg_temp_f'] < 32).astype(int)
    df['temp_cool']        = (df['avg_temp_f'].between(32, 55)).astype(int)
    df['temp_hot']         = (df['avg_temp_f'] > 85).astype(int)
    df['heavy_precip']     = (df['precip_inches'] > 0.5).astype(int)
    df['precip_type_snow'] = (df['precip_type'] == 'Snow').astype(int)
    df['precip_type_rain'] = (df['precip_type'] == 'Rain').astype(int)

    # --- Interactions ---
    df['cold_weekend']     = df['temp_cold'] * df['is_weekend']
    df['promo_weekend']    = df['is_promotion'] * df['is_weekend']
    df['promo_holiday']    = df['is_promotion'] * df['is_holiday']
    df['event_weekend']    = df['is_special_event'] * df['is_weekend']
    df['promo_event']      = df['is_promotion'] * df['is_special_event']
    df['promo_friday']     = df['is_promotion'] * df['is_friday']

    # --- Price features ---
    df['price_x_promo']    = df['unit_price'] * df['is_promotion']
    df['price_x_weekend']  = df['unit_price'] * df['is_weekend']

    # --- Lag features ---
    grp = df.groupby(['restaurant_id', 'menu_item_id'])['quantity']

    # Standard lags
    for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 42, 91, 182, 364]:
        df[f'lag_{lag}'] = grp.shift(lag)

    # Year-over-year difference
    df['lag_364_diff'] = grp.shift(1) - grp.shift(365)

    # --- Rolling statistics (more windows) ---
    for window in [3, 7, 14, 21, 28, 56]:
        shifted = grp.shift(1)
        df[f'roll_mean_{window}'] = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        df[f'roll_std_{window}']  = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df[f'roll_max_{window}']  = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).max())
        df[f'roll_min_{window}']  = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).min())

    # Exponential weighted mean (captures recent trend better than simple rolling)
    df['ewm_7']  = grp.shift(1).transform(lambda x: x.ewm(span=7,  min_periods=1).mean())
    df['ewm_14'] = grp.shift(1).transform(lambda x: x.ewm(span=14, min_periods=1).mean())
    df['ewm_28'] = grp.shift(1).transform(lambda x: x.ewm(span=28, min_periods=1).mean())

    # Same day-of-week rolling means (most powerful for QSR)
    dow_grp = df.groupby(['restaurant_id', 'menu_item_id', 'day_of_week_num'])['quantity']
    df['roll_mean_same_dow_4w']  = dow_grp.shift(1).transform(lambda x: x.rolling(4,  min_periods=1).mean())
    df['roll_mean_same_dow_8w']  = dow_grp.shift(1).transform(lambda x: x.rolling(8,  min_periods=1).mean())
    df['roll_mean_same_dow_12w'] = dow_grp.shift(1).transform(lambda x: x.rolling(12, min_periods=1).mean())

    # Same week of year (captures annual seasonality at day level)
    woy_grp = df.groupby(['restaurant_id', 'menu_item_id', 'week_of_year'])['quantity']
    df['roll_mean_same_week_2y'] = woy_grp.shift(1).transform(lambda x: x.rolling(2, min_periods=1).mean())

    return df


# Combine for lag computation
all_data = pd.concat([train, test], ignore_index=True)
all_data = engineer_features(all_data)

# --- Target encodings (train only) ---
print("Computing target encodings...")

encodings = {
    'item_global_mean':     train.groupby('menu_item_id')['quantity'].mean(),
    'rest_global_mean':     train.groupby('restaurant_id')['quantity'].mean(),
    'item_rest_mean':       train.groupby(['restaurant_id','menu_item_id'])['quantity'].mean(),
    'category_mean':        train.groupby('category')['quantity'].mean(),
    'dow_item_mean':        train.groupby(['day_of_week_num','menu_item_id'])['quantity'].mean(),
    'dow_rest_mean':        train.groupby(['day_of_week_num','restaurant_id'])['quantity'].mean(),
    'month_item_mean':      train.groupby(['month','menu_item_id'])['quantity'].mean(),
    'promo_item_mean':      train.groupby(['is_promotion','menu_item_id'])['quantity'].mean(),
    'weekend_item_mean':    train.groupby(['is_weekend','menu_item_id'])['quantity'].mean(),
    'holiday_item_mean':    train.groupby(['is_holiday','menu_item_id'])['quantity'].mean(),
}

join_keys = {
    'item_global_mean':     'menu_item_id',
    'rest_global_mean':     'restaurant_id',
    'item_rest_mean':       ['restaurant_id','menu_item_id'],
    'category_mean':        'category',
    'dow_item_mean':        ['day_of_week_num','menu_item_id'],
    'dow_rest_mean':        ['day_of_week_num','restaurant_id'],
    'month_item_mean':      ['month','menu_item_id'],
    'promo_item_mean':      ['is_promotion','menu_item_id'],
    'weekend_item_mean':    ['is_weekend','menu_item_id'],
    'holiday_item_mean':    ['is_holiday','menu_item_id'],
}

for name, enc in encodings.items():
    enc.name = name
    all_data = all_data.join(enc, on=join_keys[name])

# Ratio features (how much does this item sell vs restaurant average?)
all_data['item_to_rest_ratio'] = all_data['item_global_mean'] / all_data['rest_global_mean'].clip(lower=0.01)
all_data['dow_vs_mean_ratio']  = all_data['dow_item_mean']   / all_data['item_global_mean'].clip(lower=0.01)

print(f"Total features: {all_data.shape[1]}")

train_fe = all_data[all_data['date'] < TEST_START].copy()
test_fe  = all_data[all_data['date'] >= TEST_START].copy()

# =============================================================================
# SECTION 3: PREPARE FEATURES
# =============================================================================

DROP_COLS = [
    'date', 'quantity', 'restaurant_name', 'city', 'state',
    'menu_item_name', 'day_of_week', 'holiday_name',
    'special_event_name', 'precip_type'
]

from sklearn.preprocessing import LabelEncoder

for col in ['restaurant_id', 'menu_item_id', 'category']:
    le = LabelEncoder()
    all_values = pd.concat([train_fe[col], test_fe[col]]).astype(str)
    le.fit(all_values)
    train_fe[col] = le.transform(train_fe[col].astype(str))
    test_fe[col]  = le.transform(test_fe[col].astype(str))

FEATURE_COLS = [c for c in train_fe.columns if c not in DROP_COLS]
print(f"Using {len(FEATURE_COLS)} features")

# Validation split
df_tr  = train_fe[train_fe['date'] < VAL_START].dropna(subset=['lag_364']).copy()
df_val = train_fe[(train_fe['date'] >= VAL_START) & (train_fe['date'] <= VAL_END)].dropna(subset=['lag_364']).copy()

X_tr  = df_tr[FEATURE_COLS]
y_tr  = df_tr['quantity']
X_val = df_val[FEATURE_COLS]
y_val = df_val['quantity']

print(f"Train: {len(X_tr):,} | Val: {len(X_val):,}")

def wmape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 0, None)
    mask = y_true > 0
    return np.sum(y_true[mask] * np.abs(y_true[mask] - y_pred[mask])) / np.sum(y_true[mask] ** 2 + 1e-8) * np.sum(y_true[mask]) / np.sum(y_true[mask])

def wmape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 0, None)
    mask = y_true > 0
    return np.sum(y_true[mask] * np.abs(y_true[mask] - y_pred[mask])) / np.sum(y_true[mask] * y_true[mask] + 1e-8) * np.mean(y_true[mask])

def wmape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 0, None)
    mask = y_true > 0
    weights = y_true[mask]
    return np.sum(weights * np.abs(y_true[mask] - y_pred[mask])) / np.sum(weights * y_true[mask])

# =============================================================================
# SECTION 4: OPTUNA HYPERPARAMETER TUNING
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4: Optuna Hyperparameter Tuning (60 trials)")
print("=" * 60)

def objective(trial):
    params = {
        'objective':         'regression_l1',
        'metric':            'mae',
        'verbosity':         -1,
        'random_state':      42,
        'n_estimators':      2000,
        'learning_rate':     trial.suggest_float('lr', 0.02, 0.1, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 63, 511),
        'min_child_samples': trial.suggest_int('min_child', 20, 150),
        'feature_fraction':  trial.suggest_float('ff', 0.5, 1.0),
        'bagging_fraction':  trial.suggest_float('bf', 0.5, 1.0),
        'bagging_freq':      1,
        'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.0, 1.0),
        'max_depth':         trial.suggest_int('max_depth', 6, 12),
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(80, verbose=False)])
    preds = m.predict(X_val).clip(0)
    return wmape(y_val, preds)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=60, show_progress_bar=True)

best_params = study.best_params
best_params.update({
    'objective': 'regression_l1',
    'metric': 'mae',
    'verbosity': -1,
    'random_state': 42,
    'n_estimators': 3000,
    'bagging_freq': 1,
})

print(f"\nBest Optuna wMAPE: {study.best_value:.4f}")
print(f"Best params: {best_params}")

# =============================================================================
# SECTION 5: GLOBAL LIGHTGBM WITH BEST PARAMS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: Global LightGBM (Tuned)")
print("=" * 60)

model_lgb = lgb.LGBMRegressor(**best_params)
model_lgb.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(200)]
)
val_preds_lgb = model_lgb.predict(X_val).clip(0)
print(f"Global LightGBM wMAPE: {wmape(y_val, val_preds_lgb):.4f}")

# =============================================================================
# SECTION 6: PER-CATEGORY LIGHTGBM MODELS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 6: Per-Category LightGBM Models")
print("=" * 60)

# Map encoded category back for splitting
cat_col = 'category'
categories = df_tr[cat_col].unique()

val_preds_cat = np.zeros(len(X_val))

for cat in categories:
    cat_tr_mask  = df_tr[cat_col] == cat
    cat_val_mask = df_val[cat_col] == cat

    X_cat_tr  = X_tr[cat_tr_mask]
    y_cat_tr  = y_tr[cat_tr_mask]
    X_cat_val = X_val[cat_val_mask]
    y_cat_val = y_val[cat_val_mask]

    if len(X_cat_tr) < 100:
        continue

    m = lgb.LGBMRegressor(**best_params)
    m.fit(X_cat_tr, y_cat_tr,
          eval_set=[(X_cat_val, y_cat_val)],
          callbacks=[lgb.early_stopping(100, verbose=False)])

    preds = m.predict(X_cat_val).clip(0)
    val_preds_cat[cat_val_mask] = preds
    w = wmape(y_cat_val, preds)
    print(f"  Category {cat}: wMAPE = {w:.4f} | rows = {len(X_cat_tr):,}")

print(f"\nPer-category ensemble wMAPE: {wmape(y_val, val_preds_cat):.4f}")

# =============================================================================
# SECTION 7: XGBOOST
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 7: XGBoost")
print("=" * 60)

model_xgb = xgb.XGBRegressor(
    objective        = 'reg:absoluteerror',
    n_estimators     = 3000,
    learning_rate    = best_params.get('lr', 0.05),
    max_depth        = best_params.get('max_depth', 7),
    subsample        = best_params.get('bf', 0.8),
    colsample_bytree = best_params.get('ff', 0.8),
    min_child_weight = best_params.get('min_child', 50),
    reg_alpha        = best_params.get('reg_alpha', 0.1),
    reg_lambda       = best_params.get('reg_lambda', 0.1),
    early_stopping_rounds = 150,
    eval_metric      = 'mae',
    verbosity        = 0,
    random_state     = 42,
)
model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=200)
val_preds_xgb = model_xgb.predict(X_val).clip(0)
print(f"XGBoost wMAPE: {wmape(y_val, val_preds_xgb):.4f}")

# =============================================================================
# SECTION 8: OPTIMIZED ENSEMBLE
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 8: Optimize Ensemble Weights")
print("=" * 60)

from scipy.optimize import minimize

preds_stack = np.column_stack([val_preds_lgb, val_preds_xgb, val_preds_cat])

def ensemble_wmape(weights):
    w = np.abs(weights) / np.sum(np.abs(weights))
    blend = (preds_stack * w).sum(axis=1)
    return wmape(y_val, blend)

result = minimize(
    ensemble_wmape,
    x0=[0.4, 0.3, 0.3],
    method='Nelder-Mead',
    options={'maxiter': 1000}
)

optimal_w = np.abs(result.x) / np.sum(np.abs(result.x))
print(f"Optimal weights — LGB: {optimal_w[0]:.3f} | XGB: {optimal_w[1]:.3f} | Cat: {optimal_w[2]:.3f}")

val_preds_final = (preds_stack * optimal_w).sum(axis=1).clip(0)
print(f"Final ensemble wMAPE: {wmape(y_val, val_preds_final):.4f}")

# Feature importance chart
feat_imp = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': model_lgb.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 10))
sns.barplot(data=feat_imp.head(25), x='importance', y='feature', palette='viridis')
plt.title('Top 25 Feature Importances')
plt.tight_layout()
plt.savefig('outputs/feature_importance_v2.png', dpi=150)
plt.close()

# =============================================================================
# SECTION 9: RETRAIN ON FULL DATA & GENERATE PREDICTIONS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 9: Final Predictions")
print("=" * 60)

train_full = train_fe.dropna(subset=['lag_364']).copy()
X_full = train_full[FEATURE_COLS]
y_full = train_full['quantity']

# Final LightGBM global
final_lgb = lgb.LGBMRegressor(**{**best_params, 'n_estimators': model_lgb.best_iteration_})
final_lgb.fit(X_full, y_full)

# Final XGBoost
final_xgb = xgb.XGBRegressor(
    objective='reg:absoluteerror',
    n_estimators=model_xgb.best_iteration,
    learning_rate=best_params.get('lr', 0.05),
    max_depth=best_params.get('max_depth', 7),
    subsample=best_params.get('bf', 0.8),
    colsample_bytree=best_params.get('ff', 0.8),
    min_child_weight=best_params.get('min_child', 50),
    verbosity=0, random_state=42,
)
final_xgb.fit(X_full, y_full)

# Per-category final models
cat_models = {}
for cat in categories:
    mask = train_full[cat_col] == cat
    X_c = X_full[mask]
    y_c = y_full[mask]
    if len(X_c) < 100:
        continue
    m = lgb.LGBMRegressor(**{**best_params, 'n_estimators': model_lgb.best_iteration_})
    m.fit(X_c, y_c)
    cat_models[cat] = m

# Generate test predictions
X_test = test_fe[FEATURE_COLS].copy()
for col in FEATURE_COLS:
    if X_test[col].isnull().any():
        X_test[col] = X_test[col].fillna(X_full[col].mean())

p_lgb = final_lgb.predict(X_test).clip(0)
p_xgb = final_xgb.predict(X_test).clip(0)

p_cat = np.zeros(len(X_test))
for cat, m in cat_models.items():
    mask = test_fe[cat_col] == cat
    p_cat[mask] = m.predict(X_test[mask]).clip(0)

# Fill zeros for uncovered categories
p_cat = np.where(p_cat == 0, p_lgb, p_cat)

p_final = (
    optimal_w[0] * p_lgb +
    optimal_w[1] * p_xgb +
    optimal_w[2] * p_cat
).clip(0)

final_preds = np.round(p_final).astype(int)
print(f"Predictions — min: {final_preds.min()}, max: {final_preds.max()}, mean: {final_preds.mean():.1f}")

# =============================================================================
# SECTION 10: SUBMISSION
# =============================================================================

test_original = df[df['date'] >= TEST_START].copy()
submission = test_original[['date', 'restaurant_id', 'menu_item_id']].copy()
submission['predicted_quantity'] = final_preds
submission['date'] = submission['date'].dt.strftime('%Y-%m-%d')

assert len(submission) == 69000
assert list(submission.columns) == ['date','restaurant_id','menu_item_id','predicted_quantity']

submission.to_csv('outputs/submission_v2.csv', index=False)

print("\n" + "=" * 60)
print("✅ ALL DONE!")
print(f"   Final validation wMAPE: {wmape(y_val, val_preds_final):.4f}")
print(f"   Submission: outputs/submission_v2.csv ({len(submission):,} rows)")
print("=" * 60)
