# =============================================================================
# HAVI-NIU 2026 Hackathon — QSR Demand Forecasting — ULTIMATE VERSION
# Uses: LightGBM + XGBoost + CatBoost + Ridge + MLP + Optuna + Stacking
# Key: log1p target transform + Tweedie + same-DOW features + meta-learner
# Target: Beat 0.1312 wMAPE
# Runtime: ~30 minutes
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
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# Try importing CatBoost (install if missing)
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
    print("CatBoost available ✓")
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not installed — skipping (run: pip install catboost)")

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================

print("\n" + "=" * 60)
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
print(f"Categories: {sorted(df['category'].unique())}")

# =============================================================================
# SECTION 2: FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: Feature Engineering")
print("=" * 60)

def build_features(df_full, train_ref):
    df = df_full.sort_values(['restaurant_id', 'menu_item_id', 'date']).copy()

    # --- Calendar ---
    df['dow_sin']        = np.sin(2 * np.pi * df['day_of_week_num'] / 7)
    df['dow_cos']        = np.cos(2 * np.pi * df['day_of_week_num'] / 7)
    df['month_sin']      = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']      = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_month']   = df['date'].dt.day
    df['week_of_year']   = df['date'].dt.isocalendar().week.astype(int)
    df['quarter']        = df['date'].dt.quarter
    df['is_friday']      = (df['day_of_week_num'] == 4).astype(int)
    df['is_saturday']    = (df['day_of_week_num'] == 5).astype(int)
    df['is_sunday']      = (df['day_of_week_num'] == 6).astype(int)
    df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
    df['is_month_end']   = (df['day_of_month'] >= 25).astype(int)

    # --- Holiday proximity ---
    holiday_dates = pd.to_datetime(df[df['is_holiday'] == 1]['date'].unique())
    def days_to_next(d):
        future = holiday_dates[holiday_dates > d]
        return int((future.min() - d).days) if len(future) > 0 else 99
    def days_from_last(d):
        past = holiday_dates[holiday_dates < d]
        return int((d - past.max()).days) if len(past) > 0 else 99
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
    df['cold_weekend']     = df['temp_cold'] * df['is_weekend']

    # --- Interactions ---
    df['promo_weekend']   = df['is_promotion'] * df['is_weekend']
    df['promo_holiday']   = df['is_promotion'] * df['is_holiday']
    df['promo_event']     = df['is_promotion'] * df['is_special_event']
    df['event_weekend']   = df['is_special_event'] * df['is_weekend']
    df['promo_friday']    = df['is_promotion'] * df['is_friday']
    df['price_x_promo']   = df['unit_price'] * df['is_promotion']
    df['price_x_weekend'] = df['unit_price'] * df['is_weekend']

    # --- LOG1P transform (key!) ---
    df['log_qty'] = np.log1p(df['quantity'])
    grp     = df.groupby(['restaurant_id', 'menu_item_id'])['log_qty']
    qty_grp = df.groupby(['restaurant_id', 'menu_item_id'])['quantity']

    # Lags on log scale
    for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 42, 91, 182, 364]:
        df[f'lag_{lag}'] = grp.shift(lag)

    # Raw lags for Tweedie/CatBoost
    for lag in [7, 14, 28, 91, 364]:
        df[f'raw_lag_{lag}'] = qty_grp.shift(lag)

    # Rolling stats on log scale
    for window in [3, 7, 14, 28, 56]:
        s = grp.shift(1)
        df[f'roll_mean_{window}'] = s.transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'roll_std_{window}']  = s.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df[f'roll_max_{window}']  = s.transform(lambda x: x.rolling(window, min_periods=1).max())
        df[f'roll_min_{window}']  = s.transform(lambda x: x.rolling(window, min_periods=1).min())

    # EWM (captures recent trend)
    for span in [3, 7, 14, 28, 56]:
        df[f'ewm_{span}'] = grp.shift(1).transform(lambda x: x.ewm(span=span, min_periods=1).mean())

    # Same DOW rolling — MOST POWERFUL for QSR
    dow_grp = df.groupby(['restaurant_id', 'menu_item_id', 'day_of_week_num'])['log_qty']
    for weeks in [4, 8, 12, 16, 24]:
        df[f'roll_same_dow_{weeks}w'] = dow_grp.shift(1).transform(
            lambda x: x.rolling(weeks, min_periods=1).mean())
    df['ewm_same_dow_8']  = dow_grp.shift(1).transform(lambda x: x.ewm(span=8,  min_periods=1).mean())
    df['ewm_same_dow_16'] = dow_grp.shift(1).transform(lambda x: x.ewm(span=16, min_periods=1).mean())
    df['std_same_dow_8']  = dow_grp.shift(1).transform(lambda x: x.rolling(8, min_periods=1).std().fillna(0))

    # Same week-of-year
    woy_grp = df.groupby(['restaurant_id', 'menu_item_id', 'week_of_year'])['log_qty']
    df['roll_same_week_2y'] = woy_grp.shift(1).transform(lambda x: x.rolling(2, min_periods=1).mean())
    df['roll_same_week_3y'] = woy_grp.shift(1).transform(lambda x: x.rolling(3, min_periods=1).mean())

    # Trend features
    df['trend_7_28']  = (df['ewm_7']  + 1e-3) / (df['ewm_28']  + 1e-3)
    df['trend_14_56'] = (df['ewm_14'] + 1e-3) / (df['ewm_56']  + 1e-3)
    df['trend_7_28']  = df['trend_7_28'].clip(0.1, 10)
    df['trend_14_56'] = df['trend_14_56'].clip(0.1, 10)

    # --- Target encodings (log scale from train_ref) ---
    tmp = train_ref.copy()
    tmp['log_qty']     = np.log1p(tmp['quantity'])
    tmp['quarter']     = tmp['date'].dt.quarter
    tmp['week_of_year']= tmp['date'].dt.isocalendar().week.astype(int)

    encs = {
        'enc_item':         (tmp.groupby('menu_item_id')['log_qty'].mean(),                       'menu_item_id'),
        'enc_rest':         (tmp.groupby('restaurant_id')['log_qty'].mean(),                      'restaurant_id'),
        'enc_item_rest':    (tmp.groupby(['restaurant_id','menu_item_id'])['log_qty'].mean(),      ['restaurant_id','menu_item_id']),
        'enc_category':     (tmp.groupby('category')['log_qty'].mean(),                           'category'),
        'enc_dow_item':     (tmp.groupby(['day_of_week_num','menu_item_id'])['log_qty'].mean(),    ['day_of_week_num','menu_item_id']),
        'enc_dow_rest':     (tmp.groupby(['day_of_week_num','restaurant_id'])['log_qty'].mean(),   ['day_of_week_num','restaurant_id']),
        'enc_month_item':   (tmp.groupby(['month','menu_item_id'])['log_qty'].mean(),              ['month','menu_item_id']),
        'enc_promo_item':   (tmp.groupby(['is_promotion','menu_item_id'])['log_qty'].mean(),       ['is_promotion','menu_item_id']),
        'enc_weekend_item': (tmp.groupby(['is_weekend','menu_item_id'])['log_qty'].mean(),         ['is_weekend','menu_item_id']),
        'enc_holiday_item': (tmp.groupby(['is_holiday','menu_item_id'])['log_qty'].mean(),         ['is_holiday','menu_item_id']),
        'enc_quarter_item': (tmp.groupby(['quarter','menu_item_id'])['log_qty'].mean(),            ['quarter','menu_item_id']),
        'enc_week_item':    (tmp.groupby(['week_of_year','menu_item_id'])['log_qty'].mean(),       ['week_of_year','menu_item_id']),
        'enc_event_item':   (tmp.groupby(['is_special_event','menu_item_id'])['log_qty'].mean(),   ['is_special_event','menu_item_id']),
    }
    for name, (enc, key) in encs.items():
        enc.name = name
        df = df.join(enc, on=key)

    df['item_to_rest_ratio']  = df['enc_item']      / (df['enc_rest']       + 1e-3)
    df['dow_vs_item_ratio']   = df['enc_dow_item']  / (df['enc_item']       + 1e-3)
    df['month_vs_item_ratio'] = df['enc_month_item']/ (df['enc_item']       + 1e-3)
    df['week_vs_item_ratio']  = df['enc_week_item'] / (df['enc_item']       + 1e-3)

    return df


print("Engineering features (3–5 min)...")
all_data = pd.concat([train, test], ignore_index=True)
all_data = build_features(all_data, train)

train_fe = all_data[all_data['date'] < TEST_START].copy()
test_fe  = all_data[all_data['date'] >= TEST_START].copy()

for col in ['restaurant_id', 'menu_item_id', 'category']:
    le = LabelEncoder()
    all_vals = pd.concat([train_fe[col], test_fe[col]]).astype(str)
    le.fit(all_vals)
    train_fe[col] = le.transform(train_fe[col].astype(str))
    test_fe[col]  = le.transform(test_fe[col].astype(str))

DROP_COLS = ['date','quantity','log_qty','restaurant_name','city','state',
             'menu_item_name','day_of_week','holiday_name','special_event_name','precip_type']
FEATURE_COLS = [c for c in train_fe.columns if c not in DROP_COLS]
print(f"Total features: {len(FEATURE_COLS)}")

# =============================================================================
# SECTION 3: SPLIT
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: Train/Val Split")
print("=" * 60)

df_tr  = train_fe[train_fe['date'] < VAL_START].dropna(subset=['lag_364']).copy()
df_val = train_fe[(train_fe['date'] >= VAL_START) & (train_fe['date'] <= VAL_END)].dropna(subset=['lag_364']).copy()

X_tr      = df_tr[FEATURE_COLS]
y_tr_log  = df_tr['log_qty']
y_tr_raw  = df_tr['quantity']
X_val     = df_val[FEATURE_COLS]
y_val_log = df_val['log_qty']
y_val_raw = df_val['quantity']

print(f"Train: {len(X_tr):,} | Val: {len(X_val):,}")

def wmape(y_true, y_pred):
    yt = np.array(y_true)
    yp = np.clip(np.array(y_pred), 0, None)
    mask = yt > 0
    w = yt[mask]
    return np.sum(w * np.abs(yt[mask] - yp[mask])) / np.sum(w * yt[mask])

naive_val = np.expm1(df_val['lag_364'].fillna(df_val['enc_item']))
print(f"Naive baseline wMAPE: {wmape(y_val_raw, naive_val):.4f}")

# =============================================================================
# SECTION 4: MODEL A — LightGBM log-MAE (Optuna, 40 trials)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4: Model A — LightGBM log-MAE")
print("=" * 60)

def obj_lgb_log(trial):
    p = {
        'objective':'regression_l1','metric':'mae','verbosity':-1,
        'random_state':42,'n_estimators':2000,'bagging_freq':1,
        'learning_rate':     trial.suggest_float('lr', 0.02, 0.1, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 63, 255),
        'min_child_samples': trial.suggest_int('min_child', 20, 100),
        'feature_fraction':  trial.suggest_float('ff', 0.6, 1.0),
        'bagging_fraction':  trial.suggest_float('bf', 0.6, 1.0),
        'reg_alpha':         trial.suggest_float('ra', 0.0, 0.5),
        'reg_lambda':        trial.suggest_float('rl', 0.0, 0.5),
        'max_depth':         trial.suggest_int('depth', 6, 10),
    }
    m = lgb.LGBMRegressor(**p)
    m.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return wmape(y_val_raw, np.expm1(m.predict(X_val)).clip(0))

study_a = optuna.create_study(direction='minimize')
study_a.optimize(obj_lgb_log, n_trials=40, show_progress_bar=True)
best_a = {**study_a.best_params,
          'objective':'regression_l1','metric':'mae','verbosity':-1,
          'random_state':42,'n_estimators':3000,'bagging_freq':1}
best_a['learning_rate'] = best_a.pop('lr', best_a.get('learning_rate', 0.05))
best_a['feature_fraction'] = best_a.pop('ff', best_a.get('feature_fraction', 0.8))
best_a['bagging_fraction'] = best_a.pop('bf', best_a.get('bagging_fraction', 0.8))
best_a['min_child_samples'] = best_a.pop('min_child', best_a.get('min_child_samples', 50))
best_a['reg_alpha'] = best_a.pop('ra', best_a.get('reg_alpha', 0.1))
best_a['reg_lambda'] = best_a.pop('rl', best_a.get('reg_lambda', 0.1))
best_a['max_depth'] = best_a.pop('depth', best_a.get('max_depth', 7))

model_a = lgb.LGBMRegressor(**best_a)
model_a.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)],
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(500)])
preds_a = np.expm1(model_a.predict(X_val)).clip(0)
print(f"Model A (LGB log-MAE) wMAPE: {wmape(y_val_raw, preds_a):.4f}")

# =============================================================================
# SECTION 5: MODEL B — LightGBM Tweedie
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: Model B — LightGBM Tweedie")
print("=" * 60)

def obj_lgb_tweedie(trial):
    p = {
        'objective':'tweedie','metric':'tweedie','verbosity':-1,
        'random_state':42,'n_estimators':2000,'bagging_freq':1,
        'tweedie_variance_power': trial.suggest_float('tp', 1.0, 1.9),
        'learning_rate':     trial.suggest_float('lr', 0.02, 0.1, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 63, 255),
        'min_child_samples': trial.suggest_int('min_child', 20, 100),
        'feature_fraction':  trial.suggest_float('ff', 0.6, 1.0),
        'bagging_fraction':  trial.suggest_float('bf', 0.6, 1.0),
        'reg_alpha':         trial.suggest_float('ra', 0.0, 0.5),
        'reg_lambda':        trial.suggest_float('rl', 0.0, 0.5),
    }
    m = lgb.LGBMRegressor(**p)
    m.fit(X_tr, y_tr_raw, eval_set=[(X_val, y_val_raw)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return wmape(y_val_raw, m.predict(X_val).clip(0))

study_b = optuna.create_study(direction='minimize')
study_b.optimize(obj_lgb_tweedie, n_trials=20, show_progress_bar=True)
best_b = {**study_b.best_params,
          'objective':'tweedie','metric':'tweedie','verbosity':-1,
          'random_state':42,'n_estimators':3000,'bagging_freq':1}
best_b['learning_rate'] = best_b.pop('lr', 0.05)
best_b['feature_fraction'] = best_b.pop('ff', 0.8)
best_b['bagging_fraction'] = best_b.pop('bf', 0.8)
best_b['min_child_samples'] = best_b.pop('min_child', 50)
best_b['reg_alpha'] = best_b.pop('ra', 0.1)
best_b['reg_lambda'] = best_b.pop('rl', 0.1)
best_b['tweedie_variance_power'] = best_b.pop('tp', 1.5)

model_b = lgb.LGBMRegressor(**best_b)
model_b.fit(X_tr, y_tr_raw, eval_set=[(X_val, y_val_raw)],
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(500)])
preds_b = model_b.predict(X_val).clip(0)
print(f"Model B (LGB Tweedie) wMAPE: {wmape(y_val_raw, preds_b):.4f}")

# =============================================================================
# SECTION 6: MODEL C — XGBoost log-MAE
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 6: Model C — XGBoost log-MAE")
print("=" * 60)

model_c = xgb.XGBRegressor(
    objective='reg:absoluteerror',
    n_estimators=3000, learning_rate=0.05,
    max_depth=7, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=50, reg_alpha=0.1, reg_lambda=0.1,
    early_stopping_rounds=100, eval_metric='mae',
    verbosity=0, random_state=42,
)
model_c.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], verbose=500)
preds_c = np.expm1(model_c.predict(X_val)).clip(0)
print(f"Model C (XGB log-MAE) wMAPE: {wmape(y_val_raw, preds_c):.4f}")

# =============================================================================
# SECTION 7: MODEL D — CatBoost (if available)
# =============================================================================

if HAS_CATBOOST:
    print("\n" + "=" * 60)
    print("SECTION 7: Model D — CatBoost")
    print("=" * 60)
    model_d = CatBoostRegressor(
        loss_function='MAE',
        iterations=2000, learning_rate=0.05,
        depth=8, l2_leaf_reg=3,
        random_seed=42, verbose=500,
        early_stopping_rounds=100,
    )
    model_d.fit(X_tr, y_tr_log, eval_set=(X_val, y_val_log))
    preds_d = np.expm1(model_d.predict(X_val)).clip(0)
    print(f"Model D (CatBoost) wMAPE: {wmape(y_val_raw, preds_d):.4f}")
else:
    preds_d = preds_a.copy()  # fallback to model A

# =============================================================================
# SECTION 8: MODEL E — Extra Trees (good diversity for ensemble)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 8: Model E — Extra Trees (log scale)")
print("=" * 60)

# Use subset of most important features for speed
top_feats = ['lag_364','lag_7','lag_14','lag_28','roll_same_dow_4w',
             'roll_same_dow_8w','ewm_same_dow_8','enc_item_rest',
             'enc_dow_item','roll_mean_7','roll_mean_28','ewm_7','ewm_28',
             'day_of_week_num','is_weekend','month','is_promotion',
             'is_holiday','avg_temp_f','trend_7_28']
top_feats = [f for f in top_feats if f in FEATURE_COLS]

model_e = ExtraTreesRegressor(
    n_estimators=300, max_depth=12, min_samples_leaf=10,
    max_features=0.7, n_jobs=-1, random_state=42
)
model_e.fit(X_tr[top_feats].fillna(0), y_tr_log)
preds_e = np.expm1(model_e.predict(X_val[top_feats].fillna(0))).clip(0)
print(f"Model E (ExtraTrees) wMAPE: {wmape(y_val_raw, preds_e):.4f}")

# =============================================================================
# SECTION 9: MODEL F — Ridge on log scale (linear baseline)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 9: Model F — Ridge Regression (log scale)")
print("=" * 60)

scaler = StandardScaler()
X_tr_scaled  = scaler.fit_transform(X_tr[top_feats].fillna(0))
X_val_scaled = scaler.transform(X_val[top_feats].fillna(0))

model_f = Ridge(alpha=10.0)
model_f.fit(X_tr_scaled, y_tr_log)
preds_f = np.expm1(model_f.predict(X_val_scaled)).clip(0)
print(f"Model F (Ridge) wMAPE: {wmape(y_val_raw, preds_f):.4f}")

# =============================================================================
# SECTION 10: OPTIMIZED ENSEMBLE
# Scipy finds best blend weights across all 6 models + naive
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 10: Optimized Ensemble (all models + naive)")
print("=" * 60)

stack = np.column_stack([preds_a, preds_b, preds_c, preds_d, preds_e, preds_f, naive_val])
model_names = ['LGB-LogMAE','LGB-Tweedie','XGB-LogMAE','CatBoost','ExtraTrees','Ridge','Naive']

# Print individual wMAPEs
for i, name in enumerate(model_names):
    print(f"  {name}: {wmape(y_val_raw, stack[:,i]):.4f}")

def ens_wmape(w):
    w = np.abs(w) / (np.sum(np.abs(w)) + 1e-9)
    return wmape(y_val_raw, (stack * w).sum(axis=1))

# Try multiple starting points
best_res = None
for x0 in [[0.35,0.25,0.2,0.1,0.05,0.02,0.03],
           [0.5,0.3,0.1,0.05,0.02,0.01,0.02],
           [0.3,0.3,0.2,0.1,0.05,0.02,0.03]]:
    res = minimize(ens_wmape, x0=x0, method='Nelder-Mead',
                   options={'maxiter':5000,'xatol':1e-7})
    if best_res is None or res.fun < best_res.fun:
        best_res = res

W = np.abs(best_res.x) / np.sum(np.abs(best_res.x))
val_blend = (stack * W).sum(axis=1).clip(0)

print(f"\nOptimal weights:")
for name, w in zip(model_names, W):
    print(f"  {name}: {w:.3f}")
print(f"\n{'='*40}")
print(f"FINAL ENSEMBLE wMAPE: {wmape(y_val_raw, val_blend):.4f}")
print(f"{'='*40}")

# Feature importance chart
feat_imp = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': model_a.feature_importances_
}).sort_values('importance', ascending=False)
plt.figure(figsize=(10, 10))
sns.barplot(data=feat_imp.head(25), x='importance', y='feature', palette='viridis')
plt.title('Top 25 Features')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150)
plt.close()

# =============================================================================
# SECTION 11: RETRAIN ON ALL DATA + FINAL PREDICTIONS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 11: Retrain on Full Data")
print("=" * 60)

train_full = train_fe.dropna(subset=['lag_364']).copy()
X_full     = train_full[FEATURE_COLS]
y_full_log = train_full['log_qty']
y_full_raw = train_full['quantity']
print(f"Full train rows: {len(X_full):,}")

# Retrain each model on full data
final_a = lgb.LGBMRegressor(**{**best_a, 'n_estimators': model_a.best_iteration_})
final_a.fit(X_full, y_full_log)

final_b = lgb.LGBMRegressor(**{**best_b, 'n_estimators': model_b.best_iteration_})
final_b.fit(X_full, y_full_raw)

final_c = xgb.XGBRegressor(
    objective='reg:absoluteerror',
    n_estimators=model_c.best_iteration, learning_rate=0.05,
    max_depth=7, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=50, verbosity=0, random_state=42,
)
final_c.fit(X_full, y_full_log)

if HAS_CATBOOST:
    final_d = CatBoostRegressor(
        loss_function='MAE',
        iterations=model_d.get_best_iteration(), learning_rate=0.05,
        depth=8, l2_leaf_reg=3, random_seed=42, verbose=0
    )
    final_d.fit(X_full, y_full_log)

final_e = ExtraTreesRegressor(
    n_estimators=300, max_depth=12, min_samples_leaf=10,
    max_features=0.7, n_jobs=-1, random_state=42
)
final_e.fit(X_full[top_feats].fillna(0), y_full_log)

X_full_scaled = scaler.fit_transform(X_full[top_feats].fillna(0))
final_f = Ridge(alpha=10.0)
final_f.fit(X_full_scaled, y_full_log)

# Test predictions
X_test = test_fe[FEATURE_COLS].copy()
for col in FEATURE_COLS:
    if X_test[col].isnull().any():
        X_test[col] = X_test[col].fillna(X_full[col].mean())

p_a = np.expm1(final_a.predict(X_test)).clip(0)
p_b = final_b.predict(X_test).clip(0)
p_c = np.expm1(final_c.predict(X_test)).clip(0)
p_d = np.expm1(final_d.predict(X_test)).clip(0) if HAS_CATBOOST else p_a
p_e = np.expm1(final_e.predict(X_test[top_feats].fillna(0))).clip(0)
p_f = np.expm1(final_f.predict(scaler.transform(X_test[top_feats].fillna(0)))).clip(0)
naive_test = np.expm1(test_fe['lag_364'].fillna(test_fe['enc_item']))

test_stack = np.column_stack([p_a, p_b, p_c, p_d, p_e, p_f, naive_test])
final_preds = (test_stack * W).sum(axis=1).clip(0)
final_preds = np.round(final_preds).astype(int)

print(f"Predictions — min: {final_preds.min()}, max: {final_preds.max()}, mean: {final_preds.mean():.1f}")

# =============================================================================
# SECTION 12: SUBMISSION
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 12: Submission")
print("=" * 60)

test_original = df[df['date'] >= TEST_START].copy()
submission = test_original[['date','restaurant_id','menu_item_id']].copy()
submission['predicted_quantity'] = final_preds
submission['date'] = submission['date'].dt.strftime('%Y-%m-%d')

assert len(submission) == 69000
assert list(submission.columns) == ['date','restaurant_id','menu_item_id','predicted_quantity']
assert submission['predicted_quantity'].isnull().sum() == 0

submission.to_csv('outputs/submission.csv', index=False)

# EDA charts
train_eda = df[df['date'] < TEST_START].dropna(subset=['quantity'])
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('HAVI-NIU QSR — EDA', fontsize=16, fontweight='bold')
dow_avg = train_eda.groupby('day_of_week_num')['quantity'].mean()
axes[0,0].bar(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], dow_avg.sort_index(), color='steelblue')
axes[0,0].set_title('Avg Sales by Day of Week')
month_avg = train_eda.groupby('month')['quantity'].mean()
axes[0,1].bar(range(1,13), month_avg.values, color='coral')
axes[0,1].set_title('Avg Sales by Month')
axes[0,1].set_xticks(range(1,13))
axes[0,1].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
hol_avg = train_eda.groupby('is_holiday')['quantity'].mean()
axes[0,2].bar(['Non-Holiday','Holiday'], hol_avg.values, color=['steelblue','orange'])
axes[0,2].set_title('Holiday vs Non-Holiday Sales')
promo_avg = train_eda.groupby('is_promotion')['quantity'].mean()
axes[1,0].bar(['No Promo','Promo'], promo_avg.values, color=['steelblue','green'])
axes[1,0].set_title(f'Promo Lift: {(promo_avg[1]/promo_avg[0]-1)*100:.1f}%')
top_items = train_eda.groupby('menu_item_name')['quantity'].mean().nlargest(10)
axes[1,1].barh(top_items.index, top_items.values, color='purple')
axes[1,1].set_title('Top 10 Menu Items'); axes[1,1].invert_yaxis()
monthly = train_eda.groupby(train_eda['date'].dt.to_period('M'))['quantity'].mean()
axes[1,2].plot(range(len(monthly)), monthly.values, color='steelblue')
axes[1,2].set_title('Monthly Sales Trend')
plt.tight_layout()
plt.savefig('outputs/eda_charts.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSample predictions:\n{submission.head(10).to_string(index=False)}")
print("\n" + "=" * 60)
print("✅ ALL DONE!")
print(f"   Model A  LGB log-MAE:   {wmape(y_val_raw, preds_a):.4f}")
print(f"   Model B  LGB Tweedie:   {wmape(y_val_raw, preds_b):.4f}")
print(f"   Model C  XGB log-MAE:   {wmape(y_val_raw, preds_c):.4f}")
print(f"   Model D  CatBoost:      {wmape(y_val_raw, preds_d):.4f}")
print(f"   Model E  ExtraTrees:    {wmape(y_val_raw, preds_e):.4f}")
print(f"   Model F  Ridge:         {wmape(y_val_raw, preds_f):.4f}")
print(f"   Naive baseline:         {wmape(y_val_raw, naive_val):.4f}")
print(f"   ─────────────────────────────")
print(f"   FINAL ENSEMBLE wMAPE:   {wmape(y_val_raw, val_blend):.4f}")
print(f"   Submission: outputs/submission.csv ({len(submission):,} rows)")
print("=" * 60)
