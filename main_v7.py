import pandas as pd
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')
T0 = time.time()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)

raw = pd.read_csv('data/raw/qsr_demand_dataset.csv')
raw['date'] = pd.to_datetime(raw['date'])
raw = raw.sort_values(['restaurant_id', 'menu_item_id', 'date']).reset_index(drop=True)

tmask = raw['date'] < '2025-10-01'
raw.loc[tmask, 'quantity'] = raw.loc[tmask].groupby(
    ['restaurant_id', 'menu_item_id']
)['quantity'].transform(lambda x: x.ffill().bfill().fillna(0))

test_actual = raw.loc[~tmask, ['date', 'restaurant_id', 'menu_item_id', 'quantity']].copy()
train_raw   = raw[tmask].copy()

print(f"Train: {len(train_raw):,} | Test: {(~tmask).sum():,}")

# Quantity store (raw) for iterative prediction
QTY_STORE = {}
for _, row in train_raw[['date', 'restaurant_id', 'menu_item_id', 'quantity']].iterrows():
    QTY_STORE[(row['date'], row['restaurant_id'], row['menu_item_id'])] = row['quantity']

print(f"  QTY_STORE: {len(QTY_STORE):,} entries")


# ============================================================================
# STEP 2: STATIC FEATURES (same for both models)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: Static features...")
print("=" * 60)

def add_static_features(df, train_ref):
    df = df.copy()

    # Calendar
    df['day_of_year']  = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['day_of_month'] = df['date'].dt.day
    df['quarter']      = df['date'].dt.quarter
    df['is_month_start']= (df['date'].dt.day <= 3).astype(int)
    df['is_month_end']  = (df['date'].dt.day >= 28).astype(int)
    df['is_friday']    = (df['day_of_week_num'] == 4).astype(int)
    df['is_saturday']  = (df['day_of_week_num'] == 5).astype(int)
    df['is_sunday']    = (df['day_of_week_num'] == 6).astype(int)

    for col, p in [('day_of_year',365),('day_of_week_num',7),('month',12),('week_of_year',52)]:
        df[f'sin_{col}'] = np.sin(2*np.pi*df[col]/p)
        df[f'cos_{col}'] = np.cos(2*np.pi*df[col]/p)

    # Weather
    df['temp_bin']     = pd.cut(df['avg_temp_f'], bins=[-np.inf,20,35,50,70,np.inf],
                                labels=[0,1,2,3,4]).astype(float)
    df['is_rain']      = (df['precip_type']=='Rain').astype(int)
    df['is_snow']      = (df['precip_type']=='Snow').astype(int)
    df['heavy_precip'] = (df['precip_inches']>0.5).astype(int)
    df['temp_x_precip']= df['avg_temp_f'] * df['precip_inches']
    df['temp_squared'] = df['avg_temp_f']**2
    df['temp_cold']    = (df['avg_temp_f'] < 32).astype(int)
    df['temp_hot']     = (df['avg_temp_f'] > 85).astype(int)
    df['cold_weekend'] = df['temp_cold'] * df['is_weekend']

    # Holiday proximity
    hdates = np.sort(df[df['is_holiday']==1]['date'].unique())
    d2f, d2b = {}, {}
    for d in df['date'].unique():
        fwd = hdates[hdates > d]
        bck = hdates[hdates < d]
        d2f[d] = int((fwd[0]-d)/np.timedelta64(1,'D')) if len(fwd) else 30
        d2b[d] = int((d-bck[-1])/np.timedelta64(1,'D')) if len(bck) else 30
    df['days_to_holiday']   = df['date'].map(d2f)
    df['days_since_holiday']= df['date'].map(d2b)
    df['near_holiday']  = (df['days_to_holiday']<=3).astype(int)
    df['holiday_week']  = (df['days_to_holiday']<=7).astype(int)
    df['post_holiday']  = (df['days_since_holiday']<=2).astype(int)

    # Interactions
    df['promo_x_weekend'] = df['is_promotion'] * df['is_weekend']
    df['promo_x_holiday'] = df['is_promotion'] * df['is_holiday']
    df['promo_x_event']   = df['is_promotion'] * df['is_special_event']
    df['price_x_promo']   = df['unit_price']   * df['is_promotion']

    # IDs
    df['restaurant_num']= df['restaurant_id'].str.extract(r'(\d+)').astype(int)
    df['item_num']      = df['menu_item_id'].str.extract(r'(\d+)').astype(int)
    cat_map = {c: i for i, c in enumerate(sorted(df['category'].unique()))}
    df['category_num']  = df['category'].map(cat_map)

    # Target encodings on LOG scale (train_ref only — no leakage)
    tmp = train_ref.copy()
    tmp['log_qty'] = np.log1p(tmp['quantity'])

    for col, short in [('menu_item_id','item'),('restaurant_id','rest'),('category','cat')]:
        df[f'{short}_mean_enc'] = df[col].map(tmp.groupby(col)['log_qty'].mean())

    for cols, label in [
        (['restaurant_id','menu_item_id'],   'ri'),
        (['menu_item_id','day_of_week_num'], 'id'),
        (['menu_item_id','month'],           'im'),
        (['restaurant_id','day_of_week_num'],'rd'),
        (['restaurant_id','month'],          'rm'),
        (['category','day_of_week_num'],     'cd'),
        (['category','month'],               'cm'),
        (['menu_item_id','is_promotion'],    'ip'),
        (['menu_item_id','is_holiday'],      'ih'),
        (['restaurant_id','menu_item_id','day_of_week_num'], 'rid'),
    ]:
        avg = tmp.groupby(cols)['log_qty'].mean()
        df[f'{label}_enc'] = df.set_index(cols).index.map(lambda x, a=avg: a.get(x, np.nan))

    df['item_vol_rank'] = df['menu_item_id'].map(
        train_ref.groupby('menu_item_id')['quantity'].mean().rank(ascending=False))
    df['price_rank']    = df['unit_price'].rank(pct=True)
    df['is_value']      = (df['unit_price']<3).astype(int)
    df['is_premium']    = (df['unit_price']>8).astype(int)

    # Historical period averages (log scale) — STABLE FEATURES for Model B
    for lbl, s, e in [
        ('q4_24','2024-10-01','2024-12-31'),
        ('q4_23','2023-10-01','2023-12-31'),
        ('q4_22','2022-10-01','2022-12-31'),
        ('oct_24','2024-10-01','2024-10-31'),
        ('nov_24','2024-11-01','2024-11-30'),
        ('dec_24','2024-12-01','2024-12-31'),
        ('oct_23','2023-10-01','2023-10-31'),
        ('nov_23','2023-11-01','2023-11-30'),
        ('dec_23','2023-12-01','2023-12-31'),
        ('q3_25','2025-07-01','2025-09-30'),
    ]:
        p = train_ref[(train_ref['date']>=s)&(train_ref['date']<=e)].copy()
        p['log_qty'] = np.log1p(p['quantity'])
        pa = p.groupby(['restaurant_id','menu_item_id'])['log_qty'].mean()
        df[f'{lbl}_avg'] = df.set_index(['restaurant_id','menu_item_id']).index.map(
            lambda x, a=pa: a.get(x, np.nan))

    # Same-month average (very powerful: Oct/Nov/Dec pattern)
    tmp2 = train_ref.copy()
    tmp2['log_qty'] = np.log1p(tmp2['quantity'])
    ma = tmp2.groupby(['restaurant_id','menu_item_id','month'])['log_qty'].mean()
    df['smonth_avg'] = df.set_index(['restaurant_id','menu_item_id','month']).index.map(
        lambda x, a=ma: a.get(x, np.nan))

    # Same DOW + month (e.g., "avg Friday in October")
    tmp2['dow'] = tmp2['date'].dt.dayofweek
    dma = tmp2.groupby(['restaurant_id','menu_item_id','dow','month'])['log_qty'].mean()
    df['sdow_month_avg'] = df.set_index(
        ['restaurant_id','menu_item_id','day_of_week_num','month']
    ).index.map(lambda x, a=dma: a.get(x, np.nan))

    # Same week-of-year average (annual pattern at week level)
    tmp2['woy'] = tmp2['date'].dt.isocalendar().week.astype(int)
    wma = tmp2.groupby(['restaurant_id','menu_item_id','woy'])['log_qty'].mean()
    df['swoy_avg'] = df.set_index(['restaurant_id','menu_item_id','week_of_year']).index.map(
        lambda x, a=wma: a.get(x, np.nan))

    return df

raw = add_static_features(raw, train_raw)
print("  Done.")


# ============================================================================
# STEP 3: DEFINE FEATURE SETS FOR EACH MODEL
# Model A: ALL features including short-term lags (best for days 1-14)
# Model B: STABLE features only — no short-term lags (best for days 30-92)
# ============================================================================

# Long-term lags that remain REAL data throughout the test period
LONG_LAGS  = [364, 371]  # always real historical data, never predicted
SDOW_WEEKS = [1, 2, 3, 4, 8, 12, 16]
SHORT_LAGS = [1, 2, 3, 7, 14, 21, 28, 56, 91, 182]
ALL_LAGS   = SHORT_LAGS + LONG_LAGS
ROLL_WINDOWS = [3, 7, 14, 28, 56, 91]

PAIRS = raw[['restaurant_id','menu_item_id']].drop_duplicates().values.tolist()

# Log qty store — all lags computed on log scale
LOG_QTY_STORE = {k: np.log1p(v) for k, v in QTY_STORE.items()}


# ============================================================================
# STEP 4: BUILD TRAINING FEATURES (for both models)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: Building training lag features...")
print("=" * 60)

tdf = raw[raw['date'] < '2025-10-01'].copy()
tdf = tdf.sort_values(['restaurant_id','menu_item_id','date']).reset_index(drop=True)
tdf['log_qty'] = np.log1p(tdf['quantity'])

grp     = tdf.groupby(['restaurant_id','menu_item_id'])['log_qty']
dow_grp = tdf.groupby(['restaurant_id','menu_item_id','day_of_week_num'])['log_qty']

# All lags on log scale
for lag in ALL_LAGS:
    tdf[f'lag_{lag}'] = grp.shift(lag)

# Same-DOW lags
for k in SDOW_WEEKS:
    tdf[f'sdow_lag_{k}'] = dow_grp.shift(k)

# Rolling stats
shifted = grp.shift(1)
for w in ROLL_WINDOWS:
    rolled = shifted.rolling(w, min_periods=max(3, w//4))
    tdf[f'rmean_{w}'] = rolled.mean().values
    if w <= 28:
        tdf[f'rstd_{w}'] = rolled.std().values
        tdf[f'rmin_{w}'] = rolled.min().values
        tdf[f'rmax_{w}'] = rolled.max().values

# Same-DOW rolling
dow_shifted = dow_grp.shift(1)
for nw in [4, 8, 12, 16]:
    tdf[f'sdow_rmean_{nw}'] = dow_shifted.rolling(nw, min_periods=2).mean().values
tdf['sdow_ewm']       = dow_shifted.transform(lambda x: x.ewm(span=8, min_periods=2).mean()).values
tdf['expanding_mean'] = shifted.expanding(min_periods=7).mean().values
tdf['ewm_7']          = shifted.transform(lambda x: x.ewm(span=7,  min_periods=3).mean()).values
tdf['ewm_28']         = shifted.transform(lambda x: x.ewm(span=28, min_periods=3).mean()).values

tdf['trend_3_14']  = tdf['rmean_3']  / (tdf['rmean_14'] + 0.1)
tdf['trend_7_28']  = tdf['rmean_7']  / (tdf['rmean_28'] + 0.1)
tdf['trend_7_91']  = tdf['rmean_7']  / (tdf['rmean_91'] + 0.1)
tdf['trend_14_28'] = tdf['rmean_14'] / (tdf['rmean_28'] + 0.1)

static_in_raw = [c for c in raw.columns if c not in tdf.columns
                 and c not in ['quantity','log_qty']]
if static_in_raw:
    tdf = tdf.merge(raw[['date','restaurant_id','menu_item_id']+static_in_raw],
                    on=['date','restaurant_id','menu_item_id'], how='left')

print(f"  Shape: {tdf.shape}")

# Define feature sets
EXCLUDE = {'date','restaurant_id','restaurant_name','city','state','menu_item_id',
           'menu_item_name','category','quantity','log_qty','day_of_week','precip_type',
           'holiday_name','special_event_name'}

ALL_FEATURES = sorted([c for c in tdf.columns if c not in EXCLUDE
                       and tdf[c].dtype in ['int64','float64','int32','float32']])

# Model B stable features — NO short-term lags, no short rolling windows
# Only uses features that are REAL data throughout test period
SHORT_LAG_COLS = ([f'lag_{l}' for l in SHORT_LAGS] +
                  [f'sdow_lag_{k}' for k in SDOW_WEEKS] +
                  [f'rmean_{w}' for w in ROLL_WINDOWS] +
                  [f'rstd_{w}'  for w in [3,7,14,28]] +
                  [f'rmin_{w}'  for w in [3,7,14,28]] +
                  [f'rmax_{w}'  for w in [3,7,14,28]] +
                  [f'sdow_rmean_{nw}' for nw in [4,8,12,16]] +
                  ['sdow_ewm','ewm_7','ewm_28','expanding_mean',
                   'trend_3_14','trend_7_28','trend_7_91','trend_14_28'])

STABLE_FEATURES = [f for f in ALL_FEATURES if f not in SHORT_LAG_COLS]

print(f"  Model A features (all):    {len(ALL_FEATURES)}")
print(f"  Model B features (stable): {len(STABLE_FEATURES)}")


# ============================================================================
# STEP 5: VALIDATION SPLIT
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: Validation split...")
print("=" * 60)

vm = (tdf['date']>='2024-10-01') & (tdf['date']<='2024-12-31')
tm = tdf['date'] < '2024-10-01'

X_tr_A    = tdf.loc[tm, ALL_FEATURES]
X_tr_B    = tdf.loc[tm, STABLE_FEATURES]
y_tr_log  = tdf.loc[tm, 'log_qty']
y_tr_raw  = tdf.loc[tm, 'quantity']

X_vl_A    = tdf.loc[vm, ALL_FEATURES]
X_vl_B    = tdf.loc[vm, STABLE_FEATURES]
y_vl_log  = tdf.loc[vm, 'log_qty']
y_vl_raw  = tdf.loc[vm, 'quantity']

def wmape(yt, yp):
    yt, yp = np.array(yt, float), np.array(yp, float)
    m = yt > 0
    return np.sum(np.abs(yt[m]-yp[m])) / np.sum(yt[m]) if m.sum() else 0

iv  = train_raw.loc[train_raw['date']<'2024-10-01'].groupby('menu_item_id')['quantity'].sum()
iv  = iv / iv.max()
sw  = tdf.loc[tm, 'menu_item_id'].map(iv).fillna(0.1).values
ivf = train_raw.groupby('menu_item_id')['quantity'].sum()
ivf = ivf / ivf.max()
swf = tdf['menu_item_id'].map(ivf).fillna(0.1).values

print(f"  Train: {len(X_tr_A):,} | Val: {len(X_vl_A):,}")


# ============================================================================
# STEP 6: TRAIN MODEL A (lag-heavy — best for days 1-14)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: Model A — lag-heavy (LightGBM + XGBoost + CatBoost)...")
print("=" * 60)

models_A = {}
preds_A_val = {}

# A1: LightGBM log-MAE
print("  A1: LightGBM log-MAE...")
try:
    import lightgbm as lgb
    m = lgb.LGBMRegressor(
        objective='regression_l1', metric='mae',
        learning_rate=0.02, num_leaves=255, min_child_samples=20,
        feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.05, reg_lambda=0.1, verbose=-1,
        n_estimators=5000, random_state=42
    )
    m.fit(X_tr_A, y_tr_log, eval_set=[(X_vl_A, y_vl_log)],
          callbacks=[lgb.early_stopping(150, verbose=False)], sample_weight=sw)
    p = np.expm1(m.predict(X_vl_A)).clip(0)
    print(f"     wMAPE: {wmape(y_vl_raw, p):.4f} (iter: {m.best_iteration_})")
    preds_A_val['A_LGB'] = p

    mf = lgb.LGBMRegressor(
        objective='regression_l1', metric='mae',
        learning_rate=0.02, num_leaves=255, min_child_samples=20,
        feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.05, reg_lambda=0.1, verbose=-1,
        n_estimators=m.best_iteration_, random_state=42
    )
    mf.fit(tdf[ALL_FEATURES], tdf['log_qty'], sample_weight=swf)
    models_A['A_LGB'] = mf
    feat_imp = pd.Series(mf.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
except Exception as e:
    print(f"  Failed: {e}")

# A2: LightGBM Tweedie
print("  A2: LightGBM Tweedie...")
try:
    m = lgb.LGBMRegressor(
        objective='tweedie', tweedie_variance_power=1.5,
        metric='tweedie', learning_rate=0.02, num_leaves=255,
        min_child_samples=20, feature_fraction=0.75,
        bagging_fraction=0.8, bagging_freq=5,
        verbose=-1, n_estimators=5000, random_state=42
    )
    m.fit(X_tr_A, y_tr_raw, eval_set=[(X_vl_A, y_vl_raw)],
          callbacks=[lgb.early_stopping(150, verbose=False)], sample_weight=sw)
    p = np.clip(m.predict(X_vl_A), 0, None)
    print(f"     wMAPE: {wmape(y_vl_raw, p):.4f} (iter: {m.best_iteration_})")
    preds_A_val['A_TWE'] = p

    mf = lgb.LGBMRegressor(
        objective='tweedie', tweedie_variance_power=1.5,
        metric='tweedie', learning_rate=0.02, num_leaves=255,
        min_child_samples=20, feature_fraction=0.75,
        bagging_fraction=0.8, bagging_freq=5,
        verbose=-1, n_estimators=m.best_iteration_, random_state=42
    )
    mf.fit(tdf[ALL_FEATURES], tdf['quantity'], sample_weight=swf)
    mf._raw_scale = True
    models_A['A_TWE'] = mf
except Exception as e:
    print(f"  Failed: {e}")

# A3: XGBoost log-MAE
print("  A3: XGBoost log-MAE...")
try:
    import xgboost as xgb
    m = xgb.XGBRegressor(
        objective='reg:absoluteerror', learning_rate=0.02,
        max_depth=8, min_child_weight=20,
        subsample=0.8, colsample_bytree=0.75,
        reg_alpha=0.1, reg_lambda=1.0,
        n_estimators=5000, random_state=42,
        verbosity=0, early_stopping_rounds=150
    )
    m.fit(X_tr_A, y_tr_log, eval_set=[(X_vl_A, y_vl_log)],
          verbose=False, sample_weight=sw)
    p = np.expm1(m.predict(X_vl_A)).clip(0)
    bi = m.best_iteration if hasattr(m, 'best_iteration') else 2000
    print(f"     wMAPE: {wmape(y_vl_raw, p):.4f} (iter: {bi})")
    preds_A_val['A_XGB'] = p

    mf = xgb.XGBRegressor(
        objective='reg:absoluteerror', learning_rate=0.02,
        max_depth=8, min_child_weight=20,
        subsample=0.8, colsample_bytree=0.75,
        reg_alpha=0.1, reg_lambda=1.0,
        n_estimators=bi, random_state=42, verbosity=0
    )
    mf.fit(tdf[ALL_FEATURES], tdf['log_qty'], sample_weight=swf, verbose=False)
    models_A['A_XGB'] = mf
except Exception as e:
    print(f"  Failed: {e}")

# A4: CatBoost log-MAE
print("  A4: CatBoost log-MAE...")
try:
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(
        loss_function='MAE', learning_rate=0.02,
        depth=8, l2_leaf_reg=3,
        iterations=5000, random_seed=42, verbose=0
    )
    m.fit(X_tr_A, y_tr_log, eval_set=(X_vl_A, y_vl_log),
          early_stopping_rounds=150, sample_weight=sw)
    p = np.expm1(m.predict(X_vl_A)).clip(0)
    print(f"     wMAPE: {wmape(y_vl_raw, p):.4f} (iter: {m.best_iteration_})")
    preds_A_val['A_CAT'] = p

    mf = CatBoostRegressor(
        loss_function='MAE', learning_rate=0.02,
        depth=8, l2_leaf_reg=3,
        iterations=m.best_iteration_, random_seed=42, verbose=0
    )
    mf.fit(tdf[ALL_FEATURES], tdf['log_qty'], sample_weight=swf)
    models_A['A_CAT'] = mf
except Exception as e:
    print(f"  Failed: {e}")

# Ensemble Model A
mn_A  = list(preds_A_val.keys())
bw_A  = None; bwm_A = 999
step  = 0.05
if len(mn_A) == 1:
    bw_A = {mn_A[0]: 1.0}; bwm_A = wmape(y_vl_raw, preds_A_val[mn_A[0]])
elif len(mn_A) >= 2:
    from itertools import product
    weight_options = np.arange(0, 1.001, step)
    for ws in product(weight_options, repeat=len(mn_A)):
        if abs(sum(ws) - 1.0) > 0.001: continue
        e = sum(ws[i]*preds_A_val[mn_A[i]] for i in range(len(mn_A)))
        wm = wmape(y_vl_raw, e)
        if wm < bwm_A: bwm_A = wm; bw_A = dict(zip(mn_A, ws))

val_A = sum(bw_A[n]*preds_A_val[n] for n in mn_A)
print(f"\n  Model A ensemble wMAPE: {bwm_A:.4f}")
print(f"  Weights: {bw_A}")


# ============================================================================
# STEP 7: TRAIN MODEL B (stable — best for days 30-92)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 6: Model B — stable/direct (no short-term lags)...")
print("=" * 60)

models_B = {}
preds_B_val = {}

# B1: LightGBM log-MAE (stable features)
print("  B1: LightGBM log-MAE (stable)...")
try:
    m = lgb.LGBMRegressor(
        objective='regression_l1', metric='mae',
        learning_rate=0.02, num_leaves=127, min_child_samples=20,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.05, reg_lambda=0.1, verbose=-1,
        n_estimators=5000, random_state=42
    )
    m.fit(X_tr_B, y_tr_log, eval_set=[(X_vl_B, y_vl_log)],
          callbacks=[lgb.early_stopping(150, verbose=False)], sample_weight=sw)
    p = np.expm1(m.predict(X_vl_B)).clip(0)
    print(f"     wMAPE: {wmape(y_vl_raw, p):.4f} (iter: {m.best_iteration_})")
    preds_B_val['B_LGB'] = p

    mf = lgb.LGBMRegressor(
        objective='regression_l1', metric='mae',
        learning_rate=0.02, num_leaves=127, min_child_samples=20,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.05, reg_lambda=0.1, verbose=-1,
        n_estimators=m.best_iteration_, random_state=42
    )
    mf.fit(tdf[STABLE_FEATURES], tdf['log_qty'], sample_weight=swf)
    models_B['B_LGB'] = mf
except Exception as e:
    print(f"  Failed: {e}")

# B2: XGBoost log-MAE (stable features)
print("  B2: XGBoost log-MAE (stable)...")
try:
    m = xgb.XGBRegressor(
        objective='reg:absoluteerror', learning_rate=0.02,
        max_depth=7, min_child_weight=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        n_estimators=5000, random_state=42,
        verbosity=0, early_stopping_rounds=150
    )
    m.fit(X_tr_B, y_tr_log, eval_set=[(X_vl_B, y_vl_log)],
          verbose=False, sample_weight=sw)
    p = np.expm1(m.predict(X_vl_B)).clip(0)
    bi = m.best_iteration if hasattr(m, 'best_iteration') else 2000
    print(f"     wMAPE: {wmape(y_vl_raw, p):.4f} (iter: {bi})")
    preds_B_val['B_XGB'] = p

    mf = xgb.XGBRegressor(
        objective='reg:absoluteerror', learning_rate=0.02,
        max_depth=7, min_child_weight=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        n_estimators=bi, random_state=42, verbosity=0
    )
    mf.fit(tdf[STABLE_FEATURES], tdf['log_qty'], sample_weight=swf, verbose=False)
    models_B['B_XGB'] = mf
except Exception as e:
    print(f"  Failed: {e}")

# B3: CatBoost log-MAE (stable features)
print("  B3: CatBoost log-MAE (stable)...")
try:
    m = CatBoostRegressor(
        loss_function='MAE', learning_rate=0.02,
        depth=7, l2_leaf_reg=3,
        iterations=5000, random_seed=42, verbose=0
    )
    m.fit(X_tr_B, y_tr_log, eval_set=(X_vl_B, y_vl_log),
          early_stopping_rounds=150, sample_weight=sw)
    p = np.expm1(m.predict(X_vl_B)).clip(0)
    print(f"     wMAPE: {wmape(y_vl_raw, p):.4f} (iter: {m.best_iteration_})")
    preds_B_val['B_CAT'] = p

    mf = CatBoostRegressor(
        loss_function='MAE', learning_rate=0.02,
        depth=7, l2_leaf_reg=3,
        iterations=m.best_iteration_, random_seed=42, verbose=0
    )
    mf.fit(tdf[STABLE_FEATURES], tdf['log_qty'], sample_weight=swf)
    models_B['B_CAT'] = mf
except Exception as e:
    print(f"  Failed: {e}")

# Ensemble Model B
mn_B  = list(preds_B_val.keys())
bw_B  = None; bwm_B = 999
if len(mn_B) == 1:
    bw_B = {mn_B[0]: 1.0}; bwm_B = wmape(y_vl_raw, preds_B_val[mn_B[0]])
elif len(mn_B) >= 2:
    for ws in product(weight_options, repeat=len(mn_B)):
        if abs(sum(ws) - 1.0) > 0.001: continue
        e = sum(ws[i]*preds_B_val[mn_B[i]] for i in range(len(mn_B)))
        wm = wmape(y_vl_raw, e)
        if wm < bwm_B: bwm_B = wm; bw_B = dict(zip(mn_B, ws))

val_B = sum(bw_B[n]*preds_B_val[n] for n in mn_B)
print(f"\n  Model B ensemble wMAPE: {bwm_B:.4f}")
print(f"  Weights: {bw_B}")


# ============================================================================
# STEP 8: HORIZON-ADAPTIVE BLENDING FUNCTION
# Days 1-14:  90% A + 10% B  (short lags still mostly real)
# Days 15-30: 70% A + 30% B  (lags getting noisy)
# Days 31-60: 45% A + 55% B  (lean on stable)
# Days 61-92: 20% A + 80% B  (almost all stable)
# ============================================================================

def get_horizon_weights(day_index):
    """
    Returns (weight_A, weight_B) based on forecast horizon.
    day_index: 0-based (0 = first test day, 91 = last)
    """
    d = day_index + 1  # 1-based day number
    if d <= 7:
        return 0.95, 0.05
    elif d <= 14:
        return 0.85, 0.15
    elif d <= 21:
        return 0.70, 0.30
    elif d <= 30:
        return 0.55, 0.45
    elif d <= 45:
        return 0.40, 0.60
    elif d <= 60:
        return 0.28, 0.72
    else:
        return 0.15, 0.85

print("\n  Horizon blending schedule:")
for d in [1, 7, 14, 21, 30, 45, 60, 75, 92]:
    wa, wb = get_horizon_weights(d-1)
    print(f"    Day {d:3d}: Model A={wa:.0%}  Model B={wb:.0%}")


# ============================================================================
# STEP 9: LAG BUILDER FOR ITERATIVE PREDICTION
# ============================================================================

def compute_lags_for_day(pred_date, pairs, log_qty_store):
    """Compute ALL lag features for a given date."""
    records = []
    for rid, mid in pairs:
        row = {'restaurant_id': rid, 'menu_item_id': mid}

        # All lags (log scale)
        for lag in ALL_LAGS:
            lag_date = pred_date - pd.Timedelta(days=lag)
            row[f'lag_{lag}'] = log_qty_store.get((lag_date, rid, mid), np.nan)

        # Same-DOW lags
        for k in SDOW_WEEKS:
            sdow_date = pred_date - pd.Timedelta(weeks=k)
            row[f'sdow_lag_{k}'] = log_qty_store.get((sdow_date, rid, mid), np.nan)

        # Rolling windows
        for w in ROLL_WINDOWS:
            vals = [log_qty_store.get((pred_date - pd.Timedelta(days=d), rid, mid), None)
                    for d in range(1, w+1)]
            vals = [v for v in vals if v is not None]
            if len(vals) >= max(3, w//4):
                row[f'rmean_{w}'] = np.mean(vals)
                if w <= 28:
                    row[f'rstd_{w}'] = np.std(vals) if len(vals)>1 else 0
                    row[f'rmin_{w}'] = np.min(vals)
                    row[f'rmax_{w}'] = np.max(vals)
            else:
                row[f'rmean_{w}'] = np.nan
                if w <= 28:
                    row[f'rstd_{w}'] = row[f'rmin_{w}'] = row[f'rmax_{w}'] = np.nan

        # Same-DOW rolling
        for nw in [4, 8, 12, 16]:
            vals = [log_qty_store.get((pred_date - pd.Timedelta(weeks=k), rid, mid), None)
                    for k in range(1, nw+1)]
            vals = [v for v in vals if v is not None]
            row[f'sdow_rmean_{nw}'] = np.mean(vals) if len(vals)>=2 else np.nan

        # Same-DOW EWM
        vals = [log_qty_store.get((pred_date - pd.Timedelta(weeks=k), rid, mid), None)
                for k in range(1, 17)]
        vals = [v for v in vals if v is not None]
        if vals:
            wts = np.array([0.85**i for i in range(len(vals))])
            row['sdow_ewm'] = np.average(vals, weights=wts)
        else:
            row['sdow_ewm'] = np.nan

        # Expanding mean & EWM
        vals365 = [log_qty_store.get((pred_date - pd.Timedelta(days=d), rid, mid), None)
                   for d in range(1, 366)]
        vals365 = [v for v in vals365 if v is not None]
        row['expanding_mean'] = np.mean(vals365) if vals365 else np.nan

        vals28 = [log_qty_store.get((pred_date - pd.Timedelta(days=d), rid, mid), None)
                  for d in range(1, 29)]
        vals28 = [v for v in vals28 if v is not None]
        if vals28:
            wts = np.array([0.9**i for i in range(len(vals28))])
            row['ewm_28'] = np.average(vals28, weights=wts)
        else:
            row['ewm_28'] = np.nan

        vals7 = [log_qty_store.get((pred_date - pd.Timedelta(days=d), rid, mid), None)
                 for d in range(1, 8)]
        vals7 = [v for v in vals7 if v is not None]
        if vals7:
            wts = np.array([0.9**i for i in range(len(vals7))])
            row['ewm_7'] = np.average(vals7, weights=wts)
        else:
            row['ewm_7'] = np.nan

        # Trends
        row['trend_3_14']  = row.get('rmean_3',np.nan)  / (row.get('rmean_14',0)+0.1) if not np.isnan(row.get('rmean_3',np.nan)) else np.nan
        row['trend_7_28']  = row.get('rmean_7',np.nan)  / (row.get('rmean_28',0)+0.1) if not np.isnan(row.get('rmean_7',np.nan)) else np.nan
        row['trend_7_91']  = row.get('rmean_7',np.nan)  / (row.get('rmean_91',0)+0.1) if not np.isnan(row.get('rmean_7',np.nan)) else np.nan
        row['trend_14_28'] = row.get('rmean_14',np.nan) / (row.get('rmean_28',0)+0.1) if not np.isnan(row.get('rmean_14',np.nan)) else np.nan

        records.append(row)
    return pd.DataFrame(records)


# ============================================================================
# STEP 10: ITERATIVE FORECASTING WITH HORIZON-ADAPTIVE BLENDING
# ============================================================================
print("\n" + "=" * 60)
print("STEP 7: Iterative forecasting with horizon-adaptive blending...")
print("=" * 60)

test_dates  = sorted(raw[raw['date'] >= '2025-10-01']['date'].unique())
test_static = raw[raw['date'] >= '2025-10-01'].copy()
all_preds   = []

print(f"  {len(test_dates)} days | 750 predictions/day")

for i, pred_date in enumerate(test_dates):
    day_rows = test_static[test_static['date'] == pred_date].copy()
    lag_df   = compute_lags_for_day(pred_date, PAIRS, LOG_QTY_STORE)
    day_rows = day_rows.merge(lag_df, on=['restaurant_id','menu_item_id'], how='left')

    # Ensure all columns exist
    for f in ALL_FEATURES:
        if f not in day_rows.columns:
            day_rows[f] = np.nan
    for f in STABLE_FEATURES:
        if f not in day_rows.columns:
            day_rows[f] = np.nan

    X_day_A = day_rows[ALL_FEATURES]
    X_day_B = day_rows[STABLE_FEATURES]

    # Model A predictions
    pred_A = np.zeros(len(day_rows))
    for name, weight in bw_A.items():
        if weight > 0 and name in models_A:
            model = models_A[name]
            raw_p = model.predict(X_day_A)
            if getattr(model, '_raw_scale', False):
                pred_A += weight * np.clip(raw_p, 0, None)
            else:
                pred_A += weight * np.expm1(raw_p).clip(0)

    # Model B predictions (stable)
    pred_B = np.zeros(len(day_rows))
    for name, weight in bw_B.items():
        if weight > 0 and name in models_B:
            raw_p = models_B[name].predict(X_day_B)
            pred_B += weight * np.expm1(raw_p).clip(0)

    # Horizon-adaptive blend
    wa, wb = get_horizon_weights(i)
    day_pred = (wa * pred_A + wb * pred_B).clip(0)

    # Update log qty store for iterative prediction
    for j, (_, row) in enumerate(day_rows[['restaurant_id','menu_item_id']].iterrows()):
        LOG_QTY_STORE[(pred_date, row['restaurant_id'], row['menu_item_id'])] = np.log1p(day_pred[j])

    result = day_rows[['date','restaurant_id','menu_item_id']].copy()
    result['predicted_quantity'] = np.round(day_pred).astype(int)
    all_preds.append(result)

    if (i+1) % 7 == 0 or i == 0 or i == len(test_dates)-1:
        el = time.time() - T0
        print(f"  Day {i+1:3d}/{len(test_dates)} ({pred_date.date()}) "
              f"A={wa:.0%} B={wb:.0%}  [{el/60:.1f} min]")

predictions = pd.concat(all_preds, ignore_index=True)


# ============================================================================
# STEP 11: EVALUATE & SAVE
# ============================================================================
print("\n" + "=" * 60)
print("STEP 8: Evaluate & Save")
print("=" * 60)

ev   = test_actual.merge(predictions, on=['date','restaurant_id','menu_item_id'], how='left')
mask = ev['quantity'].notna() & (ev['quantity'] > 0)
final_wmape = wmape(ev.loc[mask,'quantity'], ev.loc[mask,'predicted_quantity'])

print(f"\n  {'='*45}")
print(f"  Model A val wMAPE:    {bwm_A:.4f}")
print(f"  Model B val wMAPE:    {bwm_B:.4f}")
print(f"  FINAL TEST wMAPE:     {final_wmape:.4f}")
print(f"  Previous best:        0.1308")
print(f"  Target:               0.10")
print(f"  {'='*45}")

print("\n  wMAPE by category:")
ci = train_raw.groupby('category')['menu_item_id'].apply(set).to_dict()
for cat in sorted(ci):
    cm = mask & ev['menu_item_id'].isin(ci[cat])
    if cm.sum():
        cw = wmape(ev.loc[cm,'quantity'], ev.loc[cm,'predicted_quantity'])
        print(f"    {cat:20s}: {cw:.4f}")

sub = predictions.copy()
sub['date'] = sub['date'].dt.strftime('%Y-%m-%d')
sub.to_csv('outputs/submission_v7.csv', index=False)
print(f"\n  Saved: outputs/submission_v7.csv ({len(sub):,} rows)")

if 'feat_imp' in dir():
    print("\n  Top 15 features (Model A):")
    for f, v in feat_imp.head(15).items():
        print(f"    {f:35s} {v:,}")

el = time.time() - T0
print(f"\n  Total time: {el/60:.1f} minutes")
print(f"  FINAL wMAPE: {final_wmape:.4f}")
