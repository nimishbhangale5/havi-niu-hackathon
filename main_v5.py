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
train_raw = raw[tmask].copy()

print(f"Train: {len(train_raw):,} | Test: {(~tmask).sum():,}")

QTY_STORE = {}
for _, row in train_raw[['date', 'restaurant_id', 'menu_item_id', 'quantity']].iterrows():
    QTY_STORE[(row['date'], row['restaurant_id'], row['menu_item_id'])] = row['quantity']

print(f"  Quantity store initialized: {len(QTY_STORE):,} entries")


# ============================================================================
# STEP 2: STATIC FEATURES
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: Building static features...")
print("=" * 60)

def add_static_features(df, train_ref):
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['day_of_month'] = df['date'].dt.day
    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = (df['date'].dt.day <= 3).astype(int)
    df['is_month_end'] = (df['date'].dt.day >= 28).astype(int)
    for col, p in [('day_of_year',365),('day_of_week_num',7),('month',12),('week_of_year',52)]:
        df[f'sin_{col}'] = np.sin(2*np.pi*df[col]/p)
        df[f'cos_{col}'] = np.cos(2*np.pi*df[col]/p)

    df['temp_bin'] = pd.cut(df['avg_temp_f'], bins=[-np.inf,20,35,50,70,np.inf], labels=[0,1,2,3,4]).astype(float)
    df['is_rain'] = (df['precip_type']=='Rain').astype(int)
    df['is_snow'] = (df['precip_type']=='Snow').astype(int)
    df['is_precip'] = (df['precip_inches']>0).astype(int)
    df['heavy_precip'] = (df['precip_inches']>0.5).astype(int)
    df['temp_x_precip'] = df['avg_temp_f'] * df['precip_inches']
    df['temp_squared'] = df['avg_temp_f']**2

    hdates = np.sort(df[df['is_holiday']==1]['date'].unique())
    d2f, d2b = {}, {}
    for d in df['date'].unique():
        fwd = hdates[hdates > d]
        bck = hdates[hdates < d]
        d2f[d] = int((fwd[0]-d)/np.timedelta64(1,'D')) if len(fwd)>0 else 30
        d2b[d] = int((d-bck[-1])/np.timedelta64(1,'D')) if len(bck)>0 else 30
    df['days_to_holiday'] = df['date'].map(d2f)
    df['days_since_holiday'] = df['date'].map(d2b)
    df['near_holiday'] = (df['days_to_holiday']<=3).astype(int)
    df['holiday_week'] = (df['days_to_holiday']<=7).astype(int)
    df['post_holiday'] = (df['days_since_holiday']<=2).astype(int)

    df['promo_x_weekend'] = df['is_promotion']*df['is_weekend']
    df['promo_x_holiday'] = df['is_promotion']*df['is_holiday']

    df['restaurant_num'] = df['restaurant_id'].str.extract(r'(\d+)').astype(int)
    df['item_num'] = df['menu_item_id'].str.extract(r'(\d+)').astype(int)
    cat_map = {c: i for i, c in enumerate(sorted(df['category'].unique()))}
    df['category_num'] = df['category'].map(cat_map)

    for col, short in [('menu_item_id','item'),('restaurant_id','rest'),('category','cat')]:
        df[f'{short}_mean_enc'] = df[col].map(train_ref.groupby(col)['quantity'].mean())
    for cols, label in [
        (['restaurant_id','menu_item_id'],'ri'), (['menu_item_id','day_of_week_num'],'id'),
        (['menu_item_id','month'],'im'), (['restaurant_id','day_of_week_num'],'rd'),
        (['restaurant_id','month'],'rm'), (['category','day_of_week_num'],'cd'),
        (['category','month'],'cm'), (['restaurant_id','category'],'rc'),
    ]:
        avg = train_ref.groupby(cols)['quantity'].mean()
        df[f'{label}_enc'] = df.set_index(cols).index.map(lambda x, a=avg: a.get(x, np.nan))

    df['item_vol_rank'] = df['menu_item_id'].map(
        train_ref.groupby('menu_item_id')['quantity'].mean().rank(ascending=False))
    df['price_rank'] = df['unit_price'].rank(pct=True)
    df['is_value'] = (df['unit_price']<3).astype(int)
    df['is_premium'] = (df['unit_price']>8).astype(int)

    for lbl, s, e in [('q4_24','2024-10-01','2024-12-31'),('q4_23','2023-10-01','2023-12-31'),
                       ('q3_25','2025-07-01','2025-09-30')]:
        p = train_ref[(train_ref['date']>=s)&(train_ref['date']<=e)]
        pa = p.groupby(['restaurant_id','menu_item_id'])['quantity'].mean()
        df[f'{lbl}_avg'] = df.set_index(['restaurant_id','menu_item_id']).index.map(lambda x,a=pa: a.get(x,np.nan))

    ma = train_ref.groupby(['restaurant_id','menu_item_id','month'])['quantity'].mean()
    df['smonth_avg'] = df.set_index(['restaurant_id','menu_item_id','month']).index.map(lambda x,a=ma: a.get(x,np.nan))

    return df

raw = add_static_features(raw, train_raw)
print("  Done.")


# ============================================================================
# STEP 3: LAG BUILDER
# ============================================================================

LAG_DAYS = [1, 2, 3, 7, 14, 21, 28, 56, 91, 182, 364, 371]
SDOW_WEEKS = [1, 2, 3, 4, 8, 12]
ROLL_WINDOWS = [3, 7, 14, 28, 56, 91]

PAIRS = raw[['restaurant_id', 'menu_item_id']].drop_duplicates().values.tolist()

def compute_lags_for_day_fast(pred_date, pairs, qty_store):
    records = []
    for rid, mid in pairs:
        row = {'restaurant_id': rid, 'menu_item_id': mid}
        for lag in LAG_DAYS:
            lag_date = pred_date - pd.Timedelta(days=lag)
            row[f'lag_{lag}'] = qty_store.get((lag_date, rid, mid), np.nan)
        for k in SDOW_WEEKS:
            sdow_date = pred_date - pd.Timedelta(weeks=k)
            row[f'sdow_lag_{k}'] = qty_store.get((sdow_date, rid, mid), np.nan)
        for w in ROLL_WINDOWS:
            vals = []
            for d in range(1, w+1):
                dt = pred_date - pd.Timedelta(days=d)
                v = qty_store.get((dt, rid, mid), None)
                if v is not None:
                    vals.append(v)
            if len(vals) >= max(3, w//4):
                row[f'rmean_{w}'] = np.mean(vals)
                if w <= 28:
                    row[f'rstd_{w}'] = np.std(vals) if len(vals) > 1 else 0
                    row[f'rmin_{w}'] = np.min(vals)
                    row[f'rmax_{w}'] = np.max(vals)
            else:
                row[f'rmean_{w}'] = np.nan
                if w <= 28:
                    row[f'rstd_{w}'] = np.nan
                    row[f'rmin_{w}'] = np.nan
                    row[f'rmax_{w}'] = np.nan
        for nw in [4, 8]:
            vals = []
            for k in range(1, nw+1):
                dt = pred_date - pd.Timedelta(weeks=k)
                v = qty_store.get((dt, rid, mid), None)
                if v is not None:
                    vals.append(v)
            row[f'sdow_rmean_{nw}'] = np.mean(vals) if len(vals) >= 2 else np.nan
        vals = []
        for d in range(1, 366):
            dt = pred_date - pd.Timedelta(days=d)
            v = qty_store.get((dt, rid, mid), None)
            if v is not None:
                vals.append(v)
        row['expanding_mean'] = np.mean(vals) if vals else np.nan
        row['trend_3_14'] = row.get('rmean_3', np.nan) / (row.get('rmean_14', 0) + 1) if not np.isnan(row.get('rmean_3', np.nan)) else np.nan
        row['trend_7_28'] = row.get('rmean_7', np.nan) / (row.get('rmean_28', 0) + 1) if not np.isnan(row.get('rmean_7', np.nan)) else np.nan
        row['trend_7_91'] = row.get('rmean_7', np.nan) / (row.get('rmean_91', 0) + 1) if not np.isnan(row.get('rmean_7', np.nan)) else np.nan
        records.append(row)
    return pd.DataFrame(records)


# ============================================================================
# STEP 4: BUILD TRAINING LAG FEATURES
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: Building lag features for training data...")
print("=" * 60)

tdf = raw[raw['date'] < '2025-10-01'].copy()
tdf = tdf.sort_values(['restaurant_id', 'menu_item_id', 'date']).reset_index(drop=True)

grp = tdf.groupby(['restaurant_id', 'menu_item_id'])['quantity']
for lag in LAG_DAYS:
    tdf[f'lag_{lag}'] = grp.shift(lag)

dow_grp = tdf.groupby(['restaurant_id', 'menu_item_id', 'day_of_week_num'])['quantity']
for k in SDOW_WEEKS:
    tdf[f'sdow_lag_{k}'] = dow_grp.shift(k)

shifted = grp.shift(1)
for w in ROLL_WINDOWS:
    rolled = shifted.rolling(w, min_periods=max(3, w//4))
    tdf[f'rmean_{w}'] = rolled.mean().values
    if w <= 28:
        tdf[f'rstd_{w}'] = rolled.std().values
        tdf[f'rmin_{w}'] = rolled.min().values
        tdf[f'rmax_{w}'] = rolled.max().values

dow_shifted = dow_grp.shift(1)
tdf['sdow_rmean_4'] = dow_shifted.rolling(4, min_periods=2).mean().values
tdf['sdow_rmean_8'] = dow_shifted.rolling(8, min_periods=3).mean().values
tdf['expanding_mean'] = shifted.expanding(min_periods=7).mean().values
tdf['trend_3_14'] = tdf['rmean_3'] / (tdf['rmean_14'] + 1)
tdf['trend_7_28'] = tdf['rmean_7'] / (tdf['rmean_28'] + 1)
tdf['trend_7_91'] = tdf['rmean_7'] / (tdf['rmean_91'] + 1)

static_in_raw = [c for c in raw.columns if c not in tdf.columns and c != 'quantity']
if static_in_raw:
    tdf = tdf.merge(raw[['date','restaurant_id','menu_item_id']+static_in_raw],
                    on=['date','restaurant_id','menu_item_id'], how='left')

print(f"  Shape: {tdf.shape}")


# ============================================================================
# STEP 5: FEATURES & VALIDATION
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: Features and validation...")
print("=" * 60)

EXCLUDE = {'date','restaurant_id','restaurant_name','city','state','menu_item_id',
           'menu_item_name','category','quantity','day_of_week','precip_type',
           'holiday_name','special_event_name'}
FEATURES = sorted([c for c in tdf.columns if c not in EXCLUDE
                    and tdf[c].dtype in ['int64','float64','int32','float32']])

print(f"  Features: {len(FEATURES)}")

vm = (tdf['date']>='2024-10-01') & (tdf['date']<='2024-12-31')
tm = tdf['date'] < '2024-10-01'

X_tr, y_tr = tdf.loc[tm, FEATURES], tdf.loc[tm, 'quantity']
X_vl, y_vl = tdf.loc[vm, FEATURES], tdf.loc[vm, 'quantity']

def wmape(yt, yp):
    yt, yp = np.array(yt, float), np.array(yp, float)
    m = yt > 0
    return np.sum(np.abs(yt[m]-yp[m])) / np.sum(yt[m]) if m.sum() else 0

iv = train_raw.loc[train_raw['date']<'2024-10-01'].groupby('menu_item_id')['quantity'].sum()
iv = iv / iv.max()
sw = tdf.loc[tm, 'menu_item_id'].map(iv).values

ivf = train_raw.groupby('menu_item_id')['quantity'].sum()
ivf = ivf / ivf.max()
swf = tdf['menu_item_id'].map(ivf).values

print(f"  Train: {len(X_tr):,} | Val: {len(X_vl):,}")


# ============================================================================
# STEP 6: TRAIN MODELS
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: Training models...")
print("=" * 60)

models = {}
vp = {}
ms = {}

print("\n  [1/3] LightGBM...")
try:
    import lightgbm as lgb
    m = lgb.LGBMRegressor(objective='mae', metric='mae', learning_rate=0.02, num_leaves=127,
        min_child_samples=20, feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.05, reg_lambda=0.1, verbose=-1, n_estimators=5000, random_state=42)
    m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
          callbacks=[lgb.early_stopping(150, verbose=False)], sample_weight=sw)
    p = np.clip(m.predict(X_vl), 0, None)
    w = wmape(y_vl, p)
    print(f"  LightGBM Val wMAPE: {w:.4f} (iter: {m.best_iteration_})")
    vp['LGB'] = p; ms['LGB'] = w

    mf = lgb.LGBMRegressor(objective='mae', metric='mae', learning_rate=0.02, num_leaves=127,
        min_child_samples=20, feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.05, reg_lambda=0.1, verbose=-1, n_estimators=m.best_iteration_, random_state=42)
    mf.fit(tdf[FEATURES], tdf['quantity'], sample_weight=swf)
    models['LGB'] = mf
    feat_imp = pd.Series(mf.feature_importances_, index=FEATURES).sort_values(ascending=False)
except Exception as e:
    print(f"  Failed: {e}")

print("\n  [2/3] XGBoost...")
try:
    import xgboost as xgb
    m = xgb.XGBRegressor(objective='reg:absoluteerror', learning_rate=0.02, max_depth=8,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=1.0,
        n_estimators=5000, random_state=42, verbosity=0, early_stopping_rounds=150)
    m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False, sample_weight=sw)
    p = np.clip(m.predict(X_vl), 0, None)
    w = wmape(y_vl, p)
    bi = m.best_iteration if hasattr(m, 'best_iteration') else 1500
    print(f"  XGBoost Val wMAPE: {w:.4f} (iter: {bi})")
    vp['XGB'] = p; ms['XGB'] = w

    mf = xgb.XGBRegressor(objective='reg:absoluteerror', learning_rate=0.02, max_depth=8,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=1.0,
        n_estimators=bi, random_state=42, verbosity=0)
    mf.fit(tdf[FEATURES], tdf['quantity'], sample_weight=swf, verbose=False)
    models['XGB'] = mf
except Exception as e:
    print(f"  Failed: {e}")

print("\n  [3/3] CatBoost...")
try:
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(loss_function='MAE', learning_rate=0.02, depth=8, l2_leaf_reg=3,
        iterations=5000, random_seed=42, verbose=0)
    m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), early_stopping_rounds=150, sample_weight=sw)
    p = np.clip(m.predict(X_vl), 0, None)
    w = wmape(y_vl, p)
    print(f"  CatBoost Val wMAPE: {w:.4f} (iter: {m.best_iteration_})")
    vp['CAT'] = p; ms['CAT'] = w

    mf = CatBoostRegressor(loss_function='MAE', learning_rate=0.02, depth=8, l2_leaf_reg=3,
        iterations=m.best_iteration_, random_seed=42, verbose=0)
    mf.fit(tdf[FEATURES], tdf['quantity'], sample_weight=swf)
    models['CAT'] = mf
except Exception as e:
    print(f"  Failed: {e}")


# ============================================================================
# STEP 7: ENSEMBLE WEIGHTS
# ============================================================================
print("\n" + "=" * 60)
print("STEP 6: Ensemble weights...")
print("=" * 60)

for n, w in sorted(ms.items(), key=lambda x: x[1]):
    print(f"  {n}: {w:.4f}")

mn = list(vp.keys())
bw, bwm = None, 999

if len(mn) >= 2:
    for w1 in np.arange(0, 1.001, 0.05):
        if len(mn) == 2:
            w2 = round(1-w1, 3)
            if w2 < 0: continue
            e = w1*vp[mn[0]] + w2*vp[mn[1]]
            wm = wmape(y_vl, e)
            if wm < bwm: bwm = wm; bw = {mn[0]: w1, mn[1]: w2}
        elif len(mn) == 3:
            for w2 in np.arange(0, 1.001-w1, 0.05):
                w3 = round(1-w1-w2, 3)
                if w3 < 0: continue
                e = w1*vp[mn[0]] + w2*vp[mn[1]] + w3*vp[mn[2]]
                wm = wmape(y_vl, e)
                if wm < bwm: bwm = wm; bw = {mn[0]: w1, mn[1]: w2, mn[2]: w3}
elif len(mn) == 1:
    bw = {mn[0]: 1.0}; bwm = ms[mn[0]]

print(f"\n  Weights: {bw}")
print(f"  Ensemble Val wMAPE: {bwm:.4f}")


# ============================================================================
# STEP 8: ITERATIVE DAY-BY-DAY PREDICTION
# ============================================================================
print("\n" + "=" * 60)
print("STEP 7: Iterative forecasting (92 days)...")
print("=" * 60)

test_dates = sorted(raw[raw['date'] >= '2025-10-01']['date'].unique())
print(f"  {len(test_dates)} days | 750 predictions/day")

test_static = raw[raw['date'] >= '2025-10-01'].copy()
all_preds = []

for i, pred_date in enumerate(test_dates):
    day_rows = test_static[test_static['date'] == pred_date].copy()
    lag_df = compute_lags_for_day_fast(pred_date, PAIRS, QTY_STORE)
    day_rows = day_rows.merge(lag_df, on=['restaurant_id', 'menu_item_id'], how='left')

    for f in FEATURES:
        if f not in day_rows.columns:
            day_rows[f] = np.nan

    X_day = day_rows[FEATURES]
    day_pred = np.zeros(len(X_day))
    for name, weight in bw.items():
        if weight > 0 and name in models:
            day_pred += weight * models[name].predict(X_day)
    day_pred = np.clip(day_pred, 0, None)

    for _, row in day_rows[['restaurant_id', 'menu_item_id']].reset_index(drop=True).iterrows():
        idx = row.name
        QTY_STORE[(pred_date, row['restaurant_id'], row['menu_item_id'])] = day_pred[idx]

    result = day_rows[['date', 'restaurant_id', 'menu_item_id']].copy()
    result['predicted_quantity'] = np.round(day_pred).astype(int)
    all_preds.append(result)

    if (i+1) % 10 == 0 or i == 0 or i == len(test_dates)-1:
        el = time.time() - T0
        print(f"  Day {i+1:3d}/{len(test_dates)} ({pred_date.date()})  [{el/60:.1f} min]")

predictions = pd.concat(all_preds, ignore_index=True)


# ============================================================================
# STEP 9: EVALUATE & SAVE
# ============================================================================
print("\n" + "=" * 60)
print("STEP 8: Evaluate & Save...")
print("=" * 60)

ev = test_actual.merge(predictions, on=['date','restaurant_id','menu_item_id'], how='left')
mask = ev['quantity'].notna() & (ev['quantity'] > 0)
final = wmape(ev.loc[mask, 'quantity'], ev.loc[mask, 'predicted_quantity'])

print(f"\n  FINAL wMAPE: {final:.4f}")

sub = predictions.copy()
sub['date'] = sub['date'].dt.strftime('%Y-%m-%d')
sub.to_csv('outputs/submission_v5.csv', index=False)
print(f"  Saved: outputs/submission_v5.csv ({len(sub):,} rows)")

if 'feat_imp' in dir():
    print("\nTop 15 features:")
    for f, v in feat_imp.head(15).items():
        print(f"  {f:30s} {v:,}")

el = time.time() - T0
print(f"\n  Total time: {el/60:.1f} minutes")
print(f"  Final wMAPE: {final:.4f}")
