# =========================
# main_strong.py - Havi Hackathon Strong Model
# =========================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# -------------------------
# wMAPE metric
# -------------------------
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

# -------------------------
# Load your data
# -------------------------
df = pd.read_csv("your_dataset.csv")  # Replace with your CSV
FEATURE_COLS = [c for c in df.columns if c != "quantity"]

# -------------------------
# Clean target
# -------------------------
df = df.copy()
df = df[df['quantity'].notna() & (df['quantity'] >= 0) & (df['quantity'] < 1e6)]
df['quantity'] = df['quantity'].fillna(0)

# -------------------------
# Split train/validation
# -------------------------
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# -------------------------
# Encode categoricals
# -------------------------
for c in train_df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    train_df[c] = le.fit_transform(train_df[c].astype(str))
    val_df[c] = le.transform(val_df[c].astype(str))

# -------------------------
# Features / Target
# -------------------------
X_train = train_df[FEATURE_COLS].values
X_val = val_df[FEATURE_COLS].values

y_train = np.clip(train_df["quantity"].values, 0, None)
y_val = np.clip(val_df["quantity"].values, 0, None)

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

# -------------------------
# LightGBM
# -------------------------
lgb_train = lgb.Dataset(X_train, y_train_log)
lgb_val = lgb.Dataset(X_val, y_val_log, reference=lgb_train)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 128,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    'verbose': -1
}

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=3000,
    valid_sets=[lgb_train, lgb_val],
    early_stopping_rounds=50,
    verbose_eval=100
)

# -------------------------
# XGBoost
# -------------------------
xgb_model = xgb.XGBRegressor(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], early_stopping_rounds=50, verbose=100)

# -------------------------
# CatBoost
# -------------------------
cat_model = CatBoostRegressor(
    iterations=2000,
    depth=8,
    learning_rate=0.05,
    eval_metric='RMSE',
    random_seed=42,
    verbose=100
)

cat_model.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), early_stopping_rounds=50)

# -------------------------
# Predictions and blending
# -------------------------
pred_lgb = np.expm1(lgb_model.predict(X_val))
pred_xgb = np.expm1(xgb_model.predict(X_val))
pred_cat = np.expm1(cat_model.predict(X_val))

# Simple weighted blend
blend_pred = 0.4*pred_lgb + 0.3*pred_xgb + 0.3*pred_cat

# -------------------------
# Evaluate wMAPE
# -------------------------
score = wmape(y_val, blend_pred)
print("Validation wMAPE:", round(score, 4))