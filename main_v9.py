import pandas as pd
import numpy as np
import warnings, time
warnings.filterwarnings('ignore')
T0 = time.time()

print("="*60); print("LOADING DATA"); print("="*60)

df = pd.read_csv('data/raw/qsr_demand_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['restaurant_id','menu_item_id','date']).reset_index(drop=True)

tmask = df['date'] < '2025-10-01'
df.loc[tmask,'quantity'] = df.loc[tmask].groupby(
    ['restaurant_id','menu_item_id'])['quantity'].transform(
    lambda x: x.ffill().bfill().fillna(0))

train = df[tmask].copy()
test  = df[~tmask].copy()
print(f"Train: {len(train):,} | Test: {len(test):,}")

QS = {}
for _,r in train[['date','restaurant_id','menu_item_id','quantity']].iterrows():
    QS[(r.date,r.restaurant_id,r.menu_item_id)] = r.quantity

print("\nBuilding features...")

def make_features(df_in, train_ref):
    d = df_in.copy()
    d['doy']  = d['date'].dt.dayofyear
    d['woy']  = d['date'].dt.isocalendar().week.astype(int)
    d['dom']  = d['date'].dt.day
    d['qtr']  = d['date'].dt.quarter
    d['dow']  = d['date'].dt.dayofweek
    d['is_fri'] = (d['dow']==4).astype(int)
    d['is_sat'] = (d['dow']==5).astype(int)
    d['is_sun'] = (d['dow']==6).astype(int)
    for c,p in [('doy',365),('dow',7),('month',12),('woy',52)]:
        d[f'sin_{c}'] = np.sin(2*np.pi*d[c]/p)
        d[f'cos_{c}'] = np.cos(2*np.pi*d[c]/p)
    d['is_thanksgiving_week'] = ((d['month']==11)&(d['dom']>=22)&(d['dom']<=28)).astype(int)
    d['is_christmas_week']    = ((d['month']==12)&(d['dom']>=22)&(d['dom']<=31)).astype(int)
    d['is_halloween']         = ((d['month']==10)&(d['dom']==31)).astype(int)
    d['is_oct'] = (d['month']==10).astype(int)
    d['is_nov'] = (d['month']==11).astype(int)
    d['is_dec'] = (d['month']==12).astype(int)
    d['temp_bin']    = pd.cut(d['avg_temp_f'],[-np.inf,20,35,50,70,np.inf],labels=[0,1,2,3,4]).astype(float)
    d['is_snow']     = (d['precip_type']=='Snow').astype(int)
    d['is_rain']     = (d['precip_type']=='Rain').astype(int)
    d['heavy_rain']  = (d['precip_inches']>0.5).astype(int)
    d['temp_sq']     = d['avg_temp_f']**2
    d['temp_precip'] = d['avg_temp_f']*d['precip_inches']
    d['cold']        = (d['avg_temp_f']<32).astype(int)
    d['cold_wknd']   = d['cold']*d['is_weekend']
    hdates = np.sort(df_in[df_in['is_holiday']==1]['date'].unique())
    d2f,d2b={},{}
    for dt in df_in['date'].unique():
        fwd=hdates[hdates>dt]; bck=hdates[hdates<dt]
        d2f[dt]=int((fwd[0]-dt)/np.timedelta64(1,'D')) if len(fwd) else 30
        d2b[dt]=int((dt-bck[-1])/np.timedelta64(1,'D')) if len(bck) else 30
    d['d2hol']    = d['date'].map(d2f)
    d['d_since']  = d['date'].map(d2b)
    d['near_hol'] = (d['d2hol']<=3).astype(int)
    d['hol_week'] = (d['d2hol']<=7).astype(int)
    d['post_hol'] = (d['d_since']<=2).astype(int)
    d['promo_wknd']  = d['is_promotion']*d['is_weekend']
    d['promo_hol']   = d['is_promotion']*d['is_holiday']
    d['promo_event'] = d['is_promotion']*d['is_special_event']
    d['price_promo'] = d['unit_price']*d['is_promotion']
    d['price_wknd']  = d['unit_price']*d['is_weekend']
    d['rest_num'] = d['restaurant_id'].str.extract(r'(\d+)').astype(int)
    d['item_num'] = d['menu_item_id'].str.extract(r'(\d+)').astype(int)
    cats={c:i for i,c in enumerate(sorted(d['category'].unique()))}
    d['cat_num'] = d['category'].map(cats)
    t=train_ref.copy(); t['lq']=np.log1p(t['quantity']); t['dow']=t['date'].dt.dayofweek
    t['woy']=t['date'].dt.isocalendar().week.astype(int)
    for col,nm in [('menu_item_id','item'),('restaurant_id','rest'),('category','cat')]:
        d[f'{nm}_enc']=d[col].map(t.groupby(col)['lq'].mean())
    for cols,nm in [
        (['restaurant_id','menu_item_id'],'ri'),(['menu_item_id','dow'],'id'),
        (['menu_item_id','month'],'im'),(['restaurant_id','dow'],'rd'),
        (['restaurant_id','month'],'rm'),(['category','dow'],'cd'),
        (['category','month'],'cm'),(['menu_item_id','is_promotion'],'ip'),
        (['menu_item_id','is_holiday'],'ih'),(['menu_item_id','is_special_event'],'ie'),
        (['restaurant_id','menu_item_id','dow'],'rid'),(['restaurant_id','menu_item_id','month'],'rim'),
    ]:
        avg=t.groupby(cols)['lq'].mean()
        d[f'{nm}_enc']=d.set_index(cols).index.map(lambda x,a=avg: a.get(x,np.nan))
    d['item_rank']=d['menu_item_id'].map(train_ref.groupby('menu_item_id')['quantity'].mean().rank(ascending=False))
    d['price_rank']=d['unit_price'].rank(pct=True)
    t2=train_ref.copy(); t2['lq']=np.log1p(t2['quantity'])
    t2['dow']=t2['date'].dt.dayofweek; t2['woy']=t2['date'].dt.isocalendar().week.astype(int)
    ma=t2.groupby(['restaurant_id','menu_item_id','month'])['lq'].mean()
    d['month_avg']=d.set_index(['restaurant_id','menu_item_id','month']).index.map(lambda x,a=ma: a.get(x,np.nan))
    dma=t2.groupby(['restaurant_id','menu_item_id','dow','month'])['lq'].mean()
    d['dow_month_avg']=d.set_index(['restaurant_id','menu_item_id','dow','month']).index.map(lambda x,a=dma: a.get(x,np.nan))
    wma=t2.groupby(['restaurant_id','menu_item_id','woy'])['lq'].mean()
    d['woy_avg']=d.set_index(['restaurant_id','menu_item_id','woy']).index.map(lambda x,a=wma: a.get(x,np.nan))
    wdma=t2.groupby(['restaurant_id','menu_item_id','woy','dow'])['lq'].mean()
    d['woy_dow_avg']=d.set_index(['restaurant_id','menu_item_id','woy','dow']).index.map(lambda x,a=wdma: a.get(x,np.nan))
    for lbl,s,e in [
        ('oct24','2024-10-01','2024-10-31'),('nov24','2024-11-01','2024-11-30'),
        ('dec24','2024-12-01','2024-12-31'),('oct23','2023-10-01','2023-10-31'),
        ('nov23','2023-11-01','2023-11-30'),('dec23','2023-12-01','2023-12-31'),
        ('oct22','2022-10-01','2022-10-31'),('nov22','2022-11-01','2022-11-30'),
        ('dec22','2022-12-01','2022-12-31'),('q3_25','2025-07-01','2025-09-30'),
    ]:
        p=t2[(t2['date']>=s)&(t2['date']<=e)]
        pa=p.groupby(['restaurant_id','menu_item_id'])['lq'].mean()
        d[f'{lbl}_avg']=d.set_index(['restaurant_id','menu_item_id']).index.map(lambda x,a=pa: a.get(x,np.nan))
    return d

all_df   = make_features(df, train)
train_fe = all_df[tmask].copy()
test_fe  = all_df[~tmask].copy()

EXCLUDE={'date','restaurant_id','restaurant_name','city','state','menu_item_id',
         'menu_item_name','category','quantity','day_of_week','precip_type',
         'holiday_name','special_event_name'}
SHORT_COLS=(
    [f'lag_{l}' for l in [1,2,3,7,14,21,28,56,91,182]]+
    [f'sdow_lag_{k}' for k in [1,2,3,4,8,12,16]]+
    [f'rmean_{w}' for w in [3,7,14,28,56,91]]+
    [f'rstd_{w}' for w in [3,7,14,28]]+[f'rmin_{w}' for w in [3,7,14,28]]+
    [f'rmax_{w}' for w in [3,7,14,28]]+
    [f'sdow_rmean_{n}' for n in [4,8,12,16]]+
    ['sdow_ewm','ewm_7','ewm_28','expanding_mean',
     'trend_3_14','trend_7_28','trend_7_91','trend_14_28'])

print("Building training lags...")
tdf=train_fe.copy().sort_values(['restaurant_id','menu_item_id','date']).reset_index(drop=True)
tdf['lq']=np.log1p(tdf['quantity'])
grp=tdf.groupby(['restaurant_id','menu_item_id'])['lq']
dgrp=tdf.groupby(['restaurant_id','menu_item_id','dow'])['lq']
for l in [1,2,3,7,14,21,28,56,91,182,364,371]: tdf[f'lag_{l}']=grp.shift(l)
for k in [1,2,3,4,8,12,16]: tdf[f'sdow_lag_{k}']=dgrp.shift(k)
sh=grp.shift(1)
for w in [3,7,14,28,56,91]:
    r=sh.rolling(w,min_periods=max(2,w//4))
    tdf[f'rmean_{w}']=r.mean().values
    if w<=28:
        tdf[f'rstd_{w}']=r.std().values; tdf[f'rmin_{w}']=r.min().values; tdf[f'rmax_{w}']=r.max().values
dsh=dgrp.shift(1)
for n in [4,8,12,16]: tdf[f'sdow_rmean_{n}']=dsh.rolling(n,min_periods=2).mean().values
tdf['sdow_ewm']=dsh.transform(lambda x: x.ewm(span=8,min_periods=2).mean()).values
tdf['expanding_mean']=sh.expanding(min_periods=7).mean().values
tdf['ewm_7']=sh.transform(lambda x: x.ewm(span=7,min_periods=3).mean()).values
tdf['ewm_28']=sh.transform(lambda x: x.ewm(span=28,min_periods=3).mean()).values
tdf['trend_3_14']=tdf['rmean_3']/(tdf['rmean_14']+0.1)
tdf['trend_7_28']=tdf['rmean_7']/(tdf['rmean_28']+0.1)
tdf['trend_7_91']=tdf['rmean_7']/(tdf['rmean_91']+0.1)
tdf['trend_14_28']=tdf['rmean_14']/(tdf['rmean_28']+0.1)

ALL_FEATS=sorted([c for c in tdf.columns if c not in EXCLUDE and c!='lq'
                  and tdf[c].dtype in ['int64','float64','int32','float32']])
STABLE_FEATS=[f for f in ALL_FEATS if f not in SHORT_COLS]
print(f"Model A: {len(ALL_FEATS)} feats | Model B: {len(STABLE_FEATS)} feats")

vm=(tdf['date']>='2024-10-01')&(tdf['date']<='2024-12-31')
tm2=tdf['date']<'2024-10-01'
Xa_tr,Xb_tr=tdf.loc[tm2,ALL_FEATS],tdf.loc[tm2,STABLE_FEATS]
ya_tr=tdf.loc[tm2,'lq']; yraw_tr=tdf.loc[tm2,'quantity']
Xa_vl,Xb_vl=tdf.loc[vm,ALL_FEATS],tdf.loc[vm,STABLE_FEATS]
ya_vl=tdf.loc[vm,'lq']; yraw_vl=tdf.loc[vm,'quantity']

def wmape(yt,yp):
    yt,yp=np.array(yt,float),np.array(yp,float); m=yt>0
    return np.sum(np.abs(yt[m]-yp[m]))/np.sum(yt[m]) if m.sum() else 0

iv=train.loc[train['date']<'2024-10-01'].groupby('menu_item_id')['quantity'].sum(); iv=iv/iv.max()
sw=tdf.loc[tm2,'menu_item_id'].map(iv).fillna(0.1).values
ivf=train.groupby('menu_item_id')['quantity'].sum(); ivf=ivf/ivf.max()
swf=tdf['menu_item_id'].map(ivf).fillna(0.1).values

import lightgbm as lgb, xgboost as xgb
try: from catboost import CatBoostRegressor; HAS_CB=True
except: HAS_CB=False

def fit_lgb(Xtr,ytr,Xvl,yvl,sw,obj='regression_l1',leaves=255,seed=42):
    m=lgb.LGBMRegressor(objective=obj,metric='mae',learning_rate=0.02,num_leaves=leaves,
        min_child_samples=20,feature_fraction=0.75,bagging_fraction=0.8,bagging_freq=5,
        reg_alpha=0.05,reg_lambda=0.1,verbose=-1,n_estimators=5000,random_state=seed)
    m.fit(Xtr,ytr,eval_set=[(Xvl,yvl)],callbacks=[lgb.early_stopping(150,verbose=False)],sample_weight=sw)
    return m

def fit_xgb(Xtr,ytr,Xvl,yvl,sw,seed=42):
    m=xgb.XGBRegressor(objective='reg:absoluteerror',learning_rate=0.02,max_depth=8,
        min_child_weight=20,subsample=0.8,colsample_bytree=0.75,reg_alpha=0.1,reg_lambda=1.0,
        n_estimators=5000,random_state=seed,verbosity=0,early_stopping_rounds=150)
    m.fit(Xtr,ytr,eval_set=[(Xvl,yvl)],verbose=False,sample_weight=sw); return m

def fit_cat(Xtr,ytr,Xvl,yvl,sw,depth=8,seed=42):
    m=CatBoostRegressor(loss_function='MAE',learning_rate=0.02,depth=depth,l2_leaf_reg=3,
        iterations=5000,random_seed=seed,verbose=0)
    m.fit(Xtr,ytr,eval_set=(Xvl,yvl),early_stopping_rounds=150,sample_weight=sw); return m

def retrain_lgb(Xf,yf,swf,iters,obj='regression_l1',leaves=255,seed=42):
    m=lgb.LGBMRegressor(objective=obj,metric='mae',learning_rate=0.02,num_leaves=leaves,
        min_child_samples=20,feature_fraction=0.75,bagging_fraction=0.8,bagging_freq=5,
        reg_alpha=0.05,reg_lambda=0.1,verbose=-1,n_estimators=iters,random_state=seed)
    m.fit(Xf,yf,sample_weight=swf); return m

def retrain_xgb(Xf,yf,swf,iters,seed=42):
    m=xgb.XGBRegressor(objective='reg:absoluteerror',learning_rate=0.02,max_depth=8,
        min_child_weight=20,subsample=0.8,colsample_bytree=0.75,reg_alpha=0.1,reg_lambda=1.0,
        n_estimators=iters,random_state=seed,verbosity=0)
    m.fit(Xf,yf,sample_weight=swf,verbose=False); return m

def retrain_cat(Xf,yf,swf,iters,depth=8,seed=42):
    m=CatBoostRegressor(loss_function='MAE',learning_rate=0.02,depth=depth,l2_leaf_reg=3,
        iterations=iters,random_seed=seed,verbose=0)
    m.fit(Xf,yf,sample_weight=swf); return m

def best_weights(preds_dict,y_true,step=0.05):
    mn=list(preds_dict.keys()); bw=None; bwm=999
    wo=np.arange(0,1.01,step)
    if len(mn)==1: return {mn[0]:1.0},wmape(y_true,preds_dict[mn[0]])
    elif len(mn)==2:
        for w1 in wo:
            w2=round(1-w1,3)
            if w2<0: continue
            wm=wmape(y_true,w1*preds_dict[mn[0]]+w2*preds_dict[mn[1]])
            if wm<bwm: bwm=wm; bw={mn[0]:w1,mn[1]:w2}
    elif len(mn)==3:
        for w1 in wo:
            for w2 in wo:
                w3=round(1-w1-w2,3)
                if w3<0: continue
                wm=wmape(y_true,w1*preds_dict[mn[0]]+w2*preds_dict[mn[1]]+w3*preds_dict[mn[2]])
                if wm<bwm: bwm=wm; bw={mn[0]:w1,mn[1]:w2,mn[2]:w3}
    elif len(mn)==4:
        for w1 in wo:
            for w2 in wo:
                for w3 in wo:
                    w4=round(1-w1-w2-w3,3)
                    if w4<0: continue
                    e=w1*preds_dict[mn[0]]+w2*preds_dict[mn[1]]+w3*preds_dict[mn[2]]+w4*preds_dict[mn[3]]
                    wm=wmape(y_true,e)
                    if wm<bwm: bwm=wm; bw={mn[0]:w1,mn[1]:w2,mn[2]:w3,mn[3]:w4}
    return bw,bwm

# MODEL A
print("\n--- MODEL A (lag-heavy) ---")
mA={}; pA={}

print("  A1: LGB log-MAE...")
m=fit_lgb(Xa_tr,ya_tr,Xa_vl,ya_vl,sw)
pA['lgb']=np.expm1(m.predict(Xa_vl)).clip(0)
print(f"     {wmape(yraw_vl,pA['lgb']):.4f} iter={m.best_iteration_}")
feat_imp=pd.Series(m.feature_importances_,index=ALL_FEATS).sort_values(ascending=False)
mA['lgb']=retrain_lgb(tdf[ALL_FEATS],tdf['lq'],swf,m.best_iteration_)

print("  A2: LGB Tweedie...")
m=fit_lgb(Xa_tr,yraw_tr,Xa_vl,yraw_vl,sw,obj='tweedie')
pA['twe']=np.clip(m.predict(Xa_vl),0,None)
print(f"     {wmape(yraw_vl,pA['twe']):.4f} iter={m.best_iteration_}")
mA['twe']=retrain_lgb(tdf[ALL_FEATS],tdf['quantity'],swf,m.best_iteration_,obj='tweedie')
mA['twe']._raw=True

print("  A3: XGB log-MAE...")
m=fit_xgb(Xa_tr,ya_tr,Xa_vl,ya_vl,sw)
pA['xgb']=np.expm1(m.predict(Xa_vl)).clip(0)
bi=m.best_iteration if hasattr(m,'best_iteration') else 2000
print(f"     {wmape(yraw_vl,pA['xgb']):.4f} iter={bi}")
mA['xgb']=retrain_xgb(tdf[ALL_FEATS],tdf['lq'],swf,bi)

if HAS_CB:
    print("  A4: CatBoost log-MAE...")
    m=fit_cat(Xa_tr,ya_tr,Xa_vl,ya_vl,sw)
    pA['cat']=np.expm1(m.predict(Xa_vl)).clip(0)
    print(f"     {wmape(yraw_vl,pA['cat']):.4f} iter={m.best_iteration_}")
    mA['cat']=retrain_cat(tdf[ALL_FEATS],tdf['lq'],swf,m.best_iteration_)

bwA,bwmA=best_weights(pA,yraw_vl)
print(f"Model A ensemble: {bwmA:.4f} | {bwA}")

# MODEL B
print("\n--- MODEL B (stable/direct) ---")
mB={}; pB={}

print("  B1: LGB log-MAE...")
m=fit_lgb(Xb_tr,ya_tr,Xb_vl,ya_vl,sw,leaves=127)
pB['lgb']=np.expm1(m.predict(Xb_vl)).clip(0)
print(f"     {wmape(yraw_vl,pB['lgb']):.4f} iter={m.best_iteration_}")
mB['lgb']=retrain_lgb(tdf[STABLE_FEATS],tdf['lq'],swf,m.best_iteration_,leaves=127)

print("  B2: XGB log-MAE...")
m=fit_xgb(Xb_tr,ya_tr,Xb_vl,ya_vl,sw)
pB['xgb']=np.expm1(m.predict(Xb_vl)).clip(0)
bi=m.best_iteration if hasattr(m,'best_iteration') else 2000
print(f"     {wmape(yraw_vl,pB['xgb']):.4f} iter={bi}")
mB['xgb']=retrain_xgb(tdf[STABLE_FEATS],tdf['lq'],swf,bi)

if HAS_CB:
    print("  B3: CatBoost log-MAE...")
    m=fit_cat(Xb_tr,ya_tr,Xb_vl,ya_vl,sw,depth=7)
    pB['cat']=np.expm1(m.predict(Xb_vl)).clip(0)
    print(f"     {wmape(yraw_vl,pB['cat']):.4f} iter={m.best_iteration_}")
    mB['cat']=retrain_cat(tdf[STABLE_FEATS],tdf['lq'],swf,m.best_iteration_,depth=7)

bwB,bwmB=best_weights(pB,yraw_vl)
print(f"Model B ensemble: {bwmB:.4f} | {bwB}")

# ITERATIVE FORECASTING
print("\n"+"="*60); print("ITERATIVE FORECASTING"); print("="*60)
PAIRS=df[['restaurant_id','menu_item_id']].drop_duplicates().values.tolist()
LQS={k:np.log1p(v) for k,v in QS.items()}
ALL_LAGS=[1,2,3,7,14,21,28,56,91,182,364,371]
SDOW_W=[1,2,3,4,8,12,16]; RW=[3,7,14,28,56,91]

def get_lags(pred_date,pairs,lqs):
    recs=[]
    for rid,mid in pairs:
        r={'restaurant_id':rid,'menu_item_id':mid}
        for l in ALL_LAGS: r[f'lag_{l}']=lqs.get((pred_date-pd.Timedelta(days=l),rid,mid),np.nan)
        for k in SDOW_W:   r[f'sdow_lag_{k}']=lqs.get((pred_date-pd.Timedelta(weeks=k),rid,mid),np.nan)
        for w in RW:
            vs=[lqs.get((pred_date-pd.Timedelta(days=d),rid,mid),None) for d in range(1,w+1)]
            vs=[v for v in vs if v is not None]
            if len(vs)>=max(2,w//4):
                r[f'rmean_{w}']=np.mean(vs)
                if w<=28: r[f'rstd_{w}']=np.std(vs) if len(vs)>1 else 0; r[f'rmin_{w}']=np.min(vs); r[f'rmax_{w}']=np.max(vs)
            else:
                r[f'rmean_{w}']=np.nan
                if w<=28: r[f'rstd_{w}']=r[f'rmin_{w}']=r[f'rmax_{w}']=np.nan
        for n in [4,8,12,16]:
            vs=[lqs.get((pred_date-pd.Timedelta(weeks=k),rid,mid),None) for k in range(1,n+1)]
            vs=[v for v in vs if v is not None]
            r[f'sdow_rmean_{n}']=np.mean(vs) if len(vs)>=2 else np.nan
        vs=[lqs.get((pred_date-pd.Timedelta(weeks=k),rid,mid),None) for k in range(1,17)]
        vs=[v for v in vs if v is not None]
        r['sdow_ewm']=np.average(vs,weights=[0.85**i for i in range(len(vs))]) if vs else np.nan
        vs365=[lqs.get((pred_date-pd.Timedelta(days=d),rid,mid),None) for d in range(1,366)]
        vs365=[v for v in vs365 if v is not None]
        r['expanding_mean']=np.mean(vs365) if vs365 else np.nan
        vs28=[lqs.get((pred_date-pd.Timedelta(days=d),rid,mid),None) for d in range(1,29)]
        vs28=[v for v in vs28 if v is not None]
        r['ewm_28']=np.average(vs28,weights=[0.9**i for i in range(len(vs28))]) if vs28 else np.nan
        vs7=[lqs.get((pred_date-pd.Timedelta(days=d),rid,mid),None) for d in range(1,8)]
        vs7=[v for v in vs7 if v is not None]
        r['ewm_7']=np.average(vs7,weights=[0.9**i for i in range(len(vs7))]) if vs7 else np.nan
        for k1,k2 in [('rmean_3','rmean_14'),('rmean_7','rmean_28'),
                       ('rmean_7','rmean_91'),('rmean_14','rmean_28')]:
            nm='trend_'+k1.split('_')[1]+'_'+k2.split('_')[1]
            r[nm]=r.get(k1,np.nan)/(r.get(k2,0)+0.1) if not np.isnan(r.get(k1,np.nan)) else np.nan
        recs.append(r)
    return pd.DataFrame(recs)

def hw(i):
    d=i+1
    if d<=7:    return 0.95,0.05
    elif d<=14:  return 0.85,0.15
    elif d<=21:  return 0.72,0.28
    elif d<=30:  return 0.55,0.45
    elif d<=45:  return 0.38,0.62
    elif d<=60:  return 0.22,0.78
    else:        return 0.10,0.90

test_dates=sorted(df[~tmask]['date'].unique())
test_static=test_fe.copy(); all_preds=[]
test_actual=df.loc[~tmask,['date','restaurant_id','menu_item_id','quantity']].copy()

print(f"Forecasting {len(test_dates)} days...")
for i,pred_date in enumerate(test_dates):
    day=test_static[test_static['date']==pred_date].copy()
    ldf=get_lags(pred_date,PAIRS,LQS)
    day=day.merge(ldf,on=['restaurant_id','menu_item_id'],how='left')
    for f in ALL_FEATS:
        if f not in day.columns: day[f]=np.nan
    pA_day=np.zeros(len(day))
    for nm,wt in bwA.items():
        if wt>0 and nm in mA:
            mo=mA[nm]; rp=mo.predict(day[ALL_FEATS])
            pA_day+=wt*(np.clip(rp,0,None) if getattr(mo,'_raw',False) else np.expm1(rp).clip(0))
    pB_day=np.zeros(len(day))
    for nm,wt in bwB.items():
        if wt>0 and nm in mB:
            pB_day+=wt*np.expm1(mB[nm].predict(day[STABLE_FEATS])).clip(0)
    wa,wb=hw(i)
    pred=(wa*pA_day+wb*pB_day).clip(0)
    for j,(rid,mid) in enumerate(zip(day['restaurant_id'],day['menu_item_id'])):
        LQS[(pred_date,rid,mid)]=np.log1p(pred[j])
    res=day[['date','restaurant_id','menu_item_id']].copy()
    res['predicted_quantity']=np.round(pred).astype(int)
    all_preds.append(res)
    if (i+1)%7==0 or i==0 or i==len(test_dates)-1:
        print(f"  Day {i+1:3d}/{len(test_dates)} ({pred_date.date()}) A={wa:.0%}/B={wb:.0%} [{(time.time()-T0)/60:.1f}min]")

predictions=pd.concat(all_preds,ignore_index=True)

print("\n"+"="*60); print("RESULTS"); print("="*60)
ev=test_actual.merge(predictions,on=['date','restaurant_id','menu_item_id'],how='left')
mask=ev['quantity'].notna()&(ev['quantity']>0)
fw=wmape(ev.loc[mask,'quantity'],ev.loc[mask,'predicted_quantity'])
print(f"\n  Model A val wMAPE: {bwmA:.4f}")
print(f"  Model B val wMAPE: {bwmB:.4f}")
print(f"  FINAL TEST wMAPE:  {fw:.4f}")
print(f"  Previous best:     0.1308")
ci=train.groupby('category')['menu_item_id'].apply(set).to_dict()
print("\n  By category:")
for cat in sorted(ci):
    cm=mask&ev['menu_item_id'].isin(ci[cat])
    if cm.sum(): print(f"    {cat:20s}: {wmape(ev.loc[cm,'quantity'],ev.loc[cm,'predicted_quantity']):.4f}")
sub=predictions.copy(); sub['date']=sub['date'].dt.strftime('%Y-%m-%d')
sub.to_csv('outputs/submission_v7.csv',index=False)
print(f"\n  Saved: outputs/submission_v7.csv ({len(sub):,} rows)")
print("\n  Top 15 features (Model A):")
for f,v in feat_imp.head(15).items(): print(f"    {f:35s} {v:,}")
print(f"\n  Time: {(time.time()-T0)/60:.1f} min | FINAL wMAPE: {fw:.4f}")
