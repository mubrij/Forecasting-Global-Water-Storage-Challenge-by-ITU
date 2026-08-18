#!/usr/bin/env python3
"""Spatial-neighborhood feature benchmark."""
from pathlib import Path
import gc
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
import lightgbm as lgb
from winning_solution import (CLIMATE, SEED, ModelConfig, load_data, fit_history_stats,
    pair_to_xy, validation_pairs, lgb_params, rmse)

SPATIAL_BASE=['TWS_t',*CLIMATE]
SPATIAL=[f'{c}_nbr{s}' for c in SPATIAL_BASE for s in (3,7)]

def augment(df):
    out=df.copy(); h,w=140,360
    land=np.zeros((h,w),np.float32)
    li=np.rint(out.lat.to_numpy()+55.5).astype(int); lj=np.rint(out.lon.to_numpy()+179.5).astype(int)
    land[li,lj]=1
    den={s:uniform_filter(land,size=s,mode=('nearest','wrap')) for s in (3,7)}
    added={name:np.empty(len(out),np.float32) for name in SPATIAL}
    for _,idx in out.groupby('month_idx',sort=False).groups.items():
        idx=np.asarray(idx); ii=li[idx];jj=lj[idx]
        for c in SPATIAL_BASE:
            grid=np.zeros((h,w),np.float32); vals=out[c].to_numpy(np.float32)[idx]
            valid=np.isfinite(vals); grid[ii[valid],jj[valid]]=vals[valid]
            # Use the structural land mask for climate; observed-value mask for TWS.
            mask=land if c!='TWS_t' else np.zeros_like(land)
            if c=='TWS_t': mask[ii[valid],jj[valid]]=1
            for s in (3,7):
                num=uniform_filter(grid,size=s,mode=('nearest','wrap'))
                d=den[s] if c!='TWS_t' else uniform_filter(mask,size=s,mode=('nearest','wrap'))
                smooth=np.divide(num,d,out=np.zeros_like(num),where=d>1e-6)
                added[f'{c}_nbr{s}'][idx]=smooth[ii,jj]
    for c,v in added.items(): out[c]=v
    return out

def make_pair(data,h):
    cols=['ID','time','month_idx','loc_key','lat','lon','month_sin','month_cos','TWS_t','Target',*CLIMATE,*SPATIAL]
    cur=data[cols]
    acols=['month_idx','loc_key','TWS_t',*CLIMATE,*SPATIAL]
    a=data[acols].copy();a.month_idx+=h-1
    a=a.rename(columns={c:f'anchor_{c}' for c in ['TWS_t',*CLIMATE,*SPATIAL]})
    return cur.merge(a,on=['month_idx','loc_key'],how='inner',validate='one_to_one')

def add_spatial_x(x,pair):
    for c in [f'{z}_nbr{s}' for z in CLIMATE for s in (3,7)]:
        x[f'current_{c}']=pair[c].to_numpy(np.float32)
        x[f'anchor_{c}']=pair[f'anchor_{c}'].to_numpy(np.float32)
        x[f'change_{c}']=x[f'current_{c}']-x[f'anchor_{c}']
    for s in (3,7):
        x[f'anchor_TWS_t_nbr{s}']=pair[f'anchor_TWS_t_nbr{s}'].to_numpy(np.float32)
        x[f'anchor_local_contrast{s}']=x.anchor_tws-x[f'anchor_TWS_t_nbr{s}']
    return x

def main():
    tr,_,_=load_data(Path('.'),need_test=False);tr=augment(tr)
    cutoff=pd.Timestamp('2012-07-01');fit=tr[tr.time<cutoff].copy();stats=fit_history_stats(fit)
    anchors=[('2012-07-01',3),('2012-12-01',3),('2013-05-01',3),('2013-11-01',3),('2014-04-01',3),('2014-09-01',7)]
    # Custom validation in anchor/horizon order.
    vxs=[];vrs=[];ys=[];hs=[];by={int(k):v for k,v in tr.groupby('month_idx',sort=False)}
    for d,mh in anchors:
        ai=pd.Timestamp(d).year*12+pd.Timestamp(d).month-1
        for h in range(1,mh+1):
            pair=make_pair(tr,h);pair=pair[(pair.month_idx==ai+h-1)&(pair['anchor_TWS_t'].notna())]
            x,r,y=pair_to_xy(pair.rename(columns={'anchor_TWS_t':'anchor_TWS_t_KEEP'}),stats,h) if False else (None,None,None)
            # pair_to_xy expects anchor_TWS_t, already present.
            from winning_solution import pair_to_xy as pxy
            x,r,y=pxy(pair,stats,h);x=add_spatial_x(x,pair)
            vxs.append(x);vrs.append(r);ys.append(y);hs.append(np.full(len(pair),h))
    vx=pd.concat(vxs,ignore_index=True);vr=np.concatenate(vrs);y=np.concatenate(ys);vh=np.concatenate(hs)
    xs=[];yrs=[]
    for h in range(1,8):
        pair=make_pair(fit,h)
        if len(pair)>60000:pair=pair.sample(60000,random_state=SEED+h)
        from winning_solution import pair_to_xy as pxy
        x,r,_=pxy(pair,stats,h);xs.append(add_spatial_x(x,pair));yrs.append(r);del pair;gc.collect()
    x=pd.concat(xs,ignore_index=True);yr=np.concatenate(yrs)
    cfg=ModelConfig(direct_estimators=500,n_jobs=20)
    model=lgb.LGBMRegressor(**lgb_params(cfg,500)).fit(x,yr,eval_set=[(vx,vr)],
        callbacks=[lgb.early_stopping(80),lgb.log_evaluation(50)])
    pred=vx.anchor_tws.to_numpy()+model.predict(vx)
    print('overall',rmse(y,pred),{h:rmse(y[vh==h],pred[vh==h]) for h in range(1,8)})

if __name__=='__main__':main()
