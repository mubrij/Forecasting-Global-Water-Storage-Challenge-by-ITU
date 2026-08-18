#!/usr/bin/env python3
"""CatBoost location-embedding experiment on the chronological holdout."""
from pathlib import Path
import gc
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from winning_solution import (load_data, fit_history_stats, validation_pairs,
    make_pair_frame, pair_to_xy, rmse, optimal_blend, SEED)


def main():
    tr,_,_=load_data(Path('.'),need_test=False)
    cutoff=pd.Timestamp('2012-07-01'); fit=tr[tr.time<cutoff].copy(); stats=fit_history_stats(fit)
    anchors=[('2012-07-01',3),('2012-12-01',3),('2013-05-01',3),
             ('2013-11-01',3),('2014-04-01',3),('2014-09-01',7)]
    vx,vr,y,vh,_=validation_pairs(tr,stats,anchors)
    # Reconstruct validation loc keys in precisely the validation_pairs order.
    vkeys=[]; by={int(k):v for k,v in tr.groupby('month_idx',sort=False)}
    for d,mh in anchors:
        ai=pd.Timestamp(d).year*12+pd.Timestamp(d).month-1; ak=by[ai][['loc_key']]
        for h in range(1,mh+1):
            cur=by.get(ai+h-1)
            if cur is None: break
            vkeys.append(cur[['loc_key']].merge(ak,on='loc_key',how='inner').loc_key.to_numpy())
    vx['loc_key']=np.concatenate(vkeys).astype(str)
    xs=[]; ys=[]
    for h in range(1,8):
        pair=make_pair_frame(fit,h)
        if len(pair)>100000: pair=pair.sample(100000,random_state=SEED+h)
        x,r,_=pair_to_xy(pair,stats,h); x['loc_key']=pair.loc_key.to_numpy().astype(str)
        xs.append(x);ys.append(r); del pair;gc.collect()
    x=pd.concat(xs,ignore_index=True); yy=np.concatenate(ys)
    model=CatBoostRegressor(iterations=1000,depth=9,learning_rate=.05,loss_function='RMSE',
        l2_leaf_reg=8,random_seed=SEED,task_type='GPU',devices='0',verbose=100,
        od_type='Iter',od_wait=100,allow_writing_files=False)
    model.fit(Pool(x,yy,cat_features=['loc_key']),eval_set=Pool(vx,vr,cat_features=['loc_key']))
    pred=vx.anchor_tws.to_numpy()+model.predict(vx)
    print('overall',rmse(y,pred))
    print({h:rmse(y[vh==h],pred[vh==h]) for h in range(1,8)})

if __name__=='__main__':main()
