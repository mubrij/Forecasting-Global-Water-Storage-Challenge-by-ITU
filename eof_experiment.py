#!/usr/bin/env python3
"""Low-rank global-field model experiment for TWS."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from winning_solution import CLIMATE, load_data, rmse


def dense_maps(df, times, locs, col, fill):
    ti = {int(t): i for i, t in enumerate(times)}
    li = {int(k): i for i, k in enumerate(locs)}
    out = np.broadcast_to(np.asarray(fill, np.float32), (len(times), len(locs))).copy()
    rows = df[df.month_idx.isin(times)]
    ii = rows.month_idx.map(ti).to_numpy(np.int32)
    jj = rows.loc_key.map(li).to_numpy(np.int32)
    out[ii, jj] = rows[col].to_numpy(np.float32)
    return out


def fit_pca(x, rank):
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    _, _, vt = np.linalg.svd((x - mean).astype(np.float64), full_matrices=False)
    basis = vt[:rank].astype(np.float32)
    return mean, basis


def score(x, mean, basis):
    return (x - mean) @ basis.T


def main():
    tr, _, _ = load_data(Path('.'), need_test=False)
    cutoff = pd.Timestamp('2012-07-01')
    cutoff_idx = cutoff.year * 12 + cutoff.month - 1
    fit = tr[tr.month_idx < cutoff_idx]
    locs = np.sort(tr.loc_key.unique())
    times = np.sort(fit.month_idx.unique())
    loc_mean = fit.groupby('loc_key').TWS_t.mean().reindex(locs).fillna(fit.TWS_t.mean()).to_numpy(np.float32)
    tws = dense_maps(fit, times, locs, 'TWS_t', loc_mean)
    target = dense_maps(fit, times, locs, 'Target', loc_mean)
    climate = {c: dense_maps(fit, times, locs, c, np.zeros(len(locs), np.float32)) for c in CLIMATE}

    anchors = [('2012-07-01',3),('2012-12-01',3),('2013-05-01',3),
               ('2013-11-01',3),('2014-04-01',3),('2014-09-01',7)]
    val_rows = []
    for ad, mh in anchors:
        ai = pd.Timestamp(ad).year * 12 + pd.Timestamp(ad).month - 1
        for h in range(1, mh+1):
            cur_idx = ai + h - 1
            cur = tr[tr.month_idx == cur_idx]
            anc = tr[tr.month_idx == ai][['loc_key','TWS_t']].rename(columns={'TWS_t':'anchor'})
            val_rows.append((h, cur.merge(anc,on='loc_key',how='inner')))

    for rank in [10, 20, 35, 50]:
        ymean, ybasis = fit_pca(np.vstack([tws, target]), rank)
        tws_s = score(tws, ymean, ybasis)
        target_s = score(target, ymean, ybasis)
        cbases = {}
        cscores = {}
        for c in CLIMATE:
            cm, cb = fit_pca(climate[c], min(rank, 25))
            cbases[c] = (cm, cb)
            cscores[c] = score(climate[c], cm, cb)
        time_pos = {int(t): i for i,t in enumerate(times)}
        preds=[]; truths=[]; hs=[]
        for h in range(1,8):
            train_pairs=[]
            for ci,t in enumerate(times):
                ai = time_pos.get(int(t-h+1))
                if ai is not None:
                    train_pairs.append((ai,ci))
            ai=np.array([z[0] for z in train_pairs]); ci=np.array([z[1] for z in train_pairs])
            angles=2*np.pi*((times[ci]+1)%12)/12
            x=np.column_stack([tws_s[ai], *[cscores[c][ci] for c in CLIMATE], np.sin(angles),np.cos(angles)])
            y=target_s[ci]
            model=Ridge(alpha=100.0).fit(x,y)
            for vh, rows in val_rows:
                if vh != h: continue
                amap=loc_mean.copy(); cmap={c:np.zeros(len(locs),np.float32) for c in CLIMATE}
                lmap={int(k):i for i,k in enumerate(locs)}
                jj=rows.loc_key.map(lmap).to_numpy(np.int32)
                amap[jj]=rows.anchor.to_numpy(np.float32)
                for c in CLIMATE: cmap[c][jj]=rows[c].to_numpy(np.float32)
                target_month=(int(rows.month_idx.iloc[0])+1)%12
                ang=2*np.pi*target_month/12
                xx=np.concatenate([score(amap[None],ymean,ybasis)[0],
                    *[score(cmap[c][None],*cbases[c])[0] for c in CLIMATE],[np.sin(ang),np.cos(ang)]])[None]
                field=ymean+model.predict(xx)[0]@ybasis
                preds.append(field[jj]); truths.append(rows.Target.to_numpy()); hs.append(np.full(len(rows),h))
        pred=np.concatenate(preds); truth=np.concatenate(truths); hh=np.concatenate(hs)
        print('rank',rank,'rmse',rmse(truth,pred))
        print({h:rmse(truth[hh==h],pred[hh==h]) for h in range(1,8)})

if __name__=='__main__': main()
