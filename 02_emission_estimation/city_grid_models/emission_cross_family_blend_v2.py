"""Combine old TabPFN ensemble and new structural strategies by validation."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from tabpfn_emission_optimization import load
ROOT=Path('${SERVER_HOME}/result')
OUT=ROOT/'emission_structural_optimization_v2'
records=[]
for scope in ['grid_random','grid_block']:
    d,x,y,groups=load(scope)
    lon=np.radians(d.lon.to_numpy());lat=np.radians(d.lat.to_numpy())
    xyz=6371*np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
    for seed in [17,42,83,20261006]:
        folder=OUT/scope/str(seed)
        nv=pd.read_csv(folder/'validation_predictions.csv');nt=pd.read_csv(folder/'test_predictions.csv')
        prior=ROOT/'emission_spatial_ensemble_v1'/scope/str(seed)
        ov=pd.read_csv(prior/'validation_predictions.csv');ot=pd.read_csv(prior/'test_predictions.csv')
        assert np.array_equal(nv.row,ov.row) and np.array_equal(nt.row,ot.row)
        val=nv.row.to_numpy();test=nt.row.to_numpy()
        vp={'Prior_TabPFN_tree_blend':np.maximum(ov.Validation_selected_blend.to_numpy(),0),
            'Structural_combination':nv.Validation_selected_combination.to_numpy()}
        tp={'Prior_TabPFN_tree_blend':np.maximum(ot.Validation_selected_blend.to_numpy(),0),
            'Structural_combination':nt.Validation_selected_combination.to_numpy()}
        rows=[]
        def save(name,v,t,**params):
            rows.append({'scope':scope,'seed':seed,'model':name,'r2_validation_log':r2_score(y[val],v),
                'r2_test_log':r2_score(y[test],t),'r2_test_original_scale':r2_score(np.expm1(y[test].astype(float)),np.expm1(t)),
                'mae_test_log':mean_absolute_error(y[test],t),**params})
            vp[name]=v;tp[name]=t
        candidates=[(r2_score(y[val],w*vp['Prior_TabPFN_tree_blend']+(1-w)*vp['Structural_combination']),w) for w in [0,.25,.5,.75,1]]
        _,w=max(candidates)
        save('Cross_family_validation_blend',w*vp['Prior_TabPFN_tree_blend']+(1-w)*vp['Structural_combination'],
            w*tp['Prior_TabPFN_tree_blend']+(1-w)*tp['Structural_combination'],prior_weight=w)
        audit=pd.read_csv(folder/'training_calibration_audit.csv');train=audit.row.to_numpy()
        cand=[]
        for ys in [0,300,1000]:
            z=np.column_stack([xyz,(d.year.to_numpy()-2017)/7*ys])
            for k in [10,30,100,300]:
                knn=KNeighborsRegressor(n_neighbors=k,weights='distance',n_jobs=8).fit(z[train],audit.residual_log)
                cv=knn.predict(z[val]);ct=knn.predict(z[test])
                for cw in [0,.25,.5,.75,1]:
                    cand.append((r2_score(y[val],np.maximum(vp['Cross_family_validation_blend']+cw*cv,0)),ys,k,cw,cv,ct))
        _,ys,k,cw,cv,ct=max(cand,key=lambda z:z[0])
        save('Cross_family_plus_calibration',np.maximum(vp['Cross_family_validation_blend']+cw*cv,0),
            np.maximum(tp['Cross_family_validation_blend']+cw*ct,0),neighbors=k,year_scale_km=ys,correction_weight=cw,prior_weight=w)
        records.extend(rows)
        pd.DataFrame({'row':test,'observed_log':y[test],**tp}).to_csv(folder/'cross_family_test_predictions.csv',index=False)
        pd.DataFrame(records).to_csv(OUT/'cross_family_metrics.csv',index=False)
        print(rows,flush=True)
pd.DataFrame(records).groupby(['scope','model']).agg(r2_log_mean=('r2_test_log','mean'),
    r2_raw_mean=('r2_test_original_scale','mean')).to_csv(OUT/'cross_family_four_split_summary.csv')
