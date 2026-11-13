"""Training OOF residual spatial smoothing; validation tunes, test stays intact."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_absolute_error
from tabpfn_emission_optimization import load
ROOT=Path('${SERVER_HOME}/result')
OUT=ROOT/'emission_spatial_calibration_v1'
for scope in ['grid_random','grid_block']:
    d,x,y,groups=load(scope)
    for seed in [17,42,83]:
        source=ROOT/'emission_outlier_robustness_v1'/scope/str(seed)
        if not (source/'test_predictions.csv').exists():
            raise RuntimeError(f'Missing training audit {source}')
        audit=pd.read_csv(source/'training_outlier_audit.csv')
        split=pd.read_csv(source/'splits.csv')
        train=audit.row.to_numpy();val=split[split.subset.eq('validation')].row.to_numpy()
        test=split[split.subset.eq('test')].row.to_numpy()
        from emission_outlier_robustness import tree
        from sklearn.impute import SimpleImputer
        imp=SimpleImputer(strategy='median',keep_empty_features=True)
        model=tree(seed);model.fit(imp.fit_transform(x[train]),y[train])
        baseline_v=model.predict(imp.transform(x[val]));baseline_t=model.predict(imp.transform(x[test]))
        residual=audit.observed_log.to_numpy()-audit.oof_prediction_log.to_numpy()
        lon=np.radians(d.lon.to_numpy());lat=np.radians(d.lat.to_numpy())
        xyz=np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])*6371
        rows=[];candidate=[]
        for year_scale in [0,300,1000]:
            z=np.column_stack([xyz,(d.year.to_numpy()-2017)/7*year_scale])
            for k in [10,30,100,300]:
                knn=KNeighborsRegressor(n_neighbors=k,weights='distance',n_jobs=8)
                knn.fit(z[train],residual)
                cv=knn.predict(z[val]);ct=knn.predict(z[test])
                for weight in [0,.25,.5,.75,1]:
                    candidate.append((r2_score(y[val],baseline_v+weight*cv),year_scale,k,weight,cv,ct))
        score,ys,k,w,cv,ct=max(candidate,key=lambda t:t[0])
        folder=OUT/scope/str(seed);folder.mkdir(parents=True,exist_ok=True)
        for name,vpred,tpred in [('HGB_full',baseline_v,baseline_t),('Validation_spatial_calibration',baseline_v+w*cv,baseline_t+w*ct)]:
            rows.append({'scope':scope,'seed':seed,'model':name,'r2_validation_log':r2_score(y[val],vpred),
                'r2_test_log':r2_score(y[test],tpred),'r2_test_original_scale':r2_score(np.expm1(y[test].astype(float)),np.expm1(tpred)),
                'mae_test_log':mean_absolute_error(y[test],tpred),'neighbors':k,'year_scale_km':ys,'correction_weight':w})
        ensemble=ROOT/'emission_spatial_ensemble_v1'/scope/str(seed)
        ve=pd.read_csv(ensemble/'validation_predictions.csv')
        te=pd.read_csv(ensemble/'test_predictions.csv')
        assert np.array_equal(ve.row.to_numpy(),val) and np.array_equal(te.row.to_numpy(),test)
        blend_v=ve.Validation_selected_blend.to_numpy();blend_t=te.Validation_selected_blend.to_numpy()
        best=max(candidate,key=lambda t:r2_score(y[val],blend_v+t[3]*t[4]))
        _,ys2,k2,w2,cv2,ct2=best
        calibrated_blend=blend_t+w2*ct2
        rows.append({'scope':scope,'seed':seed,'model':'Validation_spatial_calibration_blend',
            'r2_validation_log':r2_score(y[val],blend_v+w2*cv2),'r2_test_log':r2_score(y[test],calibrated_blend),
            'r2_test_original_scale':r2_score(np.expm1(y[test].astype(float)),np.expm1(calibrated_blend)),
            'mae_test_log':mean_absolute_error(y[test],calibrated_blend),'neighbors':k2,'year_scale_km':ys2,'correction_weight':w2})
        pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
        pd.DataFrame({'row':test,'observed_log':y[test],'baseline_log':baseline_t,'calibrated_log':baseline_t+w*ct,
            'calibrated_blend_log':calibrated_blend}).to_csv(folder/'test_predictions.csv',index=False)
        pd.DataFrame({'row':val,'observed_log':y[val],'baseline_log':baseline_v,'calibrated_log':baseline_v+w*cv,
            'calibrated_blend_log':blend_v+w2*cv2}).to_csv(folder/'validation_predictions.csv',index=False)
        print(rows,flush=True)
