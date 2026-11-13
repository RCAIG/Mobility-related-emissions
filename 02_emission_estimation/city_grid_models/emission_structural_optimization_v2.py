"""Zero hurdle, target-free mobility features and partial-pooling region models."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_absolute_error,roc_auc_score,brier_score_loss
from tabpfn_emission_optimization import load
from mobility_distance_bin_audit import FINE,mobility_features
ROOT=Path('${SERVER_HOME}/result')
OUT=ROOT/'emission_structural_optimization_v2'
def reg(seed):
    return HistGradientBoostingRegressor(max_iter=500,max_leaf_nodes=31,min_samples_leaf=40,
        learning_rate=.04,l2_regularization=15,random_state=seed)
def run(scope):
    d,base,y,groups=load(scope)
    extra=pd.read_parquet(OUT/'target_free_mobility_features.parquet')
    assert np.array_equal(d[['cell_id','year']].to_numpy(),extra[['cell_id','year']].to_numpy())
    fine=extra[FINE].to_numpy(float)
    share=fine/np.maximum(fine.sum(axis=1)[:,None],1e-9)
    mid=np.array([.5,1.5,2.5,4,6.5,9,12.5,17.5,25,35,50,80,150])
    # These are activity-weighted route-length proxies; open-ended last bin uses 150 km.
    shape=np.column_stack([share[:,:4].sum(axis=1),share[:,8:].sum(axis=1),share[:,10:].sum(axis=1),
        share@np.log1p(mid),(share**2).sum(axis=1),np.cumsum(share,axis=1)[:,:-1]])
    road_cols=[c for c in extra if c not in ['cell_id','year','total']+FINE]
    extended=np.column_stack([base,mobility_features(fine,expanded=True),shape,extra[road_cols].to_numpy()]).astype(np.float32)
    extended[~np.isfinite(extended)]=np.nan
    regions=[c for c in d if c.startswith('region_')]
    regionvalues=d[regions].to_numpy(float)
    region=np.argmax(regionvalues,axis=1)
    region[regionvalues.sum(axis=1)<=0]=-1
    for seed in [17,42,83,20261006]:
        folder=OUT/scope/str(seed);folder.mkdir(parents=True,exist_ok=True)
        pool,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(base,groups=groups))
        it,iv=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=731).split(base[pool],groups=groups[pool]))
        train,val=pool[it],pool[iv]
        assert set(groups[train]).isdisjoint(groups[test]) and set(groups[val]).isdisjoint(groups[test])
        if seed!=20261006:
            old=pd.read_csv(ROOT/'emission_outlier_robustness_v1'/scope/str(seed)/'splits.csv')
            assert np.array_equal(old.row,np.r_[train,val,test])
        pd.DataFrame({'row':np.r_[train,val,test],'subset':['train']*len(train)+['validation']*len(val)+['test']*len(test),
            'group':groups[np.r_[train,val,test]]}).to_csv(folder/'splits.csv',index=False)
        rows=[];pv={};pt={}
        def save(name,v,t,**params):
            t=np.maximum(t,0);v=np.maximum(v,0)
            pv[name]=v;pt[name]=t
            rows.append({'scope':scope,'seed':seed,'model':name,'r2_validation_log':r2_score(y[val],v),
                'r2_test_log':r2_score(y[test],t),'r2_test_original_scale':r2_score(np.expm1(y[test].astype(float)),np.expm1(t.astype(float))),
                'mae_test_log':mean_absolute_error(y[test],t),'n_test':len(test),**params})
            pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
            print(json.dumps(rows[-1],default=float),flush=True)
        im=SimpleImputer(strategy='median',keep_empty_features=True)
        a=im.fit_transform(base[train]);v=im.transform(base[val]);t=im.transform(base[test])
        baseline=reg(seed).fit(a,y[train]);save('Baseline_matched',baseline.predict(v),baseline.predict(t))
        im=SimpleImputer(strategy='median',keep_empty_features=True)
        a=im.fit_transform(extended[train]);v=im.transform(extended[val]);t=im.transform(extended[test])
        global_model=reg(seed).fit(a,y[train])
        gv=global_model.predict(v);gt=global_model.predict(t)
        save('Mobility_restructured',gv,gt)
        positive=y[train]>0
        clf=HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=20,min_samples_leaf=60,
            learning_rate=.04,l2_regularization=20,random_state=seed).fit(a,positive)
        cpv=clf.predict_proba(v)[:,1];cpt=clf.predict_proba(t)[:,1]
        posmodel=reg(seed).fit(a[positive],y[train][positive])
        vp=np.maximum(posmodel.predict(v),0);tp=np.maximum(posmodel.predict(t),0)
        save('Hurdle_expected_log',cpv*vp,cpt*tp)
        opts=[]
        for threshold in [.1,.25,.5,.75,.9]:
            opts.append((r2_score(y[val],np.where(cpv>=threshold,vp,0)),threshold))
        _,threshold=max(opts)
        save('Hurdle_hard_validation',np.where(cpv>=threshold,vp,0),np.where(cpt>=threshold,tp,0),threshold=threshold)
        # Raw expected emission and expected log are different estimands.
        save('Hurdle_expected_raw',np.log1p(cpv*np.expm1(vp)),np.log1p(cpt*np.expm1(tp)))
        pd.DataFrame([{'validation_auc':roc_auc_score(y[val]>0,cpv),'test_auc':roc_auc_score(y[test]>0,cpt),
            'test_brier':brier_score_loss(y[test]>0,cpt),'train_zero_fraction':float((~positive).mean()),
            'test_zero_fraction':float((y[test]==0).mean())}]).to_csv(folder/'zero_classification.csv',index=False)
        lv=gv.copy();lt=gt.copy();region_audit=[]
        for rid in np.unique(region):
            mask=region[train]==rid;vm=region[val]==rid;tm=region[test]==rid
            n=int(mask.sum());weight=n/(n+2000)
            used=n>=500
            if used:
                local=reg(seed).fit(a[mask],y[train][mask])
                if vm.any():lv[vm]=weight*local.predict(v[vm])+(1-weight)*gv[vm]
                if tm.any():lt[tm]=weight*local.predict(t[tm])+(1-weight)*gt[tm]
            region_audit.append({'region':regions[rid] if rid>=0 else 'unassigned','n_train':n,'local_used':used,'pooling_weight':weight})
        candidates=[(r2_score(y[val],np.maximum(gv+w*(lv-gv),0)),w) for w in [0,.25,.5,.75,1]]
        _,rw=max(candidates)
        save('Regional_partial_pooling',gv+rw*(lv-gv),gt+rw*(lt-gt),region_blend_weight=rw)
        pd.DataFrame(region_audit).to_csv(folder/'region_support.csv',index=False)
        # Family and combination are selected on validation, never test.
        cand=[];keys=list(pv)
        for i,left in enumerate(keys):
            for right in keys[i:]:
                for w in [0,.25,.5,.75,1]:
                    cand.append((r2_score(y[val],w*pv[left]+(1-w)*pv[right]),left,right,w))
        _,left,right,w=max(cand,key=lambda z:z[0])
        save('Validation_selected_combination',w*pv[left]+(1-w)*pv[right],w*pt[left]+(1-w)*pt[right],left=left,right=right,weight_left=w)
        # Separately evaluate calibration using training-only OOF regional bias.
        oof=np.full(len(train),np.nan)
        for ft,fv in GroupKFold(3).split(extended[train],groups=groups[train]):
            oi=SimpleImputer(strategy='median',keep_empty_features=True)
            om=reg(seed).fit(oi.fit_transform(extended[train[ft]]),y[train[ft]])
            oof[fv]=om.predict(oi.transform(extended[train[fv]]))
        residual=y[train]-oof
        lon=np.radians(d.lon.to_numpy());lat=np.radians(d.lat.to_numpy())
        xyz=6371*np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
        candidates=[]
        for ys in [0,300,1000]:
            loc=np.column_stack([xyz,(d.year.to_numpy()-2017)/7*ys])
            for k in [10,30,100,300]:
                knn=KNeighborsRegressor(n_neighbors=k,weights='distance',n_jobs=8).fit(loc[train],residual)
                cv=knn.predict(loc[val]);ct=knn.predict(loc[test])
                for cw in [0,.25,.5,.75,1]:
                    prediction=np.maximum(pv['Validation_selected_combination']+cw*cv,0)
                    candidates.append((r2_score(y[val],prediction),ys,k,cw,cv,ct))
        _,ys,k,cw,cv,ct=max(candidates,key=lambda z:z[0])
        save('Combination_plus_spatial_calibration',pv['Validation_selected_combination']+cw*cv,
            pt['Validation_selected_combination']+cw*ct,year_scale_km=ys,neighbors=k,correction_weight=cw)
        pd.DataFrame({'row':train,'observed_log':y[train],'mobility_oof_log':oof,'residual_log':residual}).to_csv(folder/'training_calibration_audit.csv',index=False)
        pd.DataFrame({'row':test,'observed_log':y[test],**pt}).to_csv(folder/'test_predictions.csv',index=False)
        pd.DataFrame({'row':val,'observed_log':y[val],**pv}).to_csv(folder/'validation_predictions.csv',index=False)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',required=True)
    run(parser.parse_args().scope)
