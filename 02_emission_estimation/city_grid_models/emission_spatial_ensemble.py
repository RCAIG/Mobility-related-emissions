"""Validation-selected spatial regression and blending, with intact tests."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from tabpfn import TabPFNRegressor
from tabpfn_emission_optimization import load, WEIGHT, predict_batches

OUT=Path('${SERVER_HOME}/result/emission_spatial_ensemble_v1')
def run(scope,device,seeds=None):
    d,x,y,groups=load(scope)
    for seed in ([17,42,83] if seeds is None else seeds):
        pool,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(x,groups=groups))
        it,iv=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=731).split(x[pool],groups=groups[pool]))
        train,val=pool[it],pool[iv]
        imp=SimpleImputer(strategy='median',keep_empty_features=True)
        a=imp.fit_transform(x[train]);v=imp.transform(x[val]);b=imp.transform(x[test])
        folder=OUT/scope/str(seed);folder.mkdir(parents=True,exist_ok=True)
        rows=[];pv={};pt={}
        def save(name,vpred,tpred):
            pv[name]=vpred;pt[name]=tpred
            rows.append({'scope':scope,'seed':seed,'model':name,'r2_validation_log':r2_score(y[val],vpred),
                'r2_test_log':r2_score(y[test],tpred),'mae_test_log':mean_absolute_error(y[test],tpred),
                'r2_test_original_scale':r2_score(np.expm1(y[test].astype(float)),np.expm1(tpred.astype(float)))})
            pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
            print(rows[-1],flush=True)
        models={
            'HGB_medium':HistGradientBoostingRegressor(max_iter=700,max_leaf_nodes=31,min_samples_leaf=40,l2_regularization=10,learning_rate=.035,random_state=seed),
            'HGB_flexible':HistGradientBoostingRegressor(max_iter=900,max_leaf_nodes=63,min_samples_leaf=25,l2_regularization=20,learning_rate=.025,random_state=seed),
            'ExtraTrees_regularized':ExtraTreesRegressor(n_estimators=300,min_samples_leaf=5,max_features=.8,n_jobs=8,random_state=seed)}
        for name,m in models.items():
            m.fit(a,y[train]);save(name,m.predict(v),m.predict(b))
        reg=TabPFNRegressor(n_estimators=8,device=device,model_path=WEIGHT,ignore_pretraining_limits=True,
            random_state=seed,fit_mode='fit_preprocessors')
        keep=np.random.default_rng(seed).choice(len(a),min(len(a),8192),replace=False)
        reg.fit(a[keep],y[train][keep]);save('TabPFN_8k_8estimators',predict_batches(reg,v),predict_batches(reg,b))
        # Configuration and blend weights selected strictly on internal validation.
        candidates=[]
        keys=list(pv)
        for i,left in enumerate(keys):
            for right in keys[i:]:
                for w in [0,.25,.5,.75,1]:
                    pred=w*pv[left]+(1-w)*pv[right]
                    candidates.append((r2_score(y[val],pred),left,right,w))
        _,left,right,w=max(candidates,key=lambda z:z[0])
        save('Validation_selected_blend',w*pv[left]+(1-w)*pv[right],w*pt[left]+(1-w)*pt[right])
        pd.DataFrame([{'left':left,'right':right,'weight_left':w}]).to_csv(folder/'blend_rule.csv',index=False)
        pd.DataFrame({'row':test,'observed_log':y[test],**pt}).to_csv(folder/'test_predictions.csv',index=False)
        pd.DataFrame({'row':val,'observed_log':y[val],**pv}).to_csv(folder/'validation_predictions.csv',index=False)
        del reg
        import gc,torch
        gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',required=True);p.add_argument('--device',required=True)
    p.add_argument('--seed',type=int)
    args=p.parse_args();run(args.scope,args.device,None if args.seed is None else [args.seed])
