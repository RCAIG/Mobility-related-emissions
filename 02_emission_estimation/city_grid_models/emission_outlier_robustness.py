"""Training-only outlier sensitivity with untouched grouped external holdouts."""
from pathlib import Path
import argparse
import gc
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from tabpfn import TabPFNRegressor
from tabpfn.finetuning import FinetunedTabPFNRegressor
from tabpfn_emission_optimization import load, predict_batches, WEIGHT, OUT as BASE

OUT = Path('${SERVER_HOME}/result/emission_outlier_robustness_v1')

def tree(seed, loss='squared_error'):
    return HistGradientBoostingRegressor(max_iter=350, max_leaf_nodes=20,
        min_samples_leaf=80, l2_regularization=20, learning_rate=.035,
        random_state=seed, loss=loss)

def run(scope, device, seed):
    completed = OUT / scope / str(seed) / 'metrics.csv'
    if completed.exists() and len(pd.read_csv(completed)) == 11:
        return
    d, x, y, groups = load(scope)
    pool, test = next(GroupShuffleSplit(n_splits=1, test_size=.2 if scope.startswith('grid') else .25,
        random_state=seed).split(x, groups=groups))
    it, iv = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=731).split(x[pool], groups=groups[pool]))
    train, val = pool[it], pool[iv]
    folder = OUT / scope / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    split = pd.DataFrame({'row': np.r_[train,val,test],
        'subset': ['train']*len(train)+['validation']*len(val)+['test']*len(test),
        'group': groups[np.r_[train,val,test]]})
    split.to_csv(folder / 'splits.csv', index=False)
    old_split = pd.read_csv(BASE / scope / str(seed) / 'splits.csv')
    assert split.row.tolist() == old_split.row.tolist()
    assert split.subset.tolist() == old_split.subset.tolist()
    # Each residual is predicted by a model that never saw that row's target.
    oof = np.full(len(train), np.nan)
    for ft, fv in GroupKFold(3).split(x[train], groups=groups[train]):
        imp = SimpleImputer(strategy='median', keep_empty_features=True)
        a = imp.fit_transform(x[train[ft]])
        b = imp.transform(x[train[fv]])
        reg = tree(seed)
        reg.fit(a, y[train[ft]])
        oof[fv] = reg.predict(b)
    error = np.abs(y[train]-oof)
    assert np.isfinite(error).all()
    order = np.argsort(error, kind='stable')
    trim_masks = {}
    for fraction in [.005, .01, .02]:
        keep = np.ones(len(train), dtype=bool)
        keep[order[-int(np.ceil(len(train)*fraction)):]] = False
        trim_masks[f'Trim_{fraction*100:g}pct'] = keep
    audit = pd.DataFrame({'row':train, 'group':groups[train], 'observed_log':y[train],
        'oof_prediction_log':oof, 'absolute_oof_error_log':error})
    for name, keep in trim_masks.items():
        audit[name+'_excluded'] = ~keep
    for col in ['ID_UC_G0','GC_UCN_MAI_2025','GC_CNT_GAD_2025','year','cell_id','lon','lat','population','gdp','area']:
        if col in d:
            audit[col] = d.iloc[train][col].to_numpy()
    audit.to_csv(folder / 'training_outlier_audit.csv', index=False)
    rows = []
    pred = pd.DataFrame({'row':test,'observed_log':y[test]})
    def record(name, pval, ptest, nfit, removed=0):
        observed_raw = np.expm1(y[test].astype(float))
        predicted_raw = np.expm1(ptest.astype(float))
        row = {'scope':scope,'seed':seed,'model':name,'n_train_before':len(train),
            'n_train_used':nfit,'n_removed':removed,'removed_fraction':removed/len(train),
            'n_validation':len(val),'n_test':len(test),
            'r2_validation_log':r2_score(y[val],pval), 'r2_test_log':r2_score(y[test],ptest),
            'r2_test_original_scale':r2_score(observed_raw,predicted_raw),
            'mae_test_log':mean_absolute_error(y[test],ptest)}
        e = (y[test]-ptest)**2
        raw_e = (observed_raw-predicted_raw)**2
        k = max(1,int(np.ceil(len(test)*.01)))
        row['top1pct_test_sse_share_log'] = np.sort(e)[-k:].sum()/max(e.sum(),1e-12)
        row['top1pct_test_sse_share_original'] = np.sort(raw_e)[-k:].sum()/max(raw_e.sum(),1e-12)
        rows.append(row)
        pred[name] = ptest
        pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
        pred.to_csv(folder/'test_predictions.csv',index=False)
        print(json.dumps(row,default=float),flush=True)
    options = [('HGB_full',np.ones(len(train),bool),'squared_error',False),
        ('HGB_absolute_loss',np.ones(len(train),bool),'absolute_error',False),
        ('HGB_downweight',np.ones(len(train),bool),'squared_error',True)]
    options += [('HGB_'+name,mask,'squared_error',False) for name,mask in trim_masks.items()]
    for name, keep, loss, weighted in options:
        imp = SimpleImputer(strategy='median',keep_empty_features=True)
        a = imp.fit_transform(x[train[keep]])
        v,b = imp.transform(x[val]), imp.transform(x[test])
        reg = tree(seed,loss)
        weights = np.where(trim_masks['Trim_2pct'],1.,.2) if weighted else None
        reg.fit(a,y[train[keep]],sample_weight=weights)
        record(name,reg.predict(v),reg.predict(b),int(keep.sum()),int((~keep).sum()))
    # Try each mild threshold on validation; retain all external results for audit.
    tab_options = [('TabPFN_full',np.ones(len(train),bool))]
    tab_options += [('TabPFN_'+name,mask) for name,mask in trim_masks.items()]
    for name,keep in tab_options:
        imp = SimpleImputer(strategy='median',keep_empty_features=True)
        a = imp.fit_transform(x[train[keep]])
        v,b = imp.transform(x[val]),imp.transform(x[test])
        context = np.random.default_rng(seed).choice(len(a), min(len(a),4096),replace=False)
        reg = TabPFNRegressor(n_estimators=4,device=device,model_path=WEIGHT,
            ignore_pretraining_limits=True,random_state=seed,fit_mode='fit_preprocessors')
        reg.fit(a[context],y[train[keep]][context])
        record(name,predict_batches(reg,v),predict_batches(reg,b),int(keep.sum()),int((~keep).sum()))
        del reg
        gc.collect()
    best = max([r for r in rows if r['model'].startswith('TabPFN_')],key=lambda r:r['r2_validation_log'])
    key = best['model'].replace('TabPFN_','')
    keep = trim_masks.get(key,np.ones(len(train),bool))
    imp = SimpleImputer(strategy='median',keep_empty_features=True)
    a = imp.fit_transform(x[train[keep]])
    v,b = imp.transform(x[val]),imp.transform(x[test])
    reg = FinetunedTabPFNRegressor(device=device,epochs=8,time_limit=1200,
        learning_rate=1e-5,weight_decay=.05,early_stopping=True,early_stopping_patience=3,
        n_finetune_ctx_plus_query_samples=512,finetune_ctx_query_split_ratio=.25,
        n_inference_subsample_samples=4096,n_estimators_finetune=1,n_estimators_validation=2,
        n_estimators_final_inference=4,random_state=seed,
        extra_regressor_kwargs={'model_path':WEIGHT,'ignore_pretraining_limits':True})
    reg.fit(a,y[train[keep]],X_val=v,y_val=y[val],output_dir=folder/'finetuned')
    record('TabPFN_finetuned_validation_rule',predict_batches(reg,v),predict_batches(reg,b),int(keep.sum()),int((~keep).sum()))
    (folder/'selected_training_rule.json').write_text(json.dumps(best,indent=2,default=float))
    del reg
    gc.collect()
    import torch
    torch.cuda.empty_cache()

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--scope',required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    for seed in [17,42,83]:
        run(args.scope,args.device,seed)
