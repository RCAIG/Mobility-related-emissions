"""Compare regularization and genuine TabPFN fine-tuning on grouped holdouts."""
from pathlib import Path
import argparse
import gc
import json
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from tabpfn import TabPFNRegressor
from tabpfn.finetuning import FinetunedTabPFNRegressor
from mobility_distance_bin_audit import SOCIO, ROAD, mobility_features

ROOT = Path('${SERVER_HOME}/result')
OUT = ROOT / 'tabpfn_emission_optimization_v1'
WEIGHT = '${TABPFN_CACHE_ROOT}/tabpfn-v2.5-regressor-v2.5_default.ckpt'

def load(scope):
    if scope.startswith('grid'):
        d = pd.read_parquet(ROOT / 'spatial_mobility_carbon_validation_v2/analysis/global_cell_features_v2.parquet')
        for direction in ['lon', 'lat']:
            d[f'sin_{direction}'] = np.sin(np.radians(d[direction]))
            d[f'cos_{direction}'] = np.cos(np.radians(d[direction]))
        cols = ['weighted_log_population', 'weighted_log_area', 'weighted_log_gdp',
                'sin_lon', 'cos_lon', 'sin_lat', 'cos_lat', 'city_contribution_hhi',
                'contributing_cities'] + [c for c in d if c.startswith(('dev_', 'region_'))]
        context = np.column_stack([d[cols], d.year.eq(2024)])
        values = d[[f'activity_c{i}' for i in range(1, 6)]].to_numpy()
        x = np.column_stack([context, mobility_features(values, expanded=True)])
        y = np.log1p(d.reference_emission.clip(lower=0).to_numpy())
        groups = d.cell_id.to_numpy() if scope == 'grid_random' else (
            np.floor((d.lon+180)/5).astype(int) + 100*np.floor((d.lat+90)/5).astype(int)).to_numpy()
    else:
        d = pd.read_csv(ROOT / 'city_mobility_carbon_validation_v2/input/city_year_validation_dataset.csv', low_memory=False)
        d = d[d.year.eq(int(scope.split('_')[1]))].dropna(subset=['log_transport_ce'] + SOCIO[:3]).reset_index(drop=True)
        context = np.column_stack([d[SOCIO + ROAD], pd.get_dummies(d[['development_group', 'urban_type', 'Global_NS']].astype(str), dtype=float)])
        values = d[[f'mode_{i}_km' for i in range(5)]].to_numpy(dtype=float)
        total = np.nansum(values, axis=1)
        x = np.column_stack([context, mobility_features(values, expanded=True),
                             np.log1p(total/np.maximum(d.population, 1)),
                             np.log1p(total/np.maximum(d.area, 1e-3))])
        y = d.log_transport_ce.to_numpy()
        groups = d.ID_UC_G0.to_numpy()
    x = np.asarray(x, np.float32)
    x[~np.isfinite(x)] = np.nan
    return d, x, np.asarray(y, np.float32), groups

def predict_batches(reg, x):
    return np.concatenate([reg.predict(x[i:i+1024]) for i in range(0, len(x), 1024)])

def run(scope, device, seed):
    d, x, y, groups = load(scope)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2 if scope.startswith('grid') else .25,
                                    random_state=seed).split(x, groups=groups))
    inner_tr, inner_val = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=731).split(x[tr], groups=groups[tr]))
    train, val = tr[inner_tr], tr[inner_val]
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    a = imp.fit_transform(x[train]); v = imp.transform(x[val]); b = imp.transform(x[te])
    folder = OUT / scope / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'row': np.r_[train, val, te], 'subset': ['train']*len(train)+['validation']*len(val)+['test']*len(te),
                  'group': groups[np.r_[train, val, te]]}).to_csv(folder / 'splits.csv', index=False)
    rows = []
    predictions = pd.DataFrame({'row': te, 'observed_log': y[te]})
    # All configurations are declared before reading their external test scores.
    strategies = ['HGBR_regularized', 'TabPFN_pretrained', 'TabPFN_finetuned', 'TabPFN_finetuned_longer']
    for name in strategies:
        started = time.time()
        try:
            if name == 'HGBR_regularized':
                candidates = []
                for leaf, penalty in [(12, 10), (20, 20), (28, 30)]:
                    m = HistGradientBoostingRegressor(max_iter=350, max_leaf_nodes=leaf, min_samples_leaf=80,
                        l2_regularization=penalty, learning_rate=.035, random_state=seed)
                    m.fit(a, y[train])
                    candidates.append((r2_score(y[val], m.predict(v)), m))
                score, reg = max(candidates, key=lambda z: z[0])
                pred = reg.predict(b)
            elif name == 'TabPFN_pretrained':
                # Match the final inference-context limit of the fine-tuned model.
                keep = np.random.default_rng(seed).choice(len(a), min(len(a), 4096), replace=False)
                reg = TabPFNRegressor(n_estimators=4, device=device, model_path=WEIGHT,
                    ignore_pretraining_limits=True, random_state=seed, fit_mode='fit_preprocessors')
                reg.fit(a[keep], y[train][keep])
                score = r2_score(y[val], predict_batches(reg, v))
                pred = predict_batches(reg, b)
            else:
                longer = name.endswith('longer')
                reg = FinetunedTabPFNRegressor(device=device, epochs=16 if longer else 8, time_limit=1200,
                    learning_rate=3e-5 if longer else 1e-5, weight_decay=.05, early_stopping=True, early_stopping_patience=3,
                    n_finetune_ctx_plus_query_samples=1024 if longer else 512, finetune_ctx_query_split_ratio=.25,
                    n_inference_subsample_samples=4096, n_estimators_finetune=1,
                    n_estimators_validation=2, n_estimators_final_inference=4,
                    random_state=seed, extra_regressor_kwargs={'model_path': WEIGHT,
                        'ignore_pretraining_limits': True})
                reg.fit(a, y[train], X_val=v, y_val=y[val], output_dir=folder / name)
                score = r2_score(y[val], predict_batches(reg, v))
                pred = predict_batches(reg, b)
            predictions[name] = pred
            row = {'scope':scope, 'seed':seed, 'model':name, 'n_train':len(train), 'n_validation':len(val),
                   'n_test':len(te), 'r2_validation_log':float(score), 'r2_test_log':r2_score(y[te], pred),
                   'mae_test_log':mean_absolute_error(y[te], pred), 'seconds':time.time()-started, 'status':'ok'}
        except Exception as exc:
            row = {'scope':scope, 'model':name, 'status':'failed', 'error':repr(exc), 'seconds':time.time()-started}
            traceback.print_exc()
        rows.append(row)
        print(json.dumps(row), flush=True)
        pd.DataFrame(rows).to_csv(folder / 'metrics.csv', index=False)
        predictions.to_csv(folder / 'test_predictions.csv', index=False)
        if 'reg' in locals():
            del reg
        gc.collect()
        import torch
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeat', action='store_true')
    args = parser.parse_args()
    for seed in [17, 42, 83] if args.repeat else [args.seed]:
        run(args.scope, args.device, seed)
