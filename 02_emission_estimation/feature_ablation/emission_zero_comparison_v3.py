"""Matched all-reference and positive-reference emission prediction experiments."""
from pathlib import Path
import argparse
import gc
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from tabpfn import TabPFNRegressor
from tabpfn_emission_optimization import load, WEIGHT, predict_batches
from mobility_distance_bin_audit import FINE, ROAD, SOCIO, mobility_features

ROOT = Path('${SERVER_HOME}/result')
OUT = ROOT / 'emission_zero_comparison_v3'
SEEDS = [17, 42, 83, 20261006]


def inputs(scope):
    d, base, y, groups = load(scope)
    d = d.reset_index(drop=True)
    if scope.startswith('grid'):
        extra = pd.read_parquet(ROOT / 'emission_structural_optimization_v2/target_free_mobility_features.parquet')
        assert np.array_equal(d[['cell_id', 'year']].to_numpy(), extra[['cell_id', 'year']].to_numpy())
        fine = extra[FINE].to_numpy(float)
        share = fine / np.maximum(fine.sum(axis=1)[:, None], 1e-9)
        mid = np.array([.5, 1.5, 2.5, 4, 6.5, 9, 12.5, 17.5, 25, 35, 50, 80, 150])
        shape = np.column_stack([share[:, :4].sum(axis=1), share[:, 8:].sum(axis=1),
            share[:, 10:].sum(axis=1), share @ np.log1p(mid), (share**2).sum(axis=1),
            np.cumsum(share, axis=1)[:, :-1]])
        ec = [c for c in extra if c not in ['cell_id', 'year', 'total'] + FINE]
        x = np.column_stack([base, mobility_features(fine, expanded=True), shape, extra[ec]])
        raw = d.reference_emission.to_numpy(float)
        regions = [c for c in d if c.startswith('region_')]
        rv = d[regions].to_numpy(float)
        region = np.argmax(rv, axis=1)
        region[rv.sum(axis=1) <= 0] = -1
        meta = {'base_grid_columns': ['weighted_log_population', 'weighted_log_area', 'weighted_log_gdp',
            'sin_lon', 'cos_lon', 'sin_lat', 'cos_lat', 'city_contribution_hhi', 'contributing_cities']
            + [c for c in d if c.startswith(('dev_', 'region_'))] + ['year_2024'],
            'fine_activity_columns': FINE, 'extra_columns': ec,
            'mobility_transform': 'log1p bins; shares; log1p total; CLR with +1 pseudocount; normalized entropy; active count; first/last two bin aggregates',
            'shape_features': 'short/long/extreme-long shares; weighted log-distance proxy; HHI; 12 cumulative shares',
            'road_support': 'city background transferred to grid, not locally measured grid road metrics'}
    else:
        x, raw = base, d.transport_ce.to_numpy(float)
        region, regions = None, []
        meta = {'numeric_context_columns': SOCIO + ROAD,
            'categorical_columns': ['development_group', 'urban_type', 'Global_NS'],
            'activity_columns': [f'mode_{i}_km' for i in range(5)],
            'mobility_transform': 'log1p bins; shares; log1p total; CLR with +1 pseudocount; normalized entropy; active count; first/last two bin aggregates; log1p activity per person and per area',
            'activity_class_support': 'existing city five-class inputs; not silently replaced with grid absolute-distance classes'}
    assert np.isfinite(raw).all() and (raw >= 0).all()
    assert np.allclose(y, np.log1p(raw) if scope.startswith('grid') else np.log(raw), atol=2e-5)
    x = np.asarray(x, dtype=np.float32)
    x[~np.isfinite(x)] = np.nan
    return d, x, y, raw, groups, region, regions, meta


def metrics(y, raw, pred, scope):
    pred = np.maximum(pred, 0)
    pr = np.expm1(pred.astype(float)) if scope.startswith('grid') else np.exp(pred.astype(float))
    return {'r2_log': float(r2_score(y, pred)), 'r2_raw': float(r2_score(raw, pr)),
        'mae_log': float(mean_absolute_error(y, pred)),
        'rmse_log': float(np.sqrt(mean_squared_error(y, pred))),
        'mae_raw': float(mean_absolute_error(raw, pr)),
        'spearman': float(spearmanr(y, pred).statistic) if np.ptp(y) > 0 else None}


def hgb(seed):
    return HistGradientBoostingRegressor(max_iter=500, max_leaf_nodes=31,
        min_samples_leaf=40, learning_rate=.04, l2_regularization=15, random_state=seed)


def run(scope, device):
    d, x, y, raw, groups, region, regions, meta = inputs(scope)
    out = OUT / scope
    out.mkdir(parents=True, exist_ok=True)
    (out / 'input_audit.json').write_text(json.dumps({'scope': scope, 'n': len(d),
        'n_zero': int((raw == 0).sum()), 'n_positive': int((raw > 0).sum()),
        'target': 'ln(1 + reference emission)' if scope.startswith('grid') else 'ln(reference emission)',
        'n_features': x.shape[1], **meta}, indent=2))
    for seed in SEEDS:
        pool, test = next(GroupShuffleSplit(n_splits=1, test_size=.2 if scope.startswith('grid') else .25,
            random_state=seed).split(x, groups=groups))
        it, iv = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=731).split(x[pool], groups=groups[pool]))
        train, val = pool[it], pool[iv]
        assert set(groups[train]).isdisjoint(groups[test])
        assert set(groups[val]).isdisjoint(groups[test])
        folder = out / str(seed)
        folder.mkdir(exist_ok=True)
        pd.DataFrame({'row': np.r_[train, val, test],
            'subset': ['train']*len(train)+['validation']*len(val)+['test']*len(test),
            'group': groups[np.r_[train, val, test]]}).to_csv(folder/'splits.csv', index=False)
        rows, predictions, rules = [], [], []
        for regime in ['All_reference', 'Positive_reference_only']:
            tr = train if regime == 'All_reference' else train[raw[train] > 0]
            va = val if regime == 'All_reference' else val[raw[val] > 0]
            assert len(tr) > 100 and len(va) > 10
            imp = SimpleImputer(strategy='median', keep_empty_features=True)
            a, v, t = imp.fit_transform(x[tr]), imp.transform(x[va]), imp.transform(x[test])
            pv, pt = {}, {}
            fitted = {}
            candidates = {'HGB': hgb(seed),
                'ExtraTrees': ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5,
                    max_features=.8, n_jobs=8, random_state=seed)}
            for name, model in candidates.items():
                model.fit(a, y[tr])
                pv[name], pt[name] = np.maximum(model.predict(v), 0), np.maximum(model.predict(t), 0)
                fitted[name] = model
            keep = np.random.default_rng(seed).choice(len(a), min(len(a), 8192), replace=False)
            tab = TabPFNRegressor(n_estimators=8, device=device, model_path=WEIGHT,
                ignore_pretraining_limits=True, random_state=seed, fit_mode='fit_preprocessors')
            tab.fit(a[keep], y[tr][keep])
            pv['TabPFN'], pt['TabPFN'] = np.maximum(predict_batches(tab, v), 0), np.maximum(predict_batches(tab, t), 0)
            if region is not None:
                gv, gt = pv['HGB'], pt['HGB']
                lv, lt = gv.copy(), gt.copy()
                support = []
                for rid in np.unique(region):
                    mask, vm, tm = region[tr] == rid, region[va] == rid, region[test] == rid
                    n = int(mask.sum())
                    if n >= 500:
                        local = hgb(seed).fit(a[mask], y[tr][mask])
                        w = n/(n+2000)
                        if vm.any(): lv[vm] = w*local.predict(v[vm])+(1-w)*gv[vm]
                        if tm.any(): lt[tm] = w*local.predict(t[tm])+(1-w)*gt[tm]
                    support.append({'region': regions[rid] if rid >= 0 else 'unassigned',
                        'n_train': n, 'local_used': n >= 500})
                _, w = max((r2_score(y[va], np.maximum(gv+w*(lv-gv), 0)), w) for w in [0, .25, .5, .75, 1])
                pv['Regional'], pt['Regional'] = np.maximum(gv+w*(lv-gv), 0), np.maximum(gt+w*(lt-gt), 0)
                pd.DataFrame(support).to_csv(folder/f'{regime}_region_support.csv', index=False)
                rules.append({'regime': regime, 'regional_strength': w})
            options = []
            keys = list(pv)
            for i, left in enumerate(keys):
                for right in keys[i:]:
                    for w in [0, .25, .5, .75, 1]:
                        options.append((r2_score(y[va], w*pv[left]+(1-w)*pv[right]), left, right, w))
            _, left, right, weight = max(options, key=lambda q: q[0])
            pv['Validation_blend'] = weight*pv[left]+(1-weight)*pv[right]
            pt['Validation_blend'] = weight*pt[left]+(1-weight)*pt[right]
            rules.append({'regime': regime, 'left': left, 'right': right, 'weight_left': weight,
                'validation_population': 'all' if regime == 'All_reference' else 'positive only'})
            ids = ['cell_id', 'year', 'lon', 'lat'] if scope.startswith('grid') else [
                'ID_UC_G0', 'year', 'GC_UCN_MAI_2025', 'GC_CNT_GAD_2025']
            p = d.iloc[test][ids].copy()
            p.insert(0, 'row', test)
            p['regime'], p['observed_log'], p['reference_raw'] = regime, y[test], raw[test]
            for name in pt:
                p[name] = pt[name]
                for population in ['All_test', 'Positive_test']:
                    mask = np.ones(len(test), bool) if population == 'All_test' else raw[test] > 0
                    rows.append({'scope': scope, 'seed': seed, 'regime': regime, 'model': name,
                        'evaluation': population, 'n_train': len(tr), 'n_validation': len(va),
                        'n_test': int(mask.sum()), 'r2_validation_log': r2_score(y[va], pv[name]),
                        **metrics(y[test][mask], raw[test][mask], pt[name][mask], scope)})
            # An apparent-fit diagnostic is explicitly distinct from held-out validation.
            ap = np.maximum(fitted['HGB'].predict(a), 0)
            rows.append({'scope': scope, 'seed': seed, 'regime': regime, 'model': 'HGB',
                'evaluation': 'Training_apparent_fit', 'n_train': len(tr), 'n_test': len(tr),
                **metrics(y[tr], raw[tr], ap, scope)})
            predictions.append(p)
            pd.DataFrame(rows).to_csv(folder/'metrics.csv', index=False)
            pd.concat(predictions, ignore_index=True).to_csv(folder/'test_predictions.csv', index=False)
            (folder/'validation_rules.json').write_text(json.dumps(rules, indent=2))
            print(json.dumps({'scope': scope, 'seed': seed, 'regime': regime,
                'selected': [left, right, weight], 'test_all': metrics(y[test], raw[test], pt['Validation_blend'], scope),
                'test_positive': metrics(y[test][raw[test] > 0], raw[test][raw[test] > 0], pt['Validation_blend'][raw[test] > 0], scope)}), flush=True)
            del tab, fitted, candidates
            gc.collect()
            import torch
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', required=True)
    parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args()
    for scope in args.scope.split(','):
        run(scope, args.device)
