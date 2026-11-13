"""Year-specific grid models trained only on positive reference emissions."""
from pathlib import Path
import argparse
import gc
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from tabpfn import TabPFNRegressor
from tabpfn_emission_optimization import load, WEIGHT, predict_batches
from mobility_distance_bin_audit import FINE, mobility_features

ROOT = Path('${SERVER_HOME}/result')
OUT = Path('${SERVER_HOME}/result/emission_grid_year_positive_log_v1')
SEED = 20261006
MIN_REFERENCE = 100.0


def inputs(scope, year):
    d, base, _, groups = load(scope)
    d = d.reset_index(drop=True)
    extra = pd.read_parquet(ROOT / 'emission_structural_optimization_v2/target_free_mobility_features.parquet')
    assert np.array_equal(d[['cell_id', 'year']].to_numpy(), extra[['cell_id', 'year']].to_numpy())
    fine = extra[FINE].to_numpy(float)
    share = fine / np.maximum(fine.sum(axis=1)[:, None], 1e-9)
    mid = np.array([.5, 1.5, 2.5, 4, 6.5, 9, 12.5, 17.5, 25, 35, 50, 80, 150])
    shape = np.column_stack([
        share[:, :4].sum(axis=1), share[:, 8:].sum(axis=1),
        share[:, 10:].sum(axis=1), share @ np.log1p(mid),
        (share**2).sum(axis=1), np.cumsum(share, axis=1)[:, :-1],
    ])
    extra_columns = [c for c in extra if c not in ['cell_id', 'year', 'total'] + FINE]
    x = np.column_stack([base, mobility_features(fine, expanded=True), shape, extra[extra_columns]])
    raw = d.reference_emission.to_numpy(float)
    region_columns = [c for c in d if c.startswith('region_')]
    region_values = d[region_columns].to_numpy(float)
    region = np.argmax(region_values, axis=1)
    region[region_values.sum(axis=1) <= 0] = -1

    # Apply one fixed detection floor before splitting. This removes all zeros
    # and the bottom ~0.4% of positive cells while keeping the rule identical
    # across years. The target is then the ordinary natural log C.
    keep = d.year.eq(year).to_numpy() & (raw > MIN_REFERENCE)
    d = d.loc[keep].reset_index(drop=True)
    x = np.asarray(x[keep], dtype=np.float32)
    raw = raw[keep]
    y = np.log(raw).astype(np.float32)
    groups = groups[keep]
    region = region[keep]
    x[~np.isfinite(x)] = np.nan
    assert len(d) and d.year.eq(year).all() and np.all(raw > MIN_REFERENCE)
    assert np.allclose(y, np.log(raw), atol=2e-5)
    return d, x, y, raw, groups, region


def metrics(y, raw, pred):
    predicted_raw = np.exp(pred.astype(float))
    return {
        'r2_log': float(r2_score(y, pred)),
        'r2_raw': float(r2_score(raw, predicted_raw)),
        'mae_log': float(mean_absolute_error(y, pred)),
        'rmse_log': float(np.sqrt(mean_squared_error(y, pred))),
        'mae_raw': float(mean_absolute_error(raw, predicted_raw)),
        'spearman': float(spearmanr(y, pred).statistic),
    }


def hgb(seed):
    return HistGradientBoostingRegressor(
        max_iter=500, max_leaf_nodes=31, min_samples_leaf=40,
        learning_rate=.04, l2_regularization=15, random_state=seed,
    )


def run(scope, year, device):
    d, x, y, raw, groups, region = inputs(scope, year)
    out = OUT / f'{scope}_{year}' / str(SEED)
    out.mkdir(parents=True, exist_ok=True)

    pool, test = next(GroupShuffleSplit(
        n_splits=1, test_size=.2, random_state=SEED
    ).split(x, groups=groups))
    inner_train, inner_val = next(GroupShuffleSplit(
        n_splits=1, test_size=.15, random_state=731
    ).split(x[pool], groups=groups[pool]))
    train, val = pool[inner_train], pool[inner_val]
    assert set(groups[train]).isdisjoint(groups[test])
    assert set(groups[val]).isdisjoint(groups[test])

    pd.DataFrame({
        'row': np.r_[train, val, test],
        'subset': ['train']*len(train) + ['validation']*len(val) + ['test']*len(test),
        'group': groups[np.r_[train, val, test]],
    }).to_csv(out / 'splits.csv', index=False)

    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    a = imputer.fit_transform(x[train])
    v = imputer.transform(x[val])
    t = imputer.transform(x[test])
    pv, pt = {}, {}

    candidates = {
        'HGB': hgb(SEED),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=.8,
            n_jobs=8, random_state=SEED,
        ),
    }
    for name, model in candidates.items():
        model.fit(a, y[train])
        pv[name] = model.predict(v)
        pt[name] = model.predict(t)

    sample = np.random.default_rng(SEED).choice(len(a), min(len(a), 8192), replace=False)
    tabpfn = TabPFNRegressor(
        n_estimators=8, device=device, model_path=WEIGHT,
        ignore_pretraining_limits=True, random_state=SEED,
        fit_mode='fit_preprocessors',
    )
    tabpfn.fit(a[sample], y[train][sample])
    pv['TabPFN'] = predict_batches(tabpfn, v)
    pt['TabPFN'] = predict_batches(tabpfn, t)

    global_val, global_test = pv['HGB'], pt['HGB']
    local_val, local_test = global_val.copy(), global_test.copy()
    for region_id in np.unique(region):
        tr_mask = region[train] == region_id
        va_mask = region[val] == region_id
        te_mask = region[test] == region_id
        support = int(tr_mask.sum())
        if support >= 500:
            local = hgb(SEED).fit(a[tr_mask], y[train][tr_mask])
            local_weight = support / (support + 2000)
            if va_mask.any():
                local_val[va_mask] = local_weight*local.predict(v[va_mask]) + (1-local_weight)*global_val[va_mask]
            if te_mask.any():
                local_test[te_mask] = local_weight*local.predict(t[te_mask]) + (1-local_weight)*global_test[te_mask]
    _, regional_weight = max(
        (r2_score(y[val], global_val + w*(local_val-global_val)), w)
        for w in [0, .25, .5, .75, 1]
    )
    pv['Regional'] = global_val + regional_weight*(local_val-global_val)
    pt['Regional'] = global_test + regional_weight*(local_test-global_test)

    options = []
    names = list(pv)
    for i, left in enumerate(names):
        for right in names[i:]:
            for weight in [0, .25, .5, .75, 1]:
                options.append((r2_score(y[val], weight*pv[left] + (1-weight)*pv[right]), left, right, weight))
    _, left, right, weight = max(options, key=lambda item: item[0])
    pv['Validation_blend'] = weight*pv[left] + (1-weight)*pv[right]
    pt['Validation_blend'] = weight*pt[left] + (1-weight)*pt[right]

    output = d.iloc[test][['cell_id', 'year', 'lon', 'lat']].copy()
    output.insert(0, 'row', test)
    output['regime'] = 'Reference_above_100'
    output['observed_log'] = y[test]
    output['reference_raw'] = raw[test]
    for name, values in pt.items():
        output[name] = values
    output.to_csv(out / 'test_predictions.csv', index=False)

    rows = []
    for name, values in pt.items():
        rows.append({
            'scope': f'{scope}_{year}', 'seed': SEED,
            'regime': 'Reference_above_100', 'model': name,
            'evaluation': 'Positive_test', 'n_train': len(train),
            'n_validation': len(val), 'n_test': len(test),
            'r2_validation_log': r2_score(y[val], pv[name]),
            **metrics(y[test], raw[test], values),
        })
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    rule = {
        'year': year, 'scope': scope, 'target': 'ln(C)',
        'minimum_reference_exclusive': MIN_REFERENCE,
        'low_reference_excluded_before_split': True,
        'left': left, 'right': right, 'weight_left': weight,
        'regional_strength': regional_weight,
    }
    (out / 'validation_rules.json').write_text(json.dumps(rule, indent=2))
    print(json.dumps({
        **rule, 'n_train': len(train), 'n_validation': len(val),
        'n_test': len(test), 'test': metrics(y[test], raw[test], pt['Validation_blend']),
    }), flush=True)
    del tabpfn
    gc.collect()
    import torch
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=['grid_random', 'grid_block'], required=True)
    parser.add_argument('--years', default='2017,2024')
    parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args()
    for selected_year in map(int, args.years.split(',')):
        run(args.scope, selected_year, args.device)
