"""Train grid emission models independently for one year at a time."""
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
OUT = Path('${SERVER_HOME}/result/emission_grid_year_separate_v1')
SEED = 20261006


def inputs(scope, year):
    d, base, y, groups = load(scope)
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

    keep = d.year.to_numpy() == year
    d = d.loc[keep].reset_index(drop=True)
    x = np.asarray(x[keep], dtype=np.float32)
    y = np.asarray(y[keep], dtype=np.float32)
    raw = raw[keep]
    groups = groups[keep]
    region = region[keep]
    x[~np.isfinite(x)] = np.nan
    assert len(d) and d.year.eq(year).all()
    assert np.allclose(y, np.log1p(raw), atol=2e-5)
    return d, x, y, raw, groups, region, region_columns


def metrics(y, raw, pred):
    pred = np.maximum(pred, 0)
    predicted_raw = np.expm1(pred.astype(float))
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
    d, x, y, raw, groups, region, region_names = inputs(scope, year)
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
        'subset': ['train'] * len(train) + ['validation'] * len(val) + ['test'] * len(test),
        'group': groups[np.r_[train, val, test]],
    }).to_csv(out / 'splits.csv', index=False)

    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    a = imputer.fit_transform(x[train])
    v = imputer.transform(x[val])
    t = imputer.transform(x[test])
    validation_predictions, test_predictions = {}, {}

    candidates = {
        'HGB': hgb(SEED),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=.8,
            n_jobs=8, random_state=SEED,
        ),
    }
    for name, model in candidates.items():
        model.fit(a, y[train])
        validation_predictions[name] = np.maximum(model.predict(v), 0)
        test_predictions[name] = np.maximum(model.predict(t), 0)

    sample = np.random.default_rng(SEED).choice(len(a), min(len(a), 8192), replace=False)
    tabpfn = TabPFNRegressor(
        n_estimators=8, device=device, model_path=WEIGHT,
        ignore_pretraining_limits=True, random_state=SEED,
        fit_mode='fit_preprocessors',
    )
    tabpfn.fit(a[sample], y[train][sample])
    validation_predictions['TabPFN'] = np.maximum(predict_batches(tabpfn, v), 0)
    test_predictions['TabPFN'] = np.maximum(predict_batches(tabpfn, t), 0)

    global_val = validation_predictions['HGB']
    global_test = test_predictions['HGB']
    local_val, local_test = global_val.copy(), global_test.copy()
    for region_id in np.unique(region):
        train_mask = region[train] == region_id
        val_mask = region[val] == region_id
        test_mask = region[test] == region_id
        n_region = int(train_mask.sum())
        if n_region >= 500:
            local = hgb(SEED).fit(a[train_mask], y[train][train_mask])
            local_weight = n_region / (n_region + 2000)
            if val_mask.any():
                local_val[val_mask] = local_weight * local.predict(v[val_mask]) + (1-local_weight) * global_val[val_mask]
            if test_mask.any():
                local_test[test_mask] = local_weight * local.predict(t[test_mask]) + (1-local_weight) * global_test[test_mask]
    _, regional_weight = max(
        (r2_score(y[val], np.maximum(global_val + w*(local_val-global_val), 0)), w)
        for w in [0, .25, .5, .75, 1]
    )
    validation_predictions['Regional'] = np.maximum(global_val + regional_weight*(local_val-global_val), 0)
    test_predictions['Regional'] = np.maximum(global_test + regional_weight*(local_test-global_test), 0)

    options = []
    names = list(validation_predictions)
    for i, left in enumerate(names):
        for right in names[i:]:
            for weight in [0, .25, .5, .75, 1]:
                blend = weight*validation_predictions[left] + (1-weight)*validation_predictions[right]
                options.append((r2_score(y[val], blend), left, right, weight))
    _, left, right, weight = max(options, key=lambda item: item[0])
    validation_predictions['Validation_blend'] = weight*validation_predictions[left] + (1-weight)*validation_predictions[right]
    test_predictions['Validation_blend'] = weight*test_predictions[left] + (1-weight)*test_predictions[right]

    prediction_frame = d.iloc[test][['cell_id', 'year', 'lon', 'lat']].copy()
    prediction_frame.insert(0, 'row', test)
    prediction_frame['regime'] = 'All_reference'
    prediction_frame['observed_log'] = y[test]
    prediction_frame['reference_raw'] = raw[test]
    for name, values in test_predictions.items():
        prediction_frame[name] = values
    prediction_frame.to_csv(out / 'test_predictions.csv', index=False)

    rows = []
    for name, values in test_predictions.items():
        for evaluation, mask in {
            'All_test': np.ones(len(test), dtype=bool),
            'Positive_test': raw[test] > 0,
        }.items():
            rows.append({
                'scope': f'{scope}_{year}', 'seed': SEED,
                'regime': 'All_reference', 'model': name,
                'evaluation': evaluation, 'n_train': len(train),
                'n_validation': len(val), 'n_test': int(mask.sum()),
                'r2_validation_log': r2_score(y[val], validation_predictions[name]),
                **metrics(y[test][mask], raw[test][mask], values[mask]),
            })
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    (out / 'validation_rules.json').write_text(json.dumps({
        'year': year, 'scope': scope, 'left': left, 'right': right,
        'weight_left': weight, 'regional_strength': regional_weight,
        'year_filtered_before_split': True,
    }, indent=2))
    print(json.dumps({
        'scope': scope, 'year': year, 'selected': [left, right, weight],
        'n_train': len(train), 'n_validation': len(val), 'n_test': len(test),
        'positive_test': metrics(
            y[test][raw[test] > 0], raw[test][raw[test] > 0],
            test_predictions['Validation_blend'][raw[test] > 0],
        ),
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
