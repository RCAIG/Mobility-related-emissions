"""Grouped feature ablation for the grid HGB emission model.

This uses the exact grouped splits, target transform and HGB settings of the
zero-reference comparison experiment. It deliberately does not use target
derived features or refit feature selections on the external test set.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from tabpfn_emission_optimization import load
from mobility_distance_bin_audit import FINE, mobility_features

ROOT = Path('${SERVER_HOME}/result')
OUT = ROOT / 'grid_feature_ablation_v1'
SEEDS = [17, 42, 83]


def metric(observed_log, observed_raw, prediction_log):
    prediction_log = np.maximum(prediction_log, 0)
    prediction_raw = np.expm1(prediction_log.astype(float))
    return {
        'r2_log': float(r2_score(observed_log, prediction_log)),
        'r2_raw': float(r2_score(observed_raw, prediction_raw)),
        'rmse_log': float(np.sqrt(mean_squared_error(observed_log, prediction_log))),
        'mae_log': float(mean_absolute_error(observed_log, prediction_log)),
        'mae_raw': float(mean_absolute_error(observed_raw, prediction_raw)),
        'spearman': float(spearmanr(observed_log, prediction_log).statistic),
    }


def hgb(seed):
    return HistGradientBoostingRegressor(
        max_iter=500, max_leaf_nodes=31, min_samples_leaf=40,
        learning_rate=.04, l2_regularization=15, random_state=seed,
    )


def inputs(scope):
    d, base, target, groups = load(scope)
    d = d.reset_index(drop=True)
    extra = pd.read_parquet(
        ROOT / 'emission_structural_optimization_v2/target_free_mobility_features.parquet'
    )
    assert np.array_equal(d[['cell_id', 'year']].to_numpy(), extra[['cell_id', 'year']].to_numpy())
    fine = extra[FINE].to_numpy(float)
    activity = mobility_features(fine, expanded=True)
    share = fine / np.maximum(fine.sum(axis=1)[:, None], 1e-9)
    midpoint = np.array([.5, 1.5, 2.5, 4, 6.5, 9, 12.5, 17.5, 25, 35, 50, 80, 150])
    shape = np.column_stack([
        share[:, :4].sum(axis=1), share[:, 8:].sum(axis=1), share[:, 10:].sum(axis=1),
        share @ np.log1p(midpoint), (share**2).sum(axis=1), np.cumsum(share, axis=1)[:, :-1],
    ])
    extra_cols = [c for c in extra if c not in ['cell_id', 'year', 'total'] + FINE]
    structural = extra[extra_cols].to_numpy(float)
    raw = d.reference_emission.to_numpy(float)
    assert np.allclose(target, np.log1p(raw), atol=2e-5)
    blocks = {
        'Context and geography': np.asarray(base, dtype=float),
        'Distance activity and composition': activity,
        'Distance shape': shape,
        'Extended mobility and road structure': structural,
    }
    for name, values in blocks.items():
        values[~np.isfinite(values)] = np.nan
        blocks[name] = values.astype(np.float32)
    return d, blocks, target.astype(np.float32), raw, groups, extra_cols


def run(scope):
    data, blocks, target, raw, groups, extra_cols = inputs(scope)
    feature_sets = {
        'Context and geography only': ['Context and geography'],
        'Distance activity only': ['Distance activity and composition'],
        'Activity plus distance shape': ['Distance activity and composition', 'Distance shape'],
        'Context plus activity': ['Context and geography', 'Distance activity and composition'],
        'Context plus activity plus shape': ['Context and geography', 'Distance activity and composition', 'Distance shape'],
        'Context plus activity plus structure': ['Context and geography', 'Distance activity and composition', 'Extended mobility and road structure'],
        'Full model': list(blocks),
    }
    out = OUT / scope
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        pool, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(target, groups=groups))
        subtrain, validation = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=731).split(target[pool], groups=groups[pool]))
        train, validation = pool[subtrain], pool[validation]
        assert set(groups[train]).isdisjoint(groups[test])
        assert set(groups[validation]).isdisjoint(groups[test])
        for label, required in feature_sets.items():
            x = np.column_stack([blocks[name] for name in required])
            imp = SimpleImputer(strategy='median', keep_empty_features=True)
            a, v, t = imp.fit_transform(x[train]), imp.transform(x[validation]), imp.transform(x[test])
            reg = hgb(seed).fit(a, target[train])
            prediction = np.maximum(reg.predict(t), 0)
            for evaluation, mask in [('All_test', np.ones(len(test), dtype=bool)), ('Positive_test', raw[test] > 0)]:
                row = {
                    'scope': scope, 'seed': seed, 'model': 'HGB', 'feature_set': label,
                    'feature_blocks': ' + '.join(required), 'n_features': int(x.shape[1]),
                    'n_train': len(train), 'n_validation': len(validation), 'n_test': int(mask.sum()),
                    'evaluation': evaluation,
                    'r2_validation_log': float(r2_score(target[validation], np.maximum(reg.predict(v), 0))),
                    **metric(target[test][mask], raw[test][mask], prediction[mask]),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
        pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    (out / 'feature_dictionary.json').write_text(json.dumps({
        'blocks': {key: int(value.shape[1]) for key, value in blocks.items()},
        'extended_structure_columns': extra_cols,
        'target': 'ln(1 + EDGAR reference emission)',
        'zero_training': 'all reference values, including reference-zero cells',
        'test_groups': 'grid IDs' if scope == 'grid_random' else '5-degree spatial blocks',
    }, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=['grid_random', 'grid_block', 'both'], default='both')
    args = parser.parse_args()
    for scope in ['grid_random', 'grid_block'] if args.scope == 'both' else [args.scope]:
        run(scope)
