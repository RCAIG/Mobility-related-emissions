"""Test activity-weighted transfer of city fleet, energy and transit conditions to grids.

City technology variables are joined to each city-grid route-activity record.
For a shared grid, they are averaged by each city's route activity in that grid.
No EDGAR target or reference-emission-derived weight is used in this process.
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

from grid_feature_ablation_v1 import inputs

ROOT = Path('${SERVER_HOME}/result')
OUT = ROOT / 'grid_technical_context_downscale_v1'
SEEDS = [17, 42, 83]

# These are the non-background, city-scale fleet/energy/transit variables used
# in the city models. They are all available before any grid EDGAR comparison.
TECHNICAL = [
    'logistics_index', 'vehicle_co2_intensity', 'log_bev_ownership',
    'public_transport_efficiency', 'train_service_efficiency', 'transport_tax',
    'rise_index', 'oil_share', 'electricity_share',
]


def hgb(seed):
    return HistGradientBoostingRegressor(
        max_iter=500, max_leaf_nodes=31, min_samples_leaf=40,
        learning_rate=.04, l2_regularization=15, random_state=seed,
    )


def metric(y_log, y_raw, p_log):
    p_log = np.maximum(p_log, 0)
    p_raw = np.expm1(p_log)
    return {
        'r2_log': float(r2_score(y_log, p_log)),
        'r2_raw': float(r2_score(y_raw, p_raw)),
        'rmse_log': float(np.sqrt(mean_squared_error(y_log, p_log))),
        'mae_log': float(mean_absolute_error(y_log, p_log)),
        'mae_raw': float(mean_absolute_error(y_raw, p_raw)),
        'spearman': float(spearmanr(y_log, p_log).statistic),
    }


def technical_grid_features(data):
    cells = pd.read_parquet(
        ROOT / 'spatial_mobility_carbon_validation_v2/analysis/city_cell_activity_reference_v2.parquet',
        columns=['cell_id', 'year', 'city_id', 'activity_total'],
    )
    city = pd.read_csv(
        ROOT / 'city_mobility_carbon_validation_v2/input/city_year_validation_dataset.csv',
        low_memory=False,
    )
    city = city[['ID_UC_G0', 'year'] + TECHNICAL].drop_duplicates(['ID_UC_G0', 'year'])
    assert not city.duplicated(['ID_UC_G0', 'year']).any()
    cells = cells.merge(
        city, left_on=['city_id', 'year'], right_on=['ID_UC_G0', 'year'], how='left', validate='many_to_one'
    )
    # Route activity supplies a target-free mixing weight for cities sharing a grid.
    weight = cells.activity_total.clip(lower=0).fillna(0)
    total = cells.assign(_weight=weight).groupby(['cell_id', 'year'])['_weight'].sum()
    out = pd.DataFrame(index=total.index)
    for col in TECHNICAL:
        value = pd.to_numeric(cells[col], errors='coerce')
        numerator = (value.fillna(0) * weight).groupby([cells.cell_id, cells.year]).sum()
        valid_weight = (weight * value.notna()).groupby([cells.cell_id, cells.year]).sum()
        out[f'tech_{col}'] = numerator / valid_weight.replace(0, np.nan)
        out[f'tech_coverage_{col}'] = valid_weight / total.replace(0, np.nan)
    out = out.reset_index()
    aligned = data[['cell_id', 'year']].merge(out, on=['cell_id', 'year'], how='left', validate='one_to_one')
    features = aligned.drop(columns=['cell_id', 'year']).to_numpy(np.float32)
    features[~np.isfinite(features)] = np.nan
    return features, aligned.columns.drop(['cell_id', 'year']).tolist()


def run(scope):
    data, blocks, target, raw, groups, _ = inputs(scope)
    technical, technical_names = technical_grid_features(data)
    blocks['Fleet, energy and transit context'] = technical
    feature_sets = {
        'Existing full model': [k for k in blocks if k != 'Fleet, energy and transit context'],
        'Existing full + fleet, energy and transit context': list(blocks),
        'Context + activity + structure + fleet, energy and transit context': [
            'Context and geography', 'Distance activity and composition',
            'Extended mobility and road structure', 'Fleet, energy and transit context',
        ],
    }
    out = OUT / scope
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        pool, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(target, groups=groups))
        inner, validation = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=731).split(target[pool], groups=groups[pool]))
        train = pool[inner]
        for label, required in feature_sets.items():
            x = np.column_stack([blocks[key] for key in required])
            imp = SimpleImputer(strategy='median', keep_empty_features=True)
            x_train = imp.fit_transform(x[train])
            x_test = imp.transform(x[test])
            reg = hgb(seed).fit(x_train, target[train])
            pred = np.maximum(reg.predict(x_test), 0)
            for evaluation, mask in [('All_test', np.ones(len(test), dtype=bool)), ('Positive_test', raw[test] > 0)]:
                row = {
                    'scope': scope, 'seed': seed, 'model': 'HGB', 'feature_set': label,
                    'n_features': int(x.shape[1]), 'n_train': len(train), 'n_validation': len(validation),
                    'n_test': int(mask.sum()), 'evaluation': evaluation,
                    **metric(target[test][mask], raw[test][mask], pred[mask]),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
        pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    (out / 'feature_definition.json').write_text(json.dumps({
        'technical_variables': TECHNICAL,
        'derived_features': technical_names,
        'downscaling': 'For each grid, values are the route-activity-weighted average across contributing cities. Coverage indicators show the share of route activity with non-missing source values.',
        'limitation': 'Within a city, this transfer assumes technology conditions are spatially homogeneous. It is a city-to-grid transfer, not a direct grid observation of vehicle fleet or fuel sales.',
    }, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=['grid_random', 'grid_block', 'both'], default='both')
    args = parser.parse_args()
    for scope in ['grid_random', 'grid_block'] if args.scope == 'both' else [args.scope]:
        run(scope)
