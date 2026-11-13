"""Test activity-conserving socioeconomic downscaling for grid emission models.

This is deliberately an intermediate, target-free downscaling test.  A city's
population, GDP and area are distributed over its active grid cells by its own
route-activity share, then contributions from all cities are summed.  It is
not labelled as an observed gridded population product.
"""
from pathlib import Path
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
OUT = ROOT / 'grid_socioeconomic_downscale_v1'
SEEDS = [17, 42, 83]


def hgb(seed):
    return HistGradientBoostingRegressor(
        max_iter=500, max_leaf_nodes=31, min_samples_leaf=40,
        learning_rate=.04, l2_regularization=15, random_state=seed,
    )


def metrics(y_log, y_raw, pred_log):
    pred_log = np.maximum(pred_log, 0)
    pred_raw = np.expm1(pred_log)
    return {
        'r2_log': float(r2_score(y_log, pred_log)),
        'r2_raw': float(r2_score(y_raw, pred_raw)),
        'rmse_log': float(np.sqrt(mean_squared_error(y_log, pred_log))),
        'mae_log': float(mean_absolute_error(y_log, pred_log)),
        'mae_raw': float(mean_absolute_error(y_raw, pred_raw)),
        'spearman': float(spearmanr(y_log, pred_log).statistic),
    }


def downscaled_socio(data):
    """Allocate city totals to active cells using within-city route activity."""
    p = ROOT / 'spatial_mobility_carbon_validation_v2/analysis/city_cell_activity_reference_v2.parquet'
    c = pd.read_parquet(p, columns=[
        'cell_id', 'year', 'city_id', 'activity_share_city',
        'log_population', 'log_gdp', 'log_area', 'lon', 'lat',
    ])
    # log inputs are natural logs in the parent city table.
    pop = np.expm1(c.log_population.clip(lower=0).fillna(0))
    gdp = np.expm1(c.log_gdp.clip(lower=0).fillna(0))
    area = np.expm1(c.log_area.clip(lower=0).fillna(0))
    share = c.activity_share_city.clip(lower=0).fillna(0)
    c['allocated_population'] = pop * share
    c['allocated_gdp'] = gdp * share
    c['allocated_area'] = area * share
    c['allocated_city_count'] = (share > 0).astype(float)
    g = c.groupby(['cell_id', 'year'], as_index=False)[[
        'allocated_population', 'allocated_gdp', 'allocated_area', 'allocated_city_count'
    ]].sum()
    g['grid_area_km2'] = 123.64 * np.cos(np.radians(g.lat if 'lat' in g else 0))
    # Coordinates are restored from the global table below; all features remain target-free.
    z = data[['cell_id', 'year', 'lat']].merge(g, on=['cell_id', 'year'], how='left', validate='one_to_one')
    z['grid_area_km2'] = 123.64 * np.cos(np.radians(z.lat))
    pop = z.allocated_population.fillna(0).to_numpy(float)
    gdp = z.allocated_gdp.fillna(0).to_numpy(float)
    area = z.allocated_area.fillna(0).to_numpy(float)
    activity = data.activity_total.to_numpy(float)
    return np.column_stack([
        np.log1p(pop), np.log1p(gdp), np.log1p(area),
        np.log1p(pop / np.maximum(z.grid_area_km2.to_numpy(float), 1e-6)),
        np.log1p(gdp / np.maximum(pop, 1e-6)),
        np.log1p(activity / np.maximum(pop, 1e-6)),
        np.log1p(activity / np.maximum(gdp, 1e-6)),
        z.allocated_city_count.fillna(0).to_numpy(float),
    ]).astype(np.float32)


def run(scope):
    data, blocks, target, raw, groups, _ = inputs(scope)
    socio = downscaled_socio(data)
    socio[~np.isfinite(socio)] = np.nan
    sets = {
        'Existing full model': list(blocks),
        'Existing full + activity-allocated socioeconomic totals': list(blocks) + ['downscaled socioeconomic totals'],
        'Context + activity + structure + allocated socioeconomic totals': [
            'Context and geography', 'Distance activity and composition',
            'Extended mobility and road structure', 'downscaled socioeconomic totals'
        ],
    }
    blocks['downscaled socioeconomic totals'] = socio
    rows = []
    out = OUT / scope
    out.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        pool, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(target, groups=groups))
        inner, validation = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=731).split(target[pool], groups=groups[pool]))
        train = pool[inner]
        for label, required in sets.items():
            x = np.column_stack([blocks[name] for name in required])
            imp = SimpleImputer(strategy='median', keep_empty_features=True)
            a = imp.fit_transform(x[train]); b = imp.transform(x[test])
            reg = hgb(seed).fit(a, target[train])
            pred = np.maximum(reg.predict(b), 0)
            for evaluation, mask in [('All_test', np.ones(len(test), bool)), ('Positive_test', raw[test] > 0)]:
                row = {
                    'scope': scope, 'seed': seed, 'model': 'HGB', 'feature_set': label,
                    'n_features': int(x.shape[1]), 'n_train': len(train), 'n_validation': len(validation),
                    'n_test': int(mask.sum()), 'evaluation': evaluation,
                    **metrics(target[test][mask], raw[test][mask], pred[mask]),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
        pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    (out / 'method.json').write_text(json.dumps({
        'method': 'Each city total is allocated across its active cells by activity_share_city, then summed over contributing cities.',
        'variables': ['allocated population', 'allocated GDP', 'allocated area', 'population density', 'GDP per allocated population', 'activity per allocated population', 'activity per allocated GDP', 'number of allocated city contributions'],
        'restriction': 'This is activity-weighted socioeconomic downscaling, not an independently observed gridded socioeconomic surface.',
    }, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=['grid_random', 'grid_block', 'both'], default='both')
    args = parser.parse_args()
    for scope in ['grid_random', 'grid_block'] if args.scope == 'both' else [args.scope]:
        run(scope)
