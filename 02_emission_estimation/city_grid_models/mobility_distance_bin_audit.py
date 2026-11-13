"""Compare route-distance bins while keeping emission targets and context fixed."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path('${SERVER_HOME}/result')
V1 = ROOT / 'spatial_mobility_carbon_validation_v1/city_cache'
V2 = ROOT / 'spatial_mobility_carbon_validation_v2/analysis'
CITY = ROOT / 'city_mobility_carbon_validation_v2/input/city_year_validation_dataset.csv'
OUT = ROOT / 'distance_bin_audit_v1'
FINE = [f'pkm_b{i:02d}' for i in range(13)]
# Endpoints are indices into the 13 disjoint bands, not interpolated distances.
SCHEMES = {'5': [0, 2, 4, 7, 10, 13],
           '8': [0, 2, 4, 5, 7, 9, 10, 12, 13],
           '10': [0, 1, 2, 3, 4, 6, 7, 9, 10, 12, 13],
           '13': list(range(14))}
SOCIO = ['log_population', 'log_area', 'log_gdp', 'logistics_index',
         'vehicle_co2_intensity', 'log_bev_ownership', 'public_transport_efficiency',
         'train_service_efficiency', 'transport_tax', 'rise_index', 'oil_share',
         'electricity_share']
ROAD = ['road_n_nodes', 'road_n_edges', 'road_total_length_km', 'road_beta_index',
        'road_gamma_index', 'road_alpha_index', 'road_avg_node_degree',
        'road_dead_end_ratio', 'road_highway_length_ratio', 'road_primary_length_ratio',
        'road_secondary_length_ratio', 'road_local_length_ratio',
        'road_hierarchy_entropy', 'road_hierarchy_entropy_norm',
        'road_length_weighted_avg_speed', 'road_speed_std',
        'road_road_density_km_per_km2', 'road_highway_density_km_per_km2',
        'road_intersection_density_per_km2', 'road_avg_edge_length_m',
        'road_edge_length_std_m']


def bins(frame, bounds):
    a = frame[FINE].to_numpy(dtype=np.float64)
    return np.stack([a[:, lo:hi].sum(axis=1) for lo, hi in zip(bounds[:-1], bounds[1:])], axis=1)


def mobility_features(values, expanded=False):
    total = values.sum(axis=1)
    share = values / np.maximum(total[:, None], 1e-9)
    z = np.column_stack([np.log1p(values), share, np.log1p(total)])
    if expanded:
        proportions = (values + 1.) / (total[:, None] + values.shape[1])
        entropy = -(proportions * np.log(proportions)).sum(axis=1) / np.log(values.shape[1])
        z = np.column_stack([z, np.log(proportions) - np.log(proportions).mean(axis=1)[:, None],
                             entropy, (values > 0).sum(axis=1),
                             np.log1p(values[:, :2].sum(axis=1)),
                             np.log1p(values[:, -2:].sum(axis=1))])
    return z.astype(np.float32)


def evaluate(scope, frame, values, context, y, groups, split, seed=42):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2 if scope == 'grid' else .25,
                                    random_state=seed).split(frame, groups=groups))
    followup = os.environ.get('DISTANCE_FOLLOWUP') == '1'
    names = ['5_expanded', '13_expanded'] if followup else ['context', '5', '8', '10', '13', '5_expanded', '10_expanded', '13_expanded']
    for name in names:
        if name == 'context':
            x = context
        else:
            n = name.split('_')[0]
            x = np.column_stack([context, mobility_features(bins(values, SCHEMES[n]),
                                                       expanded=name.endswith('expanded'))])
            if followup and scope.startswith('city'):
                total = values[FINE].sum(axis=1).to_numpy()
                x = np.column_stack([x, np.log1p(total / np.maximum(frame.population.to_numpy(), 1)),
                                     np.log1p(total / np.maximum(frame.area.to_numpy(), 1e-3))])
        imp = SimpleImputer(strategy='median', keep_empty_features=True)
        a = imp.fit_transform(x[tr]); b = imp.transform(x[te])
        model = HistGradientBoostingRegressor(learning_rate=.05,
            max_iter=220 if scope == 'grid' else 240,
            max_leaf_nodes=28 if scope == 'grid' else 24,
            min_samples_leaf=35 if scope == 'grid' else 30,
            l2_regularization=1.2 if scope == 'grid' else 1.0, random_state=seed)
        models = {'HGBR': model}
        if followup:
            models = {'HGBR_deeper': HistGradientBoostingRegressor(learning_rate=.04, max_iter=600,
                        max_leaf_nodes=45, min_samples_leaf=20, l2_regularization=3, random_state=seed),
                      'ExtraTrees': ExtraTreesRegressor(n_estimators=180, min_samples_leaf=3,
                        max_features=.85, n_jobs=8, random_state=seed)}
        for model_name, model in models.items():
            model.fit(a, y[tr]); pred = model.predict(b)
            row = dict(scope=scope, split=split, seed=seed, bins=name, model=model_name, n_train=len(tr),
                       n_test=len(te), r2_log=r2_score(y[te], pred),
                       r2_train_log=r2_score(y[tr], model.predict(a)),
                       mae_log=mean_absolute_error(y[te], pred))
            print(row, flush=True)
            yield row


def main():
    OUT.mkdir(exist_ok=True)
    result_name = 'results_followup.csv' if os.environ.get('DISTANCE_FOLLOWUP') == '1' else 'results_repeated.csv'
    grid = pd.read_parquet(V2 / 'global_cell_features_v2.parquet')
    city_cells = ds.dataset([str(p) for year in [2017, 2024]
                             for p in (V1 / str(year)).glob('city_*.parquet')],
                            format='parquet').to_table(columns=['cell_id', 'year', 'city_id'] + FINE).to_pandas()
    city_fine = city_cells.groupby(['city_id', 'year'], as_index=False)[FINE].sum()
    grid_fine = city_cells.groupby(['cell_id', 'year'], as_index=False)[FINE].sum()
    del city_cells
    merged = grid.merge(grid_fine, on=['cell_id', 'year'], how='left', validate='one_to_one')
    assert merged[FINE].notna().all().all()
    old_total = merged[[f'activity_c{i}' for i in range(1, 6)]].sum(axis=1)
    new_total = merged[FINE].sum(axis=1)
    discrepancy = np.abs(old_total - new_total) / np.maximum(old_total, 1)
    print('grid cache reconciliation', discrepancy.quantile([0, .5, .95, 1]).to_dict(), flush=True)
    assert discrepancy.max() < .00001
    for direction in ['lon', 'lat']:
        merged[f'sin_{direction}'] = np.sin(np.radians(merged[direction]))
        merged[f'cos_{direction}'] = np.cos(np.radians(merged[direction]))
    context_cols = ['weighted_log_population', 'weighted_log_area', 'weighted_log_gdp',
                    'sin_lon', 'cos_lon', 'sin_lat', 'cos_lat', 'city_contribution_hhi',
                    'contributing_cities'] + [x for x in grid if x.startswith(('dev_', 'region_'))]
    context = merged[context_cols].to_numpy(dtype=np.float32)
    context = np.column_stack([context, (merged.year == 2024).to_numpy(np.float32)])
    y = np.log1p(merged.reference_emission.clip(lower=0).to_numpy())
    rows = []
    for split, groups in [('random_cell', merged.cell_id),
                          ('block_5deg', np.floor((merged.lon + 180) / 5).astype(int)
                           + 100 * np.floor((merged.lat + 90) / 5).astype(int))]:
        for seed in [17, 42, 83]:
            rows.extend(evaluate('grid', merged, merged, context, y, groups, split, seed))
            pd.DataFrame(rows).to_csv(OUT / result_name, index=False)
    city = pd.read_csv(CITY, low_memory=False).merge(city_fine, left_on=['ID_UC_G0', 'year'],
                                                     right_on=['city_id', 'year'], how='inner', validate='one_to_one')
    old = city[[f'mode_{i}_km' for i in range(5)]].sum(axis=1)
    new = city[FINE].sum(axis=1)
    ratio = new / old.replace(0, np.nan)
    pd.DataFrame({'city_id': city.ID_UC_G0, 'year': city.year, 'old_total_km': old,
                  'new_total_km': new, 'ratio': ratio}).to_csv(OUT / 'city_total_reconciliation.csv', index=False)
    print('city totals ratio', ratio.quantile([0, .05, .5, .95, 1]).to_dict(),
          'correlation', old.corr(new, method='spearman'), flush=True)
    # The two city totals may differ in their population coverage. Evaluate the
    # finer bins separately rather than silently treating them as the old modes.
    city = city.dropna(subset=['log_transport_ce'] + SOCIO[:3])
    categorical = pd.get_dummies(city[['year', 'development_group', 'urban_type', 'Global_NS']].astype(str), dtype=float)
    context = np.column_stack([city[SOCIO + ROAD].to_numpy(dtype=np.float32), categorical.to_numpy(dtype=np.float32)])
    y = city.log_transport_ce.to_numpy()
    for year in [2017, 2024]:
        mask = city.year.eq(year).to_numpy()
        part = city.loc[mask]
        for seed in [17, 42, 83]:
            rows.extend(evaluate(f'city_{year}', part, part, context[mask], y[mask], part.ID_UC_G0,
                                 'heldout_city', seed))
            pd.DataFrame(rows).to_csv(OUT / result_name, index=False)


if __name__ == '__main__':
    main()
