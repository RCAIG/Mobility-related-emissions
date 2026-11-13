"""Audit zero references and build target-free grid mobility descriptors."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from mobility_distance_bin_audit import FINE, ROAD
ROOT=Path('${SERVER_HOME}/result')
OUT=ROOT/'emission_structural_optimization_v2'
OUT.mkdir(exist_ok=True)
grid=pd.read_parquet(ROOT/'spatial_mobility_carbon_validation_v2/analysis/global_cell_features_v2.parquet')
ref=pd.read_parquet(ROOT/'spatial_mobility_carbon_validation_v1/analysis/spatial_cell_activity_reference.parquet')
print('reference columns',ref.columns.tolist(),flush=True)
ref=ref[['cell_id','year','reference_emission']].drop_duplicates(['cell_id','year'])
audit=grid[['cell_id','year','reference_emission','lon','lat','activity_total']].merge(
    ref.rename(columns={'reference_emission':'original_reference'}),on=['cell_id','year'],how='left',indicator=True,validate='one_to_one')
audit['unmatched_reference']=audit['_merge'].eq('left_only')
audit['missing_original_reference']=audit.original_reference.isna()
audit['zero_prepared_reference']=audit.reference_emission.eq(0)
audit.to_csv(OUT/'reference_zero_audit.csv',index=False)
audit.groupby('year').agg(n=('cell_id','size'),zeros=('zero_prepared_reference','sum'),
    unmatched=('unmatched_reference','sum'),missing=('missing_original_reference','sum')).to_csv(OUT/'reference_zero_summary.csv')
paths=[str(p) for year in [2017,2024] for p in (ROOT/'spatial_mobility_carbon_validation_v1/city_cache'/str(year)).glob('city_*.parquet')]
cells=ds.dataset(paths,format='parquet').to_table(columns=['cell_id','year','city_id']+FINE).to_pandas()
cells=cells.groupby(['cell_id','year','city_id'],as_index=False)[FINE].sum()
cells['total']=cells[FINE].sum(axis=1)
citytot=cells.groupby(['city_id','year']).total.transform('sum')
cells['within_city_activity_share']=cells.total/np.maximum(citytot,1e-9)
cells['within_city_activity_hhi']=(cells.within_city_activity_share**2).groupby([cells.city_id,cells.year]).transform('sum')
cells['within_city_peak_share']=cells.groupby(['city_id','year']).within_city_activity_share.transform('max')
cells['within_city_active_cells']=cells.groupby(['city_id','year']).cell_id.transform('nunique')
context=pd.read_csv(ROOT/'city_mobility_carbon_validation_v2/input/city_year_validation_dataset.csv',low_memory=False)
available=[c for c in ROAD if c in context]
cells=cells.merge(context[['ID_UC_G0','year']+available].drop_duplicates(['ID_UC_G0','year']),
    left_on=['city_id','year'],right_on=['ID_UC_G0','year'],how='left',validate='many_to_one')
extra=['within_city_activity_share','within_city_activity_hhi','within_city_peak_share','within_city_active_cells']+available
for c in extra:
    cells[c]=cells[c]*cells.total
agg=cells.groupby(['cell_id','year'],as_index=False)[FINE+extra+['total']].sum(min_count=1)
for c in extra:
    agg[c]=agg[c]/np.maximum(agg.total,1e-9)
merged=grid[['cell_id','year']].merge(agg,on=['cell_id','year'],how='left',validate='one_to_one')
assert merged[FINE].notna().all().all()
old=grid[[f'activity_c{i}' for i in range(1,6)]].sum(axis=1).to_numpy()
assert np.max(np.abs(merged.total.to_numpy()-old)/np.maximum(old,1))<1e-5
merged.to_parquet(OUT/'target_free_mobility_features.parquet',index=False)
(OUT/'feature_provenance.json').write_text(json.dumps({'fine_bounds_km':[0,1,2,3,5,8,10,15,20,30,40,60,100,'inf'],
    'road_columns':available,'extra_columns':extra,'road_support':'city-level road descriptors, mobility-weighted onto grid; not local measured road counts',
    'speed_support':'road design speed summaries only; no measured behavior labels',
    'socioeconomic_context':'unchanged from existing load()'},indent=2))
print('prepared',len(merged),len(extra),flush=True)
