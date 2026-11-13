import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,'/tmp')
import existing_road_reconstruction as old

ROOT=Path('${SERVER_HOME}/result/external_validation_asia_africa_v1/nairobi_reconstruction')
BASE=Path('${SERVER_HOME}/result/road_level_mobility_carbon_reconstruction_spatial_examples_v2')
old.CITY_META[1717]={'name':'Nairobi','income':'lower_middle'}
old.ROUTE_ROOTS['lower_middle']='${SERVER_HOME}/result/trace_genetare_lower_middle_{year}'
old.CITY_CELL_REFERENCE=Path('${SERVER_HOME}/result/spatial_mobility_carbon_validation_v2/analysis/city_cell_activity_reference_v2.parquet')
result=old.extract_city_year(1717,2024,str(ROOT))
print(result,flush=True)
if result['status']=='failed': raise RuntimeError(result)
edges=pd.read_parquet(ROOT/'activity/road_activity_1717_2024.parquet')
links=pd.read_parquet(ROOT/'activity/edge_cell_activity_1717_2024.parquet')
global_data=pd.read_parquet(BASE/'results/full_fit_spatial_predictions.parquet')
grid=links.groupby(['city_id','year','cell_id'],as_index=False)[old.CLASS_COLS].sum()
grid['road_activity_total']=grid[old.CLASS_COLS].sum(axis=1)
grid=grid.merge(global_data[['cell_id','year','activity_total','spatial_predicted_emission']],on=['cell_id','year'],how='left')
city=pd.read_parquet(old.CITY_CELL_REFERENCE,columns=['cell_id','year','city_id','activity_total'])
city=city[city.city_id==1717].rename(columns={'activity_total':'city_activity'})
grid=grid.merge(city,on=['cell_id','year','city_id'],how='left')
grid['city_share']=(grid.city_activity/grid.activity_total.clip(lower=old.EPS)).fillna(grid.road_activity_total/grid.activity_total.clip(lower=old.EPS)).clip(0,1)
missing=grid.spatial_predicted_emission.isna()
ratio=np.nanmedian(grid.loc[~missing,'spatial_predicted_emission']/grid.loc[~missing,'road_activity_total'].clip(lower=old.EPS))
grid.loc[missing,'spatial_predicted_emission']=grid.loc[missing,'road_activity_total']*ratio
grid.loc[missing,'city_share']=1
grid['grid_emission']=grid.spatial_predicted_emission*grid.city_share
coef=pd.read_csv(BASE/'tables/distance_class_response_coefficients.csv').sort_values('class')
z=np.log1p(grid[old.CLASS_COLS].to_numpy(float))
w=z*coef.linear_log_activity_coefficient.to_numpy()+z**2*coef.quadratic_log_activity_coefficient.to_numpy()
shares=w/np.maximum(w.sum(axis=1,keepdims=True),old.EPS)
for i in range(1,6): grid[f'class_emission_c{i}']=grid.grid_emission*shares[:,i-1]
a=links.merge(grid[['cell_id']+[f'class_emission_c{i}' for i in range(1,6)]],on='cell_id')
den=grid.set_index('cell_id')[old.CLASS_COLS]
for i in range(1,6):
    a[f'edge_emission_c{i}']=a[f'class_emission_c{i}']*a[f'activity_c{i}']/a.cell_id.map(den[f'activity_c{i}']).clip(lower=old.EPS)
ec=[f'edge_emission_c{i}' for i in range(1,6)]
e=a.groupby('edge_code',as_index=False)[ec].sum()
e['reconstructed_emission']=e[ec].sum(axis=1)
out=edges.merge(e,on='edge_code',how='left').fillna({'reconstructed_emission':0})
(ROOT/'results').mkdir(parents=True,exist_ok=True)
out.to_parquet(ROOT/'results/road_emission_1717_2024.parquet',index=False)
grid.to_csv(ROOT/'results/grid_emission_1717_2024.csv',index=False)
print({'roads':len(out),'cells':len(grid),'missing_grid_predictions':int(missing.sum()),'grid_total':float(grid.grid_emission.sum()),'road_total':float(out.reconstructed_emission.sum()),'note':'No city-total rescaling; spatial-pattern validation only.'},flush=True)
