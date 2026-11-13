import pandas as pd
import geopandas as gpd
import pyogrio
from pathlib import Path
d=pd.read_csv('${SERVER_HOME}/result/city_mobility_carbon_validation_v2/input/city_year_validation_dataset.csv')
print(d[d.GC_UCN_MAI_2025.str.contains('Nairobi',case=False,na=False)][['ID_UC_G0','GC_UCN_MAI_2025','development_group','year']].to_string(index=False))
p='${SERVER_HOME}/result/external_validation_asia_africa_v1/gadm41_KEN.gpkg'
print(pyogrio.list_layers(p))
g=gpd.read_file(p,layer='ADM_ADM_2')
print(g.columns)
print(g[g.NAME_1.str.contains('Nairobi',case=False,na=False)].drop(columns='geometry').to_string(index=False))
