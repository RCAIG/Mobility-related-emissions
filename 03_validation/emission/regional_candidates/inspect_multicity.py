import json
from pathlib import Path
import pandas as pd
R=Path('${SERVER_HOME}/result')
d=pd.read_csv(R/'city_mobility_carbon_validation_v2/input/city_year_validation_dataset.csv')
print('COLUMNS',list(d.columns),flush=True)
names=['Indore','Tiruppur','Chandigarh','Jaipur','Vadodara','Pune','Mangaluru','Kolkata','Guwahati','Lucknow','Chennai','Delhi','Hyderabad','Mumbai','Bengaluru','Bangalore','Cape Town','Mombasa','Kisumu','Durban','Los Angeles','Baltimore','Indianapolis','Salt Lake','Sao Paulo','Rio de Janeiro','Brasilia','Curitiba','Salvador','Recife','Fortaleza','Belo Horizonte','Porto Alegre','Nakuru','Kampala','Accra','Johannesburg']
for name in names:
    z=d[d.GC_UCN_MAI_2025.str.contains(name,case=False,na=False)&d.year.eq(2024)]
    print(name,z[['ID_UC_G0','GC_UCN_MAI_2025','development_group']].to_dict('records'),flush=True)
for root in [R/'Experiments/external_validation_v1',R/'road_level_mobility_carbon_reconstruction_spatial_examples_v2/results']:
    print('FILES',root,flush=True)
    for p in root.rglob('*'):
        if p.is_file() and (p.suffix in ['.py','.csv','.json','.tif','.gpkg','.gz']):
            print(p,p.stat().st_size,flush=True)
