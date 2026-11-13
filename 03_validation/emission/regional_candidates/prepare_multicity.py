import json, requests
from pathlib import Path
import pandas as pd
R=Path('${SERVER_HOME}/result')
O=R/'external_validation_multicity_v1'; O.mkdir(exist_ok=True)
d=pd.read_csv(R/'city_mobility_carbon_validation_v2/input/city_year_validation_dataset.csv')
z=d[(d.year==2024)&d.GC_CNT_GAD_2025.str.contains('India',case=False,na=False)]
print(z[z.GC_UCN_MAI_2025.str.contains('Delhi|Kolk|Calcutta|Hyderabad',case=False,na=False)][['ID_UC_G0','GC_UCN_MAI_2025','GC_CNT_GAD_2025']].to_dict('records'),flush=True)
for country in ['Brazil','Kenya']:
    s=d[(d.year==2024)&d.GC_CNT_GAD_2025.eq(country)].copy()
    s.sort_values('population',ascending=False).head(15).to_csv(O/(country+'_candidates.csv'),index=False)
    print(country,s.sort_values('population',ascending=False).head(15)[['ID_UC_G0','GC_UCN_MAI_2025']].to_dict('records'),flush=True)
manifest=pd.read_csv(R/'Experiments/external_validation_v1/hestia_multicity_candidate_manifest.csv')
for _,row in manifest.iterrows():
    p=O/row.File
    if not p.exists():
        r=requests.get(row.URL,timeout=90); r.raise_for_status(); p.write_bytes(r.content)
    print('Hestia downloaded',p,flush=True)
