from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio
ROOT=Path('${SERVER_HOME}/result')
out={}
for year in [2017,2024]:
    with rasterio.open(ROOT/f'spatial_mobility_carbon_validation_v1/reference_tiff/{year}.tif') as s:
        out[str(year)]={'shape':[s.height,s.width],'crs':str(s.crs),'transform':list(s.transform),
            'tags':s.tags(),'band_tags':s.tags(1),'units':s.units,'descriptions':s.descriptions,'nodata':s.nodata}
d=pd.read_csv(ROOT/'city_mobility_carbon_validation_v2/input/city_year_validation_dataset.csv',low_memory=False)
out['city']={'records':len(d),'cities':int(d.ID_UC_G0.nunique()),'year_counts':d.groupby('year').size().to_dict(),
 'target_zero_counts':d.assign(z=d.transport_ce.eq(0)).groupby('year').z.sum().to_dict(),
 'target_matches_ln1p':bool(np.allclose(d.log_transport_ce,np.log1p(d.transport_ce),atol=2e-5)),
 'target_min_max':[float(d.transport_ce.min()),float(d.transport_ce.max())],
 'ln1p_max_difference':float(np.max(abs(d.log_transport_ce-np.log1p(d.transport_ce)))),
 'ln_max_difference':float(np.max(abs(d.log_transport_ce-np.log(d.transport_ce)))),
 'first_targets':d[['transport_ce','log_transport_ce']].head().to_dict('records')}
path=ROOT/'emission_zero_comparison_v3/reference_metadata_audit.json'
path.write_text(json.dumps(out,indent=2,default=str))
print(path.read_text())
