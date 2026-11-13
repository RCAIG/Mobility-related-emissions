from netCDF4 import Dataset
from pathlib import Path
import numpy as np
import json
root=Path('${SERVER_HOME}/result/external_validation_asia_africa_v1')
rows=[]
for p in sorted(root.glob('chetna_*.nc')):
    with Dataset(p) as f:
        row={'file':p.name,'attrs':f.__dict__,'variables':{k:{'shape':v.shape,'attrs':v.__dict__} for k,v in f.variables.items()}}
        for k,v in f.variables.items():
            if 'lat' in k.lower() or 'lon' in k.lower() or k.lower() in ['x','y']:
                a=v[:]
                row[k+'_range']=[float(a.min()),float(a.max())]
        rows.append(row)
print(json.dumps(rows, default=str, indent=2))
(root/'chetna_netcdf_inventory.json').write_text(json.dumps(rows,default=str,indent=2))
