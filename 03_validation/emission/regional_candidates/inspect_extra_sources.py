import os,subprocess,sys,json
from pathlib import Path
import geopandas as gpd
import pandas as pd
import pyogrio
code="""
from netCDF4 import Dataset
import json,gzip
from pathlib import Path
root=Path('${SERVER_HOME}/result/external_validation_multicity_v1')
files=[Path('/tmp/BRAVES_Total_2017.nc')]+list(root.glob('*.nc*'))
for p in files:
 if p.suffix=='.gz':
  q=p.with_suffix('');q.write_bytes(gzip.decompress(p.read_bytes()));p=q
 with Dataset(p) as f:
  print(str(p),json.dumps({'attrs':f.__dict__,'vars':{k:{'shape':v.shape,'attrs':v.__dict__} for k,v in f.variables.items()}},default=str),flush=True)
"""
env=os.environ.copy();env['PYTHONPATH']='/tmp/netcdf_validation_deps'
subprocess.run([sys.executable,'-c',code],env=env,check=True)
R=Path('${SERVER_HOME}/result')
g=gpd.read_file(R/'city_boundaries_2024.gpkg',engine='pyogrio')
print('US largest',g[g.GC_CNT_GAD.eq('United States')].sort_values('GC_POP_TOT',ascending=False).head(8).drop(columns='geometry').to_dict('records'),flush=True)
print('Kenya',g[g.GC_CNT_GAD.eq('Kenya')].sort_values('GC_POP_TOT',ascending=False).head(12).drop(columns='geometry').to_dict('records'),flush=True)
print('Berlin layers',pyogrio.list_layers('/tmp/traffic_ghg_emissions_Berlin.gpkg'),flush=True)
