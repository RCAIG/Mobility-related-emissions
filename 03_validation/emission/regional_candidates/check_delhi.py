import json
import os
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

code = """
from netCDF4 import Dataset
import numpy as np,json
f=Dataset('/tmp/Delhi_Road_Traffic_2018.nc')
x=f['LON'][:].filled(np.nan); y=f['LAT'][:].filled(np.nan)
ix=np.floor((x+180)*10+1e-7).astype(int)
iy=np.floor((y+90)*10+1e-7).astype(int)
out={}
for s in ['CH4','N2O']:
 v=np.sum(f[s][:].filled(0),axis=0)
 out[s]=[[int(i),int(j),float(v[(ix==i)&(iy==j)].sum())] for i,j in set(zip(ix.ravel(),iy.ravel()))]
print(json.dumps(out))
"""
env = os.environ.copy()
env['PYTHONPATH'] = '/tmp/netcdf_validation_deps'
data = json.loads(subprocess.check_output(['/apps/miniconda3/envs/ce_trace/bin/python', '-c', code], env=env))
p = pd.read_parquet('${SERVER_HOME}/result/road_level_mobility_carbon_reconstruction_spatial_examples_v2/results/full_fit_spatial_predictions.parquet')
p['ix'] = np.floor((p.lon+180)*10+1e-7).astype(int)
p['iy'] = np.floor((p.lat+90)*10+1e-7).astype(int)
for s, rows in data.items():
 m = p.merge(pd.DataFrame(rows, columns=['ix','iy','ref']), on=['ix','iy'])
 for year in [2017,2024]:
  v=m[(m.year==year)&(m.ref>0)&(m.spatial_predicted_emission>0)]
  print(s, year, len(v), 'rho', spearmanr(v.ref,v.spatial_predicted_emission).statistic, 'log_r', pearsonr(np.log(v.ref),np.log(v.spatial_predicted_emission)).statistic)
