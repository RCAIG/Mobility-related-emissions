import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from shapely import wkt
from shapely.geometry import box
from shapely.ops import transform
from shapely.strtree import STRtree
from pyproj import Transformer

ROOT=Path('${SERVER_HOME}/result/external_validation_asia_africa_v1')
SOURCE=Path('${SERVER_HOME}/result/road_level_mobility_carbon_reconstruction_spatial_examples_v2/results')
code="""
from netCDF4 import Dataset
import numpy as np
f=Dataset('${SERVER_HOME}/result/external_validation_asia_africa_v1/chetna_52545992.nc')
a=np.asarray(f['co2'][:].filled(np.nan))
np.savez('${SERVER_HOME}/result/external_validation_asia_africa_v1/mumbai_reference.npz',lat=f['lat'][:],lon=f['lon'][:],emission=np.nansum(a,axis=0),valid_days=np.isfinite(a).sum(axis=0))
"""
env=os.environ.copy(); env['PYTHONPATH']='/tmp/netcdf_validation_deps'
subprocess.run(['/apps/miniconda3/envs/ce_trace/bin/python','-c',code],env=env,check=True)
f=np.load(ROOT/'mumbai_reference.npz')
lat,lon,ref,days=[f[k] for k in ['lat','lon','emission','valid_days']]
ny,nx=ref.shape
dx=float(np.median(np.diff(lon))); dy=float(np.median(np.diff(lat)))
proj=Transformer.from_crs(4326,32643,always_xy=True).transform
cells=[transform(proj,box(x-dx/2,y-dy/2,x+dx/2,y+dy/2)) for y in lat for x in lon]
tree=STRtree(cells)
metrics=[]
for year in [2017,2024]:
    d=pd.read_parquet(SOURCE/f'road_emission_7599_{year}.parquet')
    model=np.zeros(ny*nx)
    outside=0.; in_box=0.
    for shape,emission in zip(d.geometry_wkt,d.reconstructed_emission):
        if emission<=0: continue
        g=transform(proj,wkt.loads(shape))
        if g.length<=0: continue
        allocated=0.
        for idx in tree.query(g):
            piece=g.intersection(cells[int(idx)]).length
            if piece>0:
                v=emission*piece/g.length
                model[int(idx)]+=v; allocated+=v
        in_box+=allocated; outside+=emission-allocated
    base=pd.DataFrame({'row':np.repeat(np.arange(ny),nx),'col':np.tile(np.arange(nx),ny),
                       'lon':np.tile(lon,ny),'lat':np.repeat(lat,nx),'reference_tCO2_2021':ref.ravel(),
                       'valid_days':days.ravel(),'model_emission':model})
    base.to_csv(ROOT/f'mumbai_native_grid_comparison_{year}.csv',index=False)
    for factor in [1,2,4]:
        z=base[base.valid_days>0].copy()
        z['row_block']=z.row//factor; z['col_block']=z.col//factor
        z=z.groupby(['row_block','col_block'])[['reference_tCO2_2021','model_emission']].sum()
        for support in ['all_reference_valid','positive_union','both_positive']:
            r=z.reference_tCO2_2021.to_numpy(); p=z.model_emission.to_numpy()
            keep=np.ones(len(r),bool)
            if support=='positive_union': keep=(r>0)|(p>0)
            if support=='both_positive': keep=(r>0)&(p>0)
            r,p=r[keep],p[keep]
            rt=set(np.argsort(r)[-max(1,int(np.ceil(len(r)*.1))):])
            pt=set(np.argsort(p)[-max(1,int(np.ceil(len(p)*.1))):])
            metrics.append({'city':'Mumbai','reference':'CHETNA-Road','reference_year':2021,
                'model_year':year,'grid_step_degrees':dx*factor,'nominal_resolution_m':500*factor,
                'support':support,'n':len(r),'spearman_rho':float(spearmanr(r,p).statistic),
                'log1p_pearson_r':float(pearsonr(np.log1p(r),np.log1p(p)).statistic),
                'top_decile_recall':len(rt&pt)/len(rt),'top_decile_jaccard':len(rt&pt)/len(rt|pt),
                'reference_total_tCO2':float(r.sum()),'model_total_unharmonized_units':float(p.sum()),
                'model_mass_in_bbox':in_box,'model_mass_outside_bbox':outside})
    z=base[base.valid_days>0]
    fig,ax=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for a,key,label in zip(ax[:2],['reference_tCO2_2021','model_emission'],['CHETNA-Road 2021','Road reconstruction '+str(year)]):
        values=z[key]/z[key].sum()
        c=a.scatter(z.lon,z.lat,c=np.log10(values+1e-8),s=7,cmap='viridis',vmin=-8,vmax=-2)
        a.set_title(label); a.set_xlabel('Longitude'); a.set_ylabel('Latitude'); a.set_aspect(1/np.cos(np.deg2rad(19)))
    fig.colorbar(c,ax=ax[:2],label='log10 share of total within common valid footprint',shrink=.7)
    r=z.reference_tCO2_2021/z.reference_tCO2_2021.sum(); p=z.model_emission/z.model_emission.sum()
    ax[2].hexbin(np.log10(r+1e-8),np.log10(p+1e-8),gridsize=40,mincnt=1,cmap='viridis')
    ax[2].plot([-8,-2],[-8,-2],color='.5',lw=1)
    ax[2].set_xlabel('Reference log10 emission share'); ax[2].set_ylabel('Model log10 emission share')
    ax[2].set_title('Spatial-pattern comparison')
    fig.savefig(ROOT/f'Mumbai_CHETNA_native_500m_{year}.png',dpi=300)
    plt.close(fig)
pd.DataFrame(metrics).to_csv(ROOT/'mumbai_validation_metrics.csv',index=False)
print(pd.DataFrame(metrics).to_string(index=False))
