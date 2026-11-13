import sys,os,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from scipy.stats import spearmanr,pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('${SERVER_HOME}/result');O=R/'external_validation_multicity_v1';O.mkdir(exist_ok=True)
g=gpd.read_file(R/'city_boundaries_2024.gpkg',engine='pyogrio')
p=pd.read_parquet(R/'road_level_mobility_carbon_reconstruction_spatial_examples_v2/results/full_fit_spatial_predictions.parquet')
p=p[p.year.eq(2024)].copy()
p['ix']=np.floor((p.lon+180)*10+1e-7).astype(int);p['iy']=np.floor((p.lat+90)*10+1e-7).astype(int)
p=p.drop_duplicates(['ix','iy'])
pg=gpd.GeoDataFrame(p,geometry=[box(-180+i*.1,-90+j*.1,-180+(i+1)*.1,-90+(j+1)*.1) for i,j in zip(p.ix,p.iy)],crs=4326)

code="""
from netCDF4 import Dataset
import numpy as np
f=Dataset('/tmp/BRAVES_Total_2017.nc');v=np.asarray(f['CO2_INV'][0,0].filled(np.nan))
x=float(f.XORIG)+(np.arange(v.shape[1])+.5)*float(f.XCELL)
y=float(f.YORIG)+(np.arange(v.shape[0])+.5)*float(f.YCELL)
np.savez('${SERVER_HOME}/result/external_validation_multicity_v1/braves_arrays.npz',lon=x,lat=y,value=v)
"""
env=os.environ.copy();env['PYTHONPATH']='/tmp/netcdf_validation_deps'
subprocess.run([sys.executable,'-c',code],env=env,check=True)
f=np.load(O/'braves_arrays.npz');lon,lat,v=[np.asarray(f[k], dtype=np.float64) for k in ['lon','lat','value']]
from scipy.sparse import csr_matrix
def overlap_matrix(centres, origin, spherical=False):
    spacing = abs(float(np.median(np.diff(centres))))
    low, high = centres-spacing/2, centres+spacing/2
    first = int(np.floor((low.min()-origin)*10))
    last = int(np.floor((high.max()-origin)*10))
    rows, cols, weights = [], [], []
    for j, (a, z) in enumerate(zip(low, high)):
        for i in range(int(np.floor((a-origin)*10)), int(np.floor((z-origin)*10))+1):
            left, right = max(a, origin+i*.1), min(z, origin+(i+1)*.1)
            if right <= left:
                continue
            fraction = (np.sin(np.deg2rad(right))-np.sin(np.deg2rad(left)))/(np.sin(np.deg2rad(z))-np.sin(np.deg2rad(a))) if spherical else (right-left)/(z-a)
            rows.append(i-first); cols.append(j); weights.append(fraction)
    return csr_matrix((weights, (rows, cols)), shape=(last-first+1, len(centres))), np.arange(first, last+1)
wx, ix = overlap_matrix(lon, -180)
wy, iy = overlap_matrix(lat, -90, spherical=True)
valid = np.isfinite(v) & (v >= 0)
aggregated = wx.dot(wy.dot(np.where(valid, v, 0)).T).T
coverage = wx.dot(wy.dot(valid.astype(float)).T).T
assert np.isclose(aggregated.sum(), np.where(valid, v, 0).sum(), rtol=1e-8)
xx, yy = np.meshgrid(ix, iy)
b = pd.DataFrame({'ix': xx.ravel(), 'iy': yy.ravel(), 'reference': aggregated.ravel(), 'coverage': coverage.ravel()})
b = b[b.coverage > 0]
bp=pg.merge(b,on=['ix','iy'])
metrics=[];status=[]
def report(city,country,name,ref,model,labels,scale):
    ref=np.asarray(ref);model=np.asarray(model)
    keep=np.isfinite(ref)&np.isfinite(model)
    ref,model=ref[keep],model[keep]
    result={'country':country,'city':name,'city_id':int(city),'n':len(ref),'comparison_scale':scale}
    if len(ref)<3 or np.unique(ref).size<2 or np.unique(model).size<2:
        result.update(status='insufficient_spatial_units');status.append(result);return
    result.update(status='completed',spearman_rho=float(spearmanr(ref,model).statistic),log1p_pearson_r=float(pearsonr(np.log1p(ref),np.log1p(model)).statistic))
    metrics.append(result);status.append(result)
    folder=O/country;folder.mkdir(exist_ok=True)
    pd.DataFrame({'spatial_unit':np.asarray(labels)[keep],'reference':ref,'model_unharmonized_units':model}).to_csv(folder/f'{city}_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(4.5,4.5),layout='constrained')
    rs=ref/ref.sum();ms=model/model.sum()
    ax.scatter(np.log10(rs+1e-8),np.log10(ms+1e-8),s=20,color='#478A92',alpha=.7)
    ax.set_xlabel('Reference log10 spatial share');ax.set_ylabel('Model log10 spatial share')
    ax.set_title(f'{name}: n={len(ref)}, rho={result["spearman_rho"]:.3f}',fontsize=10)
    fig.savefig(folder/f'{city}_comparison.png',dpi=200);plt.close(fig)

# Use the complete set of Brazilian GHSL urban centres for which both inventories overlap.
for _,city in g[g.GC_CNT_GAD.eq('Brazil')].iterrows():
    s=bp[bp.intersects(city.geometry)]
    report(city.ID_UC_G0,'Brazil',city.GC_UCN_MAI,s.reference,s.spatial_predicted_emission,s.cell_id,'0.1 degree conservative area-overlap remapping; BRAVES native ~5 km')
pd.DataFrame(metrics).to_csv(O/'brazil_all_city_metrics.csv',index=False)
pd.DataFrame(status).to_csv(O/'brazil_all_city_status.csv',index=False)
print('BRAZIL',len(metrics),'completed from',len(status),'cities',flush=True)

metrics=[];status=[]
districts=gpd.read_file(R/'external_validation_asia_africa_v1/gadm41_KEN.gpkg',layer='ADM_ADM_2',engine='pyogrio')
sources=json.loads((R/'external_validation_asia_africa_v1/ClimateTRACE_KEN_road_CO2_2024.json').read_text())
lookup={x['name']:x for x in sources}
districts['reference']=districts.NAME_2.map(lambda x:lookup.get(x+' Constituency',{}).get('emissionsQuantity',np.nan))
# Common administrative areas: integrate coarse model cells by actual area intersection.
proj=pg.to_crs(6933);dp=districts.to_crs(6933)
values=[]
for poly in dp.geometry:
    candidates=proj.iloc[proj.sindex.query(poly)]
    fractions=candidates.geometry.intersection(poly).area/candidates.geometry.area
    values.append(float((candidates.spatial_predicted_emission*fractions.to_numpy()).sum()))
districts['model']=values
for _,city in g[g.GC_CNT_GAD.eq('Kenya')].iterrows():
    s=districts[districts.intersects(city.geometry)&districts.reference.notna()].copy()
    if len(s):
        # Keep the full selected administrative polygons in both model and reference sums.
        report(city.ID_UC_G0,'Kenya',city.GC_UCN_MAI,s.reference,s.model,s.NAME_2,'subcounty/constituency; coarse-grid area integration, not road-native')
pd.DataFrame(metrics).to_csv(O/'kenya_all_city_coarse_metrics.csv',index=False)
pd.DataFrame(status).to_csv(O/'kenya_all_city_coarse_status.csv',index=False)
print('KENYA',len(metrics),'completed from',len(status),'cities',flush=True)
print(pd.DataFrame(metrics).to_string(index=False),flush=True)
