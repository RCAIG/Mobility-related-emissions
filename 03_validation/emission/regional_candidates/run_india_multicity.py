import os, sys, json, subprocess, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from shapely import wkt
from shapely.geometry import box
from shapely.ops import transform
from shapely.strtree import STRtree
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/tmp')
import existing_road_reconstruction as old

R=Path('${SERVER_HOME}/result')
REF=R/'external_validation_asia_africa_v1'
OUT=R/'external_validation_multicity_v1'; OUT.mkdir(exist_ok=True)
BASE=R/'road_level_mobility_carbon_reconstruction_spatial_examples_v2'
old.CITY_CELL_REFERENCE=R/'spatial_mobility_carbon_validation_v2/analysis/city_cell_activity_reference_v2.parquet'

def reconstruct(city_id,name):
    existing=BASE/'results'/f'road_emission_{city_id}_2024.parquet'
    if existing.exists(): return existing, 'existing_road_reconstruction'
    output=OUT/'reconstruction'
    cached=output/'results'/f'road_emission_{city_id}_2024.parquet'
    if cached.exists(): return cached,'cached_multicity_reconstruction'
    matches=[]
    for disk in ['${SERVER_DISK_ROOT}','${SERVER_DISK_ROOT}']:
        for p in (Path(disk)/'zhixing/result').glob('trace*2024'):
            if next(p.glob(f'od_paths_flows_{city_id}_chunk_*.parquet'),None): matches.append(p)
    if not matches: raise FileNotFoundError(f'No original simulated route chunks for city {city_id} ({name})')
    old.CITY_META[city_id]={'name':name,'income':'multicity'}
    old.ROUTE_ROOTS['multicity']=str(matches[0])
    diagnostic=old.extract_city_year(city_id,2024,str(output))
    if diagnostic['status']=='failed': raise RuntimeError(diagnostic)
    edges=pd.read_parquet(output/'activity'/f'road_activity_{city_id}_2024.parquet')
    links=pd.read_parquet(output/'activity'/f'edge_cell_activity_{city_id}_2024.parquet')
    global_data=pd.read_parquet(BASE/'results/full_fit_spatial_predictions.parquet')
    grid=links.groupby(['city_id','year','cell_id'],as_index=False)[old.CLASS_COLS].sum()
    grid['road_activity_total']=grid[old.CLASS_COLS].sum(axis=1)
    grid=grid.merge(global_data[['cell_id','year','activity_total','spatial_predicted_emission']],on=['cell_id','year'],how='left')
    city=pd.read_parquet(old.CITY_CELL_REFERENCE,columns=['cell_id','year','city_id','activity_total'])
    city=city[city.city_id.eq(city_id)].rename(columns={'activity_total':'city_activity'})
    grid=grid.merge(city,on=['cell_id','year','city_id'],how='left')
    grid['city_share']=(grid.city_activity/grid.activity_total.clip(lower=old.EPS)).fillna(grid.road_activity_total/grid.activity_total.clip(lower=old.EPS)).clip(0,1)
    missing=grid.spatial_predicted_emission.isna()
    if missing.all(): raise ValueError('No global predictions matched this city')
    ratio=np.nanmedian(grid.loc[~missing,'spatial_predicted_emission']/grid.loc[~missing,'road_activity_total'].clip(lower=old.EPS))
    grid.loc[missing,'spatial_predicted_emission']=grid.loc[missing,'road_activity_total']*ratio
    grid.loc[missing,'city_share']=1
    grid['grid_emission']=grid.spatial_predicted_emission*grid.city_share
    coef=pd.read_csv(BASE/'tables/distance_class_response_coefficients.csv').sort_values('class')
    z=np.log1p(grid[old.CLASS_COLS].to_numpy(float))
    weight=z*coef.linear_log_activity_coefficient.to_numpy()+z**2*coef.quadratic_log_activity_coefficient.to_numpy()
    shares=weight/np.maximum(weight.sum(axis=1,keepdims=True),old.EPS)
    for i in range(1,6): grid[f'class_emission_c{i}']=grid.grid_emission*shares[:,i-1]
    a=links.merge(grid[['cell_id']+[f'class_emission_c{i}' for i in range(1,6)]],on='cell_id')
    den=grid.set_index('cell_id')[old.CLASS_COLS]
    for i in range(1,6): a[f'edge_emission_c{i}']=a[f'class_emission_c{i}']*a[f'activity_c{i}']/a.cell_id.map(den[f'activity_c{i}']).clip(lower=old.EPS)
    ec=[f'edge_emission_c{i}' for i in range(1,6)]
    e=a.groupby('edge_code',as_index=False)[ec].sum()
    e['reconstructed_emission']=e[ec].sum(axis=1)
    result=edges.merge(e,on='edge_code',how='left').fillna({'reconstructed_emission':0})
    cached.parent.mkdir(parents=True,exist_ok=True)
    result.to_parquet(cached,index=False)
    grid.to_csv(cached.parent/f'grid_emission_{city_id}_2024.csv',index=False)
    audit={'city_id':city_id,'city':name,'missing_global_grid_predictions':int(missing.sum()),'n_global_cells':len(grid),
           'road_total':float(result.reconstructed_emission.sum()),'grid_total':float(grid.grid_emission.sum()),'route_directory':str(matches[0])}
    (output/'tables'/f'reconstruction_audit_{city_id}_2024.json').write_text(json.dumps(audit,indent=2))
    if not np.isclose(audit['road_total'],audit['grid_total'],rtol=1e-6): raise ValueError('Road/grid mass conservation failed')
    return cached,'new_road_reconstruction'

def compare(item):
    name=item['reference']['attrs']['city']
    city_id=int(item['candidates'][0]['ID_UC_G0'])
    folder=OUT/'India'/name; folder.mkdir(parents=True,exist_ok=True)
    try:
        road_path,status=reconstruct(city_id,name)
        f=np.load(OUT/'reference_arrays'/(name+'.npz'))
        lat,lon,ref,days=[f[k] for k in ['lat','lon','emission','valid_days']]
        ny,nx=ref.shape
        dx=float(np.median(np.diff(lon))); dy=float(np.median(np.diff(lat)))
        epsg=32600+int((np.mean(lon)+180)//6)+1
        proj=Transformer.from_crs(4326,epsg,always_xy=True).transform
        cells=[transform(proj,box(x-dx/2,y-dy/2,x+dx/2,y+dy/2)) for y in lat for x in lon]
        tree=STRtree(cells); model=np.zeros(ny*nx)
        roads=pd.read_parquet(road_path,columns=['geometry_wkt','reconstructed_emission'])
        for shape,amount in zip(roads.geometry_wkt,roads.reconstructed_emission):
            if amount<=0: continue
            line=transform(proj,wkt.loads(shape))
            if line.length<=0: continue
            for idx in tree.query(line):
                length=line.intersection(cells[int(idx)]).length
                if length>0: model[int(idx)]+=amount*length/line.length
        base=pd.DataFrame({'row':np.repeat(np.arange(ny),nx),'col':np.tile(np.arange(nx),ny),
              'lon':np.tile(lon,ny),'lat':np.repeat(lat,nx),'reference_tCO2_2021':ref.ravel(),
              'valid_days':days.ravel(),'model_emission_unharmonized_units':model})
        base.to_csv(folder/'native_500m_comparison.csv',index=False)
        metrics=[]
        for factor in [1,2,4]:
            z=base[base.valid_days>0].copy()
            z['row_block']=z.row//factor;z['col_block']=z.col//factor
            z=z.groupby(['row_block','col_block'])[['reference_tCO2_2021','model_emission_unharmonized_units']].sum()
            for support in ['all_reference_valid','positive_union','both_positive']:
                r=z.reference_tCO2_2021.to_numpy();p=z.model_emission_unharmonized_units.to_numpy()
                keep=np.ones(len(r),bool)
                if support=='positive_union': keep=(r>0)|(p>0)
                if support=='both_positive': keep=(r>0)&(p>0)
                r,p=r[keep],p[keep]
                if len(r)<3 or np.unique(p).size<2: continue
                n_top=max(1,int(np.ceil(len(r)*.1)))
                rt=set(np.argsort(r)[-n_top:]);pt=set(np.argsort(p)[-n_top:])
                metrics.append({'city':name,'city_id':city_id,'reference':'CHETNA-Road','reference_year':2021,'model_year':2024,
                    'nominal_resolution_m':factor*500,'support':support,'n':len(r),'spearman_rho':float(spearmanr(r,p).statistic),
                    'log1p_pearson_r':float(pearsonr(np.log1p(r),np.log1p(p)).statistic),'top_decile_recall':len(rt&pt)/n_top,
                    'reference_mass_with_model_positive_fraction':float(r[p>0].sum()/r.sum()),
                    'model_positive_cell_fraction':float((p>0).mean()),'min_valid_days':int(base.loc[base.valid_days>0,'valid_days'].min()),
                    'reconstruction_status':status})
        pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False)
        z=base[base.valid_days>0]; fig,ax=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for a,key,title in zip(ax[:2],['reference_tCO2_2021','model_emission_unharmonized_units'],['CHETNA-Road 2021','Our reconstruction 2024']):
            share=z[key]/z[key].sum()
            c=a.scatter(z.lon,z.lat,c=np.log10(share+1e-8),s=5,cmap='viridis',vmin=-8,vmax=-2)
            a.set_title(title,fontsize=10);a.set_aspect(1/np.cos(np.deg2rad(np.mean(lat))));a.set_xlabel('Longitude');a.set_ylabel('Latitude')
        fig.colorbar(c,ax=ax[:2],label='log10 emission share within reference valid footprint',shrink=.7)
        rs=z.reference_tCO2_2021/z.reference_tCO2_2021.sum();ps=z.model_emission_unharmonized_units/z.model_emission_unharmonized_units.sum()
        ax[2].hexbin(np.log10(rs+1e-8),np.log10(ps+1e-8),gridsize=40,mincnt=1,cmap='viridis')
        ax[2].plot([-8,-2],[-8,-2],color='.5',lw=1)
        ax[2].set_xlabel('Reference log10 share');ax[2].set_ylabel('Model log10 share')
        ax[2].set_title(f"{name}; rho = {metrics[0]['spearman_rho']:.3f}",fontsize=10)
        fig.savefig(folder/'comparison_500m.png',dpi=250);plt.close(fig)
        return {'city':name,'city_id':city_id,'status':'completed','metrics':metrics}
    except Exception as e:
        failure={'city':name,'city_id':city_id,'status':'failed','reason':str(e),'traceback':traceback.format_exc()}
        (folder/'failure.json').write_text(json.dumps(failure,indent=2))
        return failure

if __name__=='__main__':
    items=json.loads((OUT/'india_city_matching_audit.json').read_text())
    (OUT/'reference_arrays').mkdir(exist_ok=True)
    code="""
from pathlib import Path
import json,numpy as np
from netCDF4 import Dataset
r=Path('${SERVER_HOME}/result')
items=json.loads((r/'external_validation_multicity_v1/india_city_matching_audit.json').read_text())
for item in items:
 name=item['reference']['attrs']['city']
 with Dataset(r/'external_validation_asia_africa_v1'/item['reference']['file']) as f:
  a=np.asarray(f['co2'][:].filled(np.nan))
  np.savez(r/'external_validation_multicity_v1/reference_arrays'/(name+'.npz'),lat=f['lat'][:],lon=f['lon'][:],emission=np.nansum(a,axis=0),valid_days=np.isfinite(a).sum(axis=0))
"""
    env=os.environ.copy();env['PYTHONPATH']='/tmp/netcdf_validation_deps'
    subprocess.run([sys.executable,'-c',code],env=env,check=True)
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(compare,x):x for x in items}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print({k:v for k,v in result.items() if k not in ['metrics','traceback']},flush=True)
            (OUT/'india_progress.json').write_text(json.dumps(results,indent=2))
    rows=[row for result in results for row in result.get('metrics',[])]
    pd.DataFrame(rows).to_csv(OUT/'india_all_city_metrics.csv',index=False)
    pd.DataFrame([{k:v for k,v in x.items() if k not in ['metrics','traceback']} for x in results]).to_csv(OUT/'india_city_status.csv',index=False)
    print(pd.DataFrame(rows).query("support=='all_reference_valid'").to_string(index=False),flush=True)
