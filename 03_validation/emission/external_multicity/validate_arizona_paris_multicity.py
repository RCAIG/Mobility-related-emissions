from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box
from shapely import wkt
from scipy.stats import spearmanr,pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('${SERVER_HOME}/result');O=R/'external_validation_multicity_v1';O.mkdir(exist_ok=True)
g=gpd.read_file(R/'city_boundaries_2024.gpkg',engine='pyogrio')
p=pd.read_parquet(R/'road_level_mobility_carbon_reconstruction_spatial_examples_v2/results/full_fit_spatial_predictions.parquet')
p=p[p.year.eq(2024)].drop_duplicates('cell_id').copy()
cells=gpd.GeoDataFrame(p,geometry=[box(-180+(int(c)%3600)*.1,90-(int(c)//3600+1)*.1,-180+(int(c)%3600+1)*.1,90-(int(c)//3600)*.1) for c in p.cell_id],crs=4326)
rows=[];statuses=[]
def finish(city,country,ref,pred,labels,scale):
    ref=np.asarray(ref,float);pred=np.asarray(pred,float)
    valid=np.isfinite(ref)&np.isfinite(pred);ref=ref[valid];pred=pred[valid]
    item={'country':country,'city':city.GC_UCN_MAI,'city_id':int(city.ID_UC_G0),'n':len(ref),'comparison_scale':scale}
    if len(ref)<3 or np.unique(ref).size<2 or np.unique(pred).size<2:
        item['status']='insufficient_spatial_units';statuses.append(item);return
    item.update(status='completed',spearman_rho=float(spearmanr(ref,pred).statistic),log1p_pearson_r=float(pearsonr(np.log1p(ref),np.log1p(pred)).statistic))
    rows.append(item);statuses.append(item)
    folder=O/country;folder.mkdir(exist_ok=True)
    pd.DataFrame({'unit':np.asarray(labels)[valid],'reference':ref,'model_unharmonized_units':pred}).to_csv(folder/f'{int(city.ID_UC_G0)}_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(4.5,4.5),layout='constrained')
    ax.scatter(np.log10(ref/ref.sum()+1e-8),np.log10(pred/pred.sum()+1e-8),s=20,alpha=.7,color='#478A92')
    ax.set_xlabel('Reference log10 spatial share');ax.set_ylabel('Model log10 spatial share')
    ax.set_title(f'{city.GC_UCN_MAI}: n={len(ref)}, rho={item["spearman_rho"]:.3f}',fontsize=10)
    fig.savefig(folder/f'{int(city.ID_UC_G0)}_comparison.png',dpi=200);plt.close(fig)

# Hestia Arizona on-road, all GHSL cities with any reference road coverage.
source=R/'Experiments/external_validation_v1/inputs/hestia_arizona_onroad'
aggregate={};extent=None
for filename,layer in [('Local_AZ_v1_1.gpkg','Local_AZ_v1.1'),('NonLocal_AZ_v1_1.gpkg','NonLocal_AZ_v1.1')]:
    roads=gpd.read_file(source/filename,layer=layer,columns=['FFCO2tC2017','geometry'],engine='pyogrio')
    tx=Transformer.from_crs(roads.crs,4326,always_xy=True)
    for shape,amount in zip(roads.geometry,roads.FFCO2tC2017):
        if shape is None or not np.isfinite(amount) or amount<=0: continue
        parts=list(shape.geoms) if shape.geom_type=='MultiLineString' else [shape]
        total=sum(x.length for x in parts)
        if total<=0: continue
        for part in parts:
            xy=np.asarray(part.coords);delta=np.diff(xy,axis=0);lengths=np.linalg.norm(delta,axis=1)
            for start,change,length in zip(xy[:-1],delta,lengths):
                if length<=0:continue
                n=max(1,int(np.ceil(length/100)))
                samples=start+(np.arange(n)[:,None]+.5)/n*change
                lon,lat=tx.transform(samples[:,0],samples[:,1])
                col=np.floor((np.asarray(lon)+180)*10).astype(int);row=np.floor((90-np.asarray(lat))*10).astype(int)
                ids=row*3600+col
                for c in ids: aggregate[int(c)]=aggregate.get(int(c),0)+float(amount)*length/total/n
    print(filename,'done',len(aggregate),'coarse reference cells',flush=True)
az=cells.copy();az['reference']=az.cell_id.map(aggregate)
az=az[az.reference.notna()]
for _,city in g[g.GC_CNT_GAD.eq('United States')].iterrows():
    s=az[az.intersects(city.geometry)]
    if len(s):finish(city,'US_Arizona',s.reference,s.spatial_predicted_emission,s.cell_id,'0.1 degree, Hestia Arizona on-road 2017 vs model 2024; not 100m')
pd.DataFrame(rows).to_csv(O/'arizona_all_city_metrics.csv',index=False)
pd.DataFrame(statuses).to_csv(O/'arizona_all_city_status.csv',index=False)
print('ARIZONA',len(rows),'from',len(statuses),'cities',flush=True)

# AIRPARIF covers multiple cities in Ile-de-France, traffic-only CO2.
rows=[];statuses=[]
raw=pd.read_csv('/tmp/Paris_Traffic.csv',sep=';')
raw=raw[pd.to_numeric(raw.annee).eq(2019)].copy()
ref=raw.groupby('iris',as_index=False).agg(reference=('emission_kg_CO2','sum'),geom=('geom','first'))
reference=gpd.GeoDataFrame(ref,geometry=ref.geom.map(wkt.loads),crs=2154).to_crs(4326)
cp=cells.to_crs(2154);rp=reference.to_crs(2154)
values=[]
for poly in rp.geometry:
    s=cp.iloc[cp.sindex.query(poly)]
    fraction=s.geometry.intersection(poly).area/s.geometry.area
    values.append(float((s.spatial_predicted_emission*fraction.to_numpy()).sum()))
reference['model']=values
for _,city in g[g.GC_CNT_GAD.eq('France')].iterrows():
    s=reference[reference.intersects(city.geometry)].copy()
    if len(s):finish(city,'France_IDF',s.reference,s.model,s.iris,'IRIS units, coarse 0.1-degree model area integration; not road-native')
pd.DataFrame(rows).to_csv(O/'france_idf_all_city_metrics.csv',index=False)
pd.DataFrame(statuses).to_csv(O/'france_idf_all_city_status.csv',index=False)
print('FRANCE',len(rows),'from',len(statuses),'cities',flush=True)
