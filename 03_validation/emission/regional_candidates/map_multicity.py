from pathlib import Path
import json
import geopandas as gpd
import pyogrio
from shapely.geometry import box
R=Path('${SERVER_HOME}/result')
p=R/'city_boundaries_2024.gpkg'
print(pyogrio.list_layers(p),flush=True)
g=gpd.read_file(p,engine='pyogrio')
print('columns',list(g.columns),flush=True)
inv=json.loads((R/'external_validation_asia_africa_v1/chetna_netcdf_inventory.json').read_text())
out=[]
for x in inv:
    a,b=x['lat_range']; c,d=x['lon_range']; bbox=box(c,a,d,b)
    z=g[g.intersects(bbox)].copy()
    zz=z.to_crs(6933)
    area=zz.geometry.intersection(gpd.GeoSeries([bbox],crs=4326).to_crs(6933).iloc[0]).area
    z['overlap_km2']=area.to_numpy()/1e6
    columns=[c for c in z.columns if c!='geometry']
    rows=z.sort_values('overlap_km2',ascending=False).head(4)[columns].to_dict('records')
    print(x['attrs']['city'],rows,flush=True)
    out.append({'reference':x,'candidates':rows})
(R/'external_validation_multicity_v1/india_city_matching_audit.json').write_text(json.dumps(out,default=str,indent=2))
