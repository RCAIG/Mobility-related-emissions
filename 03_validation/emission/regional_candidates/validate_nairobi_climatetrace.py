import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.ops import transform
from shapely.strtree import STRtree
from pyproj import Transformer
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('${SERVER_HOME}/result/external_validation_asia_africa_v1')
districts = gpd.read_file(ROOT/'gadm41_KEN.gpkg', layer='ADM_ADM_2', engine='pyogrio')
districts = districts[districts.NAME_1.eq('Nairobi')].sort_values('NAME_2').reset_index(drop=True)
sources = json.loads((ROOT/'ClimateTRACE_KEN_road_CO2_2024.json').read_text())
lookup = {x['name']: x for x in sources}
selected = [lookup[x+' Constituency'] for x in districts.NAME_2]
assert len(districts) == 17
assert all(x['year'] == 2024 and x['gas'] == 'co2' and x['subsector'] == 'road-transportation' for x in selected)
districts['reference_tCO2'] = [x['emissionsQuantity'] for x in selected]
districts['reference_road_km'] = [x['capacity'] for x in selected]
districts['source_id'] = [x['id'] for x in selected]
projected = districts.to_crs(32737)
polygons = list(projected.geometry)
tree = STRtree(polygons)
roads = pd.read_parquet(ROOT/'nairobi_reconstruction/results/road_emission_1717_2024.parquet')
project = Transformer.from_crs(4326, 32737, always_xy=True).transform
values = np.zeros(17)
lengths = np.zeros(17)
counts = np.zeros(17, dtype=int)
geometries = []
for shape, value in zip(roads.geometry_wkt, roads.reconstructed_emission):
    line = transform(project, wkt.loads(shape))
    if line.length <= 0:
        continue
    geometries.append(line)
    for index in tree.query(line):
        i = int(index)
        length = line.intersection(polygons[i]).length
        if length > 0:
            values[i] += float(value) * length / line.length
            lengths[i] += length / 1000
            counts[i] += 1
districts['model_emission_unharmonized_units'] = values
districts['model_active_directed_road_km'] = lengths
districts['model_intersecting_edge_count'] = counts
districts['area_km2'] = projected.area.to_numpy()/1e6
# This extent audit is not an estimate of inventory coverage; inactive roads are absent.
footprint = gpd.GeoSeries(geometries, crs=32737).union_all().convex_hull
districts['active_road_convex_hull_area_fraction'] = [p.intersection(footprint).area/p.area for p in polygons]
districts['reference_share'] = districts.reference_tCO2/districts.reference_tCO2.sum()
districts['model_share'] = values/values.sum()
metrics = []
for mode in ['district_totals', 'district_density_per_km2']:
    ref = districts.reference_tCO2.to_numpy()
    pred = values.copy()
    if mode.endswith('km2'):
        ref = ref/districts.area_km2.to_numpy()
        pred = pred/districts.area_km2.to_numpy()
    rho = float(spearmanr(ref, pred).statistic)
    loo = [float(spearmanr(np.delete(ref,i), np.delete(pred,i)).statistic) for i in range(17)]
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(3000):
        idx = rng.integers(0,17,17)
        if np.unique(ref[idx]).size > 1 and np.unique(pred[idx]).size > 1:
            samples.append(spearmanr(ref[idx], pred[idx]).statistic)
    high_ref = set(np.argsort(ref)[-2:]); high_pred = set(np.argsort(pred)[-2:])
    metrics.append({'city':'Nairobi','reference':'Climate TRACE road transportation CO2',
        'reference_year':2024,'model_year':2024,'spatial_units':'17 constituencies, GADM 4.1',
        'comparison':mode,'n':17,'spearman_rho':rho,
        'spearman_bootstrap_95_low':float(np.quantile(samples,.025)),
        'spearman_bootstrap_95_high':float(np.quantile(samples,.975)),
        'leave_one_out_rho_min':min(loo),'leave_one_out_rho_max':max(loo),
        'log1p_pearson_r':float(pearsonr(np.log1p(ref),np.log1p(pred)).statistic),
        'top_two_overlap_fraction':len(high_ref&high_pred)/2,
        'note':'External spatial consistency; no reference-based fitting or total rescaling. Administrative vs GHSL footprint differences remain.'})
districts.drop(columns='geometry').to_csv(ROOT/'nairobi_district_comparison_2024.csv',index=False)
pd.DataFrame(metrics).to_csv(ROOT/'nairobi_validation_metrics.csv',index=False)
districts.to_file(ROOT/'nairobi_comparison.gpkg',driver='GPKG',engine='pyogrio')
fig, ax = plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
upper = max(districts.reference_share.max(),districts.model_share.max())
for a, key, title in zip(ax[:2], ['reference_share','model_share'], ['Climate TRACE 2024','Our road reconstruction 2024']):
    districts.plot(column=key,ax=a,cmap='YlOrRd',vmin=0,vmax=upper,edgecolor='white',linewidth=.6)
    a.set_title(title,fontsize=11); a.set_axis_off()
    for i in [int(districts.reference_share.idxmax()),int(districts.model_share.idxmax())]:
        p=districts.geometry.iloc[i].representative_point()
        a.annotate(districts.NAME_2.iloc[i],(p.x,p.y),fontsize=7)
fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,upper),cmap='YlOrRd'),ax=ax[:2],label='Share of Nairobi district total',shrink=.7)
ax[2].scatter(districts.reference_share, districts.model_share, s=40, color='#478A92',edgecolor='white')
ax[2].plot([0,upper],[0,upper],color='.6',linewidth=1)
for i in np.argsort(np.abs(districts.reference_share-districts.model_share))[-4:]:
    ax[2].annotate(districts.NAME_2.iloc[i],(districts.reference_share.iloc[i],districts.model_share.iloc[i]),xytext=(3,3),textcoords='offset points',fontsize=7)
ax[2].set_xlabel('Climate TRACE emission share'); ax[2].set_ylabel('Our emission share')
ax[2].set_title(f"17 districts; Spearman rho = {metrics[0]['spearman_rho']:.3f}",fontsize=11)
fig.savefig(ROOT/'Nairobi_ClimateTRACE_district_2024.png',dpi=300)
plt.close(fig)
print(pd.DataFrame(metrics).to_string(index=False),flush=True)
print(districts[['NAME_2','reference_share','model_share','active_road_convex_hull_area_fraction']].to_string(index=False),flush=True)
