from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
R = Path('${SERVER_HOME}/result')
O = R/'external_validation_multicity_v1'
p = pd.read_parquet(R/'road_level_mobility_carbon_reconstruction_spatial_examples_v2/results/full_fit_spatial_predictions.parquet')
p = p[p.year.eq(2024)].copy()
p['ix'] = np.floor((p.lon+180)*10+1e-7).astype(int)
p['iy'] = np.floor((p.lat+90)*10+1e-7).astype(int)
p = p.drop_duplicates(['ix', 'iy'])
rows = []
for path in sorted((O/'reference_arrays').glob('*.npz')):
    f = np.load(path)
    lat, lon, val, days = [f[k] for k in ['lat','lon','emission','valid_days']]
    dx, dy = abs(np.median(np.diff(lon))), abs(np.median(np.diff(lat)))
    totals = {}
    coverage = {}
    for j, y in enumerate(lat):
        for k, x in enumerate(lon):
            if days[j,k] == 0:
                continue
            a, b, c, d = x-dx/2, x+dx/2, y-dy/2, y+dy/2
            for ix in range(int(np.floor((a+180)*10)), int(np.floor((b+180)*10))+1):
                left, right = max(a,-180+ix*.1), min(b,-180+(ix+1)*.1)
                for iy in range(int(np.floor((c+90)*10)), int(np.floor((d+90)*10))+1):
                    low, high = max(c,-90+iy*.1), min(d,-90+(iy+1)*.1)
                    if right <= left or high <= low:
                        continue
                    weight = (right-left)/(b-a)*(np.sin(np.deg2rad(high))-np.sin(np.deg2rad(low)))/(np.sin(np.deg2rad(d))-np.sin(np.deg2rad(c)))
                    totals[ix,iy] = totals.get((ix,iy),0)+float(val[j,k])*weight
                    target_area = .1*(np.sin(np.deg2rad(-90+(iy+1)*.1))-np.sin(np.deg2rad(-90+iy*.1)))
                    covered_area = (right-left)*(np.sin(np.deg2rad(high))-np.sin(np.deg2rad(low)))
                    coverage[ix,iy] = coverage.get((ix,iy),0)+covered_area/target_area
    reference = pd.DataFrame([{'ix':i,'iy':j,'reference':v,'covered_fraction':min(1,coverage[i,j])} for (i,j),v in totals.items()])
    s = reference.merge(p[['ix','iy','spatial_predicted_emission']],on=['ix','iy'])
    s['spatial_predicted_emission'] *= s.covered_fraction
    item = {'city':path.stem,'reference_year':2021,'model_year':2024,'n':len(s),'comparison_scale':'0.1 degree; conservative reference remapping; model area-weighted to valid reference footprint', 'status':'insufficient_spatial_units'}
    if len(s)>=3 and s.reference.nunique()>1 and s.spatial_predicted_emission.nunique()>1:
        item.update(status='completed',spearman_rho=float(spearmanr(s.reference,s.spatial_predicted_emission).statistic))
    rows.append(item)
    folder = O/'India_coarse'; folder.mkdir(exist_ok=True)
    s.to_csv(folder/(path.stem+'_comparison.csv'),index=False)
pd.DataFrame(rows).to_csv(O/'india_coarse_all_city_metrics.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False),flush=True)
