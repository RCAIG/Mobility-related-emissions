from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

O = Path('${SERVER_HOME}/result/external_validation_multicity_v1')
cached = O/'india_city_spatial_block_uncertainty.csv'
rows = pd.read_csv(cached).to_dict('records') if cached.exists() else []
completed = {row['city'] for row in rows}
for path in sorted((O/'India').glob('*/native_500m_comparison.csv')):
    if path.parent.name in completed:
        continue
    frame = pd.read_csv(path)
    frame = frame[frame.valid_days.gt(0)].copy()
    for factor in [1, 2, 4]:
        z = frame.assign(ry=frame.row//factor, cx=frame.col//factor).groupby(['ry','cx'],as_index=False)[['reference_tCO2_2021','model_emission_unharmonized_units']].sum()
        z['block_y'] = z.ry // max(1,4//factor)
        z['block_x'] = z.cx // max(1,4//factor)
        groups = [group.index.to_numpy() for _,group in z.groupby(['block_y','block_x'])]
        reference = z.reference_tCO2_2021.to_numpy()
        model = z.model_emission_unharmonized_units.to_numpy()
        rng = np.random.default_rng(42)
        estimates = []
        for _ in range(1000):
            selected = rng.integers(0,len(groups),size=len(groups))
            indices = np.concatenate([groups[k] for k in selected])
            if np.unique(model[indices]).size > 1 and np.unique(reference[indices]).size > 1:
                estimates.append(float(spearmanr(reference[indices],model[indices]).statistic))
        low, high = np.quantile(estimates,[.025,.975])
        rows.append({'city':path.parent.name,'nominal_resolution_m':factor*500,'n':len(z),'spearman_rho':float(spearmanr(reference,model).statistic),'spatial_block_bootstrap_ci_low':low,'spatial_block_bootstrap_ci_high':high,'bootstrap_repetitions':len(estimates),'nominal_bootstrap_block_m':2000,'number_spatial_blocks':len(groups)})
table = pd.DataFrame(rows)
table.to_csv(O/'india_city_spatial_block_uncertainty.csv',index=False)
print(table.query('nominal_resolution_m==500').to_string(index=False),flush=True)
