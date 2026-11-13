import pandas as pd
from pathlib import Path
root=Path('${SERVER_HOME}/result/road_level_mobility_carbon_reconstruction_spatial_examples_v2')
for p in [root/'results/road_emission_7599_2024.parquet',root/'activity/edge_activity_7599_2024.parquet']:
    if p.exists():
        d=pd.read_parquet(p)
        print(str(p), d.shape, d.dtypes.to_string(), d.head(2).to_string(), sep='\n')
print([p.name for p in (root/'activity').iterdir()])
