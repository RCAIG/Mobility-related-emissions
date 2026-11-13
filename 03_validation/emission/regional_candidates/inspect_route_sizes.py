from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
root = Path('${SERVER_HOME}/result/trace_genetare_lower_middle_2024')
records = []
for city in [10300, 9524, 9558]:
    files = sorted(root.glob(f'od_paths_flows_{city}_chunk_*.parquet'))
    for p in files:
        records.append({'city_id':city,'file':p.name,'routes':pq.ParquetFile(p).metadata.num_rows,'bytes':p.stat().st_size})
frame = pd.DataFrame(records)
frame.to_csv('${SERVER_HOME}/result/external_validation_multicity_v1/large_city_route_inventory.csv',index=False)
print(frame.groupby('city_id').agg(chunks=('file','count'), routes=('routes','sum'), bytes=('bytes','sum')).to_string(),flush=True)
