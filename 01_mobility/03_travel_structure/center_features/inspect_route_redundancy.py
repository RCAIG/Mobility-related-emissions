from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PATHS = [
    Path("${SERVER_HOME}/result/trace_genetare_hgih_income_2017/od_paths_flows_1262_chunk_0000.parquet"),
    Path("${SERVER_HOME}/result/trace_genetare_upper_middle_2017/od_paths_flows_2335_chunk_0000.parquet"),
    Path("${SERVER_HOME}/result/trace_genetare_lower_middle_2017/od_paths_flows_1301_chunk_0000.parquet"),
]


for path in PATHS:
    arrays = pq.read_table(path, columns=["edge_sequence"])["edge_sequence"].to_pylist()
    hashes = np.fromiter((hash(tuple(seq)) for seq in arrays), dtype=np.int64, count=len(arrays))
    lengths = np.fromiter((len(seq) for seq in arrays), dtype=np.int32, count=len(arrays))
    unique = np.unique(hashes).size
    print(
        path.name,
        f"rows={len(arrays)}",
        f"unique_hashes={unique}",
        f"unique_share={unique / len(arrays):.4f}",
        f"mean_edges={lengths.mean():.1f}",
        f"p95_edges={np.percentile(lengths, 95):.0f}",
    )
