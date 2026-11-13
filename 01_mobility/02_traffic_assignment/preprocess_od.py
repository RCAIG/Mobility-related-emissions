import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def snap_points(kdtree, node_df, lons, lats):
    q = np.column_stack([lons, lats])
    dist, idx = kdtree.query(q, k=1, workers=-1)
    return idx.astype(np.int32), dist


def load_and_group_od(node_df, kdtree, od_csv_path, min_flow=1.0, chunksize=200_000):
    """
    参数:
        node_df, kdtree: 同原版
        od_csv_path: Path 或 str，OD csv 文件路径
        min_flow: 最小流量阈值，默认 1.0
        chunksize: 读取分块大小
    """
    usecols = ["o_lon", "o_lat", "d_lon", "d_lat", "flow"]
    pair_flow = defaultdict(float)

    for chunk in tqdm(pd.read_csv(od_csv_path, usecols=usecols, chunksize=chunksize),
                      desc="Read+snap+aggregate OD"):
        chunk["flow"] = pd.to_numeric(chunk["flow"], errors="coerce").fillna(0.0)

        if min_flow > 0:
            chunk = chunk[chunk["flow"] >= min_flow]
        if len(chunk) == 0:
            continue

        o_idx, _ = snap_points(kdtree, node_df, chunk["o_lon"].values, chunk["o_lat"].values)
        d_idx, _ = snap_points(kdtree, node_df, chunk["d_lon"].values, chunk["d_lat"].values)

        tmp = pd.DataFrame({"o": o_idx, "d": d_idx, "flow": chunk["flow"].values})
        tmp = tmp[tmp["o"] != tmp["d"]]

        g = tmp.groupby(["o", "d"], as_index=False)["flow"].sum()

        for o, d, f in g.itertuples(index=False):
            pair_flow[(int(o), int(d))] += float(f)

    origin_to_dests = defaultdict(list)
    for (o, d), f in pair_flow.items():
        origin_to_dests[o].append((d, f))

    return origin_to_dests, len(pair_flow)
