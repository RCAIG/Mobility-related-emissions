import numpy as np
import pandas as pd
import osmnx as ox
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def load_graph_and_prepare(graphml_path, WEIGHT_COL="travel_time"):
    """
        参数:
            graphml_path: Path 或 str，graphml 文件路径
            weight_col: 使用的权重列名，默认为 "travel_time"
        """
    # 直接使用传入的路径
    G = ox.load_graphml(graphml_path)

    # 转成 GeoDataFrame
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    # 节点：给一个连续 idx
    node_ids = gdf_nodes.index.to_numpy()
    node_df = pd.DataFrame({
        "node_id": node_ids,
        "x": gdf_nodes["x"].to_numpy(),
        "y": gdf_nodes["y"].to_numpy(),
    })
    node_df["idx"] = np.arange(len(node_df), dtype=np.int32)

    node_id_to_idx = dict(zip(node_df["node_id"].values, node_df["idx"].values))

    # 边：OSMnx 是 MultiDiGraph，(u,v,key) 多条边并存
    # 我们压成“简单有向图”：对同一 (u,v) 取最小权重（最快那条）
    # 这样 CSR 才好建，且更快
    edges = gdf_edges.reset_index()[["u", "v", WEIGHT_COL, "length"]].copy()
    edges[WEIGHT_COL] = pd.to_numeric(edges[WEIGHT_COL], errors="coerce")

    edges = edges.dropna(subset=[WEIGHT_COL])

    # 取同一(u,v)最小权重
    edges = edges.groupby(["u", "v"], as_index=False)[WEIGHT_COL].min()

    # 映射到 idx
    edges["u_idx"] = edges["u"].map(node_id_to_idx).astype(np.int32)
    edges["v_idx"] = edges["v"].map(node_id_to_idx).astype(np.int32)

    n = len(node_df)

    # 构建 CSR：每条记录对应一条有向边 u_idx -> v_idx
    row = edges["u_idx"].to_numpy()
    col = edges["v_idx"].to_numpy()
    data = edges[WEIGHT_COL].to_numpy(dtype=np.float64)

    csr_base = csr_matrix((data, (row, col)), shape=(n, n))

    # KDTree：用于吸附 lon/lat 到最近节点
    coords = np.column_stack([node_df["x"].to_numpy(), node_df["y"].to_numpy()])
    kdtree = cKDTree(coords)

    # 为了“从 predecessors 回溯到边”时快速累加，我们需要：
    # 对于每个 CSR 非零元素，知道它对应的 (u_idx, v_idx)
    # csr_base.indices 是 v_idx，csr_base.indptr 定位每个 u_idx 的邻接范围
    # 我们展开出 data 的对应 u_idx
    indptr = csr_base.indptr
    indices = csr_base.indices
    edge_u_idx = np.empty_like(indices)
    for u in range(n):
        edge_u_idx[indptr[u]:indptr[u+1]] = u
    edge_v_idx = indices.copy()

    return node_df, csr_base, edge_u_idx, edge_v_idx, kdtree, gdf_edges



