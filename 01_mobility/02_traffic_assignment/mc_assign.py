import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm
import gc
from config import N_SIM, SIGMA, DETOUR_RATIO, RANDOM_SEED


# ===== 辅助函数 =====
def get_path_edges(pred, origin, dest):
    edges = []
    v = dest
    while v != origin and v != -9999:
        u = pred[v]
        if u < 0:
            return None
        edges.append((u, v))
        v = u
    if v == origin:
        return edges[::-1]
    return None


def make_noisy_csr(csr_base, rng, sigma):
    base_data = csr_base.data
    z = rng.standard_normal(size=base_data.shape[0])
    noisy_data = base_data * np.exp(sigma * z)
    return csr_matrix((noisy_data, csr_base.indices, csr_base.indptr), shape=csr_base.shape)


# ===== 核心优化：生成器模式 + 内存安全累加 =====
class PathFlowAccumulator:
    """
    内存优化的路径流量累加器（生成器模式）
    - 每 flush 一次直接返回当前 chunk 数据
    - 不再累积全局数据，避免内存爆炸
    """

    def __init__(self):
        self.path_to_id = {}  # {(u,v,...): int_id}
        self.id_to_path = []  # [path_tuple, ...]
        self.od_path_flows = defaultdict(lambda: defaultdict(float))  # {(o,d): {path_id: flow}}
        self.total_flow = 0.0

    def add_flow(self, o, d, path_edges, flow):
        if path_edges is None:
            return
        path_tuple = tuple(path_edges)

        if path_tuple not in self.path_to_id:
            path_id = len(self.id_to_path)
            self.path_to_id[path_tuple] = path_id
            self.id_to_path.append(path_tuple)
        else:
            path_id = self.path_to_id[path_tuple]

        self.od_path_flows[(o, d)][path_id] += flow
        self.total_flow += flow

    def flush_and_reset(self):
        """彻底重置累加器，避免闭包和引用残留"""
        if not self.od_path_flows:
            return {}

        # 1. 构建返回数据（深拷贝路径元组，解除引用）
        chunk_data = {}
        for (o, d), path_flows in self.od_path_flows.items():
            chunk_data[(o, d)] = {
                tuple(path_tuple): flow
                for path_id, flow in path_flows.items()
                for path_tuple in [self.id_to_path[path_id]]
            }

        # 2. 彻底销毁旧对象
        del self.od_path_flows
        del self.path_to_id
        del self.id_to_path

        # 3. 重建全新对象
        self.od_path_flows = defaultdict(lambda: defaultdict(float))
        self.path_to_id = {}
        self.id_to_path = []
        self.total_flow = 0.0

        # 4. 强制垃圾回收
        gc.collect()

        return chunk_data


def mc_assign_origin_level(csr_base, origin_to_dests, flush_every=100, start_origin_idx=0):
    """
    路径级MC分配（支持跳过已处理 origins）

    参数:
        origin_to_dests: 完整的 {origin: [(dest, flow), ...]} 字典
        start_origin_idx: 从第 N 个 origin 开始处理（按字典 keys() 顺序）
    """
    rng = np.random.default_rng(RANDOM_SEED)
    origins = list(origin_to_dests.keys())
    total_origins = len(origins)

    # === 跳过已处理的 origins（同步随机数状态）===
    if start_origin_idx > 0:
        if start_origin_idx >= total_origins:
            print(f"  ⚠ 跳过所有 origins（start_origin_idx={start_origin_idx} >= total={total_origins}）")
            return  # 无数据可处理

        # 消耗随机数以同步状态
        n_edges = csr_base.data.shape[0]
        randoms_to_skip = start_origin_idx * N_SIM * n_edges

        if randoms_to_skip > 0:
            batch_size = 10_000_000
            batches = (randoms_to_skip + batch_size - 1) // batch_size

            print(f"  → 跳过前 {start_origin_idx:,} 个 origins（同步随机数状态）")
            for i in range(batches):
                size = min(batch_size, randoms_to_skip - i * batch_size)
                rng.standard_normal(size)

            print(f"  ✓ 随机数状态同步完成")

        # ✅ 关键：仅截断 origins 列表，不修改 origin_to_dests 字典
        origins = origins[start_origin_idx:]
        # 注意：origin_to_dests 保持完整，后续通过 origins 列表索引访问

    accumulator = PathFlowAccumulator()
    total_remaining = len(origins)

    # 使用 tqdm 显示全局进度（包含已跳过部分）
    for idx, o in enumerate(tqdm(
            origins,
            desc="Monte Carlo by origin",
            initial=start_origin_idx,
            total=total_origins,
            unit="origin"
    )):
        dests = origin_to_dests[o]  # 从完整字典中获取
        d_list = [d for d, _ in dests]
        f_list = np.array([f for _, f in dests], dtype=np.float64)

        # 基准最短路
        dist0, pred0 = dijkstra(csr_base, directed=True, indices=o, return_predecessors=True)
        dist0_d = np.array([dist0[d] for d in d_list], dtype=np.float64)

        reachable_mask = np.isfinite(dist0_d)
        if not reachable_mask.any():
            continue

        d_list = [d for d, ok in zip(d_list, reachable_mask) if ok]
        f_list = f_list[reachable_mask]
        dist0_d = dist0_d[reachable_mask]

        # Monte Carlo 模拟
        for _ in range(N_SIM):
            csr_noisy = make_noisy_csr(csr_base, rng, SIGMA)
            dist, pred = dijkstra(csr_noisy, directed=True, indices=o, return_predecessors=True)

            for d, flow, c0 in zip(d_list, f_list, dist0_d):
                c = dist[d]
                use_base = (not np.isfinite(c)) or (c > DETOUR_RATIO * c0)
                path_edges = get_path_edges(pred0, o, d) if use_base else get_path_edges(pred, o, d)

                if path_edges is not None:
                    accumulator.add_flow(o, d, path_edges, flow / N_SIM)

        # 到达 flush 点：返回当前 chunk 并重置
        if (idx + 1) % flush_every == 0 or idx == total_remaining - 1:
            chunk_data = accumulator.flush_and_reset()
            if chunk_data:
                yield chunk_data

    # 确保最后剩余数据被返回
    final_chunk = accumulator.flush_and_reset()
    if final_chunk:
        yield final_chunk
    gc.collect()