import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
import warnings
from config import OUT_DIR, MIN_FLOW
from build_network import load_graph_and_prepare
from preprocess_od import load_and_group_od
from mc_assign import mc_assign_origin_level
import numpy as np
import os
import gc
import pandas as pd
import time
import sys  # 新增：用于进程退出

# ====================== 路径配置 ======================
GRAPHML_ROOT = Path(
    "${SERVER_HOME}/result/road_2024")
OD_ROOT = Path(
    "${SERVER_DATA_ROOT}/prediction_travel_mode/generate-od-pubtools-main/generate-od-pubtools-main/assets/example_data/my_test/result/od_flows_wgs84_lower_middle_2024")

# 输出目录
OUTPUT_DIR = OUT_DIR
print(OUTPUT_DIR)

FLUSH_EVERY = 100  # 与 mc_assign.py 保持一致


def _get_expected_chunk_count(n_origins, flush_every=FLUSH_EVERY):
    """计算预期的 chunk 数量"""
    return (n_origins + flush_every - 1) // flush_every


def _all_chunks_exist(area_id, chunk_count):
    """检查该区域所有预期 chunk 文件是否都已存在"""
    for chunk_id in range(chunk_count):
        chunk_path = OUTPUT_DIR / f"od_paths_flows_{area_id}_chunk_{chunk_id:04d}.parquet"
        if not chunk_path.is_file():
            return False
    return True


def _process_chunk_to_df(chunk_data, idx_to_node):
    """将chunk数据转换为纯表格DataFrame（无几何）"""
    records = []

    for (o_idx, d_idx), path_dict in chunk_data.items():
        for path_tuple, flow in path_dict.items():
            # === 核心修改：将边序列编码为整数列表 ===
            edge_sequence = []
            for u_idx, v_idx in path_tuple:
                # 使用64位整数编码：高32位=u_idx, 低32位=v_idx
                # 假设节点索引 < 2^32 (40亿节点，实际路网远小于此)
                edge_code = (int(u_idx) << 32) | int(v_idx)
                edge_sequence.append(edge_code)

            if not edge_sequence:
                continue

            records.append({
                "o_idx": int(o_idx),
                "d_idx": int(d_idx),
                "o_node_id": idx_to_node.get(int(o_idx)),  # OSM ID (稳定标识)
                "d_node_id": idx_to_node.get(int(d_idx)),  # OSM ID (稳定标识)
                "flow": float(flow),
                "edge_sequence": edge_sequence  # List[int64] - Parquet原生支持
            })

    if not records:
        return None

    return pd.DataFrame(records)


def process_one_area(area_id: str):
    """处理单个区域，支持快速预检跳过 + chunk级断点续算 + 分块重启策略"""
    # 1. 确定文件路径
    od_path = OD_ROOT / f"{area_id}_od_flows_wgs84.csv"
    graph_path = GRAPHML_ROOT / f"road_network_drive_{area_id}.graphml"

    # 2. 检查输入文件
    if not od_path.is_file():
        print(f"跳过 {area_id}：OD 文件不存在 → {od_path}")
        return False

    if not graph_path.is_file():
        print(f"跳过 {area_id}：GraphML 文件不存在 → {graph_path}")
        return False

    # === 核心优化：快速预检（毫秒级）===
    # 仅读取 origin_id 列，统计唯一值数量（不过滤 MIN_FLOW，作为起点数上界）
    try:
        # 使用 chunksize 避免大文件内存峰值，同时快速统计唯一值
        origin_ids = set()
        for chunk in pd.read_csv(od_path, usecols=["origin_id"], chunksize=100000):
            origin_ids.update(chunk["origin_id"].unique())
        n_origins_upper_bound = len(origin_ids)
        total_expected_chunks = _get_expected_chunk_count(n_origins_upper_bound, FLUSH_EVERY)

        print(
            f"\n[快速预检] 区域 {area_id} | 估计起点数 ≤ {n_origins_upper_bound:,} | 预期 chunks ≤ {total_expected_chunks}")

        # 快速跳过：所有 chunk 已存在（保守判断：即使实际起点数略少，chunk文件仍完整）
        if total_expected_chunks > 0 and _all_chunks_exist(area_id, total_expected_chunks):
            print(f"  ✓ 跳过 {area_id}：所有 {total_expected_chunks} 个 chunk 已存在（快速预检，耗时 <100ms）")
            return True

    except Exception as e:
        print(f"  ⚠ 快速预检失败（{e}），回退到完整流程...")
        # 继续执行完整流程（不影响正确性）

    # === 完整流程：仅当需要实际处理时才加载路网 ===
    print(f"\n开始处理区域：{area_id}")
    print(f"  OD:   {od_path}")
    print(f"  Net:  {graph_path}")

    try:
        # 3. 加载路网（昂贵操作，仅在必要时执行）
        node_df, csr_base, edge_u_idx, edge_v_idx, kdtree, gdf_edges = load_graph_and_prepare(
            graphml_path=graph_path
        )

        # 4. 读取并吸附OD点（含 MIN_FLOW 过滤）
        origin_to_dests, n_pairs = load_and_group_od(
            node_df=node_df,
            kdtree=kdtree,
            od_csv_path=od_path,
            min_flow=MIN_FLOW
        )
        n_origins = len(origin_to_dests)
        total_expected_chunks = _get_expected_chunk_count(n_origins, FLUSH_EVERY)

        print(f"  Unique OD pairs after snap+aggregate: {n_pairs:,}")
        print(f"  Unique origins (after MIN_FLOW={MIN_FLOW}): {n_origins:,}")
        print(f"  Expected total chunks: {total_expected_chunks}")

        # === 二级跳过：精确检查所有 chunk 是否存在 ===
        if total_expected_chunks > 0 and _all_chunks_exist(area_id, total_expected_chunks):
            print(f"  ✓ 跳过 {area_id}：所有 {total_expected_chunks} 个 chunk 已存在（精确检查）")
            return True

        # === 检测连续存在的 chunk（从0开始）===
        skipped_chunks = 0
        while (OUTPUT_DIR / f"od_paths_flows_{area_id}_chunk_{skipped_chunks:04d}.parquet").exists():
            skipped_chunks += 1

        print(f"  已存在连续 chunk: {skipped_chunks} 个（chunk_0000 到 chunk_{skipped_chunks - 1:04d}）")

        skip_origin_count = skipped_chunks * FLUSH_EVERY
        if skip_origin_count >= n_origins:
            print(f"  ✓ 跳过 {area_id}：已处理所有 {n_origins} 个 origins（{skipped_chunks} 个 chunk）")
            return True

        # 保持完整字典 + 通过 start_origin_idx 跳过
        remaining_origin_to_dests = origin_to_dests
        print(f"  → 跳过前 {skip_origin_count:,} 个 origins（对应 {skipped_chunks} 个 chunk）")
        print(f"  → 剩余待处理 origins: {n_origins - skip_origin_count:,} 个")

        # 准备映射
        idx_to_node = dict(zip(node_df["idx"], node_df["node_id"]))
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # 分块重启策略
        enable_restart_strategy = (total_expected_chunks > 2)
        if enable_restart_strategy:
            print(f"  ⚡ 启用分块重启策略：总 chunk 数 {total_expected_chunks} > 2")

        total_records = 0
        successful_chunks = 0
        start_time = time.time()

        try:
            for relative_chunk_id, chunk_data in enumerate(
                    mc_assign_origin_level(
                        csr_base,
                        remaining_origin_to_dests,
                        flush_every=FLUSH_EVERY,
                        start_origin_idx=skip_origin_count
                    )
            ):
                absolute_chunk_id = skipped_chunks + relative_chunk_id
                chunk_path = OUTPUT_DIR / f"od_paths_flows_{area_id}_chunk_{absolute_chunk_id:04d}.parquet"

                if chunk_path.exists():
                    print(f"  ⚠ Chunk {absolute_chunk_id:04d} 已存在，跳过保存")
                    del chunk_data
                    gc.collect()
                    continue

                df_chunk = _process_chunk_to_df(chunk_data, idx_to_node)

                if df_chunk is not None and len(df_chunk) > 0:
                    df_chunk.to_parquet(chunk_path, compression="snappy", index=False)
                    chunk_records = len(df_chunk)
                    total_records += chunk_records
                    successful_chunks += 1

                    elapsed = time.time() - start_time
                    rate = total_records / elapsed if elapsed > 0 else 0
                    print(f"  ✓ Chunk {absolute_chunk_id:04d} 完成: {chunk_records:,} 条路径 "
                          f"(累计 {total_records:,} 条, 速率 {rate:,.0f}/s) → {chunk_path.name}")
                else:
                    print(f"  ⚪ Chunk {absolute_chunk_id:04d} 为空，跳过保存")

                del chunk_data, df_chunk
                gc.collect()

                # 自动重启：处理完第一个新 chunk 后立即重启
                if enable_restart_strategy and relative_chunk_id == 0:
                    print(f"\n  ⚡ 自动重启触发：已处理第1个新 chunk (chunk_{absolute_chunk_id:04d})")
                    print(f"  → 2秒后自动重启进程以释放内存并继续处理...")
                    time.sleep(2)
                    os.execl(sys.executable, sys.executable, *sys.argv)

        except KeyboardInterrupt:
            print(f"\n  ⏸️ 用户中断处理 {area_id}（已成功保存 {successful_chunks} 个新 chunk）")
            return False
        except Exception as e:
            print(f"  ✗ 处理 {area_id} 时发生错误：{e}")
            import traceback
            traceback.print_exc()
            return False

        total_time = time.time() - start_time
        if successful_chunks == 0 and (n_origins - skip_origin_count) > 0:
            print(f"  ⚠ {area_id}：剩余 {n_origins - skip_origin_count} 个 origins 未生成有效路径")

        print(f"  ✓ 区域 {area_id} 处理完成！"
              f"新增 {successful_chunks} 个 chunk ({total_records:,} 条路径), "
              f"跳过 {skipped_chunks} 个 chunk, "
              f"耗时 {total_time:.1f}s (平均 {total_records / total_time:,.0f} 条/秒)")
        return True

    except Exception as e:
        print(f"  ✗ 处理 {area_id} 时发生错误：{e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # 发现所有可能的 OD 文件
    od_files = list(OD_ROOT.glob("*_od_flows_wgs84.csv"))
    area_ids = []
    for f in od_files:
        stem = f.stem
        if stem.endswith("_od_flows_wgs84"):
            area_id = stem[:-len("_od_flows_wgs84")]
            area_ids.append(area_id)

    if not area_ids:
        print("在 OD 目录下没有找到任何 *_od_flows_wgs84.csv 文件")
        return

    print(f"发现 {len(area_ids)} 个待处理区域：")
    print(", ".join(sorted(area_ids)))
    print("-" * 80)

    success = 0
    skipped = 0
    failed = 0

    for area_id in tqdm(area_ids, desc="处理区域"):
        ok = process_one_area(area_id)
        if ok:
            success += 1
        else:
            # 检查是否因 chunk 已存在而跳过（在 process_one_area 内部已处理）
            # 这里仅统计显式失败
            failed += 1

    print("\n" + "═" * 80)
    print("批量处理完成总结：")
    print(f"  成功完成：{success:3d} 个区域")
    print(f"  失败/异常：{failed:3d} 个区域")
    print("  注意：已存在完整 chunk 集合的区域会计入'成功完成'")
    print("  注意：启用分块重启策略的区域会在处理1个 chunk 后自动重启（非失败）")
    print("═" * 80)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()