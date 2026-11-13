import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import LineString, MultiLineString, Point
import warnings
import hashlib
import sys
import pyarrow.parquet as pq
from tqdm import tqdm  # 可选：更美观的进度条

# ====================== 配置 ======================
TABLE_PARQUET_PATH = Path(
    "${SERVER_HOME}/result/trace_genetate/od_paths_flows_2869_chunk_0020.parquet"
)

GRAPHML_ROOT = Path(
    "${SERVER_DATA_ROOT}/prediction_travel_mode/trace_generate/road_network_clip/data/all_data/test"
)

# 每个输出文件包含的轨迹数量
CHUNK_SIZE = 10000

# CRS 配置
CRS_EPSG = "EPSG:4326"

# 是否跳过已存在的输出文件（断点续传）
SKIP_EXISTING = True


# ====================== 辅助函数 ======================

def decode_edge_sequence(edge_sequence):
    """将64位编码的边序列解码为 (u_idx, v_idx) 元组列表"""
    edges = []
    for code in edge_sequence:
        u_idx = (code >> 32) & 0xFFFFFFFF
        v_idx = code & 0xFFFFFFFF
        edges.append((int(u_idx), int(v_idx)))
    return edges


def load_road_network(area_id, graphml_root):
    """加载路网并构建边几何字典"""
    import osmnx as ox

    graph_path = graphml_root / f"road_network_drive_{area_id}.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(f"路网文件不存在: {graph_path}")

    print(f"  → 加载路网: {graph_path.name}")
    
    # 计算路网指纹
    with open(graph_path, 'rb') as f:
        graph_hash = hashlib.md5(f.read()).hexdigest()

    G = ox.load_graphml(graph_path)
    node_id_to_idx = {node: i for i, node in enumerate(G.nodes())}
    
    edge_geom_dict = {}
    for u, v, data in G.edges(data=True):
        u_idx = node_id_to_idx[u]
        v_idx = node_id_to_idx[v]
        geom = data.get('geometry')
        if geom is None:
            u_node = G.nodes[u]
            v_node = G.nodes[v]
            geom = LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])
        edge_geom_dict[(u_idx, v_idx)] = geom

    print(f"  → 路网加载完成: {len(G.nodes):,} 节点, {len(G.edges):,} 条边 | MD5: {graph_hash[:16]}...")
    return edge_geom_dict, node_id_to_idx, graph_hash


def rebuild_geometries(df_chunk, edge_geom_dict, chunk_index, total_chunks, global_start_idx):
    """
    重建单个数据块的几何对象
    
    参数:
        df_chunk: 当前数据块 (DataFrame)
        edge_geom_dict: 路网边几何字典
        chunk_index: 当前块索引 (0-based)
        total_chunks: 总块数
        global_start_idx: 全局起始行索引 (用于错误定位)
    """
    geometries = []
    invalid_count = 0
    total = len(df_chunk)
    
    print(f"\n[{chunk_index+1}/{total_chunks}] 处理轨迹 {global_start_idx+1} ~ {global_start_idx+total} (共 {total:,} 条)...")

    # 使用tqdm显示进度（如未安装则回退到简单打印）
    try:
        iterator = tqdm(enumerate(df_chunk.itertuples()), total=total, desc="  几何重建", ncols=80, leave=False)
    except:
        iterator = enumerate(df_chunk.itertuples())
        print(f"  → 进度: ", end="", flush=True)

    for local_idx, row in iterator:
        # 每1000条显示一次进度（当无tqdm时）
        if 'tqdm' not in sys.modules and local_idx % 1000 == 0:
            print(f"{local_idx}/{total} ", end="", flush=True)
        
        edge_sequence = row.edge_sequence
        edges = decode_edge_sequence(edge_sequence)
        geoms = []
        
        for u_idx, v_idx in edges:
            geom = edge_geom_dict.get((u_idx, v_idx))
            if geom is not None:
                geoms.append(geom)
            else:
                invalid_count += 1
                if invalid_count <= 3 and chunk_index == 0:  # 仅首块显示前3条警告
                    global_idx = global_start_idx + local_idx
                    print(f"\n    ⚠ 警告: 边 ({u_idx}, {v_idx}) 不存在 (全局轨迹ID: {global_idx})")
        
        if geoms:
            merged_geom = geoms[0] if len(geoms) == 1 else MultiLineString(geoms)
            geometries.append(merged_geom)
        else:
            geometries.append(Point())  # 空几何占位
            invalid_count += 1

    if 'tqdm' not in sys.modules:
        print(f"{total}/{total}")  # 完成进度行

    gdf = gpd.GeoDataFrame(
        df_chunk.drop(columns=['edge_sequence']), 
        geometry=geometries, 
        crs=CRS_EPSG
    )

    # 统计报告
    valid = total - invalid_count
    print(f"  → 有效几何: {valid:,}/{total:,} ({valid/total*100:.1f}%)", end="")
    if invalid_count > 0:
        print(f" | 无效边: {invalid_count:,}")
    else:
        print()
    
    return gdf


def get_output_path(base_path, chunk_index):
    """生成分块输出文件路径"""
    stem = base_path.stem
    # 移除可能已存在的 _vector 后缀（避免重复添加）
    if stem.endswith("_vector"):
        stem = stem[:-7]
    new_stem = f"{stem}_vector_{chunk_index:04d}"
    return base_path.parent / f"{new_stem}.parquet"


def main():
    print("=" * 80)
    print("表格型 Parquet → 矢量型 Parquet (分块保存) 转换工具")
    print(f"  配置: 每 {CHUNK_SIZE:,} 条轨迹保存为一个文件 | 跳过已存在: {SKIP_EXISTING}")
    print("=" * 80)

    # 1. 验证输入文件
    if not TABLE_PARQUET_PATH.exists():
        print(f"✗ 输入文件不存在: {TABLE_PARQUET_PATH}")
        sys.exit(1)

    # 2. 获取总行数（快速元数据读取）
    try:
        total_rows = pq.ParquetFile(TABLE_PARQUET_PATH).metadata.num_rows
        total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"\n1. 输入文件分析: {total_rows:,} 条轨迹 → 将分为 {total_chunks} 个输出文件")
    except Exception as e:
        print(f"✗ 无法读取Parquet元数据: {e}")
        sys.exit(1)

    # 3. 解析 area_id
    stem = TABLE_PARQUET_PATH.stem
    if "od_paths_flows_" in stem:
        area_id = stem.split("_")[-3]  # 格式: ..._2869_chunk_XXXX
        print(f"2. 检测到 area_id: {area_id}")
    else:
        area_id = input("  无法自动解析 area_id，请手动输入: ").strip()
        if not area_id:
            print("✗ area_id 不能为空")
            sys.exit(1)

    # 4. 加载路网（仅一次）
    print(f"\n3. 加载路网 (area_id={area_id})...")
    try:
        edge_geom_dict, _, graph_hash = load_road_network(area_id, GRAPHML_ROOT)
    except Exception as e:
        print(f"✗ 路网加载失败: {e}")
        print("💡 提示: 请确认 GRAPHML_ROOT 配置正确")
        sys.exit(1)

    # 5. 分块处理与保存
    print(f"\n4. 开始分块处理 (每块 {CHUNK_SIZE:,} 条)...")
    processed_chunks = 0
    skipped_chunks = 0
    total_invalid = 0

    # 使用PyArrow分块读取（内存高效）
    pf = pq.ParquetFile(TABLE_PARQUET_PATH)
    for chunk_index in range(total_chunks):
        # 计算当前块的行范围
        start_row = chunk_index * CHUNK_SIZE
        end_row = min(start_row + CHUNK_SIZE, total_rows)
        current_size = end_row - start_row

        # 生成输出路径
        output_path = get_output_path(TABLE_PARQUET_PATH, chunk_index)
        
        # 检查是否跳过已存在文件
        if SKIP_EXISTING and output_path.exists():
            skipped_chunks += 1
            if skipped_chunks <= 5 or chunk_index == total_chunks - 1:
                print(f"  [{chunk_index+1}/{total_chunks}] 跳过已存在: {output_path.name}")
            elif skipped_chunks == 6:
                print(f"  ... (后续跳过文件省略显示) ...")
            continue

        # 读取当前数据块
        chunk_table = pf.read_row_groups([chunk_index])  # 注意：这里假设row group大小≈CHUNK_SIZE
        chunk_df = chunk_table.to_pandas()
        
        # 重建几何
        gdf = rebuild_geometries(
            chunk_df, 
            edge_geom_dict, 
            chunk_index, 
            total_chunks, 
            global_start_idx=start_row
        )
        
        # 统计无效几何
        invalid_in_chunk = gdf.geometry.is_empty.sum()
        total_invalid += invalid_in_chunk
        
        # 保存结果
        try:
            gdf.to_parquet(
                output_path,
                compression="snappy",
                geometry_encoding="WKB",
                index=False,
                schema_version="1.0.0"
            )
            file_size_mb = output_path.stat().st_size / (1024 ** 2)
            print(f"  → 保存成功: {output_path.name} ({file_size_mb:.2f} MB)")
            processed_chunks += 1
        except Exception as e:
            print(f"  ✗ 保存失败 {output_path.name}: {e}")
            continue

    # 6. 最终统计
    print("\n" + "=" * 80)
    print("✓ 转换完成!")
    print(f"  总轨迹数: {total_rows:,}")
    print(f"  输出文件: {processed_chunks} 个新文件 + {skipped_chunks} 个跳过文件 (共 {total_chunks} 个)")
    print(f"  无效几何: {total_invalid:,} 条 ({total_invalid/total_rows*100:.2f}%)")
    print(f"  路网指纹: {graph_hash[:16]}...")
    print(f"  输出目录: {TABLE_PARQUET_PATH.parent}")
    print("=" * 80)

    # 7. 抽样验证（仅验证最后一个生成的文件）
    if processed_chunks > 0:
        last_chunk_idx = total_chunks - 1 if skipped_chunks == total_chunks else total_chunks - skipped_chunks - 1
        last_output = get_output_path(TABLE_PARQUET_PATH, last_chunk_idx)
        if last_output.exists():
            print(f"\n8. 抽样验证最后文件: {last_output.name}")
            try:
                sample_gdf = gpd.read_parquet(last_output)
                for i, row in sample_gdf.head(2).iterrows():
                    print(f"  [{i}] o={row['o_node_id']}, d={row['d_node_id']}, "
                          f"flow={row['flow']:.2f}, geom={row.geometry.geom_type}")
            except Exception as e:
                print(f"  验证失败: {e}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="shapely")
    warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
    main()