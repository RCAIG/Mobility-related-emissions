import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['GDAL_CACHEMAX'] = '2048'
os.environ['GDAL_SWATH_SIZE'] = '2048'
os.environ['GDAL_NUM_THREADS'] = '1'

import glob
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import re
import warnings
import math
from typing import List, Dict, Any, Optional, Tuple
import json
import multiprocessing
import atexit
import sys
import queue
import threading

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
INPUT_PARQUET_DIR = "${SERVER_HOME}/result/trace_genetare_hgih_income_2024/"
OUTPUT_PRED_DIR = "${SERVER_HOME}/result/mode_predict/high_income_2024/"
MODEL_CHECKPOINT = "urban_transformer_cls_best_a_gmm.pth"
CENTER_CSV_PATH = "${SERVER_DATA_ROOT}/prediction_travel_mode/trace_generate/road_network_clip/data/all_data/2024_pop_center/POP_Centers_2024_LocalPeak.csv"
CENTER_SCALER_JSON = "${SERVER_DATA_ROOT}/prediction_travel_mode/model/code/center_scaler_params.json"

GRAPHML_ROOT = Path(
    "${SERVER_HOME}/result/road_2024/"
)

REGION_DIRS = [
    "${SERVER_HOME}/GEE/GEE_Embeddings_high_income_2024/"
]

# ================= 新增：推理端波段选择配置 =================
USE_BAND_SELECTION_INFERENCE = True   # ✅ 启用优选波段提取
#  填入你截图中的29个优选波段索引（0-based）
SELECTED_BANDS_INFERENCE = [35, 39, 49, 24, 26, 34, 42, 27, 18, 17, 19, 44, 31, 38, 20, 25, 10, 11, 62, 41, 32, 61, 51, 15, 22, 3, 63, 57, 36]
INPUT_CHANNELS_INFERENCE = len(SELECTED_BANDS_INFERENCE) if USE_BAND_SELECTION_INFERENCE else 64
print(f"📡 推理将仅提取 {INPUT_CHANNELS_INFERENCE} 个优选波段（原64）")


# ================= 新增：推理端时序采样策略配置 =================
TEMPORAL_BASE_INTERVAL = 500.0              # ✅ 基础采样间隔 = 100m × TEMPORAL_DOWNSAMPLE_BASE(5)
MAX_TOTAL_TIMESTEPS = 100
HEAD_TAIL_RESERVED_POINTS = 50              # 首尾各保留的**点数**（按500m间隔计算）
MIDDLE_DOWNSAMPLE_RATE = 1.0                # ✅ 中间额外降采样倍率（1.0=不额外降采样，与原逻辑一致）
STRATEGY = "head_tail_focus"

def get_optimal_points_per_batch():
    if torch.cuda.is_available():
        free_mem_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        return min(32768, max(4096, int(free_mem_gb * 2200)))
    return 8192


MAX_POINTS_PER_BATCH = get_optimal_points_per_batch()
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_PROCESSES = 1 # min(8, multiprocessing.cpu_count() // 2)

USE_ENV = True
USE_LANDUSE = False
USE_TRAFFIC = False
USE_CENTER = True

CENTER_FEATURES = [
    'start_to_primary_m',
    'end_to_primary_m',
    'start_to_nearest_sub_m',
    'end_to_nearest_sub_m',
    'trip_min_to_primary_m',
    'trip_min_to_nearest_sub_m',
    'od_ratio_primary',
    'od_ratio_nearest_sub',
    'city_total_centers'
]

os.makedirs(OUTPUT_PRED_DIR, exist_ok=True)

# ==================== 全局缓存（进程内独立）====================
ROAD_NETWORK_CACHE = {}
TIFF_CACHE = {}
TIFF_BOUNDS_CACHE = {}
TIFF_BOUNDS_ARR_CACHE = {}  # ✅ 新增：预构建的NumPy边界数组缓存


def cleanup_tiff_cache():
    global TIFF_CACHE, TIFF_BOUNDS_CACHE, TIFF_BOUNDS_ARR_CACHE
    for city_id, datasets in list(TIFF_CACHE.items()):
        for ds in datasets:
            try:
                ds.close()
            except:
                pass
    TIFF_CACHE.clear()
    TIFF_BOUNDS_CACHE.clear()
    TIFF_BOUNDS_ARR_CACHE.clear()


atexit.register(cleanup_tiff_cache)


# ==================== 工具函数（保持原逻辑）====================
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_vectorized(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371000.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2_arr)
    lon2_rad = np.radians(lon2_arr)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def sample_linestring_by_haversine(line: LineString, interval_m: float = 100.0) -> List[Point]:
    coords = list(line.coords)
    if len(coords) < 2:
        return [Point(coords[0])] if coords else []

    sampled = [Point(coords[0])]
    accumulated = 0.0
    last_lat, last_lon = coords[0][1], coords[0][0]

    for i in range(1, len(coords)):
        curr_lon, curr_lat = coords[i]
        segment_dist = haversine_distance(last_lat, last_lon, curr_lat, curr_lon)

        while accumulated + segment_dist >= interval_m:
            ratio = (interval_m - accumulated) / segment_dist
            interp_lon = last_lon + ratio * (curr_lon - last_lon)
            interp_lat = last_lat + ratio * (curr_lat - last_lat)
            sampled.append(Point(interp_lon, interp_lat))
            accumulated = 0.0
            last_lon, last_lat = interp_lon, interp_lat
            segment_dist = haversine_distance(last_lat, last_lon, curr_lat, curr_lon)

        accumulated += segment_dist
        last_lon, last_lat = curr_lon, curr_lat

    if sampled[-1].distance(Point(coords[-1])) > 1e-6:
        sampled.append(Point(coords[-1]))
    return sampled


def extract_id_and_chunk_from_parquet_filename(filename: str) -> Optional[tuple]:
    match = re.search(r'od_paths_flows_(\d+)(?:_chunk_(\d+))?\.parquet$', filename)
    if match:
        city_id = match.group(1)
        chunk_id = match.group(2)
        return city_id, chunk_id
    return None


def find_tiff_files_by_id(id_str: str) -> List[str]:
    candidates = []
    for region_dir in REGION_DIRS:
        pattern = os.path.join(region_dir, f"{id_str}_*.tif")
        found = glob.glob(pattern)
        candidates.extend(found)
    return sorted(set(candidates))


def point_to_pixel_coords(dataset, lon, lat):
    x, y = ~dataset.transform * (lon, lat)
    return int(round(x)), int(round(y))


# ================= 自适应时序降采样（推理端专用 - 测地距直采版）=================
def direct_adaptive_sample_linestring(
        line: LineString,
        base_interval_m: float = 500.0,
        max_timesteps: int = 100,
        head_tail_reserved_points: int = 50,
        middle_downsample_rate: float = 1.0,
        strategy: str = "head_tail_focus"
) -> List[Point]:
    """
    ✅ 直接按自适应策略采样（原生Haversine测地距，彻底避开 Shapely 的 WGS84 度数陷阱）
    时间复杂度 O(V+T)，单次遍历，无中间冗余采样
    """
    if line is None or line.is_empty or len(line.coords) < 2:
        return []

    coords = list(line.coords)

    # 1. 预计算线段真实距离(米)与累积距离
    seg_dists = []
    cum_dists = [0.0]
    for i in range(len(coords) - 1):
        d = haversine_distance(coords[i][1], coords[i][0], coords[i + 1][1], coords[i + 1][0])
        seg_dists.append(d)
        cum_dists.append(cum_dists[-1] + d)
    total_len = cum_dists[-1]

    # 轨迹太短直接返回首尾
    if total_len < base_interval_m * 2:
        return [Point(coords[0]), Point(coords[-1])]

    # 2. 生成目标采样距离列表（米）
    head_end_dist = head_tail_reserved_points * base_interval_m
    tail_start_dist = total_len - head_end_dist
    mid_interval_m = base_interval_m * middle_downsample_rate

    targets = []
    if strategy == "head_tail_focus":
        if head_end_dist >= tail_start_dist:  # 首尾重叠 → 退回均匀
            n = min(max_timesteps, max(2, round(total_len / base_interval_m) + 1))
            targets = np.linspace(0, total_len, n).tolist()
        else:
            # 头部 [0, head_end)
            n_head = max(1, int(head_end_dist // base_interval_m))
            targets.extend([i * base_interval_m for i in range(n_head)])
            # 中间 [head_end, tail_start)
            mid_len = tail_start_dist - head_end_dist
            if mid_len > 0:
                n_mid = max(1, int(mid_len // mid_interval_m))
                targets.extend([head_end_dist + i * mid_interval_m for i in range(n_mid)])
            # 尾部 [tail_start, total]
            n_tail = max(1, int((total_len - tail_start_dist) // base_interval_m))
            targets.extend([tail_start_dist + i * base_interval_m for i in range(n_tail)])
            targets.append(total_len)  # 强制包含终点
    else:  # uniform
        n = min(max_timesteps, max(2, round(total_len / base_interval_m) + 1))
        targets = np.linspace(0, total_len, n).tolist()

    # 保证起点终点 & 长度保护
    if not targets or targets[0] > 1e-6: targets.insert(0, 0.0)
    if targets[-1] < total_len - 1e-6: targets.append(total_len)
    if len(targets) > max_timesteps:
        targets = np.linspace(0, total_len, max_timesteps).tolist()

    # 3. 将目标距离插值到原始线段（双指针 O(N) 遍历）
    sampled = []
    seg_idx = 0
    for t in targets:
        # 移动指针到包含当前目标距离的线段
        while seg_idx < len(seg_dists) and cum_dists[seg_idx + 1] < t - 1e-6:
            seg_idx += 1
        if seg_idx >= len(seg_dists):
            sampled.append(Point(coords[-1]))
            continue

        # 线性插值
        seg_len = seg_dists[seg_idx]
        ratio = (t - cum_dists[seg_idx]) / seg_len if seg_len > 1e-6 else 0.0
        ratio = max(0.0, min(1.0, ratio))
        lon = coords[seg_idx][0] + ratio * (coords[seg_idx + 1][0] - coords[seg_idx][0])
        lat = coords[seg_idx][1] + ratio * (coords[seg_idx + 1][1] - coords[seg_idx][1])
        sampled.append(Point(lon, lat))

    return sampled

# ==================== 核心优化1：TIFF边界数组预缓存 ====================
def batch_extract_ae_features_old_version(lons: List[float], lats: List[float],
                              open_datasets: List[rasterio.DatasetReader],
                              bounds_arr: np.ndarray) -> List[np.ndarray]:
    """
    ✅ 安全优化：直接接收预构建的bounds_arr (n_tiffs, 4)，避免重复构建
    """
    n_points = len(lons)
    if n_points == 0:
        return [np.zeros((64, 11, 11), dtype=np.float32) for _ in range(n_points)]

    lons_arr = np.array(lons, dtype=np.float64)
    lats_arr = np.array(lats, dtype=np.float64)

    # ✅ 直接使用预缓存的bounds_arr，消除重复构建开销
    lefts, bottoms, rights, tops = bounds_arr[:, 0], bounds_arr[:, 1], bounds_arr[:, 2], bounds_arr[:, 3]

    in_bounds = (
            (lons_arr[:, None] >= lefts) & (lons_arr[:, None] <= rights) &
            (lats_arr[:, None] >= bottoms) & (lats_arr[:, None] <= tops)
    )

    tiff_indices = np.argmax(in_bounds, axis=1)
    valid_mask = in_bounds[np.arange(n_points), tiff_indices]

    results = [np.zeros((64, 11, 11), dtype=np.float32) for _ in range(n_points)]

    for ds_idx, ds in enumerate(open_datasets):
        point_mask = (tiff_indices == ds_idx) & valid_mask
        if not np.any(point_mask):
            continue

        pts_lons = lons_arr[point_mask]
        pts_lats = lats_arr[point_mask]

        cols, rows = ~ds.transform * (pts_lons, pts_lats)
        cols = np.round(cols).astype(int)
        rows = np.round(rows).astype(int)

        half = 5
        patches = []
        for col, row in zip(cols, rows):
            if -10 <= col < ds.width + 10 and -10 <= row < ds.height + 10:
                window = Window(col - half, row - half, 11, 11)
                try:
                    data = ds.read(window=window, out_shape=(64, 11, 11),
                                   resampling=Resampling.nearest, fill_value=0)
                    patches.append(data.astype(np.float32))
                except:
                    patches.append(np.zeros((64, 11, 11), dtype=np.float32))
            else:
                patches.append(np.zeros((64, 11, 11), dtype=np.float32))

        result_indices = np.where(point_mask)[0]
        for idx, patch in zip(result_indices, patches):
            if np.any(patch != 0):
                results[idx] = patch

    return results

def batch_extract_ae_features(lons: List[float], lats: List[float],
                              open_datasets: List[rasterio.DatasetReader],
                              bounds_arr: np.ndarray,
                              selected_bands: List[int] = None) -> List[np.ndarray]:
    """
    ✅ 优化版：底层跳读未选中波段，I/O 与内存双降
    selected_bands: 0-based 波段索引列表，如 [35, 39, ...]
    """
    n_points = len(lons)
    if n_points == 0:
        C = len(selected_bands) if selected_bands else 64
        return [np.zeros((C, 11, 11), dtype=np.float32) for _ in range(n_points)]

    lons_arr = np.array(lons, dtype=np.float64)
    lats_arr = np.array(lats, dtype=np.float64)
    lefts, bottoms, rights, tops = bounds_arr[:, 0], bounds_arr[:, 1], bounds_arr[:, 2], bounds_arr[:, 3]

    in_bounds = ((lons_arr[:, None] >= lefts) & (lons_arr[:, None] <= rights) &
                 (lats_arr[:, None] >= bottoms) & (lats_arr[:, None] <= tops))
    tiff_indices = np.argmax(in_bounds, axis=1)
    valid_mask = in_bounds[np.arange(n_points), tiff_indices]

    C = len(selected_bands) if selected_bands else 64
    # rasterio 使用 1-based 索引，需转换
    rio_indexes = [b + 1 for b in selected_bands] if selected_bands else None

    results = [np.zeros((C, 11, 11), dtype=np.float32) for _ in range(n_points)]

    for ds_idx, ds in enumerate(open_datasets):
        point_mask = (tiff_indices == ds_idx) & valid_mask
        if not np.any(point_mask):
            continue

        pts_lons = lons_arr[point_mask]
        pts_lats = lats_arr[point_mask]
        cols, rows = ~ds.transform * (pts_lons, pts_lats)
        cols = np.round(cols).astype(int)
        rows = np.round(rows).astype(int)

        half = 5
        valid_indices = np.where(point_mask)[0]
        for i, (col, row) in enumerate(zip(cols, rows)):
            idx = valid_indices[i]
            if -10 <= col < ds.width + 10 and -10 <= row < ds.height + 10:
                window = Window(col - half, row - half, 11, 11)
                try:
                    # 🔑 核心：indexes 参数让 GDAL 底层直接跳过无关波段
                    data = ds.read(
                        indexes=rio_indexes,
                        window=window,
                        out_shape=(C, 11, 11),
                        resampling=Resampling.nearest,
                        fill_value=0
                    )
                    if np.any(data != 0):
                        results[idx] = data.astype(np.float32)
                except:
                    pass
    return results

def flatten_geometry(geom):
    if isinstance(geom, MultiLineString):
        coords = []
        for line in geom.geoms:
            coords.extend(line.coords)
        return LineString(coords) if len(coords) >= 2 else None
    elif isinstance(geom, LineString):
        return geom
    else:
        return None


def load_road_network_cached(area_id: str) -> Dict[tuple, LineString]:
    global ROAD_NETWORK_CACHE
    if area_id in ROAD_NETWORK_CACHE:
        return ROAD_NETWORK_CACHE[area_id]['edge_geom_dict']

    graph_path = GRAPHML_ROOT / f"road_network_drive_{area_id}.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(f"路网文件不存在: {graph_path} (area_id={area_id})")

    print(f"  → 首次加载路网 area_id={area_id} ...")
    import osmnx as ox
    import hashlib

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
            geom = LineString([(u_node['x'], u_node['y']), (v_node['x'], u_node['y'])])  # 修复：应为v_node['y']
            geom = LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])  # 修正
        edge_geom_dict[(u_idx, v_idx)] = geom

    ROAD_NETWORK_CACHE[area_id] = {
        'edge_geom_dict': edge_geom_dict,
        'graph_hash': graph_hash,
        'node_count': len(G.nodes()),
        'edge_count': len(G.edges())
    }

    print(f"  → 路网加载完成: {len(edge_geom_dict):,} 条边 | MD5: {graph_hash[:16]}...")
    return edge_geom_dict


def batch_rebuild_geometries(edge_sequences: List[np.ndarray], edge_geom_dict: Dict) -> List[Optional[LineString]]:
    geometries = []
    for edge_seq in edge_sequences:
        if len(edge_seq) == 0:
            geometries.append(None)
            continue

        u_indices = (edge_seq >> 32) & 0xFFFFFFFF
        v_indices = edge_seq & 0xFFFFFFFF

        geoms = []
        for u_idx, v_idx in zip(u_indices, v_indices):
            geom = edge_geom_dict.get((int(u_idx), int(v_idx)))
            if geom is not None:
                geoms.append(geom)

        if not geoms:
            geometries.append(None)
        elif len(geoms) == 1:
            geometries.append(geoms[0])
        else:
            geometries.append(MultiLineString(geoms))
    return geometries


# ==================== 城市中心特征计算（保持原逻辑）====================
def load_city_centers(csv_path: str) -> Dict[int, Dict]:
    print(f"Loading city centers from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'city_name': 'city_id'})

    centers_info = {}
    for city_id, group in df.groupby('city_id'):
        try:
            city_id_int = int(city_id)
            primary = group[group['center_type'] == 'primary_center'].iloc[0]
            subs = group[group['center_type'].str.startswith('sub_center_')]
            sub_coords = subs[['lon', 'lat']].values if len(subs) > 0 else np.array([]).reshape(0, 2)

            centers_info[city_id_int] = {
                'primary_lon': float(primary['lon']),
                'primary_lat': float(primary['lat']),
                'sub_centers': sub_coords,
                'total_centers': len(group)
            }
        except Exception as e:
            print(f"Warning: Skip city {city_id} due to error: {e}")
            continue

    print(f"Loaded {len(centers_info)} cities with center information")
    return centers_info


def compute_center_features_raw(points: List[Dict], city_info: Dict, total_length: float) -> np.ndarray:
    if len(points) < 2 or total_length <= 0:
        return np.zeros(len(CENTER_FEATURES), dtype=np.float32)

    lats = np.array([p['lat'] for p in points], dtype=np.float64)
    lons = np.array([p['lon'] for p in points], dtype=np.float64)

    primary_lat = city_info['primary_lat']
    primary_lon = city_info['primary_lon']
    sub_centers = city_info['sub_centers']
    total_centers = city_info['total_centers']

    start_to_primary = haversine_distance(points[0]['lat'], points[0]['lon'], primary_lat, primary_lon)
    end_to_primary = haversine_distance(points[-1]['lat'], points[-1]['lon'], primary_lat, primary_lon)

    dlat = lats - primary_lat
    dlon = lons - primary_lon
    avg_lat_rad = np.radians((lats + primary_lat) / 2.0)
    meters_per_deg_lon = 111320.0 * np.cos(avg_lat_rad)
    dists_to_primary = np.sqrt((dlat * 111320.0) ** 2 + (dlon * meters_per_deg_lon) ** 2)
    min_to_primary = float(dists_to_primary.min())

    if len(sub_centers) > 0:
        sub_lons = sub_centers[:, 0]
        sub_lats = sub_centers[:, 1]

        dists_start_sub = haversine_vectorized(
            points[0]['lat'], points[0]['lon'], sub_lats, sub_lons
        )
        start_to_nearest_sub = float(dists_start_sub.min())

        dists_end_sub = haversine_vectorized(
            points[-1]['lat'], points[-1]['lon'], sub_lats, sub_lons
        )
        end_to_nearest_sub = float(dists_end_sub.min())

        dlat_sub = lats[:, None] - sub_lats[None, :]
        dlon_sub = lons[:, None] - sub_lons[None, :]
        avg_lat_rad_sub = np.radians((lats[:, None] + sub_lats[None, :]) / 2.0)
        meters_per_deg_lon_sub = 111320.0 * np.cos(avg_lat_rad_sub)
        dists_to_subs = np.sqrt((dlat_sub * 111320.0) ** 2 + (dlon_sub * meters_per_deg_lon_sub) ** 2)
        min_to_nearest_sub = float(dists_to_subs.min())
    else:
        start_to_nearest_sub = end_to_nearest_sub = min_to_nearest_sub = -1.0

    od_ratio_primary = (start_to_primary + end_to_primary) / (total_length + 1e-6)
    od_ratio_nearest_sub = (start_to_nearest_sub + end_to_nearest_sub) / (total_length + 1e-6) \
        if start_to_nearest_sub > 0 else -1.0

    features = [
        start_to_primary,
        end_to_primary,
        start_to_nearest_sub if start_to_nearest_sub > 0 else 0.0,
        end_to_nearest_sub if end_to_nearest_sub > 0 else 0.0,
        min_to_primary,
        min_to_nearest_sub if min_to_nearest_sub > 0 else 0.0,
        od_ratio_primary,
        od_ratio_nearest_sub if od_ratio_nearest_sub > 0 else -1.0,
        float(total_centers)
    ]

    return np.array(features, dtype=np.float32)


def load_center_scaler_params(json_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"❌ 未找到标准化参数文件: {json_path}\n"
            f"   请先修改训练代码，在 load_center_data_full() 末尾添加保存逻辑:\n"
            f"   import json\n"
            f"   scaler_params = {{'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}}\n"
            f"   with open('center_scaler_params.json', 'w') as f:\n"
            f"       json.dump(scaler_params, f)\n"
            f"   然后重新运行训练以生成该文件。"
        )

    with open(json_path, 'r') as f:
        params = json.load(f)

    mean = np.array(params['mean'], dtype=np.float32)
    scale = np.array(params['scale'], dtype=np.float32)

    print(f"✓ Loaded center feature scaler parameters from {json_path}")
    print(f"  Features: {params.get('features', CENTER_FEATURES)}")
    print(f"  Mean: {mean}")
    print(f"  Scale: {scale}")

    return {'mean': mean, 'scale': scale}


def standardize_center_features(raw_features: np.ndarray, scaler_params: Dict) -> np.ndarray:
    mean = scaler_params['mean']
    scale = scaler_params['scale']
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (raw_features - mean) / scale


# ==================== 模型定义（保持原逻辑）====================
class AttentionPool(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.attn = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, x):
        B_T, C, H, W = x.shape
        weights = torch.sigmoid(self.attn(x))
        weighted = x * weights
        pooled = weighted.mean(dim=[2, 3])
        return self.proj(pooled)


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=120):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pos_embed(positions)


class SimpleResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class PatchEncoder(nn.Module):
    def __init__(self, input_channels=64, embed_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, 64)
        self.res_block = SimpleResidualBlock(64)
        self.pool = nn.MaxPool2d(2)
        self.attn_pool = AttentionPool(64, embed_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.res_block(x)
        x = self.pool(x)
        x = self.attn_pool(x)
        return x


class UrbanMultiModalClassifier(nn.Module):
    def __init__(self, num_classes=4, input_channels=64, embed_dim=128, nhead=8, num_layers=6,
                 use_env=True, use_landuse=False, use_traffic=False, use_center=True):
        super().__init__()
        self.use_env = use_env
        self.use_landuse = use_landuse
        self.use_traffic = use_traffic
        self.use_center = use_center
        self.embed_dim = embed_dim

        if self.use_env:
            self.patch_encoder = PatchEncoder(input_channels, embed_dim)
            self.self_attn_img = nn.MultiheadAttention(
                embed_dim,
                num_heads=nhead // 2,
                batch_first=True
            )

        if self.use_landuse:
            raise NotImplementedError("Landuse not used in training")

        if self.use_traffic:
            self.traffic_stats_encoder = nn.Sequential(
                nn.Linear(8, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )

        if self.use_center:
            self.center_encoder = nn.Sequential(
                nn.Linear(len(CENTER_FEATURES), embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )

        self.embed_norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_encoder = LearnablePositionalEmbedding(embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim)

        fusion_in_dim = embed_dim
        if use_env:
            fusion_in_dim = embed_dim
        if use_center:
            fusion_in_dim += embed_dim
        if use_traffic:
            fusion_in_dim += embed_dim

        if fusion_in_dim != embed_dim:
            self.modality_fusion_proj = nn.Linear(fusion_in_dim, embed_dim)
        else:
            self.modality_fusion_proj = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=512
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_img, x_land, x_traffic_stats, x_center, lengths):
        B = 0
        if x_img is not None:
            B = x_img.size(0)
        elif x_center is not None:
            B = x_center.size(0)
        else:
            raise ValueError("At least one input modality must be provided.")

        embeddings_img = None
        if self.use_env and x_img is not None:
            T = x_img.shape[1]
            x_flat = x_img.view(B * T, x_img.shape[2], x_img.shape[3], x_img.shape[4])
            embeddings_img = self.patch_encoder(x_flat)
            embeddings_img = embeddings_img.view(B, T, -1)
            embeddings_img = self.embed_norm(embeddings_img)

        center_emb = None
        if self.use_center and x_center is not None:
            center_emb = self.center_encoder(x_center).unsqueeze(1)
            center_emb = self.embed_norm(center_emb)

        combined_list = []
        T_final = 0

        if self.use_env and embeddings_img is not None:
            T_final = embeddings_img.shape[1]
            combined_list.append(embeddings_img)
            if center_emb is not None:
                combined_list.append(center_emb.expand(B, T_final, -1))
        elif center_emb is not None:
            T_final = 1
            combined_list.append(center_emb)
        else:
            T_final = 1
            combined_list.append(torch.zeros(B, 1, self.embed_dim, device=self.cls_token.device))

        if len(combined_list) == 1:
            combined = combined_list[0]
        else:
            combined = torch.cat(combined_list, dim=-1)

        if self.modality_fusion_proj is not None:
            combined = self.modality_fusion_proj(combined)

        combined = self.final_norm(combined)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat([cls_tokens, combined], dim=1)
        embeddings = self.pos_encoder(embeddings)

        total_seq_len = embeddings.shape[1] - 1
        if total_seq_len > 0 and lengths is not None:
            pad_mask = torch.arange(total_seq_len, device=embeddings.device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            pad_mask = torch.zeros(B, total_seq_len, dtype=torch.bool, device=embeddings.device)

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=embeddings.device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)

        encoded = self.transformer(embeddings, src_key_padding_mask=full_mask)
        cls_out = encoded[:, 0, :]
        logits = self.classifier(cls_out)
        return logits, cls_out


# ==================== 核心优化2：TIFF缓存返回预构建bounds_arr ====================
def get_tiff_datasets(city_id: str) -> Tuple[List[rasterio.DatasetReader], List[tuple], np.ndarray]:
    """
    ✅ 安全优化：返回预构建的bounds_arr (n_tiffs, 4) NumPy数组
    """
    global TIFF_CACHE, TIFF_BOUNDS_CACHE, TIFF_BOUNDS_ARR_CACHE

    if city_id in TIFF_CACHE:
        return TIFF_CACHE[city_id], TIFF_BOUNDS_CACHE[city_id], TIFF_BOUNDS_ARR_CACHE[city_id]

    tiff_paths = find_tiff_files_by_id(city_id)
    if not tiff_paths:
        raise FileNotFoundError(f"No TIFF files found for city {city_id}")

    datasets = []
    bounds = []
    for tif in tiff_paths:
        try:
            ds = rasterio.open(tif, 'r', sharing=False, num_threads=1)
            datasets.append(ds)
            bounds.append((ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top))
        except Exception as e:
            print(f"⚠️ 跳过损坏的 TIFF {tif}: {e}")
            continue

    if not datasets:
        raise RuntimeError(f"无法打开任何有效 TIFF for city {city_id}")

    # ✅ 预构建bounds_arr并缓存
    bounds_arr = np.array(bounds, dtype=np.float64)  # shape: (n_tiffs, 4)

    TIFF_CACHE[city_id] = datasets
    TIFF_BOUNDS_CACHE[city_id] = bounds
    TIFF_BOUNDS_ARR_CACHE[city_id] = bounds_arr

    print(f"  → 首次加载 TIFF for city {city_id}: {len(datasets)} files")
    return datasets, bounds, bounds_arr


# ==================== 动态推理主函数（集成双缓冲）====================
def process_city_file(parquet_path: str, city_centers: Dict[int, Dict], scaler_params: Dict):
    parquet_file = Path(parquet_path)
    id_chunk_result = extract_id_and_chunk_from_parquet_filename(parquet_file.name)

    if not id_chunk_result:
        print(f"⚠️ 跳过 {parquet_file.name}: 无法提取 ID/chunk 信息")
        return

    city_id_str, chunk_id = id_chunk_result
    city_id = int(city_id_str)

    if city_id not in city_centers:
        print(f"⚠️ 跳过 {city_id_str}: 未找到城市中心信息")
        return

    output_filename = f"{city_id_str}_chunk_{chunk_id}_predictions.parquet" if chunk_id else f"{city_id_str}_predictions.parquet"
    output_path = os.path.join(OUTPUT_PRED_DIR, output_filename)

    if os.path.exists(output_path):
        chunk_info = f" chunk {chunk_id}" if chunk_id else ""
        print(f"✅ 跳过 {city_id_str}{chunk_info}: 结果已存在 ({output_filename})")
        return

    chunk_info = f" chunk {chunk_id}" if chunk_id else ""
    print(f"🚀 处理城市 {city_id_str} (ID: {city_id}){chunk_info} ...")

    # 读取Parquet文件
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_file)
        df = table.to_pandas().reset_index(drop=True)
    except Exception as e:
        print(f"❌ 读取失败 {parquet_file}: {e}")
        return

    if df.empty:
        print(f"⚠️ {city_id_str}{chunk_info} 为空")
        return

    # 加载资源
    try:
        open_datasets, _, bounds_arr = get_tiff_datasets(city_id_str)  # ✅ 使用预构建bounds_arr
    except Exception as e:
        print(f"❌ TIFF 加载失败 (city_id={city_id_str}): {e}")
        return

    match = re.search(r'od_paths_flows_(\d+)(?:_chunk_\d+)?\.parquet$', parquet_file.name)
    if not match:
        print(f"❌ 无法从文件名解析 area_id: {parquet_file.name}")
        return
    area_id = match.group(1)

    try:
        edge_geom_dict = load_road_network_cached(area_id)
    except Exception as e:
        print(f"❌ 路网加载失败 (area_id={area_id}): {e}")
        return

    global model
    if 'model' not in globals() or model is None:
        raise RuntimeError("模型未在进程初始化时加载！")

    city_info = city_centers[city_id]
    all_results = []

    # ✅ 恢复tqdm进度条
    batch_points = []
    batch_metadata = []

    total_valid = 0
    total_invalid = 0

    # ✅ 使用tqdm显示进度
    pbar = tqdm(df.iterrows(), total=len(df), desc=f"City {city_id_str}{chunk_info}", leave=True)

    for idx, row in pbar:
        # 构建seg_id
        if chunk_id is not None:
            seg_id = f"{city_id_str}_chunk_{chunk_id}_{idx}"
        else:
            seg_id = f"{city_id_str}_{idx}"

        # 重建几何
        edge_sequence = row['edge_sequence']
        geom = batch_rebuild_geometries([edge_sequence], edge_geom_dict)[0]
        geom = flatten_geometry(geom) if geom is not None else None

        if geom is None or geom.is_empty:
            all_results.append({
                'seg_id': seg_id,
                'pred_label': -1,
                'point_count': 0,
                'total_length_m': 0.0,
                'chunk_id': chunk_id if chunk_id is not None else 'N/A'
            })
            total_invalid += 1
            continue

        # 采样轨迹点
        # 🔽 新增：应用自适应时序降采样 🔽
        sampled_points = direct_adaptive_sample_linestring(
            geom,
            base_interval_m=TEMPORAL_BASE_INTERVAL,  # 500.0
            max_timesteps=MAX_TOTAL_TIMESTEPS,  # 100
            head_tail_reserved_points=HEAD_TAIL_RESERVED_POINTS,  # 50
            middle_downsample_rate=MIDDLE_DOWNSAMPLE_RATE,  # 1.0
            strategy=STRATEGY  # "head_tail_focus"
        )

        # 降采样后再次检查有效性
        if len(sampled_points) < 2:
            all_results.append({
                'seg_id': seg_id,
                'pred_label': -1,
                'point_count': 0,
                'total_length_m': 0.0,
                'chunk_id': chunk_id if chunk_id is not None else 'N/A'
            })
            total_invalid += 1
            continue

        # 准备点数据（使用降采样后的点）
        lons = [pt.x for pt in sampled_points]
        lats = [pt.y for pt in sampled_points]
        points_dict = [{'lat': lat, 'lon': lon} for lat, lon in zip(lats, lons)]

        # 向量化计算总长度
        # 向量化计算总长度（使用降采样后的点）
        if len(points_dict) >= 2:
            lats_arr = np.array([p['lat'] for p in points_dict], dtype=np.float64)
            lons_arr = np.array([p['lon'] for p in points_dict], dtype=np.float64)
            dlat = np.diff(lats_arr)
            dlon = np.diff(lons_arr)
            avg_lat = (lats_arr[:-1] + lats_arr[1:]) / 2.0
            meters_per_deg_lon = 111320.0 * np.cos(np.radians(avg_lat))
            segment_dists = np.sqrt((dlat * 111320.0) ** 2 + (dlon * meters_per_deg_lon) ** 2)
            total_length = float(segment_dists.sum())
        else:
            total_length = 0.0

        # 计算中心特征
        raw_feats = compute_center_features_raw(points_dict, city_info, total_length)

        # 累积点数据
        start_idx = len(batch_points)
        batch_points.extend(zip(lons, lats))
        batch_metadata.append({
            'seg_id': seg_id,
            'start_idx': start_idx,
            'point_count': len(lons),
            'raw_feats': raw_feats,
            'total_length': total_length
        })

        # 按点数触发批处理
        if len(batch_points) >= MAX_POINTS_PER_BATCH or idx == len(df) - 1:
            if batch_points:
                # 1. 批量提取所有点的影像特征
                all_lons = [p[0] for p in batch_points]
                all_lats = [p[1] for p in batch_points]
                all_patches = batch_extract_ae_features(
                    all_lons, all_lats, open_datasets, bounds_arr,
                    selected_bands=SELECTED_BANDS_INFERENCE if USE_BAND_SELECTION_INFERENCE else None
                )

                # 2. 按轨迹重组特征
                patches_list = []
                center_list = []
                lengths_list = []

                for meta in batch_metadata:
                    pts = all_patches[meta['start_idx']:meta['start_idx'] + meta['point_count']]
                    patches_list.append(np.stack(pts))
                    center_list.append(standardize_center_features(meta['raw_feats'], scaler_params))
                    lengths_list.append(meta['point_count'])

                # 3. Padding
                max_len = max(p.shape[0] for p in patches_list) if patches_list else 0
                if max_len > 0:
                    batch_size = len(patches_list)
                    padded_patches = np.zeros((batch_size, max_len, INPUT_CHANNELS_INFERENCE, 11, 11), dtype=np.float32)

                    for i, p in enumerate(patches_list):
                        padded_patches[i, :p.shape[0]] = p

                    # 4. GPU推理
                    with torch.no_grad():
                        x_img = torch.from_numpy(padded_patches).float().to(DEVICE)
                        x_center = torch.from_numpy(np.stack(center_list)).float().to(DEVICE)
                        lengths = torch.tensor(lengths_list, dtype=torch.long).to(DEVICE)

                        logits, _ = model(
                            x_img=x_img,
                            x_land=None,
                            x_traffic_stats=None,
                            x_center=x_center,
                            lengths=lengths
                        )
                        preds = logits.argmax(dim=1).cpu().numpy()

                    # 5. 保存结果
                    for i, meta in enumerate(batch_metadata):
                        all_results.append({
                            'seg_id': meta['seg_id'],
                            'pred_label': int(preds[i]),
                            'point_count': meta['point_count'],
                            'total_length_m': round(meta['total_length'], 2),
                            'chunk_id': chunk_id if chunk_id is not None else 'N/A'
                        })
                        total_valid += 1

            # 重置批次
            batch_points = []
            batch_metadata = []

        # ✅ 更新进度条后缀信息
        if idx % 100 == 0 or idx == len(df) - 1:
            pbar.set_postfix({
                'Valid': f'{total_valid:,}',
                'Invalid': f'{total_invalid:,}',
                'Total': f'{total_valid + total_invalid:,}'
            })

    pbar.close()  # 关闭进度条

    # 保存结果
    if all_results:
        result_df = pd.DataFrame(all_results)
        result_df.to_parquet(output_path, index=False, compression='snappy')
        print(f"✅ 城市 {city_id_str}{chunk_info} 完成，共 {len(result_df)} 条轨迹，结果保存至: {output_path}")
    else:
        print(f"⚠️ 城市 {city_id_str}{chunk_info} 无有效轨迹")


# ==================== 任务处理（按chunk文件级并行）====================
def process_single_chunk(args):
    parquet_path, city_centers, scaler_params = args
    proc_name = multiprocessing.current_process().name

    # 从文件名提取city_id
    city_id_str, chunk_id = extract_id_and_chunk_from_parquet_filename(Path(parquet_path).name)
    if not city_id_str:
        print(f"[{proc_name}] ⚠️ 跳过 {Path(parquet_path).name}: 无法提取city_id")
        return

    city_id = int(city_id_str)
    if city_id not in city_centers:
        print(f"[{proc_name}] ⚠️ 跳过 {city_id_str}: 未找到城市中心信息")
        return

    chunk_info = f" chunk {chunk_id}" if chunk_id else ""
    print(f"[{proc_name}] 🚀 处理 {city_id_str}{chunk_info} ({Path(parquet_path).name})")
    process_city_file(parquet_path, city_centers, scaler_params)
    print(f"[{proc_name}] ✅ 完成 {city_id_str}{chunk_info}")


# ==================== 进程初始化 ====================
model = None


def init_worker():
    global model
    os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
    os.environ['GDAL_CACHEMAX'] = '2048'
    os.environ['GDAL_NUM_THREADS'] = '1'

    print(f"[{multiprocessing.current_process().name}] 正在预加载模型 (Max Points/Batch: {MAX_POINTS_PER_BATCH})...")
    model = UrbanMultiModalClassifier(
        num_classes=NUM_CLASSES,
        input_channels=INPUT_CHANNELS_INFERENCE,  # ✅ 动态适配通道数
        use_env=USE_ENV,
        use_landuse=USE_LANDUSE,
        use_traffic=USE_TRAFFIC,
        use_center=USE_CENTER
    ).to(DEVICE)

    state_dict = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    torch.cuda.empty_cache()
    print(f"[{multiprocessing.current_process().name}] ✅ 模型预加载完成")


# ==================== 主流程（智能任务排序）====================
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    print("=" * 70)
    print(f"🚀 启动动态推理引擎 | Max Points/Batch: {MAX_POINTS_PER_BATCH} | Processes: {NUM_PROCESSES}")
    if torch.cuda.is_available():
        free_mem_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        print(f"   GPU: {torch.cuda.get_device_name(0)} | Free Memory: {free_mem_gb:.1f} GB")
        print(f"   优化: TIFF边界缓存 + Chunk级并行 + 双缓冲预取")
    print("=" * 70)

    city_centers = load_city_centers(CENTER_CSV_PATH)
    if not city_centers:
        raise RuntimeError("未能加载任何城市中心数据！请检查CSV路径")

    try:
        scaler_params = load_center_scaler_params(CENTER_SCALER_JSON)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # ✅ 按城市ID排序chunk文件（提高缓存局部性）
    parquet_files = sorted(Path(INPUT_PARQUET_DIR).glob("od_paths_flows_*.parquet"))


    def get_city_id_key(path):
        match = re.search(r'od_paths_flows_(\d+)', path.name)
        return int(match.group(1)) if match else 999999


    parquet_files_sorted = sorted(parquet_files, key=get_city_id_key)

    print(f"🔍 发现 {len(parquet_files_sorted)} 个chunk文件，已加载 {len(city_centers)} 个城市中心信息")

    # ✅ 每个chunk文件一个独立任务
    tasks = []
    for pf in parquet_files_sorted:
        city_id_str, _ = extract_id_and_chunk_from_parquet_filename(pf.name)
        if city_id_str and int(city_id_str) in city_centers:
            tasks.append((str(pf), city_centers, scaler_params))

    print(f"\nParallelGroup: 共 {len(tasks)} 个chunk任务，使用 {NUM_PROCESSES} 个进程并行处理")
    print("   任务排序: 按城市ID分组（提高TIFF/路网缓存命中率）")
    print("-" * 70)

    with multiprocessing.Pool(
            processes=NUM_PROCESSES,
            initializer=init_worker
    ) as pool:
        pool.map(process_single_chunk, tasks)

    print("\n" + "=" * 70)
    print("🎉 所有chunk处理完毕！")
    print(f"   预测结果保存至: {OUTPUT_PRED_DIR}")
    print("=" * 70)