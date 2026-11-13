import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['GDAL_CACHEMAX'] = '8192'
os.environ['GDAL_SWATH_SIZE'] = '2048'
os.environ['GDAL_NUM_THREADS'] = 'ALL_CPUS'
os.environ['VSI_CACHE'] = 'TRUE'
os.environ['VSI_CACHE_SIZE'] = '1073741824'
os.environ['RASTERIO_NUM_THREADS'] = 'ALL_CPUS'

import glob
import time
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
from typing import List, Dict, Any, Optional, Tuple, Set
import json
import multiprocessing
import atexit

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
INPUT_PARQUET_DIR = "${SERVER_HOME}/result/trace_genetare_hgih_income_2017/"
OUTPUT_PRED_DIR = "${SERVER_HOME}/result/mode_predict/high_income_2017/"
MODEL_CHECKPOINT = "urban_transformer_cls_best_IC.pth"
CENTER_CSV_PATH = "${SERVER_DATA_ROOT}/prediction_travel_mode/trace_generate/road_network_clip/data/all_data/2017_pop_center/POP_Centers_2017_LocalPeak.csv"
CENTER_SCALER_JSON = "${SERVER_DATA_ROOT}/prediction_travel_mode/model/code/center_scaler_params.json"
GRAPHML_ROOT = Path("${SERVER_HOME}/result/road_2017/")
REGION_DIRS = ["${SERVER_HOME}/GEE/GEE_Embeddings_high_income_2017/"]


def get_optimal_points_per_batch():
    if torch.cuda.is_available():
        free_mem_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        return min(32768 * 4, max(4096, int(free_mem_gb * 2200)))
    return 8192


MAX_POINTS_PER_BATCH = get_optimal_points_per_batch()
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_PROCESSES = 1
USE_ENV = True
USE_LANDUSE = False
USE_TRAFFIC = False
USE_CENTER = True
CENTER_FEATURES = [
    'start_to_primary_m', 'end_to_primary_m', 'start_to_nearest_sub_m',
    'end_to_nearest_sub_m', 'trip_min_to_primary_m', 'trip_min_to_nearest_sub_m',
    'od_ratio_primary', 'od_ratio_nearest_sub', 'city_total_centers'
]
os.makedirs(OUTPUT_PRED_DIR, exist_ok=True)

# ==================== 全局缓存 ====================
ROAD_NETWORK_CACHE = {}
TIFF_CACHE = {}
TIFF_BOUNDS_CACHE = {}
TIFF_BOUNDS_ARR_CACHE = {}
# ✅ 新增：城市级预计算patch缓存（chunk级去重核心）
CITY_PATCH_CACHE = {}


def cleanup_tiff_cache():
    global TIFF_CACHE, TIFF_BOUNDS_CACHE, TIFF_BOUNDS_ARR_CACHE, CITY_PATCH_CACHE
    for city_id, datasets in list(TIFF_CACHE.items()):
        for ds in datasets:
            try:
                ds.close()
            except:
                pass
    TIFF_CACHE.clear()
    TIFF_BOUNDS_CACHE.clear()
    TIFF_BOUNDS_ARR_CACHE.clear()
    CITY_PATCH_CACHE.clear()


atexit.register(cleanup_tiff_cache)


# ==================== 工具函数 ====================
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
    """保持与原逻辑完全等价，确保结果一致"""
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


def extract_11x11_patch(dataset, x, y, patch_size=11):
    half = patch_size // 2
    window = Window(x - half, y - half, patch_size, patch_size)
    try:
        data = dataset.read(window=window, out_shape=(64, patch_size, patch_size), resampling=Resampling.nearest,
                            fill_value=0)
        return data.astype(np.float32)
    except Exception:
        return np.zeros((64, patch_size, patch_size), dtype=np.float32)


# ==================== 核心优化：Chunk级全局去重预计算 ====================
from numpy.lib.stride_tricks import sliding_window_view


def _coord_to_key(lon: float, lat: float, precision: int = 6) -> Tuple[int, int]:
    """将坐标转换为整数key，避免浮点精度问题，1米精度"""
    # 1度≈111km，6位小数≈0.11米精度
    scale = 10 ** precision
    return (int(round(lon * scale)), int(round(lat * scale)))


def precompute_city_edge_patches(
        city_id: str,
        edge_geom_dict: Dict[tuple, LineString],
        open_datasets: List[rasterio.DatasetReader],
        bounds_arr: np.ndarray,
        interval_m: float = 100.0,
        batch_size: int = 50000  # 控制内存峰值
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    🚀 Chunk级全局去重：预计算城市所有边的唯一采样点patch
    返回: {(lon_key, lat_key): patch_array} 查找表
    """
    cache_key = f"{city_id}_{interval_m}"
    if cache_key in CITY_PATCH_CACHE:
        return CITY_PATCH_CACHE[cache_key]

    print(f"  → 预计算城市 {city_id} 路网采样点 (去重模式)...")

    # 1. 收集所有边的唯一采样点（坐标去重）
    unique_coords: Dict[Tuple[int, int], Tuple[float, float]] = {}  # key -> (lon, lat)

    for edge_key, edge_geom in edge_geom_dict.items():
        if edge_geom is None or edge_geom.is_empty:
            continue
        # 采样（保持与原函数等价）
        sampled = sample_linestring_by_haversine(edge_geom, interval_m)
        for pt in sampled:
            key = _coord_to_key(pt.x, pt.y)
            if key not in unique_coords:
                unique_coords[key] = (pt.x, pt.y)

    print(f"  → 路网边数: {len(edge_geom_dict):,} | 唯一采样点: {len(unique_coords):,}")

    # 2. 分批批量提取patch（避免OOM）
    coord_to_patch: Dict[Tuple[int, int], np.ndarray] = {}
    coords_list = list(unique_coords.items())

    for batch_start in range(0, len(coords_list), batch_size):
        batch_end = min(batch_start + batch_size, len(coords_list))
        batch_items = coords_list[batch_start:batch_end]

        lons = [coord[0] for _, coord in batch_items]
        lats = [coord[1] for _, coord in batch_items]
        keys = [key for key, _ in batch_items]

        # 复用已有的向量化提取函数
        patches = batch_extract_ae_features(lons, lats, open_datasets, bounds_arr)

        for key, patch in zip(keys, patches):
            if np.any(patch != 0):  # 保持原逻辑：只缓存非零patch
                coord_to_patch[key] = patch

    # 缓存结果（进程内复用）
    CITY_PATCH_CACHE[cache_key] = coord_to_patch
    print(f"  → 预计算完成 | 缓存patch数: {len(coord_to_patch):,}")

    return coord_to_patch


def get_trajectory_patches_from_cache(
        geom: LineString,
        coord_to_patch: Dict[Tuple[int, int], np.ndarray],
        interval_m: float = 100.0
) -> List[np.ndarray]:
    """
    从预计算缓存中快速获取轨迹的patch序列
    保持与原逻辑完全等价：采样点顺序、插值逻辑、patch内容一致
    """
    if geom is None or geom.is_empty:
        return []

    sampled = sample_linestring_by_haversine(geom, interval_m)
    if len(sampled) < 2:
        return []

    patches = []
    for pt in sampled:
        key = _coord_to_key(pt.x, pt.y)
        if key in coord_to_patch:
            patches.append(coord_to_patch[key])
        else:
            # 边界情况：坐标未命中（理论上不应发生），返回零patch保持兼容
            patches.append(np.zeros((64, 11, 11), dtype=np.float32))

    return patches


# ==================== 原有batch_extract_ae_features（保持原逻辑）====================
def batch_extract_ae_features(
        lons: List[float],
        lats: List[float],
        open_datasets: List[rasterio.DatasetReader],
        bounds_arr: np.ndarray,
        mem_cache: Optional[Dict[int, np.ndarray]] = None
) -> List[np.ndarray]:
    """保持与原方案结果一致 + 向量化加速"""
    n = len(lons)
    if n == 0:
        return [np.zeros((64, 11, 11), dtype=np.float32) for _ in range(n)]

    lons_arr = np.asarray(lons, dtype=np.float64)
    lats_arr = np.asarray(lats, dtype=np.float64)

    # 向量化边界匹配
    lefts, bottoms, rights, tops = bounds_arr.T
    in_bounds = (
            (lons_arr[:, None] >= lefts) & (lons_arr[:, None] <= rights) &
            (lats_arr[:, None] >= bottoms) & (lats_arr[:, None] <= tops)
    )
    tiff_idx = np.argmax(in_bounds, axis=1)
    valid = in_bounds[np.arange(n), tiff_idx]
    valid_idxs = np.where(valid)[0]

    if len(valid_idxs) == 0:
        return [np.zeros((64, 11, 11), dtype=np.float32) for _ in range(n)]

    results = np.zeros((n, 64, 11, 11), dtype=np.float32)
    half = 5

    # 按数据集分组
    groups: Dict[int, List[int]] = {}
    for idx in valid_idxs:
        groups.setdefault(tiff_idx[idx], []).append(idx)

    # 函数级缓存
    if not hasattr(batch_extract_ae_features, '_mem_cache'):
        batch_extract_ae_features._mem_cache = {}

    for ds_idx, idxs in groups.items():
        ds = open_datasets[ds_idx]
        cache_key = id(ds)
        if cache_key not in batch_extract_ae_features._mem_cache:
            batch_extract_ae_features._mem_cache[cache_key] = ds.read().astype(np.float32)

        img_data = batch_extract_ae_features._mem_cache[cache_key]
        pts_lons = lons_arr[idxs]
        pts_lats = lats_arr[idxs]

        # 批量坐标转换
        cols, rows = ~ds.transform * (pts_lons, pts_lats)
        cols = np.round(cols).astype(np.int32)
        rows = np.round(rows).astype(np.int32)

        # 边缘保护
        safe_rows = np.clip(rows, half, ds.height - half - 1)
        safe_cols = np.clip(cols, half, ds.width - half - 1)

        # 向量化提取 + 转置
        windows = sliding_window_view(img_data, window_shape=(11, 11), axis=(1, 2))
        patches = windows[:, safe_rows - half, safe_cols - half, :, :]
        patches = np.transpose(patches, (1, 0, 2, 3))

        v_orig = np.array(idxs)
        nonzero_mask = np.any(patches != 0, axis=(1, 2, 3))
        results[v_orig[nonzero_mask]] = patches[nonzero_mask]

    return [results[i] for i in range(n)]


def flatten_geometry(geom):
    if isinstance(geom, MultiLineString):
        coords = []
        for line in geom.geoms:
            coords.extend(line.coords)
        return LineString(coords) if len(coords) >= 2 else None
    elif isinstance(geom, LineString):
        return geom
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
            geom = LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])
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
    df = pd.read_csv(csv_path).rename(columns={'city_name': 'city_id'})
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
    print(f"Loaded {len(centers_info)} cities with center information")
    return centers_info


def compute_center_features_raw(points: List[Dict], city_info: Dict, total_length: float) -> np.ndarray:
    if len(points) < 2 or total_length <= 0:
        return np.zeros(len(CENTER_FEATURES), dtype=np.float32)
    lats = np.array([p['lat'] for p in points], dtype=np.float64)
    lons = np.array([p['lon'] for p in points], dtype=np.float64)
    primary_lat, primary_lon = city_info['primary_lat'], city_info['primary_lon']
    sub_centers, total_centers = city_info['sub_centers'], city_info['total_centers']

    start_to_primary = haversine_distance(points[0]['lat'], points[0]['lon'], primary_lat, primary_lon)
    end_to_primary = haversine_distance(points[-1]['lat'], points[-1]['lon'], primary_lat, primary_lon)

    dlat = lats - primary_lat
    dlon = lons - primary_lon
    avg_lat_rad = np.radians((lats + primary_lat) / 2.0)
    meters_per_deg_lon = 111320.0 * np.cos(avg_lat_rad)
    dists_to_primary = np.sqrt((dlat * 111320.0) ** 2 + (dlon * meters_per_deg_lon) ** 2)
    min_to_primary = float(dists_to_primary.min())

    if len(sub_centers) > 0:
        sub_lons, sub_lats = sub_centers[:, 0], sub_centers[:, 1]
        start_to_nearest_sub = float(haversine_vectorized(points[0]['lat'], points[0]['lon'], sub_lats, sub_lons).min())
        end_to_nearest_sub = float(haversine_vectorized(points[-1]['lat'], points[-1]['lon'], sub_lats, sub_lons).min())
        dlat_sub = lats[:, None] - sub_lats[None, :]
        dlon_sub = lons[:, None] - sub_lons[None, :]
        avg_lat_rad_sub = np.radians((lats[:, None] + sub_lats[None, :]) / 2.0)
        meters_per_deg_lon_sub = 111320.0 * np.cos(avg_lat_rad_sub)
        dists_to_subs = np.sqrt((dlat_sub * 111320.0) ** 2 + (dlon_sub * meters_per_deg_lon_sub) ** 2)
        min_to_nearest_sub = float(dists_to_subs.min())
    else:
        start_to_nearest_sub = end_to_nearest_sub = min_to_nearest_sub = -1.0

    od_ratio_primary = (start_to_primary + end_to_primary) / (total_length + 1e-6)
    od_ratio_nearest_sub = (start_to_nearest_sub + end_to_nearest_sub) / (
                total_length + 1e-6) if start_to_nearest_sub > 0 else -1.0

    return np.array([
        start_to_primary, end_to_primary,
        start_to_nearest_sub if start_to_nearest_sub > 0 else 0.0,
        end_to_nearest_sub if end_to_nearest_sub > 0 else 0.0,
        min_to_primary, min_to_nearest_sub if min_to_nearest_sub > 0 else 0.0,
        od_ratio_primary, od_ratio_nearest_sub if od_ratio_nearest_sub > 0 else -1.0,
        float(total_centers)
    ], dtype=np.float32)


def load_center_scaler_params(json_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ 未找到标准化参数文件: {json_path}")
    with open(json_path, 'r') as f:
        params = json.load(f)
    mean = np.array(params['mean'], dtype=np.float32)
    scale = np.array(params['scale'], dtype=np.float32)
    print(f"✓ Loaded center feature scaler parameters from {json_path}")
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
    def __init__(self, d_model, max_len=10000):
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
        return self.attn_pool(x)


class UrbanMultiModalClassifier(nn.Module):
    def __init__(self, num_classes=4, input_channels=64, embed_dim=128, nhead=8, num_layers=6,
                 use_env=True, use_landuse=False, use_traffic=False, use_center=True):
        super().__init__()
        self.use_env, self.use_landuse, self.use_traffic, self.use_center = use_env, use_landuse, use_traffic, use_center
        self.embed_dim = embed_dim
        if self.use_env:
            self.patch_encoder = PatchEncoder(input_channels, embed_dim)
            self.self_attn_img = nn.MultiheadAttention(embed_dim, num_heads=nhead // 2, batch_first=True)
        if self.use_center:
            self.center_encoder = nn.Sequential(nn.Linear(len(CENTER_FEATURES), embed_dim), nn.ReLU(),
                                                nn.Linear(embed_dim, embed_dim))
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_encoder = LearnablePositionalEmbedding(embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim)
        fusion_in_dim = embed_dim + (embed_dim if use_center else 0)
        self.modality_fusion_proj = nn.Linear(fusion_in_dim, embed_dim) if fusion_in_dim != embed_dim else None
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True,
                                                   dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.1),
                                        nn.Linear(64, num_classes))

    def forward(self, x_img, x_land, x_traffic_stats, x_center, lengths):
        B = x_img.size(0) if x_img is not None else x_center.size(0)
        embeddings_img = None
        if self.use_env and x_img is not None:
            T = x_img.shape[1]
            x_flat = x_img.view(B * T, x_img.shape[2], x_img.shape[3], x_img.shape[4])
            embeddings_img = self.patch_encoder(x_flat).view(B, T, -1)
            embeddings_img = self.embed_norm(embeddings_img)
        center_emb = None
        if self.use_center and x_center is not None:
            center_emb = self.embed_norm(self.center_encoder(x_center).unsqueeze(1))
        combined_list, T_final = [], 0
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
        combined = combined_list[0] if len(combined_list) == 1 else torch.cat(combined_list, dim=-1)
        if self.modality_fusion_proj is not None:
            combined = self.modality_fusion_proj(combined)
        combined = self.final_norm(combined)
        embeddings = torch.cat([self.cls_token.expand(B, -1, -1), combined], dim=1)
        embeddings = self.pos_encoder(embeddings)
        total_seq_len = embeddings.shape[1] - 1
        pad_mask = torch.arange(total_seq_len, device=embeddings.device).unsqueeze(0) >= lengths.unsqueeze(
            1) if total_seq_len > 0 and lengths is not None else torch.zeros(B, total_seq_len, dtype=torch.bool,
                                                                             device=embeddings.device)
        full_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=embeddings.device), pad_mask], dim=1)
        cls_out = self.transformer(embeddings, src_key_padding_mask=full_mask)[:, 0, :]
        return self.classifier(cls_out), cls_out


# ==================== 资源加载 ====================
def get_tiff_datasets(city_id: str) -> Tuple[List[rasterio.DatasetReader], List[tuple], np.ndarray]:
    global TIFF_CACHE, TIFF_BOUNDS_CACHE, TIFF_BOUNDS_ARR_CACHE
    if city_id in TIFF_CACHE:
        return TIFF_CACHE[city_id], TIFF_BOUNDS_CACHE[city_id], TIFF_BOUNDS_ARR_CACHE[city_id]
    tiff_paths = find_tiff_files_by_id(city_id)
    if not tiff_paths:
        raise FileNotFoundError(f"No TIFF files found for city {city_id}")
    datasets, bounds = [], []
    for tif in tiff_paths:
        try:
            ds = rasterio.open(tif, 'r', sharing=False)
            datasets.append(ds)
            bounds.append((ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top))
        except Exception as e:
            print(f"⚠️ 跳过损坏的 TIFF {tif}: {e}")
    if not datasets:
        raise RuntimeError(f"无法打开任何有效 TIFF for city {city_id}")
    bounds_arr = np.array(bounds, dtype=np.float64)
    TIFF_CACHE[city_id] = datasets
    TIFF_BOUNDS_CACHE[city_id] = bounds
    TIFF_BOUNDS_ARR_CACHE[city_id] = bounds_arr
    print(f"  → 首次加载 TIFF for city {city_id}: {len(datasets)} files")
    return datasets, bounds, bounds_arr


# ==================== 诊断辅助函数 ====================
def print_diag_table(timers: dict, total_valid: int, total_invalid: int, file_time: float):
    print("\n" + "=" * 80)
    print("📊 运行时间诊断报告 | 耗时单位: 秒 (s)")
    print("-" * 80)
    total_cpu = sum(v for k, v in timers.items() if 'gpu' not in k and 'tiff' not in k)
    total_batch = sum(v for k, v in timers.items() if 'tiff' in k or 'batch' in k or 'gpu' in k)
    rows = [
        ("📥 初始化与加载", timers['load_parquet'] + timers['load_tiffs'] + timers['load_roadnet']),
        ("🗺️ 几何重建 (CPU)", timers['geo_rebuild']),
        ("📍 轨迹采样 (CPU)", timers['sample_pts']),
        ("📏 长度计算 (CPU)", timers['calc_length']),
        ("🏙️ 中心特征 (CPU)", timers['center_feat']),
        ("🖼️ TIFF I/O 提取", timers['tiff_io']),
        ("📦 Padding & 标准化", timers['prepare_batch']),
        ("🚀 CPU->GPU 传输", timers['gpu_transfer']),
        ("🧠 GPU 模型推理", timers['infer_gpu']),
        ("💾 结果保存", timers['save_result'])
    ]
    for name, t in rows:
        pct = (t / file_time * 100) if file_time > 0 else 0
        print(f"{name:<20s} | {t:>8.3f}s  | {pct:>5.1f}%")
    print("-" * 80)
    print(f"✅ 总耗时: {file_time:.3f}s | 有效轨迹: {total_valid:,} | 无效轨迹: {total_invalid:,}")
    print(f"⚠️ CPU预处理占比: {total_cpu / file_time * 100:.1f}% | Batch推理占比: {total_batch / file_time * 100:.1f}%")
    print("=" * 80 + "\n")


# ==================== 核心推理主函数（Chunk级去重优化版）====================
def process_city_file(parquet_path: str, city_centers: Dict[int, Dict], scaler_params: Dict):
    parquet_file = Path(parquet_path)
    id_chunk_result = extract_id_and_chunk_from_parquet_filename(parquet_file.name)
    if not id_chunk_result:
        print(f"⚠️ 跳过 {parquet_file.name}: 无法提取 ID/chunk 信息");
        return
    city_id_str, chunk_id = id_chunk_result
    city_id = int(city_id_str)
    if city_id not in city_centers:
        print(f"⚠️ 跳过 {city_id_str}: 未找到城市中心信息");
        return
    output_filename = f"{city_id_str}_chunk_{chunk_id}_predictions.parquet" if chunk_id else f"{city_id_str}_predictions.parquet"
    output_path = os.path.join(OUTPUT_PRED_DIR, output_filename)
    if os.path.exists(output_path):
        print(f"✅ 跳过 {city_id_str}: 结果已存在");
        return
    chunk_info = f" chunk {chunk_id}" if chunk_id else ""
    print(f"🚀 处理城市 {city_id_str} (ID: {city_id}){chunk_info} ...")

    t_start = time.perf_counter()
    timers = {k: 0.0 for k in
              ['load_parquet', 'load_tiffs', 'load_roadnet', 'geo_rebuild', 'sample_pts', 'calc_length', 'center_feat',
               'tiff_io', 'prepare_batch', 'gpu_transfer', 'infer_gpu', 'save_result']}

    # 1. 加载 Parquet
    t0 = time.perf_counter()
    try:
        import pyarrow.parquet as pq
        df = pq.read_table(parquet_file).to_pandas().reset_index(drop=True)
    except Exception as e:
        print(f"❌ 读取失败 {parquet_file}: {e}"); return
    timers['load_parquet'] += time.perf_counter() - t0
    if df.empty: print(f"⚠️ {city_id_str} 为空"); return

    # 2. 加载资源
    t0 = time.perf_counter()
    try:
        open_datasets, _, bounds_arr = get_tiff_datasets(city_id_str)
    except Exception as e:
        print(f"❌ TIFF 加载失败: {e}"); return
    timers['load_tiffs'] += time.perf_counter() - t0

    match = re.search(r'od_paths_flows_(\d+)(?:_chunk_\d+)?\.parquet$', parquet_file.name)
    if not match: print(f"❌ 无法解析 area_id"); return
    area_id = match.group(1)
    t0 = time.perf_counter()
    try:
        edge_geom_dict = load_road_network_cached(area_id)
    except Exception as e:
        print(f"❌ 路网加载失败: {e}"); return
    timers['load_roadnet'] += time.perf_counter() - t0

    # ✅ 核心优化：预计算城市所有边的唯一采样点patch（Chunk级去重）
    t0 = time.perf_counter()
    coord_to_patch = precompute_city_edge_patches(
        city_id=city_id_str,
        edge_geom_dict=edge_geom_dict,
        open_datasets=open_datasets,
        bounds_arr=bounds_arr,
        interval_m=100.0,
        batch_size=50000
    )
    timers['tiff_io'] += time.perf_counter() - t0  # 预计算计入TIFF I/O时间
    print(f"  ✓ 预计算耗时: {timers['tiff_io']:.2f}s | 唯一patch数: {len(coord_to_patch):,}")

    global model
    if 'model' not in globals() or model is None: raise RuntimeError("模型未加载！")
    city_info = city_centers[city_id]
    all_results = []
    batch_points, batch_metadata = [], []
    total_valid, total_invalid = 0, 0
    batch_cnt = 0

    pbar = tqdm(df.iterrows(), total=len(df), desc=f"City {city_id_str}{chunk_info}", leave=True)
    for idx, row in pbar:
        seg_id = f"{city_id_str}_chunk_{chunk_id}_{idx}" if chunk_id else f"{city_id_str}_{idx}"
        edge_sequence = row['edge_sequence']

        # 🕒 诊断：几何重建
        t0 = time.perf_counter()
        geom = batch_rebuild_geometries([edge_sequence], edge_geom_dict)[0]
        geom = flatten_geometry(geom) if geom is not None else None
        timers['geo_rebuild'] += time.perf_counter() - t0
        if geom is None or geom.is_empty:
            all_results.append({'seg_id': seg_id, 'pred_label': -1, 'point_count': 0, 'total_length_m': 0.0,
                                'chunk_id': chunk_id or 'N/A'})
            total_invalid += 1;
            continue

        # ✅ 核心优化：从预计算缓存快速获取轨迹patch（替代原采样+TIFF读取）
        t0 = time.perf_counter()
        patches_list_traj = get_trajectory_patches_from_cache(geom, coord_to_patch, interval_m=100.0)
        timers['sample_pts'] += time.perf_counter() - t0  # 采样+查表计入sample_pts

        if len(patches_list_traj) < 2:
            all_results.append({'seg_id': seg_id, 'pred_label': -1, 'point_count': 0, 'total_length_m': 0.0,
                                'chunk_id': chunk_id or 'N/A'})
            total_invalid += 1;
            continue

        # 保持原逻辑：生成points_dict用于中心特征计算
        lons = [pt.x for pt in sample_linestring_by_haversine(geom, interval_m=100.0)]
        lats = [pt.y for pt in sample_linestring_by_haversine(geom, interval_m=100.0)]
        points_dict = [{'lat': lat, 'lon': lon} for lat, lon in zip(lats, lons)]

        # 🕒 诊断：长度计算
        t0 = time.perf_counter()
        if len(points_dict) >= 2:
            lats_arr = np.array([p['lat'] for p in points_dict], dtype=np.float64)
            lons_arr = np.array([p['lon'] for p in points_dict], dtype=np.float64)
            dlat, dlon = np.diff(lats_arr), np.diff(lons_arr)
            avg_lat = (lats_arr[:-1] + lats_arr[1:]) / 2.0
            meters_per_deg_lon = 111320.0 * np.cos(np.radians(avg_lat))
            total_length = float(np.sqrt((dlat * 111320.0) ** 2 + (dlon * meters_per_deg_lon) ** 2).sum())
        else:
            total_length = 0.0
        timers['calc_length'] += time.perf_counter() - t0

        # 🕒 诊断：中心特征计算
        t0 = time.perf_counter()
        raw_feats = compute_center_features_raw(points_dict, city_info, total_length)
        timers['center_feat'] += time.perf_counter() - t0

        # 累积至批次（保持原batch触发逻辑）
        start_idx = len(batch_points)
        batch_points.extend(zip(lons, lats))
        batch_metadata.append({
            'seg_id': seg_id, 'start_idx': start_idx, 'point_count': len(lons),
            'raw_feats': raw_feats, 'total_length': total_length,
            'patches': patches_list_traj  # ✅ 存储预计算的patch序列
        })

        # 🔄 触发批处理（保持原逻辑）
        if len(batch_points) >= MAX_POINTS_PER_BATCH or idx == len(df) - 1:
            if batch_points:
                # ✅ 从metadata中提取预计算的patch（替代原batch_extract_ae_features调用）
                t0 = time.perf_counter()  # TIFF I/O时间已在预计算阶段计入
                all_patches = [p for meta in batch_metadata for p in meta['patches']]
                timers['tiff_io'] += 0.0  # 查表无额外I/O

                # 🕒 诊断：Padding & 标准化（保持原逻辑）
                t0 = time.perf_counter()
                patches_list, center_list, lengths_list = [], [], []
                for meta in batch_metadata:
                    pts = all_patches[meta['start_idx']:meta['start_idx'] + meta['point_count']]
                    if pts:  # 确保非空
                        patches_list.append(np.stack(pts))
                        center_list.append(standardize_center_features(meta['raw_feats'], scaler_params))
                        lengths_list.append(meta['point_count'])

                max_len = max(p.shape[0] for p in patches_list) if patches_list else 0
                if max_len > 0:
                    batch_size = len(patches_list)
                    padded_patches = np.zeros((batch_size, max_len, 64, 11, 11), dtype=np.float32)
                    for i, p in enumerate(patches_list):
                        padded_patches[i, :p.shape[0]] = p
                timers['prepare_batch'] += time.perf_counter() - t0

                # 🕒 诊断：CPU -> GPU 传输（保持原逻辑）
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if max_len > 0:
                    x_img = torch.from_numpy(padded_patches).float().to(DEVICE)
                    x_center = torch.from_numpy(np.stack(center_list)).float().to(DEVICE)
                    lengths = torch.tensor(lengths_list, dtype=torch.long).to(DEVICE)
                torch.cuda.synchronize()
                timers['gpu_transfer'] += time.perf_counter() - t0

                # 🕒 诊断：GPU 推理（保持原逻辑）
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if max_len > 0:
                    with torch.no_grad():
                        logits, _ = model(
                            x_img=x_img, x_land=None, x_traffic_stats=None,
                            x_center=x_center, lengths=lengths
                        )
                        preds = logits.argmax(dim=1).cpu().numpy()
                torch.cuda.synchronize()
                timers['infer_gpu'] += time.perf_counter() - t0

                for i, meta in enumerate(batch_metadata):
                    all_results.append({
                        'seg_id': meta['seg_id'], 'pred_label': int(preds[i]),
                        'point_count': meta['point_count'], 'total_length_m': round(meta['total_length'], 2),
                        'chunk_id': chunk_id or 'N/A'
                    })
                    total_valid += 1

                # 实时诊断输出（每5个批次）
                batch_cnt += 1
                if batch_cnt % 5 == 0:
                    cur_batch_t = timers['tiff_io'] + timers['prepare_batch'] + timers['gpu_transfer'] + timers[
                        'infer_gpu']
                    print(
                        f"\n  📊 [Batch #{batch_cnt} 实时] tiff_io:{timers['tiff_io']:.2f}s | prep:{timers['prepare_batch']:.2f}s | gpu_transfer:{timers['gpu_transfer']:.2f}s | infer:{timers['infer_gpu']:.2f}s | 当前批次总:{cur_batch_t:.2f}s")

            batch_points, batch_metadata = [], []

        if idx % 200 == 0 or idx == len(df) - 1:
            pbar.set_postfix({'Valid': f'{total_valid:,}', 'Invalid': f'{total_invalid:,}',
                              'Total': f'{total_valid + total_invalid:,}'})

    pbar.close()

    # 🕒 诊断：保存结果
    t0 = time.perf_counter()
    if all_results:
        result_df = pd.DataFrame(all_results)
        result_df.to_parquet(output_path, index=False, compression='snappy')
        print(f"✅ 城市 {city_id_str}{chunk_info} 完成，共 {len(result_df)} 条轨迹")
    else:
        print(f"⚠️ 城市 {city_id_str}{chunk_info} 无有效轨迹")
    timers['save_result'] += time.perf_counter() - t0

    # 📊 打印完整诊断报告
    print_diag_table(timers, total_valid, total_invalid, time.perf_counter() - t_start)


# ==================== 任务处理 ====================
def process_single_chunk(args):
    parquet_path, city_centers, scaler_params = args
    proc_name = multiprocessing.current_process().name
    city_id_str, chunk_id = extract_id_and_chunk_from_parquet_filename(Path(parquet_path).name)
    if not city_id_str: print(f"[{proc_name}] ⚠️ 跳过 {Path(parquet_path).name}"); return
    city_id = int(city_id_str)
    if city_id not in city_centers: print(f"[{proc_name}] ⚠️ 跳过 {city_id_str}"); return
    chunk_info = f" chunk {chunk_id}" if chunk_id else ""
    print(f"[{proc_name}] 🚀 处理 {city_id_str}{chunk_info}")
    process_city_file(parquet_path, city_centers, scaler_params)
    print(f"[{proc_name}] ✅ 完成 {city_id_str}{chunk_info}")


model = None


def init_worker():
    global model
    os.environ['GDAL_NUM_THREADS'] = 'ALL_CPUS'
    os.environ['GDAL_CACHEMAX'] = '8192'
    print(f"[{multiprocessing.current_process().name}] 正在预加载模型...")
    model = UrbanMultiModalClassifier(num_classes=NUM_CLASSES, use_env=USE_ENV, use_landuse=USE_LANDUSE,
                                      use_traffic=USE_TRAFFIC, use_center=USE_CENTER).to(DEVICE)
    state_dict = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    torch.cuda.empty_cache()
    print(f"[{multiprocessing.current_process().name}] ✅ 模型预加载完成")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    print("=" * 70)
    print(f"🚀 启动动态推理引擎 | Chunk级全局去重优化 | Processes: {NUM_PROCESSES}")
    if torch.cuda.is_available(): print(
        f"   GPU: {torch.cuda.get_device_name(0)} | Free: {torch.cuda.mem_get_info()[0] / (1024 ** 3):.1f} GB")
    print("=" * 70)

    city_centers = load_city_centers(CENTER_CSV_PATH)
    if not city_centers: raise RuntimeError("未能加载城市中心数据！")
    try:
        scaler_params = load_center_scaler_params(CENTER_SCALER_JSON)
    except FileNotFoundError as e:
        print(e); exit(1)

    parquet_files = sorted(Path(INPUT_PARQUET_DIR).glob("od_paths_flows_*.parquet"))


    def get_city_id_key(path):
        m = re.search(r'od_paths_flows_(\d+)', path.name)
        return int(m.group(1)) if m else 999999


    parquet_files_sorted = sorted(parquet_files, key=get_city_id_key)

    print(f"🔍 发现 {len(parquet_files_sorted)} 个chunk文件")
    tasks = [(str(pf), city_centers, scaler_params) for pf in parquet_files_sorted if
             int(extract_id_and_chunk_from_parquet_filename(pf.name)[0]) in city_centers]
    print(f"ParallelGroup: 共 {len(tasks)} 个任务\n" + "-" * 70)

    with multiprocessing.Pool(processes=NUM_PROCESSES, initializer=init_worker) as pool:
        pool.map(process_single_chunk, tasks)

    print("\n" + "=" * 70 + "\n🎉 所有chunk处理完毕！")