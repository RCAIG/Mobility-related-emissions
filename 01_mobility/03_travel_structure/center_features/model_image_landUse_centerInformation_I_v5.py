import os

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("MODE5_CUDA_DEVICE", "1")
import pandas as pd
import numpy as np
import glob
import gzip
import base64
import torch
from pathlib import Path
from tqdm import tqdm
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.manifold import TSNE
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GroupShuffleSplit
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import json
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')
# ================= Configuration =================
# 当前.py文件的路径
current_file = os.path.abspath(__file__)
# 获取当前文件所在目录
current_dir = os.path.dirname(current_file)
RUN_NAME = os.environ.get("MODE5_RUN_NAME", "v5_grouped_baseline")
EXPERIMENT_DIR = os.path.join(current_dir, "mode5_experiments", RUN_NAME)
N_CLUSTERS = int(os.environ.get("MODE5_N_CLUSTERS", "5"))
RANDOM_STATE = 42
CLASS_NAMES = [
    "Mode 1 (slowest)" if label == 0 else
    f"Mode {label + 1} (fastest)" if label == N_CLUSTERS - 1 else
    f"Mode {label + 1}"
    for label in range(N_CLUSTERS)
]
ENV_FEATURES_DIR = os.path.join(os.path.dirname(current_dir), "output_features_GEE")
LANDUSE_FEATURES_DIR = os.path.join(os.path.dirname(current_dir), "output_features_landuse/work")
TRAJ_FEATURES_DIR = os.path.join(os.path.dirname(current_dir), "output_features_trajectory")
CENTER_FEATURES_DIR = os.path.join(os.path.dirname(current_dir), "output_features_pop_center/work")
TRAFFIC_FEATURES_DIR = os.path.join(os.path.dirname(current_dir), "output_features_transportation/work")
# Features definition
ADVANCED_FEATURES = [
    'adv_speed_95',
    'adv_speed_99',
    'adv_high_speed_ratio_10',
    'adv_high_speed_ratio_8',
    'adv_stop_ratio',
    'adv_acc_95',
    'adv_speed_std',
    'adv_heading_change_rate'
]
BASE_FEATURES = ['avg_speed_mps', 'avg_accel_mps2']
FEATURE_COLS = BASE_FEATURES + ADVANCED_FEATURES
# Land use types
LANDUSE_TYPES = ['commercial', 'industrial', 'park', 'public', 'residential']
# City center features
CENTER_FEATURES = ['start_to_primary_m', 'end_to_primary_m', 'start_to_nearest_sub_m', 'end_to_nearest_sub_m',
                   'trip_min_to_primary_m', 'trip_min_to_nearest_sub_m', 'od_ratio_primary', 'od_ratio_nearest_sub',
                   'city_total_centers']

# ================= 新增：时序采样策略配置 =================
TEMPORAL_DOWNSAMPLE_BASE = 5          # 基础降采样率：每5个100m点→1个500m点
MAX_TOTAL_TIMESTEPS = 100             # 最大保留时间步数（对应50km，超过则触发首尾策略）
HEAD_TAIL_RESERVED = 50               # 首尾各保留的点数（500m间隔，即各25km范围）
MIDDLE_DOWNSAMPLE_RATE = 1            # 中间部分的额外降采样率（2=1000m间隔，1=不额外降采样）
STRATEGY = "head_tail_focus"          # "uniform": 均匀降采样 | "head_tail_focus": 首尾优先

# ================= 新增：波段选择配置 =================
USE_BAND_SELECTION = True
SELECTED_BANDS = None
AUTO_SELECT_BANDS = True
USE_CENTER_CONFIG = os.environ.get("MODE5_USE_CENTER", "1") == "1"
REQUIRE_CENTER_MATCH = os.environ.get("MODE5_REQUIRE_CENTER_MATCH", "0") == "1"
REUSE_BANDS_JSON = os.environ.get("MODE5_SELECTED_BANDS_JSON")
if REUSE_BANDS_JSON:
    with open(REUSE_BANDS_JSON, "r", encoding="utf-8") as handle:
        SELECTED_BANDS = json.load(handle)["selected_bands"]
    AUTO_SELECT_BANDS = False

# 🔽 修改：启用累积贡献率模式 🔽
BAND_SELECTION_MODE = "cumulative"    # "fixed_count" | "percentile" | "cumulative"
NUM_BANDS_TO_KEEP = 16
PERCENTILE_THRESHOLD = 75
CUMULATIVE_THRESHOLD = 0.90           # 🔽 新增：累积贡献率阈值（0.90 表示保留到重要性累计达 90%）

INPUT_CHANNELS = len(SELECTED_BANDS) if (USE_BAND_SELECTION and SELECTED_BANDS) else 64
print(f"📡 Environment patch input channels: {INPUT_CHANNELS}/64 (will be updated after selection)")


# ================= Helper Functions =================
def decompress_patch(b64_str: str, selected_bands: list = None) -> np.ndarray:
    """
    解压 patch，可选只保留指定波段

    Parameters
    ----------
    b64_str : str
        base64+gzip 编码的字符串
    selected_bands : list[int], optional
        要保留的波段索引列表（如 [0, 5, 12, ...]），None 表示保留全部 64 个

    Returns
    -------
    arr : np.ndarray
        形状 [C_selected, 11, 11]（如果 selected_bands 不为 None）
        或 [64, 11, 11]（如果 selected_bands 为 None）
    """
    try:
        compressed = base64.b64decode(b64_str)
        raw = gzip.decompress(compressed)
        # 原始形状: [64, 11, 11]
        arr = np.frombuffer(raw, dtype=np.float32).reshape(64, 11, 11)

        # 🔽 如果指定了波段选择，只保留选中的波段 🔽
        if selected_bands is not None and len(selected_bands) < 64:
            arr = arr[selected_bands, :, :]  # [C_selected, 11, 11]

        return arr
    except Exception:
        # 返回零填充，形状需匹配 selected_bands
        C_out = len(selected_bands) if (selected_bands and len(selected_bands) < 64) else 64
        return np.zeros((C_out, 11, 11), dtype=np.float32)


def extract_landuse_sequence(landuse_str):
    if pd.isna(landuse_str) or not isinstance(landuse_str, str):
        return np.array([0], dtype=np.int64)
    landuse_list = [x.strip() for x in landuse_str.split('|') if x.strip()]
    if len(landuse_list) == 0:
        return np.array([0], dtype=np.int64)
    sequence = []
    for lu in landuse_list:
        token_id = LANDUSE_TYPES.index(lu) + 1 if lu in LANDUSE_TYPES else 0
        sequence.append(token_id)
    return np.array(sequence, dtype=np.int64)


def adaptive_temporal_downsample(seq_list, base_rate=5, max_timesteps=100,
                                 head_tail_reserved=50, middle_rate=2,
                                 strategy="head_tail_focus"):
    """
    自适应时序降采样：针对城市出行轨迹的首尾优先策略

    Parameters
    ----------
    seq_list : list or np.ndarray
        原始序列（patch字符串列表 或 token numpy数组）
    base_rate : int
        基础降采样率（如5表示100m→500m）
    max_timesteps : int
        最大允许的时间步数（触发首尾策略的阈值）
    head_tail_reserved : int
        首尾各保留的点数（在基础降采样后的序列上）
    middle_rate : int
        中间部分的额外降采样率（2表示再2:1抽稀）
    strategy : str
        "uniform": 均匀降采样到max_timesteps
        "head_tail_focus": 首尾精细+中间稀疏

    Returns
    -------
    downsampled : list or np.ndarray
        降采样后的序列（类型与输入一致）
    """
    if not seq_list or len(seq_list) == 0:
        return seq_list

    original_type = type(seq_list)

    # Step 1: 基础降采样（100m → 500m）
    if base_rate > 1:
        if isinstance(seq_list, np.ndarray):
            seq_base = seq_list[::base_rate]
        else:  # list
            seq_base = seq_list[::base_rate]
    else:
        seq_base = seq_list

    n = len(seq_base)

    # Step 2: 如果长度仍在阈值内，直接返回
    if n <= max_timesteps:
        return seq_base

    # Step 3: 应用策略
    if strategy == "head_tail_focus":
        half_reserved = head_tail_reserved

        # 边界情况：序列太短，首尾会重叠 → 退回均匀采样
        if 2 * half_reserved >= n:
            indices = np.linspace(0, n - 1, max_timesteps, dtype=int)
        else:
            # 头部索引 [0, half_reserved)
            head_indices = list(range(0, half_reserved))
            # 尾部索引 [n-half_reserved, n)
            tail_indices = list(range(n - half_reserved, n))

            # 中间部分 [half_reserved, n-half_reserved)
            middle_start = half_reserved
            middle_end = n - half_reserved
            middle_len = middle_end - middle_start

            if middle_len > 0 and middle_rate > 1:
                # 中间额外降采样
                middle_indices = list(range(middle_start, middle_end, middle_rate))
            else:
                middle_indices = list(range(middle_start, middle_end))

            # 合并所有索引（去重+排序）
            all_indices = head_indices + middle_indices + tail_indices
            all_indices = sorted(set(all_indices))

            # 最终保护：如果还是超长，最后均匀采样到max_timesteps
            if len(all_indices) > max_timesteps:
                final_indices = np.linspace(0, len(all_indices) - 1, max_timesteps, dtype=int)
                indices = [all_indices[i] for i in final_indices]
            else:
                indices = all_indices

        # 按索引提取
        if isinstance(seq_base, np.ndarray):
            return seq_base[indices]
        else:
            return [seq_base[i] for i in indices]

    else:  # uniform strategy: 简单均匀采样
        indices = np.linspace(0, n - 1, max_timesteps, dtype=int)
        if isinstance(seq_base, np.ndarray):
            return seq_base[indices]
        else:
            return [seq_base[i] for i in indices]


def analyze_band_importance(sample_patches: np.ndarray, method="variance", num_keep=16, random_state=42):
    """
    分析 64 个波段的重要性，返回选中的波段索引（按重要性降序）

    Parameters
    ----------
    sample_patches : np.ndarray
        形状 [N_samples, 64, 11, 11] 的样本数据
    method : str
        - "variance": 基于空间方差（信息量）选择，计算最快
        - "entropy": 基于直方图熵（信息丰富度）选择
        - "pca": 基于 PCA 载荷（线性贡献度）选择，最准确但较慢
        - "random": 随机选择（用于消融实验基线）
    num_keep : int
        保留的波段数量（建议 8~24）
    random_state : int
        随机种子（仅 random 方法使用）

    Returns
    -------
    selected_indices : list[int]
        选中的波段索引列表，按重要性从高到低排序
    band_scores : np.ndarray (64,)
        每个波段的得分（可用于可视化）
    """
    N, C, H, W = sample_patches.shape
    assert C == 64, f"Expected 64 bands, got {C}"

    # 展平空间维度，便于统计：[N, 64, 121]
    patches_flat = sample_patches.reshape(N, C, -1)

    if method == "variance":
        # 方案1: 空间方差（推荐起点）- 方差大 = 信息丰富
        band_var = np.var(patches_flat, axis=(0, 2))  # [64]
        scores = band_var

    elif method == "entropy":
        # 方案2: 直方图熵 - 分布越均匀，熵越大
        scores = []
        for c in range(C):
            values = patches_flat[:, c, :].flatten()
            # 归一化到 [0,1] 避免数值问题
            if values.max() > values.min():
                values_norm = (values - values.min()) / (values.max() - values.min() + 1e-10)
            else:
                values_norm = np.zeros_like(values)
            # 50  bins 直方图
            hist, _ = np.histogram(values_norm, bins=50, range=(0, 1), density=True)
            hist = hist[hist > 0]  # 移除零概率
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            scores.append(entropy)
        scores = np.array(scores)

    elif method == "pca":
        # 方案3: PCA 载荷绝对值和 - 在主要成分上权重大的波段更重要
        from sklearn.decomposition import PCA
        # 重塑为 [样本×空间位置, 波段]
        X = patches_flat.transpose(0, 2, 1).reshape(-1, C)  # [N*121, 64]
        # 拟合 PCA，保留 num_keep 个主成分
        pca = PCA(n_components=min(num_keep, C), random_state=random_state)
        pca.fit(X)
        # 计算每个原始波段在所有主成分上的载荷绝对值之和
        scores = np.sum(np.abs(pca.components_), axis=0)  # [64]

    elif method == "random":
        # 方案4: 随机选择（基线对照）
        rng = np.random.default_rng(random_state)
        scores = rng.random(C)

    else:
        raise ValueError(f"Unknown band selection method: {method}")

    # 选择得分最高的 num_keep 个波段（按得分降序）
    selected_indices = np.argsort(scores)[-num_keep:][::-1].tolist()

    return selected_indices, scores


def select_bands_by_gradient_sensitivity(patches: np.ndarray, labels: np.ndarray,
                                         mode="cumulative", num_keep=16,
                                         percentile_threshold=75, cumulative_threshold=0.90,
                                         num_epochs=3, batch_size=64, device='cuda'):
    """
    基于梯度敏感性的波段筛选（任务驱动）
    mode: "fixed_count" | "percentile" | "cumulative"
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    X = torch.from_numpy(patches).float()
    y = torch.from_numpy(labels).long()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    num_classes = len(np.unique(labels))

    class ProxyModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, num_classes))

        def forward(self, x):
            return self.fc(self.pool(x).view(x.size(0), -1))

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = ProxyModel(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"  训练代理模型 ({num_epochs} epochs)...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(loader):.4f}")

    print("  计算波段梯度重要性...")
    model.eval()
    channel_grad_norm = torch.zeros(64, device=device)

    with torch.enable_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_X.requires_grad = True
            out = model(batch_X)
            loss = criterion(out, batch_y.to(device))
            loss.backward()
            grad = batch_X.grad.detach().abs()
            channel_grad_norm += grad.mean(dim=[0, 2, 3])

    importance = channel_grad_norm.cpu().numpy()
    importance = importance / (importance.sum() + 1e-8)

    # 🔽 核心：排序并应用策略 🔽
    sorted_indices = np.argsort(importance)[::-1]
    sorted_scores = importance[sorted_indices]
    threshold_val = 0.0

    if mode == "fixed_count":
        selected_bands = sorted_indices[:num_keep].tolist()
        threshold_val = sorted_scores[num_keep - 1] if num_keep < len(sorted_scores) else 0
        print(f"  ✅ 策略: 固定数量 Top-{num_keep}")

    elif mode == "percentile":
        threshold_val = np.percentile(importance, percentile_threshold)
        selected_bands = np.where(importance >= threshold_val)[0].tolist()
        selected_bands = sorted(selected_bands, key=lambda i: importance[i], reverse=True)
        print(f"  ✅ 策略: 百分位阈值 {percentile_threshold}%")

    elif mode == "cumulative":
        # 计算累积贡献率
        cumsum = np.cumsum(sorted_scores)
        # 找到第一个 >= cumulative_threshold 的索引
        cutoff_idx = np.argmax(cumsum >= cumulative_threshold) + 1
        selected_bands = sorted_indices[:cutoff_idx].tolist()
        threshold_val = sorted_scores[cutoff_idx - 1]
        print(f"  ✅ 策略: 累积贡献率 {cumulative_threshold * 100:.0f}% → 自动选中 {len(selected_bands)} 个波段")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return selected_bands, importance, threshold_val

# ================= 交通特征：8个全局统计量 =================
def extract_traffic_statistics(road_speeds_str: str) -> np.ndarray:
    """从速度序列提取8个全局统计量，表征路网结构特征"""
    if not isinstance(road_speeds_str, str) or not road_speeds_str.strip():
        return np.zeros(8, dtype=np.float32)

    try:
        speeds = np.array([float(x) for x in road_speeds_str.split('|') if x.strip()], dtype=np.float32)
        if len(speeds) == 0:
            return np.zeros(8, dtype=np.float32)

        # 8个统计量：均值、标准差、最大值、最小值、变异系数、高速比例(>40kph)、25%分位数、75%分位数
        stats = [
            np.mean(speeds),
            np.std(speeds) if len(speeds) > 1 else 0.0,
            np.max(speeds),
            np.min(speeds),
            (np.std(speeds) / np.mean(speeds)) if np.mean(speeds) > 1e-5 else 0.0,  # CV
            np.mean(speeds > 40.0),  # 高速比例
            np.percentile(speeds, 25),
            np.percentile(speeds, 75)
        ]
        return np.array(stats, dtype=np.float32)
    except:
        return np.zeros(8, dtype=np.float32)


# ================= Lazy Load Functions (只加载元数据，不解压) =================
def load_env_index(directory: str) -> pd.DataFrame:
    """只加载 seg_id 和 b64 字符串，不解压 patch"""
    print(f"Loading Environment index from: {directory}")
    files = glob.glob(os.path.join(directory, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {directory}")
    dfs = []
    for f in tqdm(files, desc="Reading Parquet (index only)"):
        try:
            df = pd.read_parquet(f, columns=['seg_id', 'ae_patch_11x11x64_b64'])
            df['seg_id'] = df['seg_id'].astype(str)

            def is_valid_patch(b64_str) -> bool:
                if not isinstance(b64_str, str) or b64_str.strip() == '':
                    return False
                patch_strs = [x.strip() for x in b64_str.split('|') if x.strip()]
                if not patch_strs:
                    return False
                try:
                    for p in patch_strs:
                        arr = decompress_patch(p)
                        if np.any(arr != 0):
                            return True
                    return False
                except:
                    return False

            mask = df['ae_patch_11x11x64_b64'].apply(is_valid_patch)
            valid_df = df[mask]
            if len(df) - len(valid_df) > 0:
                print(f"  {Path(f).name}: filtered {len(df) - len(valid_df)} invalid patches")
            dfs.append(valid_df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=['seg_id']).reset_index(drop=True)
    print(f"Environment index loaded: {len(full_df)} valid segments")
    return full_df  # 列: seg_id, ae_patch_11x11x64_b64 (原始字符串)


def load_landuse_index(directory: str) -> pd.DataFrame:
    """只加载 seg_id 和 nearest_landuse 字符串"""
    print(f"Loading Landuse index from: {directory}")
    files = glob.glob(os.path.join(directory, "*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {directory}")
    dfs = []
    for f in tqdm(files, desc="Reading Excel (index only)"):
        try:
            df = pd.read_excel(f, usecols=['seg_id', 'nearest_landuse'])
            df['seg_id'] = df['seg_id'].astype(str)

            def is_valid_landuse(s):
                if not isinstance(s, str) or s.strip() == '':
                    return False
                lst = [x.strip() for x in s.split('|') if x.strip()]
                return any(x in LANDUSE_TYPES for x in lst)

            mask = df['nearest_landuse'].apply(is_valid_landuse)
            valid_df = df[mask]
            if len(df) - len(valid_df) > 0:
                print(f"  {Path(f).name}: filtered {len(df) - len(valid_df)} invalid landuse")
            dfs.append(valid_df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=['seg_id']).reset_index(drop=True)
    print(f"Landuse index loaded: {len(full_df)} valid segments")
    return full_df


# 新增：加载交通特征索引（只读 road_speeds_kph）
def load_traffic_index(directory: str) -> pd.DataFrame:
    """加载新格式的交通特征（仅 road_speeds_kph）"""
    print(f"Loading Traffic index from: {directory}")
    files = glob.glob(os.path.join(directory, "*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {directory}")
    dfs = []
    for f in tqdm(files, desc="Reading Excel (traffic)"):
        try:
            # 只读取 seg_id 和 road_speeds_kph
            df = pd.read_excel(f, usecols=['seg_id', 'road_speeds_kph'])
            df['seg_id'] = df['seg_id'].astype(str)

            # 过滤：确保 road_speeds_kph 非空且为字符串
            def is_valid_row(row):
                speed_str = row.get('road_speeds_kph')
                return isinstance(speed_str, str) and '|' in speed_str

            mask = df.apply(is_valid_row, axis=1)
            valid_df = df[mask]
            if len(df) - len(valid_df) > 0:
                print(f"  {Path(f).name}: filtered {len(df) - len(valid_df)} invalid rows")
            if not valid_df.empty:
                dfs.append(valid_df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    if not dfs:
        raise ValueError(f"No valid traffic data found in {directory}. "
                         "Ensure files contain 'seg_id' and 'road_speeds_kph' columns.")
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=['seg_id']).reset_index(drop=True)
    print(f"Traffic index loaded: {len(full_df)} valid segments")
    return full_df


def load_center_data_full(directory: str) -> pd.DataFrame:
    """城市中心数据较小，可全加载并标准化"""
    print(f"Loading Center features from: {directory}")
    files = glob.glob(os.path.join(directory, "*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {directory}")
    dfs = []
    for f in tqdm(files, desc="Reading Excel"):
        try:
            df = pd.read_excel(f)
            required = ['seg_id'] + CENTER_FEATURES
            if not all(c in df.columns for c in required):
                missing = [c for c in required if c not in df.columns]
                print(f"Skipping {f}: missing {missing}")
                continue
            df = df[required].copy()
            df['seg_id'] = df['seg_id'].astype(str)
            df = df.dropna(subset=CENTER_FEATURES)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=['seg_id']).reset_index(drop=True)
    scaler = StandardScaler()
    full_df[CENTER_FEATURES] = scaler.fit_transform(full_df[CENTER_FEATURES])
    print(f"Center data loaded and standardized: {len(full_df)} segments")
    # 保存 scaler 参数到 JSON
    import json
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'features': CENTER_FEATURES
    }
    with open("center_scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=2)
    print("✓ Saved center feature scaler parameters to center_scaler_params.json")
    return full_df


def load_and_clean_traj_data(directory: str) -> pd.DataFrame:
    # 与原代码完全一致（必须全加载用于聚类）
    print(f"Loading Trajectory features from: {directory}")
    files = sorted(glob.glob(os.path.join(directory, "*.xlsx")))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {directory}")
    dfs = []
    for f in tqdm(files, desc="Reading Excel"):
        try:
            df = pd.read_excel(f, engine='openpyxl')
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    df['seg_id'] = df['seg_id'].astype(str)
    print(f"Original total segments: {len(df)}")
    if 'type' in df.columns:
        before = len(df)
        df = df[df['type'] != 'stay'].copy()
        print(f"Removed 'stay' segments: {before - len(df)}, remaining: {len(df)}")
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    valid = (
            (df.get('adv_speed_95', 0) <= 35) &
            (df.get('adv_speed_95', 0) >= 0.3) &
            (df.get('adv_acc_95', 0) <= 20) &
            (df.get('adv_stop_ratio', 0) <= 0.9) &
            (df.get('adv_heading_change_rate', 0) <= 200)
    )
    df = df[valid].copy()
    print(f"After filtering extreme values: {len(df)}")
    if 'type' in df.columns:
        df = df[df['type'] != 'stay']
    df = df.reset_index(drop=True)
    return df


def perform_advanced_clustering(df: pd.DataFrame):
    """
    简化版聚类：使用 KMeans 直接聚为 4 类，基于高区分度特征。
    """
    print("\n执行简化版 KMeans 聚类（固定 K=4）...")
    # ==================== 定义特征列 ====================
    advanced_features = [
        'adv_speed_95',
        'adv_speed_99',
        'adv_high_speed_ratio_10',
        'adv_high_speed_ratio_8'
    ]
    base_features = ['avg_speed_mps', 'avg_accel_mps2']
    feature_cols = advanced_features  # 可根据需要加入 base_features
    print(f"使用特征数量: {len(feature_cols)}")
    print("特征列表:", feature_cols)
    # ==================== 数据预处理 ====================
    print(f"原始总段数: {len(df)}")
    # 移除 'stay' 段
    if 'type' in df.columns:
        before = len(df)
        df = df[df['type'] != 'stay'].copy()
        print(f"移除停留段: {before - len(df)} 条，剩余 {len(df)}")
    # 过滤极端值（硬阈值）
    valid_conditions = []
    if 'adv_speed_95' in df.columns:
        valid_conditions.append((df['adv_speed_95'] >= 0.3) & (df['adv_speed_95'] <= 35))
    if 'adv_acc_95' in df.columns:
        valid_conditions.append(df['adv_acc_95'] <= 20)
    if 'adv_stop_ratio' in df.columns:
        valid_conditions.append(df['adv_stop_ratio'] <= 0.9)
    if 'adv_heading_change_rate' in df.columns:
        valid_conditions.append(df['adv_heading_change_rate'] <= 200)
    if valid_conditions:
        valid = valid_conditions[0]
        for cond in valid_conditions[1:]:
            valid &= cond
        df = df[valid].copy()
        print(f"移除极端异常段后剩余: {len(df)} 条")
    df = df.reset_index(drop=True)
    print(f"最终用于聚类的有效运动段: {len(df)} 条")
    print("-" * 60)
    # ==================== 特征提取与标准化 ====================
    X_raw = df[feature_cols].copy()
    X_raw.fillna(X_raw.median(numeric_only=True), inplace=True)
    X_raw.fillna(0, inplace=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    print(f"标准化后特征维度: {X.shape}")
    # 保存标准化特征
    X_df_scaled = pd.DataFrame(X, columns=feature_cols, index=df.index)
    X_df_scaled['seg_id'] = df['seg_id']
    X_df_scaled.to_csv("trajectory_features_scaled_kmeans.csv", index=False)
    # ==================== KMeans 聚类（K=4） ====================
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    # 计算轮廓系数
    sil_score = silhouette_score(X, labels)
    print(f"KMeans 聚类完成（K={k}），轮廓系数（Silhouette Score）: {sil_score:.4f}")
    # ==================== 保存结果 ====================
    df['label'] = labels
    df.to_csv("clustering_results_kmeans.csv", index=False)
    print("聚类结果已保存到 clustering_results_kmeans.csv")
    # ==================== 生成聚类报告 ====================
    cluster_stats_mean = df.groupby('label')[feature_cols].mean()
    cluster_stats_std = df.groupby('label')[feature_cols].std()
    cluster_counts = df['label'].value_counts().sort_index()
    result_df = pd.DataFrame(index=cluster_stats_mean.index)
    for col in feature_cols:
        result_df[col] = (
                cluster_stats_mean[col].round(4).astype(str) +
                " ± " +
                cluster_stats_std[col].round(4).astype(str)
        )
    result_df["Count"] = cluster_counts.values
    print(result_df.T)
    result_df.T.to_excel("cluster_comparison_kmeans.xlsx")
    print(f"\n聚类完成！共 {k} 个簇")
    print(f"各簇样本数分布: {dict(df['label'].value_counts().sort_index())}")
    return df, k


def perform_advanced_clustering_v5(df: pd.DataFrame):
    """Create leakage-controlled, speed-ordered five-class proxy labels."""
    feature_cols = [
        'adv_speed_95',
        'adv_speed_99',
        'adv_high_speed_ratio_10',
        'adv_high_speed_ratio_8',
    ]
    if 'type' in df.columns:
        df = df[df['type'] != 'stay'].copy()
    valid = (
        df['adv_speed_95'].between(0.3, 35)
        & (df.get('adv_acc_95', 0) <= 20)
        & (df.get('adv_stop_ratio', 0) <= 0.9)
        & (df.get('adv_heading_change_rate', 0) <= 200)
    )
    df = df.loc[valid].reset_index(drop=True)

    group_col = 'gpx_id' if 'gpx_id' in df.columns else 'seg_id'
    groups = df[group_col].fillna(df['seg_id']).astype(str).to_numpy()
    indices = np.arange(len(df))
    outer = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=RANDOM_STATE)
    train_idx, remainder_idx = next(outer.split(indices, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=RANDOM_STATE + 1)
    val_rel, test_rel = next(inner.split(remainder_idx, groups=groups[remainder_idx]))
    val_idx, test_idx = remainder_idx[val_rel], remainder_idx[test_rel]
    split = np.full(len(df), 'test', dtype=object)
    split[train_idx] = 'train'
    split[val_idx] = 'val'
    df['split'] = split

    X_raw = df[feature_cols].copy()
    train_medians = X_raw.iloc[train_idx].median(numeric_only=True)
    X_raw = X_raw.fillna(train_medians).fillna(0)
    scaler = StandardScaler().fit(X_raw.iloc[train_idx])
    X = scaler.transform(X_raw)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
    kmeans.fit(X[train_idx])
    raw_labels = kmeans.predict(X)
    speed_order = (
        df.iloc[train_idx].assign(_raw_label=raw_labels[train_idx])
        .groupby('_raw_label')['adv_speed_95'].mean().sort_values().index
    )
    raw_to_ordered = {int(raw): ordered for ordered, raw in enumerate(speed_order)}
    labels = np.asarray([raw_to_ordered[int(label)] for label in raw_labels], dtype=int)
    df['label'] = labels

    rng = np.random.default_rng(RANDOM_STATE)
    sil_idx = rng.choice(train_idx, size=min(20000, len(train_idx)), replace=False)
    sil_score = silhouette_score(X[sil_idx], labels[sil_idx])
    print(f"K=5 training-group silhouette score: {sil_score:.4f}")
    print(pd.crosstab(df['label'], df['split']))

    joblib.dump({
        'scaler': scaler,
        'kmeans': kmeans,
        'raw_to_ordered': raw_to_ordered,
        'feature_cols': feature_cols,
        'class_names': CLASS_NAMES,
    }, 'trajectory_cluster_model_v5.joblib')
    with open('trajectory_cluster_metadata_v5.json', 'w', encoding='utf-8') as handle:
        json.dump({
            'n_clusters': N_CLUSTERS,
            'silhouette_train': float(sil_score),
            'feature_cols': feature_cols,
            'raw_to_ordered': raw_to_ordered,
            'class_names': CLASS_NAMES,
            'split_group': group_col,
        }, handle, indent=2)

    profile_cols = ['avg_speed_mps', 'adv_speed_95', 'adv_speed_99',
                    'adv_high_speed_ratio_10', 'adv_high_speed_ratio_8',
                    'adv_stop_ratio', 'adv_acc_95', 'adv_speed_std']
    profile_cols = [column for column in profile_cols if column in df.columns]
    df.groupby('label')[profile_cols].agg(['count', 'mean', 'std', 'median']).to_excel(
        'cluster_comparison_kmeans_v5.xlsx'
    )
    df.to_csv('clustering_results_kmeans_v5.csv', index=False)
    return df, N_CLUSTERS


# ================= Lazy Dataset =================
class LazyMultiModalTrajectoryDataset(Dataset):
    def __init__(self, traj_df_with_labels, env_index_df=None, landuse_index_df=None, traffic_index_df=None,
                 center_df=None,
                 use_env=True, use_landuse=True, use_traffic=True, use_center=True, selected_bands: list = None):
        """
        traj_df_with_labels: 必须包含 seg_id 和 label（聚类后）
        其他 df 只包含 seg_id + 原始字段（字符串或数值），不包含 tensor
        """
        self.use_env = use_env
        self.use_landuse = use_landuse
        self.use_traffic = use_traffic  # 注意：现在 use_traffic 表示使用统计量
        self.use_center = use_center
        self.selected_bands = selected_bands
        # 构建样本列表（只存 seg_id 和 label）
        metadata_cols = [column for column in ['seg_id', 'label', 'split', 'gpx_id']
                         if column in traj_df_with_labels.columns]
        merged = traj_df_with_labels[metadata_cols].copy()
        if use_env and env_index_df is not None:
            merged = pd.merge(merged, env_index_df, on='seg_id', how='inner')
        if use_landuse and landuse_index_df is not None:
            merged = pd.merge(merged, landuse_index_df, on='seg_id', how='inner')
        if use_traffic and traffic_index_df is not None:
            merged = pd.merge(merged, traffic_index_df, on='seg_id', how='inner')
        if use_center and center_df is not None:
            merged = pd.merge(merged, center_df, on='seg_id', how='inner')
        print(f"Final matched samples for training: {len(merged)}")
        self.samples = merged.to_dict('records')  # 列表 of dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        seg_id = row['seg_id']
        label = int(row['label'])
        sample = {'seg_id': seg_id, 'label': label}
        # 环境 patch：按需解压
        if self.use_env:
            b64_str = row['ae_patch_11x11x64_b64']
            patch_strs = [p.strip() for p in b64_str.split('|') if p.strip()]

            # 🔽 自适应降采样 🔽
            if patch_strs:
                patch_strs = adaptive_temporal_downsample(
                    patch_strs,
                    base_rate=TEMPORAL_DOWNSAMPLE_BASE,
                    max_timesteps=MAX_TOTAL_TIMESTEPS,
                    head_tail_reserved=HEAD_TAIL_RESERVED,
                    middle_rate=MIDDLE_DOWNSAMPLE_RATE,
                    strategy=STRATEGY
                )

            # 解压为tensor（带空序列保护）
            patches = [decompress_patch(p, selected_bands=self.selected_bands) for p in patch_strs] if patch_strs else []
            if patches:
                sample['input_img'] = torch.from_numpy(np.stack(patches)).float()  # [T', 64, 11, 11]
            else:
                # 兜底：单个零patch，避免shape错误
                C_out = len(self.selected_bands) if (self.selected_bands and len(self.selected_bands) < 64) else 64
                sample['input_img'] = torch.zeros((1, C_out, 11, 11), dtype=torch.float32)
        # 土地利用：按需 tokenization
        if self.use_landuse:
            landuse_str = row['nearest_landuse']
            token_seq = extract_landuse_sequence(landuse_str)  # np.ndarray, dtype=int64

            if len(token_seq) > 0:
                token_seq = adaptive_temporal_downsample(
                    token_seq,
                    base_rate=TEMPORAL_DOWNSAMPLE_BASE,
                    max_timesteps=MAX_TOTAL_TIMESTEPS,
                    head_tail_reserved=HEAD_TAIL_RESERVED,
                    middle_rate=MIDDLE_DOWNSAMPLE_RATE,
                    strategy=STRATEGY
                )

            sample['input_land'] = torch.from_numpy(token_seq).long()  # [T']
        # ================= 核心修改2：返回交通统计量而非序列 =================
        if self.use_traffic:
            traffic_stats = extract_traffic_statistics(row['road_speeds_kph'])  # 8个统计量
            sample['input_traffic_stats'] = torch.from_numpy(traffic_stats).float()  # 字段名变更

        # 城市中心：已经是标准化数值，直接转 tensor
        if self.use_center:
            center_vals = np.array([row[c] for c in CENTER_FEATURES], dtype=np.float32)
            sample['input_center'] = torch.from_numpy(center_vals)

        return sample


# ================= collate_fn 与 my_collate_fn =================
def collate_fn(batch, use_env=True, use_landuse=True, use_traffic=True, use_center=True):
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    seg_ids = [item['seg_id'] for item in batch]
    inputs_img = None
    inputs_land = None
    inputs_traffic_stats = None  # 字段名变更
    inputs_center = None
    lengths = None

    if use_env:
        img_seqs = [item['input_img'] for item in batch if 'input_img' in item]
        if img_seqs:  # 防止空列表
            inputs_img = pad_sequence(img_seqs, batch_first=True, padding_value=0)
            lengths = torch.tensor([len(seq) for seq in img_seqs])

    if use_landuse:
        land_seqs = [item['input_land'] for item in batch if 'input_land' in item]
        if land_seqs:
            inputs_land = pad_sequence(land_seqs, batch_first=True, padding_value=0)

    # ================= 核心修改4：处理统计量（非序列，无需 pad_sequence） =================
    if use_traffic:
        traffic_stats_list = [item['input_traffic_stats'] for item in batch if 'input_traffic_stats' in item]
        if traffic_stats_list:
            inputs_traffic_stats = torch.stack(traffic_stats_list)  # [B, 8]

    if use_center:
        center_vecs = [item['input_center'] for item in batch if 'input_center' in item]
        if center_vecs:
            inputs_center = torch.stack(center_vecs)

    return {
        'input_img': inputs_img,
        'input_land': inputs_land,
        'input_traffic_stats': inputs_traffic_stats,  # 字段名变更
        'input_center': inputs_center,
        'label': labels,
        'lengths': lengths,
        'seg_ids': seg_ids
    }


def my_collate_fn(batch, use_env=True, use_landuse=True, use_traffic=True, use_center=True):
    return collate_fn(batch, use_env, use_landuse, use_traffic, use_center)


# ================= 模型定义 =================
class AttentionPool(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.attn = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, x):
        # x: [B*T, C, H, W]
        B_T, C, H, W = x.shape
        # 计算注意力权重
        weights = torch.sigmoid(self.attn(x))  # [B*T, 1, H, W]
        # 加权平均
        weighted = x * weights  # [B*T, C, H, W]
        pooled = weighted.mean(dim=[2, 3])  # [B*T, C]
        return self.proj(pooled)  # [B*T, embed_dim]


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=120):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}. "
                             f"Increase max_len in LearnablePositionalEmbedding.")
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pos_embed(positions)


class SimpleResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, channels)  # 替换 BN → GN
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(8, channels)
        self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out


class PatchEncoder(nn.Module):
    def __init__(self, input_channels=64, embed_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, 64)
        self.res_block = SimpleResidualBlock(64)
        self.pool = nn.MaxPool2d(2)
        self.attn_pool = AttentionPool(64, embed_dim)  # 替换 GlobalAvgPool

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.res_block(x)
        x = self.pool(x)  # [B*T, 64, 4, 4]
        x = self.attn_pool(x)  # [B*T, embed_dim]
        return x


class TokenEmbedder(nn.Module):
    def __init__(self, num_tokens=5, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens + 1, embed_dim, padding_idx=0)

    def forward(self, x):
        return self.embedding(x)


# ================= 模型 =================
class UrbanMultiModalClassifier(nn.Module):
    def __init__(self, num_classes=5, input_channels=64, embed_dim=128, nhead=8, num_layers=4,
                 use_env=True, use_landuse=True, use_traffic=True, use_center=True):
        super().__init__()
        self.use_env = use_env
        self.use_landuse = use_landuse
        self.use_traffic = use_traffic  # 现在表示使用统计量
        self.use_center = use_center
        self.embed_dim = embed_dim

        # 影像编码器
        if self.use_env:
            self.patch_encoder = PatchEncoder(INPUT_CHANNELS, embed_dim)
            self.self_attn_img = nn.MultiheadAttention(embed_dim, num_heads=nhead // 2, batch_first=True)

        # 土地利用（保留但未启用）
        if self.use_landuse:
            self.token_embedder = TokenEmbedder(num_tokens=5, embed_dim=embed_dim)

        # ================= 核心修改6：新增交通统计量编码器（与center_encoder结构完全一致） =================
        if self.use_traffic:
            self.traffic_stats_encoder = nn.Sequential(
                nn.Linear(8, embed_dim),  # 8个统计量
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )

        # 城市中心特征编码器
        if self.use_center:
            self.center_encoder = nn.Sequential(
                nn.Linear(len(CENTER_FEATURES), embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )

        # 归一化与位置编码
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)  # 正态初始化
        self.pos_encoder = LearnablePositionalEmbedding(embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim)

        # 融合投影层（处理拼接后维度翻倍）
        fusion_in_dim = embed_dim
        if use_env:
            fusion_in_dim = embed_dim  # 影像作为基础
            if use_center:
                fusion_in_dim += embed_dim  # + center
            if use_traffic:
                fusion_in_dim += embed_dim  # + traffic stats

        if fusion_in_dim != embed_dim:
            self.modality_fusion_proj = nn.Linear(fusion_in_dim, embed_dim)
        else:
            self.modality_fusion_proj = None

        # Transformer 与分类头
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
        # 确定 batch size
        B = 0
        if x_land is not None:
            B = x_land.size(0)
        elif x_traffic_stats is not None:
            B = x_traffic_stats.size(0)
        elif x_img is not None:
            B = x_img.size(0)
        elif x_center is not None:
            B = x_center.size(0)
        else:
            raise ValueError("At least one input modality must be provided.")

        # === 编码各模态 ===
        embeddings_img = None
        if self.use_env and x_img is not None:
            T = x_img.shape[1]
            x_flat = x_img.view(B * T, x_img.shape[2], x_img.shape[3], x_img.shape[4])
            embeddings_img = self.patch_encoder(x_flat)
            embeddings_img = embeddings_img.view(B, T, -1)
            embeddings_img = self.embed_norm(embeddings_img)

        embeddings_land = None
        if self.use_landuse and x_land is not None:
            embeddings_land = self.token_embedder(x_land)
            embeddings_land = self.embed_norm(embeddings_land)

        # ================= 核心修改7：交通统计量编码（全局特征） =================
        traffic_emb = None
        if self.use_traffic and x_traffic_stats is not None:
            traffic_emb = self.traffic_stats_encoder(x_traffic_stats)  # [B, embed_dim]
            traffic_emb = traffic_emb.unsqueeze(1)  # [B, 1, D]
            traffic_emb = self.embed_norm(traffic_emb)

        # 城市中心特征编码（全局特征）
        center_emb = None
        if self.use_center and x_center is not None:
            center_emb = self.center_encoder(x_center).unsqueeze(1)  # [B, 1, D]
            center_emb = self.embed_norm(center_emb)

        # === 多模态融合：统一构建 [B, T, embed_dim] 序列 ===
        combined_list = []
        T_final = 0

        # 情况 1: 有影像（主导时间轴）
        if self.use_env and embeddings_img is not None:
            T_final = embeddings_img.shape[1]
            combined_list.append(embeddings_img)

            # 扩展全局特征到时间步（与城市中心处理方式完全一致）
            if traffic_emb is not None:
                combined_list.append(traffic_emb.expand(B, T_final, -1))  # [B, T, D]
            if center_emb is not None:
                combined_list.append(center_emb.expand(B, T_final, -1))  # [B, T, D]

        # 情况 2: 无影像，但有其他全局特征 → 构造 T=1
        elif traffic_emb is not None or center_emb is not None:
            T_final = 1
            if traffic_emb is not None:
                combined_list.append(traffic_emb)  # [B, 1, D]
            if center_emb is not None:
                combined_list.append(center_emb)  # [B, 1, D]

        # 情况 3: 兜底（理论上不应触发）
        else:
            T_final = 1
            combined_list.append(torch.zeros(B, 1, self.embed_dim, device=self.cls_token.device))

        # 拼接所有模态
        if len(combined_list) == 1:
            combined = combined_list[0]  # [B, T, D]
        else:
            combined = torch.cat(combined_list, dim=-1)  # [B, T, D * num_modalities]

        # 投影回 embed_dim（如果需要）
        if self.modality_fusion_proj is not None:
            combined = self.modality_fusion_proj(combined)  # [B, T, embed_dim]

        # 确保 final shape: [B, T, embed_dim]
        assert combined.shape[-1] == self.embed_dim, \
            f"Combined embedding dim {combined.shape[-1]} ≠ expected {self.embed_dim}"

        # Final LayerNorm
        combined = self.final_norm(combined)

        # === 添加 CLS token ===
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        embeddings = torch.cat([cls_tokens, combined], dim=1)  # [B, 1+T, D]

        # 位置编码
        embeddings = self.pos_encoder(embeddings)

        # === Padding mask ===
        total_seq_len = embeddings.shape[1] - 1  # exclude CLS
        if total_seq_len > 0 and lengths is not None:
            pad_mask = torch.arange(total_seq_len, device=embeddings.device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            pad_mask = torch.zeros(B, total_seq_len, dtype=torch.bool, device=embeddings.device)

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=embeddings.device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)

        # Transformer 编码
        encoded = self.transformer(embeddings, src_key_padding_mask=full_mask)
        cls_out = encoded[:, 0, :]  # [B, D]
        logits = self.classifier(cls_out)
        return logits, cls_out


class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        log_softmax = F.log_softmax(pred, dim=-1)
        ce_loss = F.nll_loss(log_softmax, target, reduction='none')
        pred_prob = torch.exp(log_softmax)
        target_prob = pred_prob.gather(1, target.unsqueeze(1)).squeeze()
        focal_weight = (1 - target_prob) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def co_teach_train_epoch(model_a, model_b, dataloader, criterion, optimizer_a, optimizer_b,
                         device, epoch, max_epochs, start_noise=0.3, end_noise=0.1):
    model_a.train()
    model_b.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    # 动态 noise rate
    noise_rate = start_noise - (start_noise - end_noise) * (epoch / max_epochs)
    noise_rate = max(end_noise, noise_rate)  # 不低于 end_noise

    for batch in dataloader:
        inputs_img = batch['input_img'].to(device) if batch['input_img'] is not None else None
        inputs_land = batch['input_land'].to(device) if batch['input_land'] is not None else None
        # ================= 核心修改8：使用 input_traffic_stats =================
        inputs_traffic_stats = batch['input_traffic_stats'].to(device) if batch[
                                                                              'input_traffic_stats'] is not None else None
        inputs_center = batch['input_center'].to(device) if batch['input_center'] is not None else None
        labels = batch['label'].to(device)
        lengths = batch['lengths'].to(device)

        logits_a, _ = model_a(inputs_img, inputs_land, inputs_traffic_stats, inputs_center, lengths)
        logits_b, _ = model_b(inputs_img, inputs_land, inputs_traffic_stats, inputs_center, lengths)

        # per-sample loss (same as before)
        log_softmax_a = F.log_softmax(logits_a, dim=-1)
        ce_loss_a = F.nll_loss(log_softmax_a, labels, reduction='none')
        pred_prob_a = torch.exp(log_softmax_a)
        target_prob_a = pred_prob_a.gather(1, labels.unsqueeze(1)).squeeze()
        focal_weight_a = (1 - target_prob_a) ** criterion.gamma
        loss_a_per_sample = focal_weight_a * ce_loss_a

        log_softmax_b = F.log_softmax(logits_b, dim=-1)
        ce_loss_b = F.nll_loss(log_softmax_b, labels, reduction='none')
        pred_prob_b = torch.exp(log_softmax_b)
        target_prob_b = pred_prob_b.gather(1, labels.unsqueeze(1)).squeeze()
        focal_weight_b = (1 - target_prob_b) ** criterion.gamma
        loss_b_per_sample = focal_weight_b * ce_loss_b

        _, sorted_idx_a = loss_a_per_sample.sort()
        _, sorted_idx_b = loss_b_per_sample.sort()
        keep_num = int(len(loss_a_per_sample) * (1 - noise_rate))
        if keep_num == 0:
            keep_num = 1

        small_loss_idx_a = sorted_idx_a[:keep_num]
        small_loss_idx_b = sorted_idx_b[:keep_num]

        optimizer_b.zero_grad()
        loss_for_b = loss_b_per_sample[small_loss_idx_a].mean()
        loss_for_b.backward()
        torch.nn.utils.clip_grad_norm_(model_b.parameters(), max_norm=1.0)
        optimizer_b.step()

        optimizer_a.zero_grad()
        loss_for_a = loss_a_per_sample[small_loss_idx_b].mean()
        loss_for_a.backward()
        torch.nn.utils.clip_grad_norm_(model_a.parameters(), max_norm=1.0)
        optimizer_a.step()

        pred_a = logits_a.argmax(dim=1)
        correct += (pred_a == labels).sum().item()
        epoch_loss += loss_for_a.item()
        total += labels.size(0)

    return epoch_loss / len(dataloader), 100.0 * correct / total, noise_rate  # 返回 noise_rate 用于日志


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs_img = batch['input_img'].to(device) if batch['input_img'] is not None else None
            inputs_land = batch['input_land'].to(device) if batch['input_land'] is not None else None
            # ================= 核心修改9：使用 input_traffic_stats =================
            inputs_traffic_stats = batch['input_traffic_stats'].to(device) if batch[
                                                                                  'input_traffic_stats'] is not None else None
            inputs_center = batch['input_center'].to(device) if batch['input_center'] is not None else None
            labels = batch['label'].to(device)
            lengths = batch['lengths'].to(device)

            logits, _ = model(inputs_img, inputs_land, inputs_traffic_stats, inputs_center, lengths)
            loss = criterion(logits, labels)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            epoch_loss += loss.item()
            total += labels.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss / len(dataloader), 100.0 * correct / total, f1


def train_and_evaluate(env_index, landuse_index, traffic_index, center_data, traj_with_labels, best_k,
                       use_env=True, use_landuse=True, use_traffic=True, use_center=True):
    BATCH_SIZE = int(os.environ.get('MODE5_BATCH_SIZE', 48))
    LR = float(os.environ.get('MODE5_LR', 3e-4))
    MAX_EPOCHS = int(os.environ.get('MODE5_MAX_EPOCHS', 80))
    NUM_CLASSES = best_k
    PATIENCE = int(os.environ.get('MODE5_PATIENCE', 12))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full_dataset = LazyMultiModalTrajectoryDataset(
        traj_df_with_labels=traj_with_labels,
        env_index_df=env_index if use_env else None,
        landuse_index_df=landuse_index if use_landuse else None,
        traffic_index_df=traffic_index if use_traffic else None,
        center_df=center_data if use_center else None,
        use_env=use_env, use_landuse=use_landuse, use_traffic=use_traffic, use_center=use_center,
        selected_bands = SELECTED_BANDS if USE_BAND_SELECTION else None
    )

    all_labels = [item['label'] for item in full_dataset.samples]
    all_indices = list(range(len(full_dataset)))
    if not all('split' in item for item in full_dataset.samples):
        raise ValueError('Grouped train/val/test assignments are missing from the matched samples.')
    train_idx = [i for i, item in enumerate(full_dataset.samples) if item['split'] == 'train']
    val_idx = [i for i, item in enumerate(full_dataset.samples) if item['split'] == 'val']
    test_idx = [i for i, item in enumerate(full_dataset.samples) if item['split'] == 'test']

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    test_ds = Subset(full_dataset, test_idx)

    from collections import Counter
    print("\n类别分布（分层后）：")
    print("Train:", Counter([all_labels[i] for i in train_idx]))
    print("Val:  ", Counter([all_labels[i] for i in val_idx]))
    print("Test: ", Counter([all_labels[i] for i in test_idx]))

    class_counts = np.bincount(all_labels)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[[all_labels[i] for i in train_idx]]
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=lambda b: my_collate_fn(
            b,
            use_env=use_env,
            use_landuse=use_landuse,
            use_traffic=use_traffic,
            use_center=use_center
        ),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # val_loader 和 test_loader 同理
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: my_collate_fn(
            b,
            use_env=use_env,
            use_landuse=use_landuse,
            use_traffic=use_traffic,
            use_center=use_center
        ),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: my_collate_fn(
            b,
            use_env=use_env,
            use_landuse=use_landuse,
            use_traffic=use_traffic,
            use_center=use_center
        ),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Create two models for Co-Teaching
    model_a = UrbanMultiModalClassifier(num_classes=NUM_CLASSES,
                                        use_env=use_env, use_landuse=use_landuse,
                                        use_traffic=use_traffic, use_center=use_center).to(device)
    model_b = UrbanMultiModalClassifier(num_classes=NUM_CLASSES,
                                        use_env=use_env, use_landuse=use_landuse,
                                        use_traffic=use_traffic, use_center=use_center).to(device)

    optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=LR, weight_decay=0.01)
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=LR, weight_decay=0.01)
    criterion = ImprovedFocalLoss(gamma=2.0)  # Using Focal Loss as base

    scheduler_a = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_a, mode='min', factor=0.5, patience=5,
                                                             verbose=True)
    scheduler_b = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_b, mode='min', factor=0.5, patience=5,
                                                             verbose=True)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_val_f1 = -np.inf
    patience_counter = 0

    print("\n" + "=" * 60)
    print("START CO-TEACHING TRAINING (with dynamic noise rate)")
    print("=" * 60)

    for epoch in range(1, MAX_EPOCHS + 1):
        # 传入 epoch 和 max_epochs
        train_loss, train_acc, current_noise = co_teach_train_epoch(
            model_a, model_b, train_loader, criterion, optimizer_a, optimizer_b,
            device, epoch, MAX_EPOCHS, start_noise=0.3, end_noise=0.1
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        val_loss, val_acc, val_f1 = evaluate(model_a, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        scheduler_a.step(val_loss)
        scheduler_b.step(val_loss)

        current_lr = optimizer_a.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:6.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:6.2f}% F1: {val_f1:.4f} | "
              f"LR: {current_lr:.2e} | Noise: {current_noise:.3f}")  # 打印当前 noise_rate

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model_a.state_dict(), "urban_transformer_cls_best_a_v5.pth")
            torch.save(model_b.state_dict(), "urban_transformer_cls_best_b_v5.pth")
            print(f"    → New best models saved! (Macro-F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    → Early stopping triggered at epoch {epoch}")
                break

    print("Training finished!\n")

    # Load best model_a for testing
    model_a.load_state_dict(torch.load("urban_transformer_cls_best_a_v5.pth", map_location=device))
    model_a.eval()

    all_preds = []
    all_labels = []
    all_embeddings = []
    all_test_ids = []

    with torch.no_grad():
        for batch in test_loader:
            inputs_img = batch['input_img'].to(device) if batch['input_img'] is not None else None
            inputs_land = batch['input_land'].to(device) if batch['input_land'] is not None else None
            # ================= 核心修改10：测试时使用 input_traffic_stats =================
            inputs_traffic_stats = batch['input_traffic_stats'].to(device) if batch[
                                                                                  'input_traffic_stats'] is not None else None
            inputs_center = batch['input_center'].to(device) if batch['input_center'] is not None else None
            labels = batch['label'].to(device)
            lengths = batch['lengths'].to(device)

            logits, embeddings = model_a(inputs_img, inputs_land, inputs_traffic_stats, inputs_center, lengths)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_embeddings.extend(embeddings.cpu().numpy())
            all_test_ids.extend(batch['seg_ids'])

    all_embeddings = np.array(all_embeddings)

    print("\n=== Final Test Classification Report ===")
    print(classification_report(all_labels, all_preds, digits=4))
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    accuracy = float(np.mean(np.asarray(all_labels) == np.asarray(all_preds)))
    report = classification_report(
        all_labels, all_preds, labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES, output_dict=True, digits=4, zero_division=0
    )
    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'best_validation_macro_f1': float(best_val_f1),
        'n_test': len(all_labels),
        'classification_report': report,
    }
    with open('test_metrics_v5.json', 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    pd.DataFrame({
        'seg_id': all_test_ids,
        'true_label': all_labels,
        'pred_label': all_preds,
    }).to_csv('test_predictions_v5.csv', index=False)
    pd.DataFrame(history).to_csv('training_history_v5.csv', index=False)
    print(f"Test accuracy={accuracy:.4f}, balanced accuracy={balanced_acc:.4f}, "
          f"macro-F1={macro_f1:.4f}, weighted-F1={weighted_f1:.4f}")

    palette = ['#2A6F97', '#4D908E', '#90BE6D', '#F9C74F', '#C8553D']
    rng = np.random.default_rng(RANDOM_STATE)
    latent_idx = []
    labels_array = np.asarray(all_labels)
    for label in range(NUM_CLASSES):
        candidates = np.flatnonzero(labels_array == label)
        latent_idx.extend(rng.choice(candidates, size=min(1600, len(candidates)), replace=False).tolist())
    latent_idx = np.asarray(latent_idx)
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, init='pca',
                learning_rate='auto', perplexity=min(40, max(5, len(latent_idx) // 100)))
    emb_2d = tsne.fit_transform(all_embeddings[latent_idx])

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.linewidth': 0.7})
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for label, color, name in zip(range(NUM_CLASSES), palette, CLASS_NAMES):
        mask = labels_array[latent_idx] == label
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=11, alpha=0.52,
                   linewidth=0, color=color, label=name, rasterized=True)
    ax.set_title('b  Latent representation of held-out trajectories', loc='left', fontweight='bold')
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, markerscale=1.5, loc='best')
    fig.tight_layout()
    fig.savefig('test_latent_space_v5.png', dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    annotations = np.asarray([
        [f'{cm_norm[i, j] * 100:.1f}%\n({cm[i, j]:,})' for j in range(NUM_CLASSES)]
        for i in range(NUM_CLASSES)
    ])
    fig, ax = plt.subplots(figsize=(6.7, 5.8))
    sns.heatmap(cm_norm, annot=annotations, fmt='', cmap=sns.light_palette('#16697A', as_cmap=True),
                vmin=0, vmax=1, square=True, linewidths=0.7, linecolor='white',
                cbar_kws={'label': 'Row-normalized proportion'}, ax=ax)
    ax.set_title('c  Five-class prediction performance', loc='left', fontweight='bold')
    ax.set_ylabel('Reference proxy class')
    ax.set_xlabel('Predicted proxy class')
    ax.set_xticklabels([str(i + 1) for i in range(NUM_CLASSES)], rotation=0)
    ax.set_yticklabels([str(i + 1) for i in range(NUM_CLASSES)], rotation=0)
    fig.tight_layout()
    fig.savefig('test_confusion_matrix_v5.png', dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print("\nAll visualizations have been saved.")


def run_band_selection_analysis(env_dir: str, traj_df_with_labels: pd.DataFrame,
                                num_samples=10000, num_keep=None, percentile_threshold=75,
                                use_gmm=True, gmm_random_state=42,  # 🔽 新增参数
                                temporal_strategy="head_tail_middle",
                                save_path="selected_bands_task_aware.json"):
    """
    改进版：任务驱动的波段筛选（考虑时序信息 + GMM自适应阈值）
    """
    import random, json
    from sklearn.utils import resample

    print(f"🎯 启动任务驱动波段筛选 (Strategy={temporal_strategy}, GMM={use_gmm})")

    # 1. 加载环境索引
    print("📂 加载环境特征索引...")
    env_files = glob.glob(os.path.join(env_dir, "*.parquet"))
    if not env_files:
        raise FileNotFoundError(f"No parquet files in {env_dir}")

    env_dfs = []
    for f in env_files:
        try:
            df = pd.read_parquet(f, columns=['seg_id', 'ae_patch_11x11x64_b64'])
            df['seg_id'] = df['seg_id'].astype(str)
            env_dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")

    if not env_dfs:
        raise ValueError("No valid environment index files found!")

    env_index = pd.concat(env_dfs, ignore_index=True)
    env_index = env_index.drop_duplicates(subset=['seg_id'])
    env_index = env_index.set_index('seg_id')
    print(f"✓ 环境索引加载完成: {len(env_index)} 个有效 seg_id")

    # 2. 找出有环境特征的轨迹（关键：先取交集！）
    traj_ids = set(traj_df_with_labels['seg_id'].astype(str).tolist())
    env_ids = set(env_index.index.tolist())
    common_ids = traj_ids & env_ids

    print(f"📊 轨迹 seg_id 总数: {len(traj_ids)}")
    print(f" 环境特征 seg_id 总数: {len(env_ids)}")
    print(f" 交集 seg_id 数量: {len(common_ids)}")

    if len(common_ids) < 50:
        raise ValueError(f"交集过小！只有 {len(common_ids)} 个 seg_id 同时存在于轨迹和环境特征中。")

    # 过滤出有环境特征的轨迹
    traj_filtered = traj_df_with_labels[traj_df_with_labels['seg_id'].astype(str).isin(common_ids)].copy()

    # 3. 均衡采样轨迹（避免类别不平衡）
    all_seg_ids = traj_filtered['seg_id'].tolist()
    all_labels = traj_filtered['label'].tolist()

    unique_labels = sorted(traj_filtered['label'].unique())
    num_classes = len(unique_labels)
    samples_per_class = num_samples // num_classes

    selected_ids, selected_labels = [], []
    for cls in unique_labels:
        cls_mask = [l == cls for l in all_labels]
        cls_ids = [all_seg_ids[i] for i, m in enumerate(cls_mask) if m]
        cls_lbls = [all_labels[i] for i, m in enumerate(cls_mask) if m]

        if len(cls_ids) > samples_per_class:
            ids_s, lbls_s = resample(cls_ids, cls_lbls,
                                     n_samples=samples_per_class,
                                     random_state=42)
        else:
            ids_s, lbls_s = cls_ids, cls_lbls

        selected_ids.extend(ids_s)
        selected_labels.extend(lbls_s)

    print(f"✓ 均衡采样 {len(selected_ids)} 条轨迹（{num_classes} 类）")

    # 4. 收集 patches（考虑时序）
    collected_patches = []
    collected_labels = []
    skipped_empty = 0
    skipped_error = 0

    print("📦 解压并收集 patches...")
    for seg_id, label in tqdm(zip(selected_ids, selected_labels), desc="Loading patches"):
        try:
            seg_id_str = str(seg_id)
            if seg_id_str not in env_index.index:
                skipped_empty += 1
                continue

            row = env_index.loc[seg_id_str]
            b64_str = row['ae_patch_11x11x64_b64']

            if not isinstance(b64_str, str) or not b64_str.strip():
                skipped_empty += 1
                continue

            patch_strs = [p.strip() for p in b64_str.split('|') if p.strip()]
            if not patch_strs:
                skipped_empty += 1
                continue

            #  根据策略采样时间步 🔽
            if temporal_strategy == "first_only":
                indices = [0]
            elif temporal_strategy == "head_tail_middle":
                n = len(patch_strs)
                if n >= 3:
                    indices = [0, n // 2, n - 1]  # 首、中、尾
                else:
                    indices = list(range(n))
            elif temporal_strategy == "adaptive":
                n = len(patch_strs)
                max_samples = min(5, n)
                indices = np.linspace(0, n - 1, max_samples, dtype=int).tolist()
            else:
                raise ValueError(f"Unknown strategy: {temporal_strategy}")

            # 添加选中的 patches
            for idx in indices:
                patch = decompress_patch(patch_strs[idx])  # [64, 11, 11]
                if np.any(patch != 0):  # 跳过全零
                    collected_patches.append(patch)
                    collected_labels.append(label)

        except Exception as e:
            skipped_error += 1
            continue

    print(f"✓ 收集完成：{len(collected_patches)} 个有效 patches")
    print(f"  跳过（空/缺失）: {skipped_empty} 条")
    print(f"  跳过（异常）: {skipped_error} 条")

    if len(collected_patches) < 50:
        raise ValueError(f"有效 patches 不足！仅收集到 {len(collected_patches)} 个，需要至少 50 个。")

    # 5. 转为 numpy 数组
    sample_array = np.stack(collected_patches)  # [N, 64, 11, 11]
    print(f"✓ 样本数组形状: {sample_array.shape}")

    # 6. 运行任务驱动分析（基于梯度）
    print("\n🔬 训练轻量代理模型并计算梯度...")
    selected_bands, scores, threshold_used = select_bands_by_gradient_sensitivity(
        patches=sample_array,
        labels=np.array(collected_labels),
        mode=BAND_SELECTION_MODE,          # 🔽 传入模式
        num_keep=NUM_BANDS_TO_KEEP,
        percentile_threshold=PERCENTILE_THRESHOLD,
        cumulative_threshold=CUMULATIVE_THRESHOLD,  # 🔽 传入累积阈值
        num_epochs=3
    )

    # 7. 打印结果
    # 更新日志打印
    print(f"\n📊 任务驱动波段筛选结果: ")
    print(f"  模式: {BAND_SELECTION_MODE}")
    print(f"  选中波段数: {len(selected_bands)}")
    print(f"  判定阈值: {threshold_used:.6f}")
    print(f"  Selected bands: {selected_bands}")
    print(f"  Top 10 scores: {scores[selected_bands[:10]].round(6)}")
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")



    # 8. 保存结果
    result = {
        "method": "task_aware_gradient",
        "temporal_strategy": temporal_strategy,
        "selection_mode": BAND_SELECTION_MODE,
        "cumulative_threshold": CUMULATIVE_THRESHOLD if BAND_SELECTION_MODE == "cumulative" else None,
        "num_bands_selected": len(selected_bands),
        "selected_bands": selected_bands,
        "all_scores": scores.tolist()
    }
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✓ 结果保存至 {save_path}")

    # 9. 可视化
    plt.figure(figsize=(12, 4))
    plt.bar(range(64), scores, color='steelblue', alpha=0.7)
    plt.axhline(y=threshold_used, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold_used:.4f})')
    plt.scatter(selected_bands, scores[selected_bands],
                color='gold', s=80, zorder=5, label=f'Selected ({len(selected_bands)} bands)')
    plt.xlabel("Band Index (0-63)")
    plt.ylabel("Gradient-based Importance Score")
    plt.title(f"Task-Aware Band Selection\n({BAND_SELECTION_MODE}, {len(selected_bands)} bands, {CUMULATIVE_THRESHOLD*100:.0f}% cumulative)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("band_selection_gradient.png", dpi=300)
    plt.close()
    print("✓ 可视化保存至 band_selection_gradient.png")

    return selected_bands

# ================= 主程序 =================
if __name__ == "__main__":
    try:
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        os.chdir(EXPERIMENT_DIR)
        print(f"Experiment output: {EXPERIMENT_DIR}")
        # 🔽 新增：任务驱动波段筛选（在主流程中调用）
        if USE_BAND_SELECTION and SELECTED_BANDS is None and AUTO_SELECT_BANDS:
            print("\n" + "=" * 60)
            print("🎯 TASK-AWARE BAND SELECTION")
            print("=" * 60)

            # 先加载轨迹数据（用于均衡采样）
            traj_data_full = load_and_clean_traj_data(TRAJ_FEATURES_DIR)
            traj_with_labels_full, best_k = perform_advanced_clustering_v5(traj_data_full.copy())


            # 运行任务驱动筛选
            selected = run_band_selection_analysis(
                env_dir=ENV_FEATURES_DIR,
                traj_df_with_labels=traj_with_labels_full[traj_with_labels_full['split'] == 'train'].copy(),
                num_samples=10000,
                temporal_strategy="head_tail_middle",
                save_path="selected_bands_task_aware.json"
            )

            # 更新配置
            SELECTED_BANDS = selected
            INPUT_CHANNELS = len(SELECTED_BANDS)  # 🔽 动态通道数
            print(f"\n✓ 任务驱动筛选完成：{INPUT_CHANNELS} 个波段（自适应）")
            print(f"  波段索引: {SELECTED_BANDS}\n")

        use_env = True
        use_landuse = False
        use_traffic = False
        use_center = USE_CENTER_CONFIG

        # 懒加载索引（不解压 patch）
        env_index = load_env_index(ENV_FEATURES_DIR) if use_env else None
        landuse_index = load_landuse_index(LANDUSE_FEATURES_DIR) if use_landuse else None
        traffic_index = load_traffic_index(TRAFFIC_FEATURES_DIR) if use_traffic else None
        center_data = load_center_data_full(CENTER_FEATURES_DIR) if (use_center or REQUIRE_CENTER_MATCH) else None

        # 轨迹必须全加载（用于聚类）
        traj_data_full = load_and_clean_traj_data(TRAJ_FEATURES_DIR)

        # ================= 先在完整轨迹数据上进行聚类 =================
        print("\n=== 进行轨迹聚类（基于所有可用运动段，不受其他模态限制）===")
        traj_with_labels_full, best_k = perform_advanced_clustering_v5(traj_data_full.copy())

        # 保存完整的聚类结果（可选，方便后续分析所有轨迹）
        traj_with_labels_full.to_csv("clustering_results_full_dataset.csv", index=False)
        print(f"聚类完成，共 {best_k} 个簇，完整数据集包含 {len(traj_with_labels_full)} 条段")

        # ================= 再根据当前模态设置进行匹配过滤 =================
        valid_seg_ids = set(traj_with_labels_full['seg_id'])
        if use_env and env_index is not None:
            valid_seg_ids &= set(env_index['seg_id'])
        if use_landuse and landuse_index is not None:
            valid_seg_ids &= set(landuse_index['seg_id'])
        if use_traffic and traffic_index is not None:
            valid_seg_ids &= set(traffic_index['seg_id'])
        if (use_center or REQUIRE_CENTER_MATCH) and center_data is not None:
            valid_seg_ids &= set(center_data['seg_id'])

        traj_with_labels = traj_with_labels_full[
            traj_with_labels_full['seg_id'].isin(valid_seg_ids)
        ].copy().reset_index(drop=True)

        print(f"多模态匹配后，用于分类训练的样本数: {len(traj_with_labels)}")

        if len(traj_with_labels) == 0:
            raise ValueError("No matching segments found after multi-modal intersection!")

        # 检查类别分布是否仍然合理
        print("用于分类的样本类别分布:")
        print(traj_with_labels['label'].value_counts().sort_index())

        # ================= 开始训练（使用过滤后的带标签数据） =================
        train_and_evaluate(env_index, landuse_index, traffic_index, center_data, traj_with_labels, best_k,
                           use_env=use_env, use_landuse=use_landuse, use_traffic=use_traffic, use_center=use_center)

    except Exception as e:
        print(f"\nExecution failed: {e}")
        import traceback

        traceback.print_exc()
