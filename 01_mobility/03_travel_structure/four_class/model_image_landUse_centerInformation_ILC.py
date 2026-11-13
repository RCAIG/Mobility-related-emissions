import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"  # 关键！避开 GPU 0
import pandas as pd
import numpy as np
import glob
import gzip
import base64
import torch
from pathlib import Path
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import BayesianGaussianMixture
from umap import UMAP
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster

# Suppress warnings
warnings.filterwarnings('ignore')
# ================= Configuration =================
# 当前.py文件的路径
current_file = os.path.abspath(__file__)

# 获取当前文件所在目录
current_dir = os.path.dirname(current_file)

ENV_FEATURES_DIR = os.path.join(os.path.dirname(current_dir), "output_features_GEE")
LANDUSE_FEATURES_DIR = os.path.join(os.path.dirname(current_dir), "output_features_landuse/work")
TRAJ_FEATURES_DIR = os.path.join(os.path.dirname(current_dir), "output_features_trajectory")
CENTER_FEATURES_DIR = os.path.join(os.path.dirname(current_dir), "output_features_gdp_center/work")

# Features definition
ADVANCED_FEATURES = [
    'adv_speed_95', 'adv_speed_99', 'adv_high_speed_ratio_10',
    'adv_high_speed_ratio_8', 'adv_stop_ratio', 'adv_acc_95',
    'adv_speed_std', 'adv_heading_change_rate'
]
BASE_FEATURES = ['avg_speed_mps', 'avg_accel_mps2']
FEATURE_COLS = BASE_FEATURES + ADVANCED_FEATURES

# Land use types
LANDUSE_TYPES = ['commercial', 'industrial', 'park', 'public', 'residential']

# City center features
CENTER_FEATURES = ['start_to_primary_m', 'end_to_primary_m', 'start_to_nearest_sub_m', 'end_to_nearest_sub_m',
                   'trip_min_to_primary_m', 'trip_min_to_nearest_sub_m', 'od_ratio_primary', 'od_ratio_nearest_sub',
                   'city_total_centers']

# ================= Helper Functions =================

def decompress_patch(b64_str: str) -> np.ndarray:
    try:
        compressed = base64.b64decode(b64_str)
        raw = gzip.decompress(compressed)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(64, 11, 11)
        return arr
    except Exception:
        return np.zeros((64, 11, 11), dtype=np.float32)

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
    print(f"Environment index loaded: {len(full_df)} valid segments\n")
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
    print(f"Landuse index loaded: {len(full_df)} valid segments\n")
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
    print(f"Center data loaded and standardized: {len(full_df)} segments\n")
    return full_df

def load_and_clean_traj_data(directory: str) -> pd.DataFrame:
    # 与原代码完全一致（必须全加载用于聚类）
    print(f"Loading Trajectory features from: {directory}")
    files = glob.glob(os.path.join(directory, "*.xlsx"))
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
    新的聚类方法：基于8个高区分度特征 + 2个基础特征
    使用 BayesianGMM + UMAP降维 + 层次聚类后处理
    """
    print("\n执行新的高级聚类方法...")

    # ==================== 定义特征列 ====================
    # 8个高区分度特征
    advanced_features = [
        'adv_speed_95',
        # 'adv_speed_99',
        'adv_high_speed_ratio_10',
        # 'adv_high_speed_ratio_8',
        'adv_stop_ratio',
        # 'adv_acc_95',
        'adv_speed_std',
        'adv_heading_change_rate'
    ]

    # 2个基础特征
    base_features = ['avg_speed_mps', 'avg_accel_mps2']

    # 最终使用的特征列（共10个）
    feature_cols = base_features + advanced_features
    print(f"使用特征数量: {len(feature_cols)}")
    print("特征列表:", feature_cols)

    # ==================== 特征检查 ====================
    # 检查缺失列并填充
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"警告：以下特征缺失，将填充为0: {missing_cols}")
        for c in missing_cols:
            df[c] = 0.0

    # ==================== 数据预处理 ====================
    print(f"原始总段数: {len(df)}")

    # 1. 彻底删除所有停留段（stay）
    if 'type' in df.columns:
        before = len(df)
        df = df[df['type'] != 'stay'].copy()
        print(f"移除停留段: {before - len(df)} 条，剩余 {len(df)}")

    # 2. 过滤物理上不可能的值（硬阈值）
    valid_conditions = []

    # 检查特征是否存在，避免KeyError
    if 'adv_speed_95' in df.columns:
        valid_conditions.append(df['adv_speed_95'] <= 35)
        valid_conditions.append(df['adv_speed_95'] >= 0.3)

    if 'adv_acc_95' in df.columns:
        valid_conditions.append(df['adv_acc_95'] <= 20)

    if 'adv_stop_ratio' in df.columns:
        valid_conditions.append(df['adv_stop_ratio'] <= 0.9)

    if 'adv_heading_change_rate' in df.columns:
        valid_conditions.append(df['adv_heading_change_rate'] <= 200)

    if valid_conditions:
        # 将所有条件用&连接
        valid = valid_conditions[0]
        for condition in valid_conditions[1:]:
            valid = valid & condition

        df = df[valid].copy()
        print(f"移除极端异常段后剩余: {len(df)} 条")

    # 3. 再次检查是否还有type=='stay'残留
    if 'type' in df.columns:
        remain_stay = (df['type'] == 'stay').sum()
        if remain_stay > 0:
            print(f"警告：仍有 {remain_stay} 条 stay 段，强制删除")
            df = df[df['type'] != 'stay']

    # 重置索引
    df = df.reset_index(drop=True)
    print(f"最终用于聚类的有效运动段: {len(df)} 条")
    print("-" * 60)

    # ==================== 特征提取和标准化 ====================
    X_raw = df[feature_cols].copy()
    print(f"缺失值统计: {X_raw.isna().sum().sum()} 个缺失值")

    # 填充缺失值
    X_raw.fillna(X_raw.median(numeric_only=True), inplace=True)
    X_raw.fillna(0, inplace=True)

    # 标准化（必须！GMM对尺度敏感）
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    print(f"标准化后特征维度: {X.shape}")

    # 保存标准化特征
    X_df_scaled = pd.DataFrame(X, columns=feature_cols, index=df.index)
    X_df_scaled['seg_id'] = df['seg_id']
    X_df_scaled.to_csv("trajectory_features_scaled_new.csv", index=False)

    # ==================== UMAP降维 ====================
    try:
        print("正在进行UMAP降维...")
        umap_reducer = UMAP(
            n_components=5,
            n_neighbors=30,
            min_dist=0.1,
            metric='euclidean',
            random_state=42,
            n_jobs=-1
        )
        X_reduced = umap_reducer.fit_transform(X)
        print(f"UMAP降维后维度: {X_reduced.shape}")
    except ImportError:
        print("警告：未安装UMAP，使用原始特征进行聚类")
        X_reduced = X

    # ==================== Bayesian GMM 自动选K ====================
    print("使用Bayesian Gaussian Mixture进行聚类...")
    bgmm = BayesianGaussianMixture(
        n_components=20,
        covariance_type='full',
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=1e-4,
        mean_precision_prior=0.8,
        reg_covar=1e-4,
        random_state=42,
        max_iter=1000,
        tol=1e-4,
        init_params='kmeans',
        n_init=10
    )

    initial_labels = bgmm.fit_predict(X_reduced)

    # 有效簇个数：权重 > 阈值
    effective_k = np.sum(bgmm.weights_ > 0.01)
    print(f"GMM自动检测有效簇数: {effective_k}")
    print("各簇权重:", bgmm.weights_)

    # 重新映射标签到连续的0到effective_k-1
    unique_labels = np.unique(initial_labels)
    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
    initial_labels = np.array([label_map[l] for l in initial_labels])
    effective_k = len(unique_labels)

    # ==================== 层次聚类后处理（固定为4类） ====================
    if effective_k > 1:
        print("进行层次聚类后处理，固定合并为4个簇...")
        # 计算每个初始簇的中心
        cluster_centers = np.zeros((effective_k, X_reduced.shape[1]))
        for i in range(effective_k):
            mask = initial_labels == i
            if np.sum(mask) > 0:
                cluster_centers[i] = X_reduced[mask].mean(axis=0)

        # 进行层次聚类
        Z = linkage(cluster_centers, method='ward', metric='euclidean')

        # 直接固定切割为4个簇（如果初始有效簇少于4，则保持原样）
        if effective_k >= 4:
            meta_labels = fcluster(Z, t=4, criterion='maxclust') - 1  # t=4 表示强制分成4个簇
            best_n_clusters = 4
            print(f"强制合并为 {best_n_clusters} 个簇")
        else:
            # 如果GMM给出的有效簇少于4个，直接使用原始标签
            meta_labels = initial_labels.copy()
            best_n_clusters = effective_k
            print(f"初始有效簇数 {effective_k} < 4，保持不变，不进行合并")

        # 重新映射最终标签
        labels = np.array([meta_labels[l] for l in initial_labels])

        # 可选：仍然计算silhouette score用于参考（不影响结果）
        score = silhouette_score(X_reduced, labels)
        print(f"固定4簇后的 Silhouette Score: {score:.4f} (仅供参考)")

    # # ==================== 层次聚类后处理 ====================
    # if effective_k > 1:
    #     print("进行层次聚类后处理...")
    #     # 计算每个初始簇的中心
    #     cluster_centers = np.zeros((effective_k, X_reduced.shape[1]))
    #     for i in range(effective_k):
    #         mask = initial_labels == i
    #         if np.sum(mask) > 0:
    #             cluster_centers[i] = X_reduced[mask].mean(axis=0)
    #
    #     # 进行层次聚类
    #     Z = linkage(cluster_centers, method='ward', metric='euclidean')
    #
    #     # 动态确定最佳簇数，使用silhouette score
    #     best_score = -1
    #     best_n_clusters = 2
    #     best_meta_labels = None
    #     scores = []
    #     n_range = range(2, effective_k + 1)
    #
    #     for n_clusters in n_range:
    #         meta_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
    #         new_labels = np.array([meta_labels[l] for l in initial_labels])
    #         score = silhouette_score(X_reduced, new_labels)
    #         scores.append(score)
    #         if score > best_score:
    #             best_score = score
    #             best_n_clusters = n_clusters
    #             best_meta_labels = meta_labels
    #
    #     print(f"层次聚类后处理最佳簇数: {best_n_clusters} (silhouette score: {best_score:.4f})")
    #
    #     # 可视化silhouette score曲线
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(n_range, scores, marker='o', linestyle='-', color='b')
    #     plt.scatter(best_n_clusters, best_score, color='r', s=100, zorder=5)
    #     plt.annotate(f'Best: {best_n_clusters}', xy=(best_n_clusters, best_score),
    #                  xytext=(best_n_clusters + 0.2, best_score + 0.01),
    #                  arrowprops=dict(facecolor='red', shrink=0.05))
    #     plt.title('Silhouette Score vs. Number of Clusters (Hierarchical Merging)')
    #     plt.xlabel('Number of Clusters')
    #     plt.ylabel('Silhouette Score')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig("hierarchical_silhouette_curve.png", dpi=300, bbox_inches='tight')
    #     plt.close()
    #
    #     # 使用最佳标签
    #     labels = np.array([best_meta_labels[l] for l in initial_labels])
    #     effective_k = best_n_clusters
    # else:
    #     labels = initial_labels
    #     print("初始簇数 <=1，无需后处理")

    # ==================== 保存聚类结果 ====================
    df['label'] = labels  # 注意：这里保持用'label'列名以兼容后续代码
    probs = bgmm.predict_proba(X_reduced)

    # 保存每个簇的概率（可选）
    for i in range(bgmm.n_components):
        if i < probs.shape[1]:
            df[f'prob_cluster_{i}'] = probs[:, i]
        else:
            df[f'prob_cluster_{i}'] = 0.0

    # 保存聚类结果（保持与原代码兼容的列名）
    df.to_csv("clustering_results_new.csv", index=False)
    print("聚类结果已保存到 clustering_results_new.csv")

    # ==================== 生成聚类报告 ====================
    print("生成聚类报告...")

    # 使用原始特征计算统计
    cluster_stats_mean = df.groupby('label')[feature_cols].mean()
    cluster_stats_std = df.groupby('label')[feature_cols].std()
    cluster_stats_count = df['label'].value_counts().sort_index()

    # 创建格式化的结果DataFrame
    result_df = pd.DataFrame(index=cluster_stats_mean.index)
    for col in feature_cols:
        result_df[f"{col}"] = (
                cluster_stats_mean[col].round(4).astype(str) +
                " ± " +
                cluster_stats_std[col].round(4).astype(str)
        )
    result_df["Count"] = cluster_stats_count.values

    print(result_df.T)
    result_df.T.to_excel("cluster_comparison_raw_new.xlsx")

    # ==================== 热力图可视化 ====================
    X_vis = pd.DataFrame(X, columns=feature_cols)
    X_vis['cluster'] = df['label']
    cluster_summary = X_vis.groupby('cluster').mean()

    plt.figure(figsize=(12, 8))
    vmax = np.abs(cluster_summary.values).max()
    vmin = -vmax

    sns.heatmap(
        cluster_summary.T,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.6,
        linecolor='white',
        cbar_kws={
            'label': 'Z-score (standardized feature value)',
            'shrink': 0.8
        },
        annot_kws={'size': 10, 'weight': 'bold', 'color': 'black'}
    )
    plt.title(f"Trajectory Clusters K={effective_k} - Advanced Features")
    plt.ylabel("Features")
    plt.xlabel("Cluster")
    plt.tight_layout()
    plt.savefig("cluster_heatmap_new.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== 2D可视化 ====================
    pca = PCA(n_components=2, random_state=42)
    X_pca_2d = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 左图：层次聚类前
    scatter0 = axes[0].scatter(
        X_pca_2d[:, 0], X_pca_2d[:, 1],
        c=initial_labels, cmap='tab10',
        alpha=0.6, s=30
    )
    axes[0].set_title('Cluster Distribution Before Hierarchical Clustering')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    plt.colorbar(scatter0, ax=axes[0], label='Cluster')

    # 右图：层次聚类后
    scatter1 = axes[1].scatter(
        X_pca_2d[:, 0], X_pca_2d[:, 1],
        c=labels, cmap='tab10',
        alpha=0.6, s=30
    )
    axes[1].set_title('Cluster Distribution After Hierarchical Clustering')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    plt.colorbar(scatter1, ax=axes[1], label='Cluster')

    plt.tight_layout()
    plt.savefig("cluster_distribution_2d_new.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n聚类完成！共生成 {effective_k} 个簇")
    print(f"各簇样本数分布: {dict(df['label'].value_counts().sort_index())}")
    best_k = effective_k

    return df, best_k

# ================= Lazy Dataset =================

class LazyMultiModalTrajectoryDataset(Dataset):
    def __init__(self, traj_df_with_labels, env_index_df=None, landuse_index_df=None, center_df=None,
                 use_env=True, use_landuse=True, use_center=True):
        """
        traj_df_with_labels: 必须包含 seg_id 和 label（聚类后）
        其他 df 只包含 seg_id + 原始字段（字符串或数值），不包含 tensor
        """
        self.use_env = use_env
        self.use_landuse = use_landuse
        self.use_center = use_center

        # 构建样本列表（只存 seg_id 和 label）
        merged = traj_df_with_labels[['seg_id', 'label']].copy()
        if use_env and env_index_df is not None:
            merged = pd.merge(merged, env_index_df, on='seg_id', how='inner')
        if use_landuse and landuse_index_df is not None:
            merged = pd.merge(merged, landuse_index_df, on='seg_id', how='inner')
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
            patches = [decompress_patch(p) for p in patch_strs]
            patches_tensor = torch.from_numpy(np.stack(patches)).float()
            sample['input_img'] = patches_tensor

        # 土地利用：按需 tokenization
        if self.use_landuse:
            landuse_str = row['nearest_landuse']
            token_seq = extract_landuse_sequence(landuse_str)
            sample['input_land'] = torch.from_numpy(token_seq).long()

        # 城市中心：已经是标准化数值，直接转 tensor
        if self.use_center:
            center_vals = np.array([row[c] for c in CENTER_FEATURES], dtype=np.float32)
            sample['input_center'] = torch.from_numpy(center_vals)

        return sample

# ================= collate_fn 与 my_collate_fn 不变 =================

def collate_fn(batch, use_env=True, use_landuse=True, use_center=True):
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    seg_ids = [item['seg_id'] for item in batch]

    inputs_img = None
    inputs_land = None
    inputs_center = None
    lengths = None

    if use_env:
        img_seqs = [item['input_img'] for item in batch]
        inputs_img = pad_sequence(img_seqs, batch_first=True, padding_value=0)
        lengths = torch.tensor([len(seq) for seq in img_seqs])

    if use_landuse:
        land_seqs = [item['input_land'] for item in batch]
        inputs_land = pad_sequence(land_seqs, batch_first=True, padding_value=0)
        if lengths is None:
            lengths = torch.tensor([len(seq) for seq in land_seqs])

    if use_center:
        center_vecs = [item['input_center'] for item in batch]
        inputs_center = torch.stack(center_vecs)

    if lengths is None:
        lengths = torch.ones(len(batch), dtype=torch.int64)

    return {
        'input_img': inputs_img,
        'input_land': inputs_land,
        'input_center': inputs_center,
        'label': labels,
        'lengths': lengths,
        'seg_ids': seg_ids
    }

def my_collate_fn(batch):
    return collate_fn(batch, use_env=True, use_landuse=False, use_center=True)

# ================= 其余部分（模型、训练）完全不变（仅优化 PatchEncoder 显存） =================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            device = x.device
            pe = torch.zeros(seq_len, x.size(2), device=device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, x.size(2), 2, device=device).float() * (-math.log(10000.0) / x.size(2)))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return x + pe.unsqueeze(0)
        return x + self.pe[:seq_len, :].unsqueeze(0)

class SimpleResidualBlock(nn.Module):
    def __init__(self, channels):  # 保持通道不变
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
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
        self.bn1 = nn.BatchNorm2d(64)
        self.res_block = SimpleResidualBlock(64)  # 保持64通道，不扩张
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, embed_dim)  # 输入64通道

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.res_block(x)
        x = self.pool(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class TokenEmbedder(nn.Module):
    def __init__(self, num_tokens=5, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens + 1, embed_dim, padding_idx=0)

    def forward(self, x):
        return self.embedding(x)

class UrbanMultiModalClassifier(nn.Module):
    def __init__(self, num_classes=4, input_channels=64, embed_dim=128, nhead=8, num_layers=6,
                 use_env=True, use_landuse=True, use_center=True):
        super().__init__()
        self.use_env = use_env
        self.use_landuse = use_landuse
        self.use_center = use_center
        self.embed_dim = embed_dim

        if self.use_env:
            self.patch_encoder = PatchEncoder(input_channels, embed_dim)
        if self.use_landuse:
            self.token_embedder = TokenEmbedder(num_tokens=5, embed_dim=embed_dim)
        if self.use_center:
            self.center_encoder = nn.Sequential(
                nn.Linear(len(CENTER_FEATURES), embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )

        self.embed_norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True, dim_feedforward=512
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=nhead // 2, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_img, x_land, x_center, lengths):
        B = x_land.size(0) if x_land is not None else (x_img.size(0) if x_img is not None else x_center.size(0))

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

        center_emb = None
        if self.use_center and x_center is not None:
            center_emb = self.center_encoder(x_center).unsqueeze(1)
            center_emb = self.embed_norm(center_emb)

        if embeddings_land is not None and embeddings_img is not None:
            embeddings, _ = self.cross_attn(embeddings_land, embeddings_img, embeddings_img)
        elif embeddings_land is not None:
            embeddings = embeddings_land
        elif embeddings_img is not None:
            embeddings = embeddings_img
        else:
            embeddings = torch.zeros(B, 0, self.embed_dim, device=self.cls_token.device)

        has_center = center_emb is not None
        if has_center:
            embeddings = torch.cat((embeddings, center_emb), dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = self.pos_encoder(embeddings)

        main_T = embeddings.shape[1] - 1 - (1 if has_center else 0)
        if main_T > 0:
            pad_mask = torch.arange(main_T, device=embeddings.device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            pad_mask = torch.empty(B, 0, dtype=torch.bool, device=embeddings.device)

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=embeddings.device)
        center_mask = torch.zeros(B, 1, dtype=torch.bool, device=embeddings.device) if has_center else torch.empty(B, 0, dtype=torch.bool, device=embeddings.device)
        mask = torch.cat((cls_mask, center_mask, pad_mask), dim=1)

        encoded = self.transformer(embeddings, src_key_padding_mask=mask)
        cls_embeddings = encoded[:, 0, :]
        logits = self.classifier(cls_embeddings)
        return logits, cls_embeddings

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

def train_epoch(model, dataloader, criterion, optimizer, device, use_focal=False):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch in dataloader:
        inputs_img = batch['input_img'].to(device) if batch['input_img'] is not None else None
        inputs_land = batch['input_land'].to(device) if batch['input_land'] is not None else None
        inputs_center = batch['input_center'].to(device) if batch['input_center'] is not None else None
        labels = batch['label'].to(device)
        lengths = batch['lengths'].to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs_img, inputs_land, inputs_center, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        epoch_loss += loss.item()
        total += labels.size(0)

    return epoch_loss / len(dataloader), 100.0 * correct / total

def evaluate(model, dataloader, criterion, device, use_focal=False):
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
            inputs_center = batch['input_center'].to(device) if batch['input_center'] is not None else None
            labels = batch['label'].to(device)
            lengths = batch['lengths'].to(device)

            logits, _ = model(inputs_img, inputs_land, inputs_center, lengths)
            loss = criterion(logits, labels)

            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            epoch_loss += loss.item()
            total += labels.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    return epoch_loss / len(dataloader), 100.0 * correct / total, f1

def train_and_evaluate(env_index, landuse_index, center_data, traj_with_labels, best_k,
                       use_env=True, use_landuse=True, use_center=True):
    BATCH_SIZE = 16
    LR = 1e-4
    MAX_EPOCHS = 60
    NUM_CLASSES = 4
    PATIENCE = 10
    USE_FOCAL_LOSS = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full_dataset = LazyMultiModalTrajectoryDataset(
        traj_df_with_labels=traj_with_labels,
        env_index_df=env_index if use_env else None,
        landuse_index_df=landuse_index if use_landuse else None,
        center_df=center_data if use_center else None,
        use_env=use_env, use_landuse=use_landuse, use_center=use_center
    )

    all_labels = [item['label'] for item in full_dataset.samples]
    all_indices = list(range(len(full_dataset)))

    train_idx, temp_idx = train_test_split(all_indices, train_size=0.70, stratify=all_labels, random_state=42)
    remaining_labels = [all_labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, stratify=remaining_labels, random_state=42)

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

    # 降低 num_workers，避免 Windows 下内存泄漏
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                              collate_fn=my_collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=my_collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=my_collate_fn, num_workers=2, pin_memory=True)

    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = UrbanMultiModalClassifier(num_classes=NUM_CLASSES,
                                      use_env=use_env, use_landuse=use_landuse, use_center=use_center).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    if USE_FOCAL_LOSS:
        criterion = ImprovedFocalLoss(gamma=2.0)
    else:
        class LabelSmoothingCrossEntropy(nn.Module):
            def __init__(self, smoothing=0.1):
                super().__init__()
                self.smoothing = smoothing
            def forward(self, pred, target):
                n_classes = pred.size(1)
                log_preds = F.log_softmax(pred, dim=-1)
                with torch.no_grad():
                    true_dist = torch.zeros_like(log_preds)
                    true_dist.fill_(self.smoothing / (n_classes - 1))
                    true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
                return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_val_acc = 0.0
    patience_counter = 0

    print("\n" + "=" * 60)
    print("START TRAINING")
    print("=" * 60)

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:6.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:6.2f}% F1: {val_f1:.4f} | "
              f"LR: {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "urban_transformer_cls_best_ILC.pth")
            print(f"    → New best model saved! (Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    → Early stopping triggered at epoch {epoch}")
                break

    print("Training finished!\n")

    model.load_state_dict(torch.load("urban_transformer_cls_best_ILC.pth"))
    model.eval()

    all_preds = []
    all_labels = []
    all_embeddings = []
    with torch.no_grad():
        for batch in test_loader:
            inputs_img = batch['input_img'].to(device) if batch['input_img'] is not None else None
            inputs_land = batch['input_land'].to(device) if batch['input_land'] is not None else None
            inputs_center = batch['input_center'].to(device) if batch['input_center'] is not None else None
            labels = batch['label'].to(device)
            lengths = batch['lengths'].to(device)

            logits, embeddings = model(inputs_img, inputs_land, inputs_center, lengths)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_embeddings.extend(embeddings.cpu().numpy())

    all_embeddings = np.array(all_embeddings)
    print("\n=== Final Test Classification Report ===")
    print(classification_report(all_labels, all_preds, digits=4))
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"\nTest F1 Score: {test_f1:.4f}")

    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    emb_2d = tsne.fit_transform(all_embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.6, s=15)
    plt.colorbar(scatter, label='Class')
    plt.title("Latent Space Visualization (t-SNE)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True, alpha=0.3)
    plt.savefig("test_latent_space_ILC.png", dpi=300, bbox_inches='tight')
    plt.close()

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("test_confusion_matrix_ILC.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\nAll visualizations have been saved.")

# ================= 主程序 =================

if __name__ == "__main__":
    try:
        use_env = True
        use_landuse = True
        use_center = True

        # 懒加载索引（不解压 patch）
        env_index = load_env_index(ENV_FEATURES_DIR) if use_env else None
        landuse_index = load_landuse_index(LANDUSE_FEATURES_DIR) if use_landuse else None
        center_data = load_center_data_full(CENTER_FEATURES_DIR) if use_center else None

        # 轨迹必须全加载（用于聚类）
        traj_data_full = load_and_clean_traj_data(TRAJ_FEATURES_DIR)

        # ================= 先在完整轨迹数据上进行聚类 =================
        print("\n=== 进行轨迹聚类（基于所有可用运动段，不受其他模态限制）===\n")
        traj_with_labels_full, best_k = perform_advanced_clustering(traj_data_full.copy())

        # 保存完整的聚类结果（可选，方便后续分析所有轨迹）
        traj_with_labels_full.to_csv("clustering_results_full_dataset.csv", index=False)
        print(f"聚类完成，共 {best_k} 个簇，完整数据集包含 {len(traj_with_labels_full)} 条段")

        # ================= 再根据当前模态设置进行匹配过滤 =================
        valid_seg_ids = set(traj_with_labels_full['seg_id'])

        if use_env and env_index is not None:
            valid_seg_ids &= set(env_index['seg_id'])
        if use_landuse and landuse_index is not None:
            valid_seg_ids &= set(landuse_index['seg_id'])
        if use_center and center_data is not None:
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
        train_and_evaluate(
            env_index, landuse_index, center_data,
            traj_with_labels, best_k,
            use_env=use_env,
            use_landuse=use_landuse,
            use_center=use_center
        )

    except Exception as e:
        print(f"\nExecution failed: {e}")
        import traceback

        traceback.print_exc()