import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm

# ====================== 配置部分 ======================
NPY_INPUT_DIR = r"${SERVER_DATA_ROOT}/prediction_travel_mode/generate-od-pubtools-main/generate-od-pubtools-main/assets/example_data/my_test/result/lower_middle_2024"
CSV_OUTPUT_DIR = r"${SERVER_DATA_ROOT}/prediction_travel_mode/generate-od-pubtools-main/generate-od-pubtools-main/assets/example_data/my_test/result/od_flows_wgs84_lower_middle_2024"

# 格网 shp 文件所在的目录（每个 area_id 对应一个 {area_id}.shp）
GRID_SHP_BASE_DIR = r"${SERVER_DATA_ROOT}/prediction_travel_mode/generate-od-pubtools-main/generate-od-pubtools-main/assets/example_data/my_test/shp/ALL"

# 输出 CSV 文件名模板（可自定义）
CSV_FILENAME_TEMPLATE = "{area_id}_od_flows_wgs84.csv"

# 是否过滤很小的流量（推荐打开，减少文件体积）
FILTER_SMALL_FLOW = True
MIN_FLOW_THRESHOLD = 1e-6  # 可调，例如 0.1 或 1.0


# ======================================================

def get_area_id_from_filename(filename):
    """从文件名中提取 area_id，例如 xxx_od_matrix.npy → xxx"""
    name = Path(filename).stem
    if '_od_matrix' in name:
        return name.replace('_od_matrix', '')
    return None


def load_grid_centroids_and_ids(area_id):
    """读取对应 area_id 的格网 shp，返回中心点坐标和行索引作为 ID"""
    shp_path = Path(GRID_SHP_BASE_DIR) / f"{area_id}.shp"

    if not shp_path.exists():
        print(f"  警告：找不到格网文件 {shp_path}，跳过该 area_id")
        return None, None

    print(f"  读取格网文件: {shp_path}")
    gdf = gpd.read_file(shp_path)

    # 确保是经纬度坐标系（WGS84）
    if gdf.crs is None:
        raise ValueError(f"格网文件 {shp_path} 缺少坐标系信息")
    if not gdf.crs.is_geographic:
        print("  格网不是经纬度坐标，正在转换为 EPSG:4326...")
        gdf = gdf.to_crs(epsg=4326)

    # 计算中心点
    centroids = gdf.centroid
    coords = np.array([(p.x, p.y) for p in centroids])

    # 使用行索引（从 0 开始）作为 ID
    grid_ids = np.arange(len(gdf))

    print(f"  成功读取 {len(coords)} 个格网中心点")
    print(f"  使用行索引（0 到 {len(grid_ids) - 1}）作为 o_id / d_id")

    return coords, grid_ids


def convert_single_npy_to_csv(npy_path, output_dir):
    """处理单个 npy 文件"""
    area_id = get_area_id_from_filename(npy_path.name)
    if not area_id:
        print(f"跳过无法解析 area_id 的文件: {npy_path.name}")
        return 0

    # 构造目标 CSV 路径
    out_csv = Path(output_dir) / CSV_FILENAME_TEMPLATE.format(area_id=area_id)

    # 如果输出文件已存在，跳过
    if out_csv.exists():
        print(f"\n跳过已存在的输出文件: {out_csv.name}")
        return 0

    print(f"\n处理区域: {area_id}")

    # 读取 OD 矩阵
    try:
        od_matrix = np.load(npy_path)
    except Exception as e:
        print(f"  加载 npy 文件失败: {e}")
        return 0

    n = od_matrix.shape[0]
    if od_matrix.shape != (n, n):
        print(f"  矩阵不是方阵，跳过: {npy_path.name}")
        return 0

    # 读取对应格网的中心点和 ID（行索引）
    centroids, grid_ids = load_grid_centroids_and_ids(area_id)
    if centroids is None:
        return 0

    if len(centroids) != n or len(grid_ids) != n:
        print(f"  警告：格网数量 {len(centroids)} 与矩阵维度 {n} 不匹配，跳过")
        return 0

    # 生成记录
    records = []

    for i in range(n):
        for j in range(n):
            flow = float(od_matrix[i, j])
            if flow <= 0:
                continue

            o_lon, o_lat = centroids[i]
            d_lon, d_lat = centroids[j]
            o_id = int(grid_ids[i])  # 确保是整数
            d_id = int(grid_ids[j])

            records.append({
                'origin_id': o_id,
                'dest_id': d_id,
                'o_lon': o_lon,
                'o_lat': o_lat,
                'd_lon': d_lon,
                'd_lat': d_lat,
                'flow': flow
            })

    if not records:
        print("  该区域没有有效流量记录")
        return 0

    df = pd.DataFrame(records)

    # 过滤小流量（可选）
    if FILTER_SMALL_FLOW:
        df = df[df['flow'] > MIN_FLOW_THRESHOLD].copy()

    # 保存到 CSV
    df.to_csv(out_csv, index=False, float_format='%.8f')
    print(f"  转换完成 → {out_csv}")
    print(f"  总记录数: {len(df):,} 条")

    return len(df)


def main():
    # 确保输出目录存在
    Path(CSV_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 查找所有 _od_matrix.npy 文件
    input_path = Path(NPY_INPUT_DIR)
    npy_files = list(input_path.rglob("*_od_matrix.npy"))

    if not npy_files:
        print("在输入目录中没有找到任何 *_od_matrix.npy 文件！")
        return

    print(f"共找到 {len(npy_files)} 个 OD 矩阵文件")

    # ===== 核心优化：提前检查输出文件存在性，过滤无需处理的文件 =====
    files_to_process = []
    skipped_files = []

    for npy_file in npy_files:
        area_id = get_area_id_from_filename(npy_file.name)
        if not area_id:
            print(f"⚠️  无法从文件名提取 area_id，跳过: {npy_file.name}")
            skipped_files.append((npy_file, "无法解析area_id"))
            continue

        out_csv = Path(CSV_OUTPUT_DIR) / CSV_FILENAME_TEMPLATE.format(area_id=area_id)

        if out_csv.exists():
            skipped_files.append((npy_file, f"输出文件已存在: {out_csv.name}"))
        else:
            files_to_process.append(npy_file)

    # 打印跳过详情
    if skipped_files:
        print(f"\n{'=' * 60}")
        print(f"跳过 {len(skipped_files)} 个文件（原因如下）:")
        for i, (f, reason) in enumerate(skipped_files[:10], 1):  # 只显示前10个避免刷屏
            print(f"  {i}. {f.name} → {reason}")
        if len(skipped_files) > 10:
            print(f"  ... 还有 {len(skipped_files) - 10} 个文件被跳过（未全部显示）")
        print(f"{'=' * 60}")

    # 打印处理计划
    print(f"\n✅ 需要处理的文件数: {len(files_to_process)} / {len(npy_files)}")
    if not files_to_process:
        print("💡 所有文件输出均已存在，无需处理。程序结束。")
        return

    # 开始处理
    total_records = 0
    print(f"\n开始转换 {len(files_to_process)} 个文件...\n")

    for npy_file in tqdm(files_to_process, desc="转换进度"):
        count = convert_single_npy_to_csv(npy_file, CSV_OUTPUT_DIR)
        total_records += count

    # 最终统计
    print("\n" + "=" * 60)
    print("✅ 全部转换完成！")
    print(f"  总输入文件数: {len(npy_files)}")
    print(f"  跳过文件数:   {len(skipped_files)}")
    print(f"  实际处理数:   {len(files_to_process)}")
    print(f"  生成总记录数: {total_records:,} 条")
    print(f"  输出目录:     {CSV_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()