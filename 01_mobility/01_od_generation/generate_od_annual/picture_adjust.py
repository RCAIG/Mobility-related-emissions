import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 从 utils.py 中导入绘图函数（假设 utils.py 和本脚本在同一目录）
from utils import plot_od_arc_chart

# ==================== 配置区域 ====================

# OD 矩阵 (.npy) 和对应的 .shp 文件所在目录
# 当前设置：npy 在 “模拟结果” 文件夹，shp 在其上级的 “shp” 文件夹
data_dir = Path(
    r"${OD_OUTPUT_ROOT}"
)

# 图片输出目录
output_dir = data_dir / "arc_charts"
output_dir.mkdir(parents=True, exist_ok=True)

# 是否强制使用固定阈值？推荐 False，让每座城市自动计算分位数
force_fixed_threshold = False
fixed_low = 50  # 仅当 force_fixed_threshold=True 时生效
fixed_high = 100

# ================================================

npy_files = list(data_dir.glob("*_od_matrix.npy"))

if not npy_files:
    print(f"警告：在 {data_dir} 中未找到任何 *_od_matrix.npy 文件！")
else:
    print(f"找到 {len(npy_files)} 个 OD 矩阵文件，开始生成弧线图...\n")

for npy_path in npy_files:
    city_name = npy_path.stem.replace("_od_matrix", "")
    # 假设 shp 文件在上层目录的 shp 文件夹中
    shp_path = data_dir.parent / "shp" / f"{city_name}.shp"

    # 如果你的 shp 和 npy 在同一目录，改用下面这行：
    # shp_path = data_dir / f"{city_name}.shp"

    print(f"正在处理城市：{city_name}")

    try:
        # 读取 OD 矩阵
        od_matrix = np.load(npy_path)

        # 读取 shapefile
        if not shp_path.exists():
            print(f"  错误：未找到对应的 shp 文件 {shp_path}，跳过")
            continue

        area_gdf = gpd.read_file(shp_path)
        area_gdf = area_gdf.to_crs(epsg=4326)

        # 检查维度是否匹配
        if od_matrix.shape[0] != od_matrix.shape[1] or od_matrix.shape[0] != len(area_gdf):
            print(f"  错误：OD 矩阵不是方阵或维度 {od_matrix.shape[0]} 与区域数 {len(area_gdf)} 不匹配，跳过")
            continue

        print(f"  OD 矩阵形状: {od_matrix.shape}, 总出行量: {od_matrix.sum():,}")

        # ==================== 计算分位数阈值 ====================
        # 只取上三角（避免对角线0和重复计数）
        flows = od_matrix[np.triu_indices_from(od_matrix, k=1)]
        positive_flows = flows[flows > 0]

        if len(positive_flows) == 0:
            print("  警告：OD 矩阵中无正流量，无法绘图，跳过")
            continue

        if force_fixed_threshold:
            low = fixed_low
            high = fixed_high
        else:
            # 自动计算 33% 和 66% 分位数（与 plot_od_arc_chart 内部逻辑一致）
            low = int(np.percentile(positive_flows, 25))
            high = int(np.percentile(positive_flows, 75))

            # 防止 low >= high 或 low=0 的极端情况
            if low >= high:
                high = int(np.percentile(positive_flows, 80))
            if low == 0:
                low = 1

        print(f"  绘图阈值：低流量 <= {low}（蓝色），{low} < 中流量 <= {high}（红色），> {high}（黄色）")

        # ==================== 绘图并保存 ====================
        fig = plot_od_arc_chart(od_matrix, area_gdf, low=low, high=high)

        img_path = output_dir / f"{city_name}_od_arc_chart.png"
        fig.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"  弧线图已保存：{img_path}\n")

    except Exception as e:
        print(f"  处理 {city_name} 时出错：{e}\n")
        continue

print("所有弧线图生成完成！")