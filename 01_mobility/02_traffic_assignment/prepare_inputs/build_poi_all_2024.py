import osmnx as ox
import geopandas as gpd
from pathlib import Path
import pandas as pd
import sys

# ==================== 基本设置 ====================
date = "2024-01-01T00:00:00Z"
# ox.settings.overpass_settings = f'[out:json][timeout:300][date:"{date}"]'

DATA_DIR = Path("${SERVER_HOME}/result/poi_2024")
DATA_DIR.mkdir(exist_ok=True)

# ==================== 修改点：全量 POI 标签定义 ====================
# 这里列出了 OSM 中所有常见的 POI 相关主键 (Keys)
# 值设为 True 表示下载该键下的所有具体类型 (Values)，无需手动列举 restaurant, school 等
POI_TAGS = {
    "amenity": True,  # 核心设施 (餐厅、医院、学校等)
    "shop": True,  # 所有商店
    "tourism": True,  # 所有旅游相关
    "leisure": True,  # 所有休闲设施
    "office": True,  # 所有办公室
    # "public_transport": True,  # 公共交通
    # "building": "yes",  # 建筑物 (通常作为 POI 载体)
    "craft": True,  # 手工艺店
    "healthcare": True,  # 医疗保健
    "education": True,  # 教育设施
    "entertainment": True,  # 娱乐设施
    "historic": True,  # 历史遗迹
    "man_made": True,  # 人工设施 (塔、灯塔等)
    # "natural": True,  # 自然特征 (山峰、洞穴等，有时算 POI)
    "sport": True,  # 运动设施
    "aeroway": True,  # 航空设施
    # "railway": ["station", "halt", "tram_stop", "subway_entrance"],  # 仅下载站点，不下载铁路线
    # "highway": ["bus_stop", "elevator", "steps", "crossing"],  # 仅下载设施点，不下载道路线
}
# ===============================================================

# 1. 读取城市/区域边界 shapefile
print("读取城市/区域边界 shapefile...")
shp_path = r"${SERVER_DATA_ROOT}/prediction_travel_mode/trace_generate/road_network_clip/data/2024_urban/2024_urban.shp"
urban_areas = gpd.read_file(shp_path)
urban_areas = urban_areas.to_crs(epsg=4326)

if 'ID_UC_G0' not in urban_areas.columns:
    raise ValueError("shapefile 中缺少 'ID_UC_G0' 字段！")
urban_areas['ID_UC_G0'] = urban_areas['ID_UC_G0'].astype(int)

# ==================== 根据目录文件筛选 ID ====================
target_dir = Path(
    r"${SERVER_DATA_ROOT}/prediction_travel_mode/generate-od-pubtools-main/generate-od-pubtools-main/assets/example_data/my_test/shp/Decrease_high_income/"
)

if target_dir.exists():
    target_ids = set()
    shp_files = list(target_dir.glob("*.shp"))
    print(f"掃描目錄 {target_dir}，找到 {len(shp_files)} 個 .shp 文件")
    for file in shp_files:
        try:
            file_id = int(file.stem)
            target_ids.add(file_id)
        except ValueError:
            print(f"警告：文件 {file.name} 無法轉換為整數 ID，已跳過")

    original_count = len(urban_areas)
    urban_areas = urban_areas[urban_areas['ID_UC_G0'].isin(target_ids)]
    filtered_count = len(urban_areas)
    print(f"原始區域數量：{original_count}")
    print(f"篩選後區域數量：{filtered_count} (根據目錄中的 .shp 文件名 ID)")
    if filtered_count == 0:
        print("錯誤：沒有找到匹配的 ID，程序終止。")
        sys.exit(0)
else:
    print(f"警告：目標目錄 {target_dir} 不存在，將處理所有區域。")
# ================================================================================

# 2. 逐个处理每个区域
for idx, row in urban_areas.iterrows():
    uc_id = row['ID_UC_G0']
    print(f"\n{'=' * 60}")
    print(f"开始处理全量 POI 数据 - ID_UC_G0 = {uc_id} ({idx + 1}/{len(urban_areas)})")

    output_path = DATA_DIR / f"{uc_id}_poi.parquet"

    if output_path.exists():
        print(f"  → 已存在，跳过：{output_path.name}")
        continue

    polygon = row.geometry

    try:
        print(f"  正在下载全量 POI 数据（ID={uc_id}）...")
        gdf = ox.features_from_polygon(polygon, tags=POI_TAGS)

        if gdf.empty:
            print(f"  × 该区域无 POI 数据，生成空文件（ID={uc_id}）")
            empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            empty_gdf.to_parquet(output_path, index=False)
            continue

        print("  清理字段格式...")
        for col in gdf.columns:
            if gdf[col].dtype == object:
                gdf[col] = gdf[col].apply(
                    lambda x: ','.join(str(i) for i in x) if isinstance(x, (list, tuple))
                    else str(x) if pd.notnull(x) else ''
                )

        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        print(f"  保存 POI 数据：{output_path.name}")
        gdf.to_parquet(output_path, compression="snappy", index=False)
        print(f"  √ 完成！ID_UC_G0 = {uc_id}")

    except Exception as e:
        print(f"  × 处理 ID_UC_G0 = {uc_id} 时出错：{str(e)}")
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        empty_gdf.to_parquet(output_path, index=False)
        continue

print("\n" + "=" * 70)
print("所有区域全量 POI 数据下载完毕！")
print(f"数据保存位置：{DATA_DIR.resolve()}")
print("=" * 70)