import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import warnings
import sys

# 關閉一些常見的警告（可選）
warnings.filterwarnings("ignore", category=UserWarning, module="osmnx")

# ==================== 基本設置 ====================
DATE = "2017-01-01T00:00:00Z"
ox.settings.overpass_settings = f'[out:json][timeout:600][date:"{DATE}"]'  # 延長到 10 分鐘

DATA_DIR = Path("${SERVER_HOME}/result/road_2017")
DATA_DIR.mkdir(exist_ok=True)

# 中國大陸常見道路等級參考速度 (km/h)
DEFAULT_HWY_SPEEDS = {
    'motorway': 100,
    'trunk': 80,
    'primary': 70,
    'secondary': 60,
    'tertiary': 50,
    'residential': 40,
    'service': 30,
    'living_street': 20,
    'unclassified': 40,
    'road': 50,
}

FALLBACK_SPEED = 40

# 要處理的欄位（容易出現 list 的欄位）
PROBLEM_COLS = [
    'osmid', 'name', 'ref', 'lanes', 'highway', 'bridge', 'tunnel',
    'service', 'access', 'destination', 'maxspeed', 'junction',
    'width', 'maxheight', 'maxweight'
]


def process_single_area(row):
    """處理單一區域的核心函數（供多進程呼叫）"""
    uc_id = int(row['ID_UC_G0'])
    id_str = str(uc_id)

    print(f"[{uc_id}] 開始處理...", flush=True)

    # 定義所有可能輸出的檔案路徑
    files_to_check = [
        DATA_DIR / f"road_network_drive_{id_str}.graphml",
        DATA_DIR / f"nodes_{id_str}.parquet",
        DATA_DIR / f"edges_{id_str}.parquet",
        DATA_DIR / f"preview_road_network_{id_str}.png",
    ]

    # 如果全部檔案都存在 → 跳過
    if all(f.exists() for f in files_to_check):
        print(f"[{uc_id}] 所有目標檔案已存在，跳過", flush=True)
        return uc_id, True, "已存在，跳過"

    polygon = row.geometry

    try:
        # 下載路網
        G = ox.graph_from_polygon(
            polygon,
            network_type="all",
            simplify=True,
            retain_all=False,
            truncate_by_edge=True,
            custom_filter=None  # 如有需要可自行加上過濾條件
        )

        if len(G) == 0:
            msg = f"[{uc_id}] 路網為空，跳過"
            print(msg, flush=True)
            return uc_id, False, msg

        # 加速度與旅行時間
        G = ox.add_edge_speeds(G, hwy_speeds=DEFAULT_HWY_SPEEDS, fallback=FALLBACK_SPEED)
        G = ox.add_edge_travel_times(G)

        # 預覽圖（可選擇註解掉以進一步加速）
        try:
            fig, ax = ox.plot_graph(
                G,
                bgcolor="white",
                node_size=0,
                edge_linewidth=0.5,
                edge_color="#666666",
                show=False,
                close=False
            )
            plt.title(f"路網預覽 - ID_UC_G0 = {uc_id}")
            plt.tight_layout()
            plt.savefig(DATA_DIR / f"preview_road_network_{id_str}.png",
                        dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[{uc_id}] 繪圖失敗（不影響主流程）：{e}", flush=True)

        # 保存 GraphML
        ox.save_graphml(G, filepath=DATA_DIR / f"road_network_drive_{id_str}.graphml")

        # 轉 GeoDataFrame 並清理
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)

        # 處理 list 欄位 → 轉字串
        for gdf in [nodes_gdf, edges_gdf]:
            for col in PROBLEM_COLS:
                if col in gdf.columns:
                    gdf[col] = gdf[col].apply(
                        lambda x: ','.join(str(i) for i in x) if isinstance(x, (list, tuple))
                        else str(x) if pd.notnull(x) else ''
                    )

            # 更保險：所有 object 欄位都處理一次
            for col in gdf.select_dtypes(include=['object']).columns:
                gdf[col] = gdf[col].apply(
                    lambda x: ','.join(str(i) for i in x) if isinstance(x, (list, tuple))
                    else str(x) if pd.notnull(x) else ''
                )

        # 儲存 parquet
        nodes_gdf.to_parquet(DATA_DIR / f"nodes_{id_str}.parquet", index=False)
        edges_gdf.to_parquet(DATA_DIR / f"edges_{id_str}.parquet", index=False)

        print(f"[{uc_id}] ✓ 完成", flush=True)
        return uc_id, True, "完成"

    except Exception as e:
        msg = f"[{uc_id}] × 錯誤：{str(e)}"
        print(msg, flush=True)
        return uc_id, False, str(e)


def main():
    print("讀取城市/區域邊界 shapefile...")
    shp_path = r"${SERVER_DATA_ROOT}/prediction_travel_mode/trace_generate/road_network_clip/data/2024_urban/2024_urban.shp"

    urban_areas = gpd.read_file(shp_path)
    urban_areas = urban_areas.to_crs(epsg=4326)

    if 'ID_UC_G0' not in urban_areas.columns:
        raise ValueError("shapefile 中缺少 'ID_UC_G0' 欄位！")

    urban_areas['ID_UC_G0'] = urban_areas['ID_UC_G0'].astype(int)

    # ==================== 新增：根據目錄文件篩選 ID ====================
    target_dir = Path(r"${SERVER_DATA_ROOT}/prediction_travel_mode/generate-od-pubtools-main/generate-od-pubtools-main/assets/example_data/my_test/shp/ALL/")
    
    if target_dir.exists():
        target_ids = set()
        shp_files = list(target_dir.glob("*.shp"))
        print(f"掃描目錄 {target_dir}，找到 {len(shp_files)} 個 .shp 文件")
        
        for file in shp_files:
            try:
                # 獲取文件名不含副檔名部分，例如 "5929.shp" -> "5929"
                file_id = int(file.stem)
                target_ids.add(file_id)
            except ValueError:
                print(f"警告：文件 {file.name} 無法轉換為整數 ID，已跳過")
        
        # 篩選 urban_areas，只保留 ID 在 target_ids 中的行
        original_count = len(urban_areas)
        urban_areas = urban_areas[urban_areas['ID_UC_G0'].isin(target_ids)]
        filtered_count = len(urban_areas)
        
        print(f"原始區域數量：{original_count}")
        print(f"篩選後區域數量：{filtered_count} (根據目錄中的 .shp 文件名 ID)")
        
        if filtered_count == 0:
            print("錯誤：沒有找到匹配的 ID，程序終止。")
            return
    else:
        print(f"警告：目標目錄 {target_dir} 不存在，將處理所有區域。")
    # ================================================================

    print(f"總共發現 {len(urban_areas)} 個區域")

    # 建議進程數：CPU 核心數 × 0.7～1.0 之間，依記憶體與網路狀況調整
    # 網路下載為瓶頸時，建議不要開太高（4～12 較合理）
    MAX_WORKERS = 8

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(process_single_area, row): row['ID_UC_G0']
            for _, row in urban_areas.iterrows()
        }

        for future in concurrent.futures.as_completed(future_to_id):
            uc_id = future_to_id[future]
            try:
                id_, success, message = future.result()
                results.append((id_, success, message))
            except Exception as exc:
                print(f"[{uc_id}] 進程異常：{exc}", file=sys.stderr, flush=True)

    # 最終總結
    print("\n" + "=" * 70)
    print("全部處理結束！總結如下：")
    success_count = sum(1 for _, s, _ in results if s)
    print(f"成功：{success_count} / {len(results)}")
    print(f"失敗/跳過：{len(results) - success_count}")
    print("資料位置：", DATA_DIR.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()