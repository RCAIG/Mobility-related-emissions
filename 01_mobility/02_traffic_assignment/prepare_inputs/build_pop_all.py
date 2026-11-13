import os
import glob
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
from rasterio.io import MemoryFile
import pyproj
from shapely.ops import transform


# ------------------ 本地人口数据配置 ------------------

class LocalPopConfig:
    """本地人口栅格数据配置（LandScan 1km 或 WorldPop 100m）"""
    
    # 本地数据根目录（请根据实际情况修改）
    DATA_ROOT = r"${SERVER_DATA_ROOT}/prediction_travel_mode/generate-od-pubtools-main/generate-od-pubtools-main/assets/example_data/my_test/tif"
    
    # 年份 → 文件路径模板（支持多种命名规范）
    YEAR_TO_PATH = {
        year: f"{{root}}/landscan-global-{year}-assets/landscan-global-{year}.tif"
        for year in range(2017, 2026)  # 2017-2025
    }
    
    DEFAULT_YEAR = 2017
    NODATA_VALUE = -9999  # 根据实际数据调整，LandScan常用-9999，WorldPop常用0
    
    @classmethod
    def get_path(cls, year: int, root: str = None) -> Path:
        """根据年份获取本地 TIF 文件路径"""
        if root is None:
            root = cls.DATA_ROOT
        if year not in cls.YEAR_TO_PATH:
            raise ValueError(f"年份 {year} 未配置，可用: {list(cls.YEAR_TO_PATH.keys())}")
        path_template = cls.YEAR_TO_PATH[year]
        return Path(path_template.format(root=root, year=year))
    
    @classmethod
    def set_root(cls, root: str):
        cls.DATA_ROOT = root


# ------------------ 辅助函数 ------------------

def calculate_utm_epsg(longitude, latitude):
    """根据经纬度计算对应的UTM EPSG代码"""
    utm_zone = int((longitude + 180) / 6) + 1
    return 32600 + utm_zone if latitude >= 0 else 32700 + utm_zone


def _safe_transform_geometry(geom, src_crs, dst_crs):
    """安全地进行几何坐标转换"""
    if src_crs is None or str(src_crs).upper() == str(dst_crs).upper():
        return geom
    try:
        src_proj = pyproj.Proj(str(src_crs))
        dst_proj = pyproj.Proj(str(dst_crs))
        project = lambda x, y: pyproj.transform(src_proj, dst_proj, x, y)
        return transform(project, geom)
    except Exception:
        return geom


# ------------------ 核心处理函数 ------------------

def population_one_region(region, target_year=2017, data_root=None):
    '''
    从本地 TIF 提取单个格网的人口数据
    
    Parameters:
    - region: geopandas row (with geometry)
    - target_year: 目标年份 (默认2024)
    - data_root: 本地数据根目录
    
    Returns:
    - (population, area_km2)
    '''
    if data_root is None:
        data_root = LocalPopConfig.DATA_ROOT
    
    left, bottom, right, top = region.geometry.bounds
    region_geom = region.geometry
    
    # 获取本地TIF路径
    try:
        tif_path = LocalPopConfig.get_path(target_year, data_root)
        if not tif_path.exists():
            print(f"⚠️ 文件不存在: {tif_path}")
            return float('nan'), float('nan')
    except Exception as e:
        print(f"⚠️ 获取路径失败: {e}")
        return float('nan'), float('nan')
    
    try:
        with rasterio.open(tif_path) as raster:
            raster_crs = raster.crs
            target_crs = region.crs if hasattr(region, 'crs') else "EPSG:4326"
            
            # 坐标转换：如果区域几何与栅格CRS不一致
            geom_converted = _safe_transform_geometry(region_geom, target_crs, raster_crs)
            left, bottom, right, top = geom_converted.bounds
            
            # 使用window预读取提升效率（可选优化）
            window = from_bounds(left, bottom, right, top, raster.transform)
            
            # 裁剪并提取人口
            out_img, _ = mask(
                dataset=raster, 
                shapes=[geom_converted], 
                crop=True, 
                filled=True, 
                all_touched=True,
                pad=True
            )
            
            # 处理NoData值
            no_data = raster.nodata if raster.nodata is not None else LocalPopConfig.NODATA_VALUE
            if no_data is not None:
                valid_mask = (out_img > 0) & (out_img != no_data)
            else:
                valid_mask = out_img > 0
                
            population = float(out_img[valid_mask].sum()) if valid_mask.any() else 0.0
            
            # 计算面积（转换为UTM投影计算平方米）
            epsg = calculate_utm_epsg(left, bottom)
            geo_series = gpd.GeoSeries([region_geom], crs=target_crs)
            area_km2 = geo_series.to_crs(f"EPSG:{epsg}").area.item() / 1e6
            
            return population, area_km2
            
    except FileNotFoundError:
        print(f"⚠️ TIF文件未找到: {tif_path}")
        return float('nan'), float('nan')
    except Exception as e:
        print(f"⚠️ 处理区域时出错: {e}")
        return float('nan'), float('nan')


def process_shp_to_geoparquet_single(shp_path, output_dir, target_year=2017, data_root=None):
    """
    处理单个 .shp 文件（多进程安全版本）
    
    Parameters:
    - shp_path: 输入shapefile路径
    - output_dir: 输出目录
    - target_year: 人口数据年份
    - data_root: 本地人口数据根目录
    """
    base_name = os.path.splitext(os.path.basename(shp_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.parquet")

    # 跳过已存在的文件
    if os.path.exists(output_path):
        return f"⏭️  Skipped (exists): {base_name}"

    try:
        gdf = gpd.read_file(shp_path)
    except Exception as e:
        return f"❌ Read failed: {base_name} | {e}"

    # 确保有CRS信息
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    
    gdf = gdf.copy()
    pops = []
    areas = []

    # 逐行处理（避免在子进程中嵌套多进程）
    for idx, row in enumerate(gdf.itertuples()):
        pop, area = population_one_region(
            region=row, 
            target_year=target_year, 
            data_root=data_root
        )
        pops.append(pop)
        areas.append(area)

    gdf["pop"] = pops
    gdf["area_km2"] = areas
    gdf["pop_density"] = gdf["pop"] / gdf["area_km2"].replace(0, np.nan)

    try:
        # 只保留必要列输出，减小文件体积
        output_cols = [c for c in gdf.columns if c not in ['geometry']] + ['geometry']
        gdf[output_cols].to_parquet(output_path, compression="snappy", index=False)
        return f"✅ Saved: {base_name} | {len(gdf)} rows"
    except Exception as e:
        return f"⚠️ Save failed: {base_name} | {e}"


def main():
    # ------------------ 配置区域 ------------------
    input_dir = "${SERVER_DATA_ROOT}/prediction_travel_mode/generate-od-pubtools-main/generate-od-pubtools-main/assets/example_data/my_test/shp/ALL"
    output_dir = "${SERVER_DATA_ROOT}/prediction_travel_mode/trace_generate/road_network_clip/data/all_data/2017_pop/"
    
    # 人口数据配置
    TARGET_YEAR = 2017  # 目标年份
    DATA_ROOT = r"${SERVER_DATA_ROOT}/prediction_travel_mode/generate-od-pubtools-main/generate-od-pubtools-main/assets/example_data/my_test/tif"  # 本地TIF根目录
    
    # 并发控制（根据CPU核心数和IO性能调整）
    MAX_WORKERS = 4  # 本地文件读取建议降低并发，避免磁盘IO瓶颈
    
    # ------------------ 主逻辑 ------------------
    os.makedirs(output_dir, exist_ok=True)
    
    # 更新配置
    LocalPopConfig.set_root(DATA_ROOT)
    
    shp_files = glob.glob(os.path.join(input_dir, "*.shp"))
    if not shp_files:
        raise ValueError(f"No .shp files found in {input_dir}")
    
    print(f"📁 Found {len(shp_files)} shapefiles to process.")
    print(f"📅 Target year: {TARGET_YEAR}")
    print(f"🗂️  Data root: {DATA_ROOT}")
    print(f"🔧 Workers: {MAX_WORKERS}")
    print("-" * 60)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_shp_to_geoparquet_single, 
                shp_path, 
                output_dir, 
                target_year=TARGET_YEAR,
                data_root=DATA_ROOT
            ): shp_path
            for shp_path in shp_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                shp_path = futures[future]
                print(f"💥 Unhandled error in {os.path.basename(shp_path)}: {e}")
    
    print("-" * 60)
    print("🎉 All tasks completed!")


if __name__ == "__main__":
    main()