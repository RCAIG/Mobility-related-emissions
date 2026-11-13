# worldpop.py - 完整简化版

import os
from pathlib import Path
from io import BytesIO
from multiprocessing import Pool

import numpy as np
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds

from .utils import calculate_utm_epsg


# ============ 🔽 本地数据配置 ============
class LocalPopConfig:
    """本地人口栅格数据配置（LandScan 1km）"""

    # 本地数据根目录（请根据实际情况修改）
    DATA_ROOT = r"${SERVER_DATA_ROOT}/prediction_travel_mode/generate-od-pubtools-main/generate-od-pubtools-main/assets/example_data/my_test/tif"

    # 年份 → 文件路径模板
    YEAR_TO_PATH = {
        year: r"{root}/landscan-global-{year}-assets/landscan-global-{year}.tif"
        for year in range(2017, 2026)  # 2017-2025
    }

    DEFAULT_YEAR = 2017

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


# ============ 🔼 配置结束 ============


def population_one_region(region_geom, region_idx, bounds, target_crs, config):
    '''
    从本地 TIF 提取单个格网的人口

    Parameters:
    - region_geom: shapely geometry object (该格网的几何)
    - region_idx: 格网索引/名称 (用于日志)
    - bounds: (left, bottom, right, top) 边界框
    - target_crs: 区域数据的 CRS (用于面积计算)
    - config: 配置字典 {year, root}
    '''
    left, bottom, right, top = bounds
    region_id = region_idx if region_idx is not None else f"region_{hash(str(bounds))}"
    target_year = config.get('year', LocalPopConfig.DEFAULT_YEAR)
    data_root = config.get('root', LocalPopConfig.DATA_ROOT)

    # 获取本地 TIF 路径
    tif_path = LocalPopConfig.get_path(target_year, data_root)
    if not tif_path.exists():
        raise FileNotFoundError(f"本地人口数据不存在: {tif_path}")

    try:
        with rasterio.open(tif_path) as raster:
            # 坐标转换：如果区域几何与栅格 CRS 不一致，需要转换
            if target_crs and str(target_crs).upper() != str(raster.crs).upper():
                # 使用 geopandas 进行几何投影转换
                from shapely.ops import transform
                import pyproj

                src_proj = pyproj.Proj(str(target_crs))
                dst_proj = pyproj.Proj(str(raster.crs))
                project = lambda x, y: pyproj.transform(src_proj, dst_proj, x, y)
                region_geom = transform(project, region_geom)
                # 更新 bounds
                left, bottom, right, top = region_geom.bounds

            # 使用 window 预裁剪，提升读取效率
            window = from_bounds(left, bottom, right, top, raster.transform)

            # 裁剪并提取人口
            out_img, _ = mask(dataset=raster, shapes=[region_geom], crop=True, filled=True, all_touched=True)

            # 处理 NoData 值（LandScan 通常为 0 或 -9999）
            no_data = raster.nodata if raster.nodata is not None else -9999
            valid_mask = (out_img > 0) & (out_img != no_data)
            population = float(out_img[valid_mask].sum()) if valid_mask.any() else 0

    except Exception as e:
        raise RuntimeError(f"LandScan {target_year} region {region_id} failed") from e

    # 计算面积（使用原始 CRS，确保单位正确）
    try:
        epsg = calculate_utm_epsg(bounds[0], bounds[1])
        geo_series = gpd.GeoSeries([region_geom], crs=target_crs)
        area_km2 = geo_series.to_crs(f"EPSG:{epsg}").area.item() / 1e6
    except Exception as e:
        raise RuntimeError(f"Area calculation failed for region {region_id}") from e

    # 封装结果
    result = {
        'region_id': region_id,
        'bounds': list(bounds),
        'population': population,
        'area_km2': area_km2,
        'pop_density': population / area_km2 if area_km2 > 0 else 0,
        'year': target_year
    }

    # 打印进度
    print(f"   ✓ [{target_year}] {region_id}: pop={population:,.0f}, area={area_km2:.2f} km²")

    return population, area_km2, result


def worldpop(area_shp, token=None, num_proc=10, save_path=None,
             use_local=True, target_year=None, data_root=None):
    '''
    从本地 TIF 提取区域人口（完全替代在线 API）

    Parameters:
    - area_shp: geopandas.GeoDataFrame, 输入格网（必须含 geometry 列）
    - target_year: int, 目标年份 (2017-2025)
    - data_root: str, 本地数据根目录（可选）
    - save_path: str/Path, 结果保存路径（可选）
    - 其他参数保留以保持签名兼容
    '''
    # 配置处理
    if target_year is None:
        target_year = LocalPopConfig.DEFAULT_YEAR
    if data_root is None:
        data_root = LocalPopConfig.DATA_ROOT

    config = {'year': target_year, 'root': data_root}

    # 🔑 关键：从 GeoDataFrame 获取 CRS（而不是从 Series）
    target_crs = area_shp.crs if hasattr(area_shp, 'crs') else None
    if target_crs is None:
        print("⚠️ 警告: area_shp 未定义 CRS，假设为 EPSG:4326")
        target_crs = "EPSG:4326"

    populations = []
    areasizes = []
    detailed_results = []

    # 遍历每个格网（使用 itertuples 避免 Series 的 crs 问题）
    for idx, row in tqdm(area_shp.iterrows(), total=len(area_shp), desc=f" -- Population [{target_year}]"):
        # 🔑 关键：提取 geometry 和 索引，而不是直接传 Series
        region_geom = row.geometry
        region_idx = getattr(row, 'name', idx)  # 尝试获取 name，否则用索引
        bounds = region_geom.bounds

        population, area, result = population_one_region(
            region_geom=region_geom,
            region_idx=region_idx,
            bounds=bounds,
            target_crs=target_crs,
            config=config
        )

        populations.append(population)
        areasizes.append(area)
        detailed_results.append(result)

    # 保存结果
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(detailed_results)
        csv_path = save_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"   📁 人口结果已保存: {csv_path}")

    # 汇总打印
    total_pop = sum(populations)
    total_area = sum(areasizes)
    print(
        f"   📊 汇总 [{target_year}]: {len(populations)} 区域, 总人口={total_pop:,.0f}, 平均密度={total_pop / total_area if total_area > 0 else 0:.1f}/km²")

    # 🔑 返回值格式保持完全一致（兼容 generator.py）
    return np.array([populations, areasizes]).T
