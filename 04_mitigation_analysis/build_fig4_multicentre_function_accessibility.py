"""Multicentre sensitivity analysis for V25 population-function mismatch.

This script leaves the V25 single-centre results untouched.  It re-queries
Overture Places, detects population and native-function density peaks, and
compares the original one-centre mismatch with a population-node to nearest
function-node mismatch.  The run is resumable by 5-degree tile checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def relaunch_with_python_312() -> None:
    if sys.version_info >= (3, 10):
        return
    if os.environ.get("V25_MULTICENTRE_PY312") == "1":
        raise RuntimeError("Python 3.10 or newer is required.")
    env = os.environ.copy()
    env["V25_MULTICENTRE_PY312"] = "1"
    try:
        code = subprocess.call(["py", "-3.12", str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    except FileNotFoundError as exc:
        raise RuntimeError("Python 3.12 was not found. Run this script with py -3.12.") from exc
    raise SystemExit(code)


relaunch_with_python_312()

sys.path[:0] = [r"${WORKSPACE_ROOT}", r"${WORKSPACE_ROOT}\\.pydeps_analysis"]

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from shapely import points, wkb
from sklearn.neighbors import BallTree


ROOT = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V25")
TABLE = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V21\Tables\city_analysis_data.csv")
BOUNDARIES = Path(r"${URBAN_DATA_ROOT}\\全球城市边界\城市边界2024\2024城市边界.shp")
POP_DIR = Path(r"${URBAN_DATA_ROOT}\\全球城市边界\建筑环境和移动碳排\DrivingFactorAnalysis\data\2024_pop")
OUT = ROOT / "Figures" / "Fig4_function_accessibility"
CHECKPOINTS = OUT / "multicentre_5deg_checkpoints"
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

OVERTURE_RELEASE = "2026-08-19.0"
OVERTURE_URI = f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}/theme=places/type=place/*"
MIN_CONFIDENCE = 0.60
EARTH_RADIUS_KM = 6371.0088
TILE_DEGREES = 5.0
MIN_PLACES = 20
PEAK_RADIUS_KM = 1.5
MIN_SEPARATION_KM = 1.0
PEAK_RELATIVE_THRESHOLD = 0.30
MAX_CENTRES = 20
EPS = 1e-12

FUNCTIONS = [
    ("services_and_business", "Services and business"),
    ("shopping", "Shopping"),
    ("food_and_drink", "Food and drink"),
    ("health_care", "Health care"),
    ("lifestyle_services", "Lifestyle services"),
    ("education", "Education"),
    ("travel_and_transportation", "Travel and transportation"),
    ("sports_and_recreation", "Sports and recreation"),
    ("cultural_and_historic", "Cultural and historic"),
    ("community_and_government", "Community and government"),
    ("lodging", "Lodging"),
    ("arts_and_entertainment", "Arts and entertainment"),
]
DOMAINS = {x[0] for x in FUNCTIONS}


def haversine_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise row-wise distance for latitude/longitude arrays."""
    ar = np.deg2rad(a)
    br = np.deg2rad(b)
    dlat = br[:, 0] - ar[:, 0]
    dlon = (br[:, 1] - ar[:, 1] + np.pi) % (2 * np.pi) - np.pi
    h = np.sin(dlat / 2) ** 2 + np.cos(ar[:, 0]) * np.cos(br[:, 0]) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def local_xy(coords: np.ndarray) -> np.ndarray:
    """Approximate local kilometre coordinates, adequate within one city."""
    lat0 = np.deg2rad(float(np.nanmean(coords[:, 0])))
    return np.column_stack((coords[:, 1] * 111.320 * np.cos(lat0), coords[:, 0] * 110.574))


def density_peaks(coords: np.ndarray, weights: np.ndarray, *, kind: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Return centre coordinates, centre weights, and identification status."""
    if len(coords) == 0:
        return np.empty((0, 2)), np.empty(0), "missing"
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0)
    xy = local_xy(coords)
    radius = PEAK_RADIUS_KM
    tree = cKDTree(xy)
    neighbourhoods = tree.query_ball_point(xy, radius)
    density = np.asarray([weights[idx].sum() for idx in neighbourhoods], dtype=float)
    if kind == "function" and len(coords) < MIN_PLACES:
        centre = np.average(coords, axis=0, weights=weights + EPS)
        return centre[None, :], np.array([weights.sum()]), "fallback_sparse"
    if not np.isfinite(density).any() or density.max() <= 0:
        centre = np.average(coords, axis=0, weights=weights + EPS)
        return centre[None, :], np.array([weights.sum()]), "fallback_invalid"
    # A point is a peak when its local density is not below any neighbour.
    candidates = np.asarray([i for i, neighbours in enumerate(neighbourhoods)
                             if density[i] >= density[neighbours].max() - EPS], dtype=int)
    candidates = candidates[np.argsort(density[candidates])[::-1]]
    selected: list[int] = []
    threshold = density.max() * PEAK_RELATIVE_THRESHOLD
    for idx in candidates:
        if density[idx] < threshold:
            break
        if all(np.linalg.norm(xy[idx] - xy[j]) >= MIN_SEPARATION_KM for j in selected):
            selected.append(int(idx))
        if len(selected) >= MAX_CENTRES:
            break
    if not selected:
        selected = [int(np.argmax(density))]
    centres = coords[selected]
    # Assign source mass to the nearest retained peak; this is the centre weight.
    centre_tree = cKDTree(local_xy(centres))
    nearest = centre_tree.query(xy, k=1)[1]
    centre_weights = np.bincount(nearest, weights=weights, minlength=len(centres)).astype(float)
    status = "multicentre" if len(centres) > 1 else "single_peak"
    return centres, centre_weights, status


def population_centres(city_id: int) -> tuple[np.ndarray, np.ndarray, str]:
    path = POP_DIR / f"{city_id}.parquet"
    if not path.exists():
        return np.empty((0, 2)), np.empty(0), "missing_population_grid"
    try:
        gdf = gpd.read_parquet(path, columns=["pop", "geometry"])
        rows = []
        for pop, geom in gdf[["pop", "geometry"]].itertuples(index=False, name=None):
            if float(pop) > 10 and geom is not None and not geom.is_empty:
                c = geom.centroid
                rows.append((float(c.y), float(c.x), float(pop)))
        if not rows:
            return np.empty((0, 2)), np.empty(0), "empty_population_grid"
        arr = np.asarray(rows, dtype=float)
        return density_peaks(arr[:, :2], arr[:, 2], kind="population")
    except Exception as exc:
        return np.empty((0, 2)), np.empty(0), f"population_error_{type(exc).__name__}"


def one_centre_mismatch(pop_coords: np.ndarray, pop_weights: np.ndarray,
                        poi: pd.DataFrame, area_km2: float, domain: str) -> float:
    group = poi[poi.top_level_domain.eq(domain)]
    if group.empty or len(pop_coords) == 0:
        return np.nan
    p = np.average(pop_coords, axis=0, weights=pop_weights + EPS)
    f = group[["latitude", "longitude"]].to_numpy(float)
    q = np.average(f, axis=0)
    radius = max(math.sqrt(max(float(area_km2), EPS) / math.pi), 0.5)
    return float(haversine_km(p[None, :], q[None, :])[0] / radius)


def multi_centre_mismatch(pop_coords: np.ndarray, pop_weights: np.ndarray,
                          poi: pd.DataFrame, area_km2: float, domain: str) -> tuple[float, int, str]:
    group = poi[poi.top_level_domain.eq(domain)]
    if group.empty or len(pop_coords) == 0:
        return np.nan, 0, "missing"
    fcoords = group[["latitude", "longitude"]].to_numpy(float)
    fweights = np.ones(len(fcoords), dtype=float)
    fcentres, _, status = density_peaks(fcoords, fweights, kind="function")
    # Every population node is matched to its nearest same-function centre.
    pop_xy = local_xy(pop_coords)
    fun_xy = local_xy(fcentres)
    nearest = cKDTree(fun_xy).query(pop_xy, k=1)[1]
    matched = fcentres[nearest]
    distances = haversine_km(pop_coords, matched)
    radius = max(math.sqrt(max(float(area_km2), EPS) / math.pi), 0.5)
    return float(np.average(distances, weights=pop_weights + EPS) / radius), int(len(fcentres)), status


def query_tile(connection: duckdb.DuckDBPyConnection, bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    minx, miny, maxx, maxy = bounds
    sql = """
        SELECT id,
               list_extract(taxonomy.hierarchy, 1) AS top_level_domain,
               confidence,
               (bbox.xmin + bbox.xmax) / 2.0 AS longitude,
               (bbox.ymin + bbox.ymax) / 2.0 AS latitude
        FROM read_parquet(?)
        WHERE bbox.xmin >= ? AND bbox.xmax <= ?
          AND bbox.ymin >= ? AND bbox.ymax <= ?
          AND confidence >= ?
    """
    return connection.execute(sql, [OVERTURE_URI, minx, maxx, miny, maxy, MIN_CONFIDENCE]).fetchdf()


def city_metrics(city_id: int, area_km2: float, places: pd.DataFrame) -> dict:
    pop_coords, pop_weights, pop_status = population_centres(city_id)
    out: dict[str, object] = {
        "ID_UC_G0": int(city_id),
        "overture_places_n_multicentre": int(len(places)),
        "population_center_count": int(len(pop_coords)),
        "population_center_status": pop_status,
    }
    if len(pop_coords) == 0:
        out["multicentre_status"] = pop_status
        return out
    for domain, _ in FUNCTIONS:
        out[f"single_mismatch_{domain}"] = one_centre_mismatch(pop_coords, pop_weights, places, area_km2, domain)
        multi, n, status = multi_centre_mismatch(pop_coords, pop_weights, places, area_km2, domain)
        out[f"multicentre_mismatch_{domain}"] = multi
        out[f"function_center_count_{domain}"] = n
        out[f"function_center_status_{domain}"] = status
    single = [out[f"single_mismatch_{d}"] for d, _ in FUNCTIONS if np.isfinite(out[f"single_mismatch_{d}"])]
    multi = [out[f"multicentre_mismatch_{d}"] for d, _ in FUNCTIONS if np.isfinite(out[f"multicentre_mismatch_{d}"])]
    out["single_mismatch_mean"] = float(np.mean(single)) if single else np.nan
    out["multicentre_mismatch_mean"] = float(np.mean(multi)) if multi else np.nan
    out["multicentre_status"] = "complete" if len(multi) else "no_function_data"
    return out


def process_tile(tile_key: str, cities: gpd.GeoDataFrame, connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    target = CHECKPOINTS / f"tile_{tile_key}.csv"
    if target.exists() and target.stat().st_size > 100:
        cached = pd.read_csv(target)
        if "multicentre_status" in cached.columns:
            return cached
    minx, miny, maxx, maxy = cities.total_bounds
    places = query_tile(connection, (minx - 0.02, miny - 0.02, maxx + 0.02, maxy + 0.02))
    if len(places):
        places = places[places.top_level_domain.isin(DOMAINS)].drop_duplicates("id")
        geo = gpd.GeoDataFrame(places, geometry=points(places.longitude.to_numpy(), places.latitude.to_numpy()), crs=4326)
        assigned = gpd.sjoin(geo, cities[["ID_UC_G0", "geometry"]], how="inner", predicate="within")
        assigned = assigned.drop(columns=["index_right", "geometry"])
    else:
        assigned = pd.DataFrame(columns=["ID_UC_G0", "top_level_domain", "latitude", "longitude"])
    grouped = {int(k): v for k, v in assigned.groupby("ID_UC_G0")} if len(assigned) else {}
    rows = []
    for row in cities[["ID_UC_G0", "GC_UCA_KM2"]].itertuples(index=False):
        subset = grouped.get(int(row.ID_UC_G0), assigned.iloc[0:0])
        try:
            rows.append(city_metrics(int(row.ID_UC_G0), float(row.GC_UCA_KM2), subset))
        except Exception as exc:
            rows.append({"ID_UC_G0": int(row.ID_UC_G0), "multicentre_status": f"error_{type(exc).__name__}"})
    result = pd.DataFrame(rows)
    result.to_csv(target, index=False, encoding="utf-8-sig")
    return result


def add_outcomes(metrics: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    keep = ["ID_UC_G0", "iso3", "city_name", "GC_UCA_KM2", "Pop_2024", "logPop2024", "logGDP2024",
            "d_log_percap_travel_distance", "d_log_total_distance", "d_short_share", "d_long_share",
            "log_distance_pc", "log_distance", "short_share", "long_share"]
    q = data[[c for c in keep if c in data.columns]].copy()
    # Keep the spatial outcomes identical to the V25 figure script. The
    # source table stores 2024 distance, population, and shares separately;
    # construct the analysis-scale aliases before the paired statistics.
    distance_2024 = pd.to_numeric(data.get("total_travel_distance_m_2024"), errors="coerce")
    population_2024 = pd.to_numeric(data.get("Pop_2024"), errors="coerce")
    q["log_distance"] = np.where(distance_2024 > 0, np.log(distance_2024), np.nan)
    q["log_distance_pc"] = np.where(
        (distance_2024 > 0) & (population_2024 > 0),
        np.log(distance_2024 / population_2024),
        np.nan,
    )
    q["short_share"] = pd.to_numeric(data.get("short_share_2024"), errors="coerce")
    q["long_share"] = pd.to_numeric(data.get("long_share_2024"), errors="coerce")
    q["log_pop"] = pd.to_numeric(q["logPop2024"], errors="coerce") if "logPop2024" in q else np.nan
    log_gdp = pd.to_numeric(q["logGDP2024"], errors="coerce") if "logGDP2024" in q else np.nan
    q["log_gdppc"] = log_gdp - q["log_pop"]
    d = metrics.merge(q, on="ID_UC_G0", how="left")
    return d


def statistics(data: pd.DataFrame, x: str, y: str) -> dict:
    z = data[[x, y, "iso3", "log_pop", "log_gdppc", "GC_UCA_KM2"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 8:
        return {"n": len(z), "rho": np.nan, "within_country_rho": np.nan}
    rho = float(spearmanr(z[x], z[y]).statistic)
    xx = z[x] - z.groupby("iso3")[x].transform("mean")
    yy = z[y] - z.groupby("iso3")[y].transform("mean")
    wrho = float(spearmanr(xx, yy).statistic) if xx.nunique() > 1 and yy.nunique() > 1 else np.nan
    return {"n": len(z), "countries": z.iso3.nunique(), "rho": rho, "within_country_rho": wrho,
            "x_mean": float(z[x].mean()), "y_mean": float(z[y].mean())}


def write_comparisons(panel: pd.DataFrame) -> None:
    rows = []
    for domain, label in FUNCTIONS:
        old = f"single_mismatch_{domain}"
        new = f"multicentre_mismatch_{domain}"
        for scale, outcomes in [
            ("temporal_2017_2024", ["d_log_percap_travel_distance", "d_log_total_distance", "d_short_share", "d_long_share"]),
            ("spatial_2024", ["log_distance_pc", "log_distance", "short_share", "long_share"]),
        ]:
            for outcome in outcomes:
                for method, feature in [("single", old), ("multicentre", new)]:
                    s = statistics(panel, feature, outcome)
                    rows.append({"scale": scale, "domain": domain, "domain_label": label,
                                 "method": method, "feature": feature, "outcome": outcome, **s})
    pd.DataFrame(rows).to_csv(OUT / "multicentre_vs_single_statistics.csv", index=False, encoding="utf-8-sig")
    summary = []
    for domain, label in FUNCTIONS:
        old = panel[f"single_mismatch_{domain}"]
        new = panel[f"multicentre_mismatch_{domain}"]
        valid = pd.DataFrame({"old": old, "new": new}).replace([np.inf, -np.inf], np.nan).dropna()
        summary.append({"domain": domain, "domain_label": label, "n_valid": len(valid),
                        "single_mean": valid.old.mean(), "multicentre_mean": valid.new.mean(),
                        "single_median": valid.old.median(), "multicentre_median": valid.new.median(),
                        "mean_difference_multi_minus_single": (valid.new - valid.old).mean(),
                        "median_difference_multi_minus_single": (valid.new - valid.old).median(),
                        "spearman_single_multicentre": spearmanr(valid.old, valid.new).statistic if len(valid) >= 8 else np.nan})
    pd.DataFrame(summary).to_csv(OUT / "multicentre_single_comparison.csv", index=False, encoding="utf-8-sig")
    diagnostics = {
        "cities": int(len(panel)),
        "population_center_count_median": float(panel.population_center_count.median()),
        "population_center_count_p90": float(panel.population_center_count.quantile(.9)),
        "population_center_count_multicentre_share": float((panel.population_center_count > 1).mean()),
    }
    for domain, _ in FUNCTIONS:
        status = panel[f"function_center_status_{domain}"].fillna("missing")
        diagnostics[f"{domain}_function_center_count_median"] = float(panel[f"function_center_count_{domain}"].replace(0, np.nan).median())
        diagnostics[f"{domain}_function_multicentre_share"] = float((panel[f"function_center_count_{domain}"] > 1).mean())
        diagnostics[f"{domain}_fallback_or_missing_share"] = float((~status.eq("multicentre")).mean())
    (OUT / "multicentre_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument("--limit-tiles", type=int, default=None,
                        help="Process only the first N spatial tiles; intended for validation runs.")
    parser.add_argument("--city-id", type=int, default=None,
                        help="Process only one city; intended for validation runs.")
    args = parser.parse_args()
    data = pd.read_csv(TABLE, low_memory=False)
    boundaries = gpd.read_file(BOUNDARIES, columns=["ID_UC_G0", "geometry"]).to_crs(4326)
    needed = data[["ID_UC_G0", "GC_UCA_KM2", "longitude", "latitude"]].copy()
    needed["ID_UC_G0"] = pd.to_numeric(needed.ID_UC_G0, errors="coerce").astype("Int64")
    boundaries["ID_UC_G0"] = pd.to_numeric(boundaries.ID_UC_G0, errors="coerce").astype("Int64")
    cities = boundaries.merge(needed, on="ID_UC_G0", how="inner")
    cities["tile_x"] = np.floor((cities.longitude + 180) / TILE_DEGREES).astype(int)
    cities["tile_y"] = np.floor((cities.latitude + 90) / TILE_DEGREES).astype(int)
    cities["tile_key"] = cities.tile_x.astype(str) + "_" + cities.tile_y.astype(str)
    if args.city_id is not None:
        cities = cities[cities.ID_UC_G0.eq(args.city_id)].copy()
        if cities.empty:
            raise ValueError(f"City ID {args.city_id} was not found in the merged city table.")
    all_groups = list(cities.groupby("tile_key", sort=True))
    if args.limit_tiles is not None:
        all_groups = all_groups[:max(args.limit_tiles, 0)]
    if not all_groups:
        raise ValueError("No spatial tiles remain after applying the test filters.")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be within shard-count")
    groups = [g for i, g in enumerate(all_groups) if i % args.shard_count == args.shard_index]
    print(f"Cities={len(cities)}; blocks={len(groups)}/{len(all_groups)}; shard={args.shard_index + 1}/{args.shard_count}", flush=True)
    connection = duckdb.connect()
    outputs = []
    start = time.time()
    for i, (tile, group) in enumerate(groups, 1):
        print(f"[{i}/{len(groups)}] tile={tile}, cities={len(group)}, elapsed={(time.time()-start)/60:.1f} min", flush=True)
        outputs.append(process_tile(tile, group, connection))
    if args.no_finalize:
        return
    if args.shard_count != 1:
        raise ValueError("Finalization must run without shard arguments")
    metrics = pd.concat(outputs, ignore_index=True).drop_duplicates("ID_UC_G0")
    metrics.to_csv(OUT / "multicentre_function_metrics_all_cities.csv", index=False, encoding="utf-8-sig")
    panel = add_outcomes(metrics, data)
    panel.to_csv(OUT / "multicentre_analysis_panel.csv", index=False, encoding="utf-8-sig")
    write_comparisons(panel)
    (OUT / "README_multicentre_method.txt").write_text(
        "V25 multicentre sensitivity analysis.\n\n"
        "Population centres are local density peaks in 2024 population grids after pop>10 filtering.\n"
        "Native Overture function centres are local POI-density peaks. Peaks must be at least 1 km apart\n"
        "and reach 30% of the city-domain maximum local density; sparse domains fall back to one centre.\n"
        "The multicentre mismatch is the population-weighted distance from each population centre to its\n"
        "nearest same-domain function centre, divided by max(sqrt(area/pi), 0.5 km).\n"
        "The original single-centre metric is recomputed from the same 2024 population grid and POIs\n"
        "for a paired comparison. No V25 primary figure is overwritten.\n"
        "Overture places are counts of mapped places, not facility capacity, visits, or travel-time routing.\n"
        "The associations are descriptive and conditional sensitivity analyses, not causal estimates.\n",
        encoding="utf-8",
    )
    print(f"Wrote multicentre outputs to {OUT}", flush=True)


if __name__ == "__main__":
    main()
