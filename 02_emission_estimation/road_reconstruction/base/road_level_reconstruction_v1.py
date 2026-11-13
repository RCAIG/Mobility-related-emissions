from __future__ import annotations

import argparse
import json
import math
import re
import time
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.collections import LineCollection
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EPS = 1e-12
YEARS = (2017, 2024)
CLASS_CUTS_KM = np.asarray([2.0, 5.0, 15.0, 40.0])
CLASS_LABELS = ("<2 km", "2-5 km", "5-15 km", "15-40 km", ">=40 km")
CLASS_COLS = [f"activity_c{i}" for i in range(1, 6)]

CITY_ORDER = (4012, 2335, 1290, 1301, 3381, 353)
CITY_META = {
    4012: {"name": "Quebec", "income": "high_income"},
    2335: {"name": "Cordoba", "income": "upper_middle"},
    1290: {"name": "Brussels", "income": "high_income"},
    1301: {"name": "Lubango", "income": "lower_middle"},
    3381: {"name": "Kagoshima", "income": "high_income"},
    353: {"name": "Christchurch", "income": "high_income"},
}

ROUTE_ROOTS = {
    "high_income": "${SERVER_HOME}/result/trace_genetare_hgih_income_{year}",
    "upper_middle": "${SERVER_HOME}/result/trace_genetare_upper_middle_{year}",
    "lower_middle": "${SERVER_HOME}/result/trace_genetare_lower_middle_{year}",
    "low_income": "${SERVER_HOME}/result/trace_genetare_low_income_{year}",
}
ROAD_ROOT = "${SERVER_HOME}/result/road_{year}"

GLOBAL_FEATURES = Path(
    "${SERVER_HOME}/result/spatial_mobility_carbon_validation_v2/analysis/"
    "global_cell_features_v2.parquet"
)
CITY_CELL_REFERENCE = Path(
    "${SERVER_HOME}/result/spatial_mobility_carbon_validation_v2/analysis/"
    "city_cell_activity_reference_v2.parquet"
)
CITY_DATA = Path(
    "${SERVER_HOME}/result/city_mobility_carbon_validation_v2/input/"
    "city_year_validation_dataset.csv"
)
SPATIAL_METRICS = Path(
    "${SERVER_HOME}/result/spatial_mobility_carbon_validation_v2/analysis/"
    "global_validation_metrics.csv"
)
CITY_METRICS = Path(
    "${SERVER_HOME}/result/city_mobility_carbon_validation_v2/analysis/"
    "validation_metrics.csv"
)

DEV_LEVELS = ["Low income", "Lower Middle", "Upper Middle", "High income"]
REGION_LEVELS = [
    "Eastern and South-Eastern Asia", "Central and Southern Asia", "Sub-Saharan Africa",
    "Europe", "Latin America and the Caribbean", "Northern Africa and Western Asia",
    "Northern America", "Australia and New Zealand", "Oceania",
]

SIZE = ["log_population", "log_area", "log_gdp"]
SOCIO = [
    "logistics_index", "vehicle_co2_intensity", "log_bev_ownership",
    "public_transport_efficiency", "train_service_efficiency", "transport_tax",
    "rise_index", "oil_share", "electricity_share",
]
ROAD = [
    "road_n_nodes", "road_n_edges", "road_total_length_km", "road_beta_index",
    "road_gamma_index", "road_alpha_index", "road_avg_node_degree",
    "road_dead_end_ratio", "road_highway_length_ratio", "road_primary_length_ratio",
    "road_secondary_length_ratio", "road_local_length_ratio",
    "road_hierarchy_entropy", "road_hierarchy_entropy_norm",
    "road_length_weighted_avg_speed", "road_speed_std",
    "road_road_density_km_per_km2", "road_highway_density_km_per_km2",
    "road_intersection_density_per_km2", "road_avg_edge_length_m",
    "road_edge_length_std_m",
]
CITY_CATEGORICAL = ["year_label", "development_group", "urban_type", "Global_NS"]


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def parse_linestring(text: str | None, fallback: np.ndarray) -> np.ndarray:
    if not text or not text.startswith("LINESTRING"):
        return fallback
    body = text[text.find("(") + 1:text.rfind(")")]
    try:
        coords = np.asarray([[float(v) for v in token.strip().split()[:2]] for token in body.split(",")])
        return coords if len(coords) >= 2 else fallback
    except Exception:
        return fallback


def coords_to_wkt(coords: np.ndarray) -> str:
    return "LINESTRING (" + ", ".join(f"{x:.7f} {y:.7f}" for x, y in coords) + ")"


def wkt_to_coords(text: str) -> np.ndarray:
    return parse_linestring(text, np.empty((0, 2), dtype=float))


def segment_edge(coords: np.ndarray, edge_length_km: float, max_segment_km: float = 0.5):
    if len(coords) < 2:
        return np.empty(0, np.int64), np.empty(0, float)
    raw = haversine_km(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1])
    raw_total = float(raw.sum())
    if raw_total <= 0:
        return np.empty(0, np.int64), np.empty(0, float)
    parts: defaultdict[int, float] = defaultdict(float)
    for a, b, raw_length in zip(coords[:-1], coords[1:], raw):
        n = max(1, int(math.ceil(float(raw_length) / max_segment_km)))
        fractions = (np.arange(n) + 0.5) / n
        mids = a[None, :] + fractions[:, None] * (b - a)[None, :]
        cols = np.floor((mids[:, 0] + 180.0) * 10.0).astype(np.int64)
        rows = np.floor((90.0 - mids[:, 1]) * 10.0).astype(np.int64)
        valid = (cols >= 0) & (cols < 3600) & (rows >= 0) & (rows < 1800)
        piece = edge_length_km * float(raw_length) / raw_total / n
        for cell_id in (rows[valid] * 3600 + cols[valid]).tolist():
            parts[int(cell_id)] += piece
    return (np.fromiter(parts.keys(), np.int64), np.fromiter(parts.values(), float))


def graph_pass(path: Path, active_codes: set[int] | None = None):
    key_names: dict[str, str] = {}
    node_index: dict[str, int] = {}
    node_xy: list[tuple[float, float]] = []
    selected: dict[int, tuple] = {}
    for _, element in ET.iterparse(path, events=("end",)):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "key":
            key_names[element.attrib["id"]] = element.attrib.get("attr.name", "")
            element.clear()
        elif tag == "node":
            values = {key_names.get(d.attrib.get("key", ""), ""): d.text for d in element
                      if d.tag.rsplit("}", 1)[-1] == "data"}
            node_index[element.attrib["id"]] = len(node_xy)
            node_xy.append((float(values["x"]), float(values["y"])))
            element.clear()
        elif tag == "edge":
            source, target = element.attrib["source"], element.attrib["target"]
            if source in node_index and target in node_index:
                code = (node_index[source] << 32) | node_index[target]
                if active_codes is None or code in active_codes:
                    values = {key_names.get(d.attrib.get("key", ""), ""): d.text for d in element
                              if d.tag.rsplit("}", 1)[-1] == "data"}
                    length = float(values.get("length") or 0) / 1000.0
                    impedance = float(values.get("travel_time") or length or np.inf)
                    if active_codes is None:
                        value = (impedance, length)
                    else:
                        fallback = np.asarray([node_xy[node_index[source]], node_xy[node_index[target]]], float)
                        coords = parse_linestring(values.get("geometry"), fallback)
                        value = (impedance, length, coords, float(values.get("speed_kph") or np.nan))
                    if code not in selected or impedance < selected[code][0]:
                        selected[code] = value
            element.clear()
    return selected


def route_files(city_id: int, year: int) -> list[Path]:
    income = CITY_META[city_id]["income"]
    root = Path(ROUTE_ROOTS[income].format(year=year))
    return sorted(root.glob(f"od_paths_flows_{city_id}_chunk_*.parquet"))


def road_path(city_id: int, year: int) -> Path:
    return Path(ROAD_ROOT.format(year=year)) / f"road_network_drive_{city_id}.graphml"


def accumulate_routes(files: list[Path], sorted_codes: np.ndarray, edge_lengths: np.ndarray):
    edge_class_flow = np.zeros((len(sorted_codes), 5), dtype=np.float64)
    route_count = matched_occurrences = unmatched_occurrences = 0
    total_flow = 0.0
    class_flow = np.zeros(5)
    class_routes = np.zeros(5)
    for file in files:
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=25_000, columns=["flow", "edge_sequence"]):
            frame = batch.to_pandas()
            sequences = frame.edge_sequence.to_numpy()
            flows = frame.flow.to_numpy(float)
            lens = np.fromiter((len(x) for x in sequences), np.int64, count=len(sequences))
            valid_route = lens > 0
            if not valid_route.any():
                continue
            sequences, flows, lens = sequences[valid_route], flows[valid_route], lens[valid_route]
            flat = np.concatenate(sequences).astype(np.int64, copy=False)
            positions = np.searchsorted(sorted_codes, flat)
            safe = np.minimum(positions, len(sorted_codes) - 1)
            matched = (positions < len(sorted_codes)) & (sorted_codes[safe] == flat)
            flat_lengths = np.zeros(len(flat), dtype=float)
            flat_lengths[matched] = edge_lengths[positions[matched]]
            starts = np.r_[0, np.cumsum(lens)[:-1]]
            route_lengths = np.add.reduceat(flat_lengths, starts)
            classes = np.searchsorted(CLASS_CUTS_KM, route_lengths, side="right")
            repeated_flow = np.repeat(flows, lens)
            repeated_class = np.repeat(classes, lens)
            valid = matched & np.isfinite(repeated_flow) & (repeated_flow > 0)
            keys = positions[valid] * 5 + repeated_class[valid]
            edge_class_flow += np.bincount(
                keys, weights=repeated_flow[valid], minlength=edge_class_flow.size
            ).reshape(edge_class_flow.shape)
            route_count += len(flows)
            total_flow += float(flows.sum())
            matched_occurrences += int(matched.sum())
            unmatched_occurrences += int((~matched).sum())
            class_flow += np.bincount(classes, weights=flows, minlength=5)
            class_routes += np.bincount(classes, minlength=5)
    return edge_class_flow, {
        "route_count": route_count,
        "total_flow": total_flow,
        "matched_edge_occurrences": matched_occurrences,
        "unmatched_edge_occurrences": unmatched_occurrences,
        **{f"flow_c{i + 1}": class_flow[i] for i in range(5)},
        **{f"routes_c{i + 1}": class_routes[i] for i in range(5)},
    }


def extract_city_year(city_id: int, year: int, output_root: str, overwrite: bool = False):
    started = time.time()
    root = Path(output_root)
    edge_out = root / "activity" / f"road_activity_{city_id}_{year}.parquet"
    cell_out = root / "activity" / f"edge_cell_activity_{city_id}_{year}.parquet"
    diag_out = root / "tables" / f"extraction_diagnostic_{city_id}_{year}.json"
    if edge_out.exists() and cell_out.exists() and diag_out.exists() and not overwrite:
        return {"city_id": city_id, "year": year, "status": "cached"}
    edge_out.parent.mkdir(parents=True, exist_ok=True)
    diag_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        road = road_path(city_id, year)
        files = route_files(city_id, year)
        if not road.exists() or not files:
            raise FileNotFoundError(f"road={road.exists()} route_chunks={len(files)}")
        selected = graph_pass(road)
        codes = np.asarray(sorted(selected), dtype=np.int64)
        lengths = np.asarray([selected[int(c)][1] for c in codes], float)
        edge_class_flow, diag = accumulate_routes(files, codes, lengths)
        active_idx = np.flatnonzero(edge_class_flow.sum(axis=1) > 0)
        active_codes = codes[active_idx]
        geometry = graph_pass(road, set(active_codes.tolist()))

        edge_rows = []
        cell_rows = []
        for idx in active_idx:
            code = int(codes[idx])
            if code not in geometry:
                continue
            _, length, coords, speed = geometry[code]
            if not np.isfinite(length) or length <= 0:
                length = float(haversine_km(
                    coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1]
                ).sum())
            flows = edge_class_flow[idx]
            edge_rows.append({
                "city_id": city_id, "year": year, "edge_code": code,
                "length_km": length, "speed_kph": speed,
                "geometry_wkt": coords_to_wkt(coords),
                **{f"edge_flow_c{i + 1}": flows[i] for i in range(5)},
            })
            cell_ids, cell_lengths = segment_edge(coords, length)
            for cell_id, cell_length in zip(cell_ids, cell_lengths):
                cell_rows.append({
                    "city_id": city_id, "year": year, "edge_code": code,
                    "cell_id": int(cell_id), "edge_length_in_cell_km": cell_length,
                    **{f"activity_c{i + 1}": flows[i] * cell_length for i in range(5)},
                })
        edge_frame = pd.DataFrame(edge_rows)
        cell_frame = pd.DataFrame(cell_rows)
        edge_frame.to_parquet(edge_out, index=False, compression="snappy")
        cell_frame.to_parquet(cell_out, index=False, compression="snappy")
        diag.update({
            "city_id": city_id, "city": CITY_META[city_id]["name"], "year": year,
            "route_chunks": len(files), "graph_edges": len(codes),
            "active_edges": len(edge_frame), "edge_cell_links": len(cell_frame),
            "runtime_seconds": time.time() - started, "status": "ok",
        })
        diag_out.write_text(json.dumps(diag, indent=2), encoding="utf-8")
        return diag
    except Exception as exc:
        return {"city_id": city_id, "year": year, "status": "failed",
                "error": repr(exc), "traceback": traceback.format_exc(),
                "runtime_seconds": time.time() - started}


def spatial_engineer(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    for col in CLASS_COLS:
        x[f"log_{col}"] = np.log1p(x[col].clip(lower=0))
        x[f"share_{col}"] = x[col] / np.maximum(x.activity_total, EPS)
    x["log_activity_total"] = np.log1p(x.activity_total)
    x["log_contributing_cities"] = np.log1p(x.contributing_cities)
    x["sin_lon"], x["cos_lon"] = np.sin(np.radians(x.lon)), np.cos(np.radians(x.lon))
    x["sin_lat"], x["cos_lat"] = np.sin(np.radians(x.lat)), np.cos(np.radians(x.lat))
    x["year_2024"] = x.year.eq(2024).astype(float)
    x["target"] = np.log1p(x.reference_emission.clip(lower=0))
    return x.replace([np.inf, -np.inf], np.nan)


SPATIAL_MOBILITY = (
    [f"log_{x}" for x in CLASS_COLS] + [f"share_{x}" for x in CLASS_COLS]
    + ["log_activity_total", "city_contribution_hhi", "log_contributing_cities"]
)
SPATIAL_CONTEXT = (
    ["weighted_log_population", "weighted_log_area", "weighted_log_gdp",
     "sin_lon", "cos_lon", "sin_lat", "cos_lat", "year_2024"]
    + [f"dev_{x}" for x in DEV_LEVELS] + [f"region_{x}" for x in REGION_LEVELS]
)


def spatial_model(seed: int = 42):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=220, max_leaf_nodes=28,
            min_samples_leaf=35, l2_regularization=1.2, random_state=seed,
        )),
    ])


def polynomial_design(d: pd.DataFrame, degree: int):
    z = np.log1p(d[CLASS_COLS].clip(lower=0).to_numpy(float))
    return np.c_[z, z ** 2] if degree == 2 else z


def select_polynomial_mapping(d: pd.DataFrame, out_dir: Path):
    groups = (np.floor((d.lon + 180) / 5).astype(int)
              + 100 * np.floor((d.lat + 90) / 5).astype(int)).to_numpy()
    rows = []
    for degree in (1, 2):
        pred = np.full(len(d), np.nan)
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(d, groups=groups), 1):
            reg = LinearRegression(positive=True)
            reg.fit(polynomial_design(d.iloc[tr], degree), d.target.iloc[tr])
            pred[te] = reg.predict(polynomial_design(d.iloc[te], degree))
        rows.append({
            "degree": degree, "validation": "held-out 5-degree blocks",
            "r2_log": r2_score(d.target, pred),
            "rmse_log": math.sqrt(mean_squared_error(d.target, pred)),
            "mae_log": mean_absolute_error(d.target, pred),
        })
    scores = pd.DataFrame(rows)
    scores.to_csv(out_dir / "polynomial_mapping_cross_validation.csv", index=False)
    degree = int(scores.sort_values(["rmse_log", "degree"]).iloc[0].degree)
    reg = LinearRegression(positive=True).fit(polynomial_design(d, degree), d.target)
    coefficient_rows = []
    for i, label in enumerate(CLASS_LABELS):
        coefficient_rows.append({
            "class": i + 1, "distance_range": label,
            "linear_log_activity_coefficient": float(reg.coef_[i]),
            "quadratic_log_activity_coefficient": float(reg.coef_[i + 5]) if degree == 2 else 0.0,
            "selected_degree": degree,
        })
    coefficients = pd.DataFrame(coefficient_rows)
    coefficients.to_csv(out_dir / "distance_class_response_coefficients.csv", index=False)
    return degree, reg, coefficients, scores


def city_engineer(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["year_label"] = x.year.astype(str)
    return x.replace([np.inf, -np.inf], np.nan)


def city_preprocessor(train, test, numeric, categorical):
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    x = pre.fit_transform(train[numeric + categorical]).astype(np.float32)
    z = pre.transform(test[numeric + categorical]).astype(np.float32)
    return x, z


def predict_city_totals(data_path: Path, gpu: str, out_dir: Path):
    d = city_engineer(pd.read_csv(data_path))
    features = SIZE + SOCIO + ROAD
    rows = []
    for year in YEARS:
        year_data = d[d.year.eq(year)].dropna(subset=["log_transport_ce"] + SIZE).copy()
        test = year_data[year_data.ID_UC_G0.isin(CITY_ORDER)].copy()
        train = year_data[~year_data.ID_UC_G0.isin(CITY_ORDER)].copy()
        model_name = "TabPFN Size + context (six cities excluded)"
        try:
            from tabpfn import TabPFNRegressor
            x, z = city_preprocessor(train, test, features, CITY_CATEGORICAL)
            reg = TabPFNRegressor(
                n_estimators=4, device=gpu, ignore_pretraining_limits=True,
                fit_mode="fit_preprocessors", random_state=42,
            )
            reg.fit(x, train.log_transport_ce.to_numpy(np.float32))
            pred = reg.predict(z)
        except Exception as exc:
            model_name = f"HGBR Size + context fallback ({type(exc).__name__})"
            pre = ColumnTransformer([
                ("num", SimpleImputer(strategy="median"), features),
                ("cat", Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), CITY_CATEGORICAL),
            ])
            reg = Pipeline([("pre", pre), ("model", HistGradientBoostingRegressor(
                learning_rate=.05, max_iter=240, max_leaf_nodes=24,
                min_samples_leaf=30, l2_regularization=1.0, random_state=42,
            ))])
            reg.fit(train[features + CITY_CATEGORICAL], train.log_transport_ce)
            pred = reg.predict(test[features + CITY_CATEGORICAL])
        for (_, row), value in zip(test.iterrows(), pred):
            rows.append({
                "city_id": int(row.ID_UC_G0), "city": row.GC_UCN_MAI_2025,
                "year": year, "predicted_log_total": float(value),
                "predicted_total": max(0.0, float(np.expm1(value))),
                "observed_total_not_used_for_calibration": float(row.transport_ce),
                "model": model_name,
            })
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "city_total_predictions_excluding_examples.csv", index=False)
    return result


def response_shares(activity: np.ndarray, degree: int, reg: LinearRegression):
    z = np.log1p(np.maximum(activity, 0))
    response = z * reg.coef_[:5]
    if degree == 2:
        response += z ** 2 * reg.coef_[5:]
    fallback = activity / np.maximum(activity.sum(axis=1, keepdims=True), EPS)
    total = response.sum(axis=1, keepdims=True)
    shares = np.divide(response, total, out=np.zeros_like(response), where=total > EPS)
    shares[total[:, 0] <= EPS] = fallback[total[:, 0] <= EPS]
    return shares


def calibrate(output_root: Path, gpu: str):
    table_dir = output_root / "tables"
    result_dir = output_root / "results"
    figure_dir = output_root / "figures"
    for directory in (table_dir, result_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    global_raw = pd.read_parquet(GLOBAL_FEATURES)
    global_data = spatial_engineer(global_raw)
    features = SPATIAL_MOBILITY + SPATIAL_CONTEXT
    spatial = spatial_model().fit(global_data[features], global_data.target)
    global_data["spatial_predicted_log"] = spatial.predict(global_data[features])
    global_data["spatial_predicted_emission"] = np.maximum(
        0.0, np.expm1(global_data.spatial_predicted_log)
    )
    global_data[["cell_id", "year", "lon", "lat", "activity_total", "reference_emission",
                 "spatial_predicted_log", "spatial_predicted_emission"]].to_parquet(
        result_dir / "full_fit_spatial_predictions.parquet", index=False
    )

    degree, response_model, coefficients, poly_scores = select_polynomial_mapping(global_data, table_dir)
    city_totals = predict_city_totals(CITY_DATA, gpu, table_dir)

    city_reference = pd.read_parquet(
        CITY_CELL_REFERENCE,
        columns=["cell_id", "year", "city_id", "activity_total"],
    )
    city_reference = city_reference[city_reference.city_id.isin(CITY_ORDER)].copy()
    city_reference = city_reference.rename(columns={"activity_total": "reference_city_activity"})

    audits = []
    for city_id in CITY_ORDER:
        for year in YEARS:
            edge_path = output_root / "activity" / f"road_activity_{city_id}_{year}.parquet"
            link_path = output_root / "activity" / f"edge_cell_activity_{city_id}_{year}.parquet"
            edges = pd.read_parquet(edge_path)
            links = pd.read_parquet(link_path)
            grid = links.groupby(["city_id", "year", "cell_id"], as_index=False)[CLASS_COLS].sum()
            grid["road_activity_total"] = grid[CLASS_COLS].sum(axis=1)
            grid = grid.merge(
                global_data[["cell_id", "year", "activity_total", "spatial_predicted_emission"]],
                on=["cell_id", "year"], how="left",
            ).merge(
                city_reference, on=["cell_id", "year", "city_id"], how="left",
            )
            grid["city_share"] = grid.reference_city_activity / np.maximum(grid.activity_total, EPS)
            fallback_share = grid.road_activity_total / np.maximum(grid.activity_total, EPS)
            grid["city_share"] = grid.city_share.fillna(fallback_share).clip(0, 1)
            missing_prediction = grid.spatial_predicted_emission.isna()
            if missing_prediction.any():
                valid = ~missing_prediction & grid.road_activity_total.gt(0)
                ratio = np.nanmedian(
                    grid.loc[valid, "spatial_predicted_emission"]
                    / grid.loc[valid, "road_activity_total"]
                )
                grid.loc[missing_prediction, "spatial_predicted_emission"] = (
                    grid.loc[missing_prediction, "road_activity_total"] * max(float(ratio), EPS)
                )
                grid.loc[missing_prediction, "city_share"] = 1.0
            grid["raw_city_grid_emission"] = grid.spatial_predicted_emission * grid.city_share
            target_row = city_totals[(city_totals.city_id.eq(city_id)) & (city_totals.year.eq(year))]
            if target_row.empty:
                raise ValueError(f"No city total prediction for {city_id}, {year}")
            city_target = float(target_row.predicted_total.iloc[0])
            raw_total = float(grid.raw_city_grid_emission.sum())
            if raw_total <= 0:
                grid["raw_city_grid_emission"] = grid.road_activity_total
                raw_total = float(grid.raw_city_grid_emission.sum())
            scale = city_target / max(raw_total, EPS)
            grid["calibrated_grid_emission"] = grid.raw_city_grid_emission * scale
            shares = response_shares(grid[CLASS_COLS].to_numpy(float), degree, response_model)
            for i in range(5):
                grid[f"class_emission_c{i + 1}"] = grid.calibrated_grid_emission * shares[:, i]
            grid.to_parquet(result_dir / f"calibrated_grid_emission_{city_id}_{year}.parquet", index=False)

            merge_cols = ["cell_id"] + [f"class_emission_c{i}" for i in range(1, 6)]
            allocation = links.merge(grid[merge_cols], on="cell_id", how="left")
            denominators = grid.set_index("cell_id")[CLASS_COLS]
            for i in range(1, 6):
                denom = allocation.cell_id.map(denominators[f"activity_c{i}"]).to_numpy(float)
                allocation[f"edge_emission_c{i}"] = (
                    allocation[f"class_emission_c{i}"].to_numpy(float)
                    * allocation[f"activity_c{i}"].to_numpy(float)
                    / np.maximum(denom, EPS)
                )
            emission_cols = [f"edge_emission_c{i}" for i in range(1, 6)]
            edge_emission = allocation.groupby("edge_code", as_index=False)[emission_cols].sum()
            edge_emission["reconstructed_emission"] = edge_emission[emission_cols].sum(axis=1)
            edge_result = edges.merge(edge_emission, on="edge_code", how="left")
            edge_result[emission_cols + ["reconstructed_emission"]] = (
                edge_result[emission_cols + ["reconstructed_emission"]].fillna(0)
            )
            edge_result["emission_per_road_km"] = (
                edge_result.reconstructed_emission / np.maximum(edge_result.length_km, EPS)
            )
            edge_result["total_assigned_flow"] = edge_result[
                [f"edge_flow_c{i}" for i in range(1, 6)]
            ].sum(axis=1)
            edge_result.to_parquet(
                result_dir / f"road_emission_{city_id}_{year}.parquet",
                index=False, compression="snappy",
            )
            road_total = float(edge_result.reconstructed_emission.sum())
            audits.append({
                "city_id": city_id, "city": CITY_META[city_id]["name"], "year": year,
                "active_edges": len(edge_result), "active_cells": len(grid),
                "spatial_raw_total": raw_total, "city_model_total": city_target,
                "city_reconciliation_factor": scale, "road_reaggregated_total": road_total,
                "absolute_conservation_error": abs(road_total - city_target),
                "relative_conservation_error": abs(road_total - city_target) / max(city_target, EPS),
                "spatial_model": "HGBR mobility + city context, full-data reconstruction fit",
                "city_model": target_row.model.iloc[0],
                "distance_response_degree": degree,
            })
    audit = pd.DataFrame(audits)
    audit.to_csv(table_dir / "road_reconstruction_conservation_audit.csv", index=False)
    copy_validation_evidence(table_dir)
    plot_response_curves(global_data, degree, response_model, coefficients, figure_dir)
    plot_road_maps(output_root, audit)
    return audit


def copy_validation_evidence(table_dir: Path):
    spatial = pd.read_csv(SPATIAL_METRICS)
    spatial = spatial[(spatial.model.eq("HGBR"))
                      & (spatial.feature_set.eq("Mobility + city context"))]
    spatial.to_csv(table_dir / "source_spatial_validation_evidence.csv", index=False)
    city = pd.read_csv(CITY_METRICS)
    city = city[(city.model.eq("TabPFN")) & (city.feature_set.eq("Size + context"))]
    city.to_csv(table_dir / "source_city_validation_evidence.csv", index=False)


def set_plot_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13.5,
        "axes.labelsize": 12, "text.color": "#202B33", "axes.labelcolor": "#202B33",
        "axes.titlecolor": "#202B33", "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def plot_response_curves(d, degree, model, coefficients, figure_dir):
    set_plot_style()
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.4), constrained_layout=True)
    colors = ["#287271", "#3A7D8C", "#6B8E75", "#C17C3A", "#9B4A3F"]
    for i, (ax, label, color) in enumerate(zip(axes, CLASS_LABELS, colors)):
        positive = d.loc[d[CLASS_COLS[i]].gt(0), CLASS_COLS[i]]
        hi = float(positive.quantile(.99)) if len(positive) else 1.0
        a = np.geomspace(max(float(positive.quantile(.02)), 1e-4), max(hi, 1e-3), 160)
        z = np.log1p(a)
        response = model.coef_[i] * z
        if degree == 2:
            response += model.coef_[i + 5] * z ** 2
        ax.plot(a, response, color=color, linewidth=2.3)
        ax.fill_between(a, 0, response, color=color, alpha=.13)
        ax.set_xscale("log")
        ax.set_title(f"C{i + 1}  {label}", loc="left", fontweight="bold")
        ax.set_xlabel("Class activity (log scale)")
        if i == 0:
            ax.set_ylabel("Fitted log-emission response")
        ax.grid(color="#D8DEDE", linewidth=.55, alpha=.75)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Empirical distance-class response used for within-grid allocation",
                 fontsize=15, fontweight="bold")
    fig.savefig(figure_dir / "Figure_distance_class_emission_response.png", dpi=320, bbox_inches="tight")
    fig.savefig(figure_dir / "Figure_distance_class_emission_response.pdf", bbox_inches="tight")
    plt.close(fig)


def map_norm(output_root: Path):
    values = []
    for city_id in CITY_ORDER:
        for year in YEARS:
            p = output_root / "results" / f"road_emission_{city_id}_{year}.parquet"
            x = pd.read_parquet(p, columns=["emission_per_road_km"])
            x = x.emission_per_road_km.to_numpy(float)
            values.append(x[np.isfinite(x) & (x > 0)])
    all_values = np.concatenate(values)
    log_values = np.log10(all_values)
    return mpl.colors.Normalize(np.quantile(log_values, .02), np.quantile(log_values, .995))


ROAD_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "road_carbon", ["#C9D8D5", "#5E9A96", "#176B70", "#D18A3C", "#9E3D32"]
)
YEAR_BORDER = {2017: "#497C83", 2024: "#B65C43"}


def draw_road_panel(ax, data, city_id, year, norm, show_total=True):
    coords, values, flows = [], [], []
    for row in data.itertuples(index=False):
        xy = wkt_to_coords(row.geometry_wkt)
        if len(xy) >= 2 and row.reconstructed_emission > 0:
            coords.append(xy)
            values.append(math.log10(max(row.emission_per_road_km, EPS)))
            flows.append(row.total_assigned_flow)
    if not coords:
        ax.text(.5, .5, "No reconstructed links", ha="center", va="center")
        return
    values = np.asarray(values)
    flows = np.log1p(np.asarray(flows))
    widths = .16 + .70 * (flows - flows.min()) / max(float(np.ptp(flows)), EPS)
    background = LineCollection(coords, colors="#DDE2E0", linewidths=.22, alpha=.55, zorder=1)
    foreground = LineCollection(coords, cmap=ROAD_CMAP, norm=norm, linewidths=widths,
                                alpha=.84, zorder=2, capstyle="round")
    foreground.set_array(values)
    ax.add_collection(background)
    ax.add_collection(foreground)
    all_xy = np.vstack(coords)
    lo, hi = np.quantile(all_xy, [.005, .995], axis=0)
    dx, dy = max(hi[0] - lo[0], .01), max(hi[1] - lo[1], .01)
    ax.set_xlim(lo[0] - .04 * dx, hi[0] + .04 * dx)
    ax.set_ylim(lo[1] - .04 * dy, hi[1] + .04 * dy)
    midlat = float(np.mean([lo[1], hi[1]]))
    ax.set_aspect(1 / max(math.cos(math.radians(midlat)), .2), adjustable="datalim")
    ax.set_anchor("C")
    total = float(data.reconstructed_emission.sum())
    ax.set_title(f"{CITY_META[city_id]['name']}  |  {year}", loc="left", pad=6, fontweight="bold")
    if show_total:
        ax.text(.015, .025, f"Reconstructed total: {total:,.0f}", transform=ax.transAxes,
                fontsize=9.4, color="#34464D", bbox=dict(facecolor="white", alpha=.78,
                edgecolor="none", pad=2.0), zorder=4)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(YEAR_BORDER[year]); spine.set_linewidth(.65)


def plot_road_maps(output_root: Path, audit: pd.DataFrame):
    set_plot_style()
    figure_dir = output_root / "figures"
    norm = map_norm(output_root)
    tasks = [(city_id, year) for year in YEARS for city_id in CITY_ORDER]
    fig, axes = plt.subplots(3, 4, figsize=(15.6, 11.35))
    for ax, (city_id, year) in zip(axes.flat, tasks):
        data = pd.read_parquet(output_root / "results" / f"road_emission_{city_id}_{year}.parquet")
        draw_road_panel(ax, data, city_id, year, norm)
    fig.subplots_adjust(left=.035, right=.985, top=.900, bottom=.145, wspace=.065, hspace=.17)
    fig.suptitle("Reconstructed road-network transport CO$_2$ emissions",
                 fontsize=16.5, fontweight="bold", y=.970)
    cax = fig.add_axes([.25, .072, .50, .017])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=ROAD_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Road emission intensity (log$_{10}$ reconstructed emission per road km)")
    fig.text(.035, .016,
             "Border: 2017 (teal), 2024 (vermillion). Width indicates assigned route flow. "
             "Values are model-constrained reconstructions, not link-level observations.",
             fontsize=10.3, color="#526168")
    fig.savefig(figure_dir / "Figure_reconstructed_road_emissions_6cities_2017_2024.png",
                dpi=320, bbox_inches="tight")
    fig.savefig(figure_dir / "Figure_reconstructed_road_emissions_6cities_2017_2024.pdf",
                bbox_inches="tight")
    plt.close(fig)

    individual = figure_dir / "individual"
    individual.mkdir(exist_ok=True)
    for city_id, year in tasks:
        data = pd.read_parquet(output_root / "results" / f"road_emission_{city_id}_{year}.parquet")
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        draw_road_panel(ax, data, city_id, year, norm)
        fig.subplots_adjust(left=.035, right=.975, top=.925, bottom=.11)
        cax = fig.add_axes([.20, .045, .60, .022])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=ROAD_CMAP), cax=cax,
                          orientation="horizontal")
        cb.set_label("Log$_{10}$ road emission intensity")
        fig.savefig(individual / f"Road_emission_{city_id}_{CITY_META[city_id]['name']}_{year}.png",
                    dpi=320, bbox_inches="tight")
        plt.close(fig)


def write_readme(output_root: Path):
    text = """# Road-level mobility-carbon reconstruction V1

## Scientific interpretation

The outputs are **reconstructed road-level transport CO2 emissions constrained by simulated
mobility, the validated 0.1-degree spatial model, and the validated city-total model**. They are
not observations of emissions on individual road links.

## Estimation chain

1. Each simulated route is assigned to one of five network-distance classes: <2, 2-5, 5-15,
   15-40, and >=40 km. Route flow is accumulated on every traversed directed road edge.
2. Edge flow is multiplied by the portion of edge length in each 0.1-degree cell, producing
   class-specific road activity. This follows the activity-by-distance logic of link-based road
   emission inventories, while avoiding unsupported assumptions about local fleet and speed cycles.
3. The selected spatial model is HGBR with mobility and city context. Its strict held-out
   5-degree-block R2 reached 0.604 (seed 42), exceeding the corresponding TabPFN result (0.592).
4. A non-negative first- or second-order polynomial in log class activity is selected by
   five-fold held-out 5-degree-block validation. It supplies only the relative class response used
   to allocate a cell total; it is not interpreted as a causal or laboratory emission factor.
5. The selected city-total model is TabPFN with size and socioeconomic/road context. The six
   displayed cities are excluded when their totals are predicted. Published validation files show
   R2=0.733 in 2017 and R2=0.688 in 2024 for this model family.
6. Spatial predictions determine the within-city distribution. A single multiplicative city
   reconciliation factor makes the sum of all road links equal the held-out city-total prediction.
   Within each cell, emissions are allocated to edges by class-specific activity and the empirical
   polynomial response. Conservation is checked in `road_reconstruction_conservation_audit.csv`.

## Why TabPFN is not used directly on road links

There is no link-level emission truth for training. Direct road-link regression would therefore
create pseudo-precision and would not guarantee that link predictions sum to the validated grid and
city totals. Machine learning is used only at the scales where a reference target exists; explicit
mass-preserving reconciliation performs the downscaling.

## Key limitations

- EDGAR at 0.1 degree is a gridded reference, not road-link truth.
- The five categories are network-distance proxies. They must not be renamed as observed transport
  modes unless route-level mode predictions are joined in a later version.
- Flow-weighted distance captures activity but not local fleet composition, congestion, gradients,
  cold starts, or speed trajectories.
- The class response is associational. The grid total is identifiable; the decomposition among five
  classes is not unique without independent mode/fleet observations.
- Validation performance must be quoted from held-out spatial and city tests. The full-data spatial
  fit used for reconstruction is not an independent validation result.

## Methodological anchors

- Smit, Ntziachristos & Boulter (2010), *Atmospheric Environment* 44, 2943-2953,
  https://doi.org/10.1016/j.atmosenv.2010.05.022.
- Barth & Boriboonsomsin (2008), *Transportation Research Record* 2058, 163-171,
  https://doi.org/10.3141/2058-20.
- EMEP/EEA Air Pollutant Emission Inventory Guidebook, Road Transport chapter.
- US EPA MOVES and the Handbook Emission Factors for Road Transport (HBEFA), as reference
  frameworks for future speed-, fleet-, and fuel-resolved extensions.
- Roberts et al. (2017), *Ecography* 40, 913-929, https://doi.org/10.1111/ecog.02881,
  for spatially structured cross-validation.
- Hollmann et al. (2025), *Nature* 637, 319-326,
  https://doi.org/10.1038/s41586-024-08328-6, for TabPFN.

## Main outputs

- `activity/`: edge-level class flow and edge-cell class activity.
- `results/`: calibrated cell totals and reconstructed directed-edge emissions.
- `tables/`: model evidence, polynomial selection, city totals, and conservation audit.
- `figures/`: response curves, 12 individual maps, and the six-city 2017/2024 panel.
"""
    (output_root / "README_method.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=["extract", "calibrate", "plot", "all"], default="all")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--city-ids", nargs="+", type=int, default=list(CITY_ORDER))
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_readme(args.output_root)
    if args.stage in ("extract", "all"):
        tasks = [(c, y, str(args.output_root), args.overwrite)
                 for c in args.city_ids for y in args.years]
        log_path = args.output_root / "tables" / "extraction_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        current_results = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor, log_path.open("a", encoding="utf-8") as log:
            futures = {executor.submit(extract_city_year, *task): task for task in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                current_results.append(result)
                log.write(json.dumps(result) + "\n"); log.flush()
                print(f"[{i}/{len(tasks)}] {result.get('city_id')} {result.get('year')} "
                      f"{result.get('status')} {result.get('runtime_seconds', 0):.1f}s "
                      f"{result.get('error', '')}", flush=True)
        failed = [result for result in current_results if result.get("status") == "failed"]
        if failed:
            raise RuntimeError(f"Extraction failures: {failed}")
    if args.stage in ("calibrate", "all"):
        audit = calibrate(args.output_root, args.gpu)
        print(audit.to_string(index=False), flush=True)
    elif args.stage == "plot":
        audit = pd.read_csv(args.output_root / "tables" / "road_reconstruction_conservation_audit.csv")
        plot_road_maps(args.output_root, audit)


if __name__ == "__main__":
    main()
