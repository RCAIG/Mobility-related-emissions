"""Country-balanced WorldMove validation using original 2024 OD matrices.

All four World Bank income groups are read from the same sparse OD schema.
Up to five high-population matched cities are retained per country so that
large countries do not dominate the external validation.
"""

import io
import json
from pathlib import Path
import re
import unicodedata
import zipfile

import numpy as np
import pandas as pd


ZIP_PATH = Path("${SERVER_HOME}/worldmove/data.zip")
WORK = Path("${SERVER_HOME}/worldmove")
OUT = WORK / "worldmove_validation_all_income"
CATALOG = WORK / "global_city_catalog.csv"
CODE_MAP = WORK / "country_code_to_iso3.json"
OD_RESULT_ROOT = Path(
    "${SERVER_DATA_ROOT}/prediction_travel_mode/"
    "generate-od-pubtools-main/generate-od-pubtools-main/assets/"
    "example_data/my_test/result"
)
ROOTS = {
    "low_income": OD_RESULT_ROOT / "od_flows_wgs84_low_income_2024",
    "lower_middle": OD_RESULT_ROOT / "od_flows_wgs84_lower_middle_2024",
    "upper_middle": OD_RESULT_ROOT / "od_flows_wgs84_upper_middle_2024",
    "high_income": OD_RESULT_ROOT / "od_flows_wgs84_high_income_2024",
}
DISTANCE_BINS_KM = np.asarray(
    [0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, np.inf], dtype=float
)
MAX_CITIES_PER_COUNTRY = 5
MATCH_RADIUS_KM = 120.0
RNG_SEED = 20261023
FILE_RE = re.compile(r"(\d+)_od_flows_wgs84\.csv$")
PREFERRED_EXAMPLES = [
    "Philadelphia", "Belo Horizonte", "Birmingham", "Madrid", "Kinshasa",
    "Mumbai", "Chongqing", "Singapore", "Ankara",
]


def normalize(value):
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.minimum(1, np.sqrt(a)))


def js_divergence(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p / max(p.sum(), 1e-15), q / max(q.sum(), 1e-15)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-15))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cpc(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p / max(p.sum(), 1e-15), q / max(q.sum(), 1e-15)
    return float(np.minimum(p, q).sum())


def rank_bins(values, n_bins=20):
    values = np.asarray(values, float)
    values = np.sort(values[np.isfinite(values) & (values > 0)])[::-1]
    if not len(values):
        return np.zeros(n_bins)
    out = np.asarray([part.sum() for part in np.array_split(values, n_bins)])
    return out / max(out.sum(), 1e-15)


def power_correct(model, target):
    model, target = np.asarray(model, float), np.asarray(target, float)
    best_p, best_loss = 1.0, np.inf
    for exponent in np.linspace(0.2, 5.0, 481):
        trial = np.power(np.maximum(model, 1e-15), exponent)
        trial /= max(trial.sum(), 1e-15)
        loss = np.sum((trial - target) ** 2)
        if loss < best_loss:
            best_p, best_loss = float(exponent), float(loss)
    corrected = np.power(np.maximum(model, 1e-15), best_p)
    return corrected / max(corrected.sum(), 1e-15), best_p


def load_worldmove_grid(zf, city_key):
    raw = json.loads(zf.read(f"mobility_data/grid/{city_key}.json").decode("utf-8"))
    n = max(map(int, raw)) + 1
    xy = np.full((n, 2), np.nan)
    for key, value in raw.items():
        xy[int(key)] = value[:2]
    return xy


def endpoint_matrix(traj, n_locations):
    oi, di = traj[:, 0].astype(int), traj[:, -1].astype(int)
    valid = (oi >= 0) & (di >= 0) & (oi < n_locations) & (di < n_locations)
    out = np.zeros((n_locations, n_locations), dtype=np.float32)
    np.add.at(out, (oi[valid], di[valid]), 1)
    return out


def trajectory_distributions(traj, xy, mask):
    data = traj[mask]
    oi, di = data[:, 0].astype(int), data[:, -1].astype(int)
    valid = (oi >= 0) & (di >= 0) & (oi < len(xy)) & (di < len(xy))
    oi, di = oi[valid], di[valid]
    valid = np.isfinite(xy[oi]).all(axis=1) & np.isfinite(xy[di]).all(axis=1)
    oi, di = oi[valid], di[valid]
    origin = np.bincount(oi, minlength=len(xy)).astype(float)
    destination = np.bincount(di, minlength=len(xy)).astype(float)
    distance = haversine_km(xy[oi, 0], xy[oi, 1], xy[di, 0], xy[di, 1])
    hist = np.histogram(distance, bins=DISTANCE_BINS_KM)[0].astype(float)
    return origin, destination, hist


def build_model_index():
    index = {}
    for income, root in ROOTS.items():
        for path in root.glob("*_od_flows_wgs84.csv"):
            match = FILE_RE.match(path.name)
            if match:
                index.setdefault(int(match.group(1)), (income, path))
    return index


def model_distributions(city_id, matrix_path):
    od = pd.read_csv(
        matrix_path,
        usecols=["origin_id", "dest_id", "o_lon", "o_lat", "d_lon", "d_lat", "flow"],
    )
    od = od.replace([np.inf, -np.inf], np.nan).dropna()
    od = od[od.flow > 0]
    flow = od.flow.to_numpy(float)
    origin = od.groupby("origin_id", sort=False).flow.sum().to_numpy(float)
    destination = od.groupby("dest_id", sort=False).flow.sum().to_numpy(float)
    distance = haversine_km(
        od.o_lon.to_numpy(float), od.o_lat.to_numpy(float),
        od.d_lon.to_numpy(float), od.d_lat.to_numpy(float),
    )
    distance_hist = np.histogram(distance, bins=DISTANCE_BINS_KM, weights=flow)[0]
    n_zones = int(max(od.origin_id.max(), od.dest_id.max()) + 1)
    return origin, destination, distance_hist, float(flow.sum()), n_zones


def match_and_select(zf, catalog, code_map, model_index):
    catalog = catalog[catalog.ID_UC_G0.isin(model_index)].copy()
    catalog["norm"] = catalog.city_name.map(normalize)
    by_country = {code: group.reset_index(drop=True) for code, group in catalog.groupby("iso3")}
    matches, coverage = [], []
    members = [n for n in zf.namelist() if re.fullmatch(r"mobility_data/traj/.+\.npy", n)]
    for member in members:
        key = Path(member).stem
        parts = key.split("_", 2)
        if len(parts) != 3:
            continue
        raw_code, raw_city = parts[1].upper(), parts[2].replace("_", " ")
        iso3 = code_map.get(raw_code)
        if not iso3 or iso3 not in by_country:
            coverage.append({"worldmove_city": key, "iso3": iso3 or "", "status": "no_local_country"})
            continue
        wm_xy = load_worldmove_grid(zf, key)
        center = np.nanmean(wm_xy, axis=0)
        candidates = by_country[iso3]
        dist = haversine_km(center[0], center[1], candidates.longitude, candidates.latitude)
        exact = candidates["norm"].to_numpy() == normalize(raw_city)
        eligible = exact | (dist <= MATCH_RADIUS_KM)
        if not eligible.any():
            coverage.append({"worldmove_city": key, "iso3": iso3, "status": "no_city_within_radius"})
            continue
        score = np.where(exact, dist - 10000.0, dist)
        score[~eligible] = np.inf
        local = candidates.iloc[int(np.argmin(score))]
        matches.append({
            "wm": key, "id": int(local.ID_UC_G0), "city": local.city_name,
            "worldmove_city_name": raw_city, "country": local.country, "iso3": iso3,
            "lon": float(center[0]), "lat": float(center[1]),
            "match_distance_km": float(dist[int(np.argmin(score))]),
            "exact_name": bool(exact[int(np.argmin(score))]),
            "population_2024": float(local.GC_POP_TOT_2024),
            "income_group": model_index[int(local.ID_UC_G0)][0],
            "matrix_path": str(model_index[int(local.ID_UC_G0)][1]),
        })
    matches = pd.DataFrame(matches).sort_values(["match_distance_km", "population_2024"], ascending=[True, False])
    matches = matches.drop_duplicates("id")
    selected = []
    for _, group in matches.groupby("iso3", sort=True):
        group = group.sort_values("population_2024", ascending=False)
        selected.append(group.head(MAX_CITIES_PER_COUNTRY))
    selected = pd.concat(selected, ignore_index=True)
    return matches, selected, pd.DataFrame(coverage)


def choose_examples(selected):
    chosen = []
    normalized = selected.worldmove_city_name.map(normalize)
    for preferred in PREFERRED_EXAMPLES:
        hit = selected[normalized == normalize(preferred)]
        if len(hit):
            chosen.append(hit.sort_values("population_2024", ascending=False).iloc[0])
    if len(chosen) < 9:
        used = {row.iso3 for row in chosen}
        for _, row in selected.sort_values("population_2024", ascending=False).iterrows():
            if row.iso3 not in used:
                chosen.append(row); used.add(row.iso3)
            if len(chosen) == 9:
                break
    return chosen[:9]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(CATALOG)
    catalog["ID_UC_G0"] = catalog.ID_UC_G0.astype(int)
    code_map = json.loads(CODE_MAP.read_text())
    model_index = build_model_index()
    rng = np.random.default_rng(RNG_SEED)
    records, example_payload, example_labels = [], {}, {}
    with zipfile.ZipFile(ZIP_PATH) as zf:
        all_matches, selected, unmatched = match_and_select(zf, catalog, code_map, model_index)
        selected.to_csv(OUT / "worldmove_country_balanced_manifest.csv", index=False)
        all_matches.to_csv(OUT / "worldmove_all_candidate_matches.csv", index=False)
        unmatched.to_csv(OUT / "worldmove_unmatched_audit.csv", index=False)
        examples = choose_examples(selected)
        for number, (_, row) in enumerate(selected.iterrows(), 1):
            try:
                traj = np.load(io.BytesIO(zf.read(f"mobility_data/traj/{row.wm}.npy")), allow_pickle=False)
                wm_xy = load_worldmove_grid(zf, row.wm)
                model_o, model_d, model_dist, model_total, model_grid_cells = model_distributions(
                    int(row.id), Path(row.matrix_path)
                )
                split = rng.random(len(traj)) < 0.5
                cal_o, cal_d, cal_dist = trajectory_distributions(traj, wm_xy, split)
                test_o, test_d, test_dist = trajectory_distributions(traj, wm_xy, ~split)
                dimensions = {
                    "origin": (rank_bins(model_o), rank_bins(cal_o), rank_bins(test_o)),
                    "destination": (rank_bins(model_d), rank_bins(cal_d), rank_bins(test_d)),
                    "distance": (
                        model_dist / max(model_dist.sum(), 1e-15),
                        cal_dist / max(cal_dist.sum(), 1e-15),
                        test_dist / max(test_dist.sum(), 1e-15),
                    ),
                }
                result = {
                    "worldmove_city": row.wm, "city": row.city, "country": row.country,
                    "iso3": row.iso3, "id_uc_g0": int(row.id),
                    "income_group": row.income_group, "match_distance_km": row.match_distance_km,
                    "worldmove_trajectories": len(traj), "model_od_total": model_total,
                    "model_grid_cells": model_grid_cells,
                }
                for label, (model, calibration, test) in dimensions.items():
                    corrected, exponent = power_correct(model, calibration)
                    result[f"{label}_cpc_raw"] = cpc(model, test)
                    result[f"{label}_js_raw"] = js_divergence(model, test)
                    result[f"{label}_cpc_calibrated_holdout"] = cpc(corrected, test)
                    result[f"{label}_js_calibrated_holdout"] = js_divergence(corrected, test)
                    result[f"{label}_power"] = exponent
                records.append(result)
            except Exception as exc:
                print("SKIP", row.wm, repr(exc), flush=True)
                continue
            if number % 20 == 0:
                print("processed", number, "of", len(selected), flush=True)
        completed = {record["worldmove_city"] for record in records}
        for row in examples:
            if row.wm not in completed:
                continue
            traj = np.load(io.BytesIO(zf.read(f"mobility_data/traj/{row.wm}.npy")), allow_pickle=False)
            xy = load_worldmove_grid(zf, row.wm)
            example_payload[row.wm] = endpoint_matrix(traj, len(xy))
            example_payload[row.wm + "__xy"] = xy
            example_labels[row.wm] = row.worldmove_city_name

    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT / "WorldMove_OD_validation_city_level.csv", index=False)
    locations = selected[selected.wm.isin(metrics.worldmove_city)][
        ["wm", "city", "country", "iso3", "lon", "lat", "income_group"]
    ].rename(columns={"wm": "worldmove_city"})
    locations.to_csv(OUT / "worldmove_matched_city_locations.csv", index=False)
    np.savez_compressed(OUT / "worldmove_nine_city_od_matrices.npz", **example_payload)
    (OUT / "worldmove_nine_city_labels.json").write_text(json.dumps(example_labels, indent=2))

    metric_columns = [c for c in metrics if c.endswith(("_holdout", "_raw"))]
    summary = {
        "worldmove_cities": 1621,
        "candidate_matches": int(len(all_matches)),
        "selected_cities": int(len(selected)),
        "completed_cities": int(len(metrics)),
        "covered_countries": int(metrics.iso3.nunique()),
        "maximum_cities_per_country": MAX_CITIES_PER_COUNTRY,
        "sampling": "up to five highest-population matched cities per country",
        "model_data": "original 2024 city OD matrices from all four income groups",
        "calibration": "power correction fitted on one random half of WorldMove trajectories and evaluated on the other half",
        "metrics": {c: {"mean": float(metrics[c].mean()), "median": float(metrics[c].median())} for c in metric_columns},
        "example_cities": list(example_labels.values()),
    }
    (OUT / "worldmove_od_validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
