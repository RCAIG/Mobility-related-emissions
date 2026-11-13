import csv
import io
import json
import math
import os
import re
import subprocess
import tempfile
import zipfile
from collections import Counter

import numpy as np
from osgeo import ogr, osr

ZIP_PATH = "${SERVER_HOME}/worldmove/data.zip"
MANIFEST = "${SERVER_HOME}/worldmove/worldmove_matched_manifest.csv"
OUT = "${SERVER_HOME}/worldmove/worldmove_validation"
os.makedirs(OUT, exist_ok=True)

LOCAL_SHP_ROOT = (
    "${SERVER_DATA_ROOT}/prediction_travel_mode/"
    "generate-od-pubtools-main/generate-od-pubtools-main/assets/"
    "example_data/my_test/shp/ALL"
)
DISTANCE_BINS_KM = np.asarray(
    [0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, np.inf], dtype=float
)
EXAMPLE_CITIES = {
    "187_GB_London": "London",
    "269_FR_Paris": "Paris",
    "380_US_New_York": "New York",
    "144_US_Los_Angeles": "Los Angeles",
    "564_MX_Mexico_City": "Mexico City",
    "764_BR_Brasilia": "Brasilia",
    "396_CN_Shanghai": "Shanghai",
    "1062_IN_Mumbai": "Mumbai",
    "394_JP_Tokyo": "Tokyo",
}


def js_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-15)
    q = q / max(q.sum(), 1e-15)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-15))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cpc(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-15)
    q = q / max(q.sum(), 1e-15)
    return float(np.minimum(p, q).sum())


def rank_bins(values, n_bins=20):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0:
        return np.zeros(n_bins, dtype=float)
    values = np.sort(values)[::-1]
    chunks = np.array_split(values, n_bins)
    out = np.array([x.sum() for x in chunks], dtype=float)
    return out / max(out.sum(), 1e-15)


def power_correct(model, target):
    """Fit a monotone concentration correction on a calibration split."""
    model = np.asarray(model, dtype=float)
    target = np.asarray(target, dtype=float)
    best_p, best_loss = 1.0, float("inf")
    for p in np.linspace(0.2, 5.0, 481):
        z = np.power(np.maximum(model, 1e-15), p)
        z /= max(z.sum(), 1e-15)
        loss = float(np.sum((z - target) ** 2))
        if loss < best_loss:
            best_p, best_loss = float(p), loss
    z = np.power(np.maximum(model, 1e-15), best_p)
    return z / max(z.sum(), 1e-15), best_p


def load_matrix(path):
    with open(path, newline="") as f:
        rows = [[float(x) for x in row] for row in csv.reader(f) if row]
    a = np.asarray(rows, dtype=float)
    n = min(a.shape)
    a = a[:n, :n]
    a[~np.isfinite(a)] = 0
    a[a < 0] = 0
    return a


def load_worldmove(zf, member):
    raw = zf.read(member)
    return np.load(io.BytesIO(raw), allow_pickle=False)


def load_worldmove_grid(zf, city_key):
    member = "mobility_data/grid/" + city_key + ".json"
    raw = json.loads(zf.read(member).decode("utf-8"))
    n = max(int(k) for k in raw) + 1
    xy = np.full((n, 2), np.nan, dtype=float)
    for k, value in raw.items():
        xy[int(k)] = value[:2]
    return xy


def load_local_centroids(city_id, expected_n):
    path = os.path.join(LOCAL_SHP_ROOT, str(city_id) + ".shp")
    ds = ogr.Open(path)
    if ds is None:
        raise FileNotFoundError(path)
    layer = ds.GetLayer(0)
    source = layer.GetSpatialRef()
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    if source is not None:
        source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = None if source is None or source.IsSame(target) else osr.CoordinateTransformation(source, target)
    xy = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            xy.append((np.nan, np.nan))
            continue
        point = geom.Centroid().Clone()
        if transform is not None:
            point.Transform(transform)
        xy.append((point.GetX(), point.GetY()))
    ds = None
    xy = np.asarray(xy, dtype=float)
    if len(xy) < expected_n:
        raise ValueError("centroid count is smaller than OD matrix: %s < %s" % (len(xy), expected_n))
    return xy[:expected_n]


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.minimum(1, np.sqrt(a)))


def distance_hist_from_trajectories(traj, xy, mask=None):
    if mask is None:
        mask = np.ones(len(traj), dtype=bool)
    x = traj[mask]
    if not len(x):
        return np.zeros(len(DISTANCE_BINS_KM) - 1, dtype=float)
    oi = x[:, 0].astype(int)
    di = x[:, -1].astype(int)
    valid = (oi >= 0) & (di >= 0) & (oi < len(xy)) & (di < len(xy))
    oi, di = oi[valid], di[valid]
    valid_xy = np.isfinite(xy[oi]).all(axis=1) & np.isfinite(xy[di]).all(axis=1)
    oi, di = oi[valid_xy], di[valid_xy]
    dist = haversine_km(xy[oi, 0], xy[oi, 1], xy[di, 0], xy[di, 1])
    return np.histogram(dist, bins=DISTANCE_BINS_KM)[0].astype(float)


def distance_hist_from_matrix(mat, xy):
    oi, di = np.nonzero(mat > 0)
    weights = mat[oi, di]
    valid = np.isfinite(xy[oi]).all(axis=1) & np.isfinite(xy[di]).all(axis=1)
    oi, di, weights = oi[valid], di[valid], weights[valid]
    dist = haversine_km(xy[oi, 0], xy[oi, 1], xy[di, 0], xy[di, 1])
    return np.histogram(dist, bins=DISTANCE_BINS_KM, weights=weights)[0].astype(float)


def endpoint_matrix(traj, n_locations):
    oi = traj[:, 0].astype(int)
    di = traj[:, -1].astype(int)
    valid = (oi >= 0) & (di >= 0) & (oi < n_locations) & (di < n_locations)
    out = np.zeros((n_locations, n_locations), dtype=np.float32)
    np.add.at(out, (oi[valid], di[valid]), 1)
    return out


def trajectory_marginals(a, mask=None):
    if mask is None:
        mask = np.ones(len(a), dtype=bool)
    x = a[mask]
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) == 0:
        return np.array([]), np.array([]), 0
    origin = Counter(x[:, 0].astype(int).tolist())
    destination = Counter(x[:, -1].astype(int).tolist())
    return np.asarray(list(origin.values()), dtype=float), np.asarray(list(destination.values()), dtype=float), len(x)


def main():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig", newline="")))
    # Keep one first-stage city output for each WorldMove city.
    unique = {}
    for r in rows:
        unique.setdefault(r["wm"], r)
    rows = list(unique.values())
    records = []
    city_locations = []
    example_payload = {}
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = set(zf.namelist())
        rng = np.random.default_rng(20261023)
        for j, r in enumerate(rows):
            member = "mobility_data/traj/" + r["wm"] + ".npy"
            # Some archive names contain a filename suffix only, while the manifest has no extension.
            if member not in names:
                member = "mobility_data/traj/" + r["wm"]
            if member not in names:
                continue
            try:
                traj = load_worldmove(zf, member)
                mat = load_matrix(r["path"])
                wm_xy = load_worldmove_grid(zf, r["wm"])
                local_xy = load_local_centroids(r["id"], mat.shape[0])
            except Exception as exc:
                print("skip", r["wm"], repr(exc), flush=True)
                continue
            if traj.ndim != 2 or traj.shape[1] < 2 or mat.ndim != 2:
                continue
            n = len(traj)
            mask = rng.random(n) < 0.5
            wo, wd, n0 = trajectory_marginals(traj)
            wo1, wd1, n1 = trajectory_marginals(traj, mask)
            wo2, wd2, n2 = trajectory_marginals(traj, ~mask)
            mo = rank_bins(mat.sum(axis=1))
            md = rank_bins(mat.sum(axis=0))
            wo_r, wd_r = rank_bins(wo), rank_bins(wd)
            wo1_r, wd1_r = rank_bins(wo1), rank_bins(wd1)
            wo2_r, wd2_r = rank_bins(wo2), rank_bins(wd2)
            co, po = power_correct(mo, wo1_r)
            cd, pd = power_correct(md, wd1_r)
            co = co / max(co.sum(), 1e-15)
            cd = cd / max(cd.sum(), 1e-15)
            wm_dist = distance_hist_from_trajectories(traj, wm_xy)
            wm_dist_1 = distance_hist_from_trajectories(traj, wm_xy, mask)
            wm_dist_2 = distance_hist_from_trajectories(traj, wm_xy, ~mask)
            model_dist = distance_hist_from_matrix(mat, local_xy)
            corrected_dist, p_dist = power_correct(
                model_dist / max(model_dist.sum(), 1e-15),
                wm_dist_1 / max(wm_dist_1.sum(), 1e-15),
            )
            wm_dist_n = wm_dist / max(wm_dist.sum(), 1e-15)
            wm_dist_2_n = wm_dist_2 / max(wm_dist_2.sum(), 1e-15)
            records.append({
                "worldmove_city": r["wm"], "city": r["city"], "country": r["country"],
                "id_uc_g0": r["id"], "worldmove_trajectories": n0,
                "model_od_total": float(mat.sum()), "model_grid_cells": int(mat.shape[0]),
                "origin_cpc_raw": cpc(mo, wo_r), "origin_js_raw": js_divergence(mo, wo_r),
                "destination_cpc_raw": cpc(md, wd_r), "destination_js_raw": js_divergence(md, wd_r),
                "origin_cpc_calibrated_holdout": cpc(co, wo2_r), "origin_js_calibrated_holdout": js_divergence(co, wo2_r),
                "destination_cpc_calibrated_holdout": cpc(cd, wd2_r), "destination_js_calibrated_holdout": js_divergence(cd, wd2_r),
                "distance_cpc_raw": cpc(model_dist, wm_dist_n), "distance_js_raw": js_divergence(model_dist, wm_dist_n),
                "distance_cpc_calibrated_holdout": cpc(corrected_dist, wm_dist_2_n),
                "distance_js_calibrated_holdout": js_divergence(corrected_dist, wm_dist_2_n),
                "origin_power": po, "destination_power": pd, "distance_power": p_dist,
                "worldmove_origin_top10_share": float(np.sort(wo)[::-1][:max(1, int(np.ceil(len(wo)*0.1)))].sum()/max(wo.sum(),1e-15)),
                "model_origin_top10_share": float(np.sort(mat.sum(axis=1))[::-1][:max(1, int(np.ceil(len(mo)*0.1)))].sum()/max(mat.sum(),1e-15)),
                "worldmove_destination_top10_share": float(np.sort(wd)[::-1][:max(1, int(np.ceil(len(wd)*0.1)))].sum()/max(wd.sum(),1e-15)),
                "model_destination_top10_share": float(np.sort(mat.sum(axis=0))[::-1][:max(1, int(np.ceil(len(md)*0.1)))].sum()/max(mat.sum(),1e-15)),
            })
            city_locations.append({
                "worldmove_city": r["wm"], "city": r["city"], "country": r["country"],
                "lon": float(np.nanmean(wm_xy[:, 0])), "lat": float(np.nanmean(wm_xy[:, 1])),
            })
            if (j + 1) % 50 == 0:
                print("processed", j + 1, flush=True)
        # Export nine familiar cities directly from WorldMove for the dataset overview.
        for city_key, display_name in EXAMPLE_CITIES.items():
            traj = load_worldmove(zf, "mobility_data/traj/" + city_key + ".npy")
            xy = load_worldmove_grid(zf, city_key)
            example_payload[city_key] = endpoint_matrix(traj, len(xy))
            example_payload[city_key + "__xy"] = xy
    out_csv = os.path.join(OUT, "worldmove_od_validation_city_level.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader(); w.writerows(records)
    with open(os.path.join(OUT, "worldmove_matched_city_locations.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(city_locations[0]))
        w.writeheader(); w.writerows(city_locations)
    np.savez_compressed(os.path.join(OUT, "worldmove_nine_city_od_matrices.npz"), **example_payload)
    with open(os.path.join(OUT, "worldmove_nine_city_labels.json"), "w") as f:
        json.dump(EXAMPLE_CITIES, f, indent=2)
    numeric = [k for k in records[0] if k not in {"worldmove_city", "city", "country"}]
    summary = {"n_matched_cities": len(records), "matching": "exact normalized city name plus ISO country code; one local city output per WorldMove city", "rank_bins": 20, "distance_bins_km": [str(x) for x in DISTANCE_BINS_KM], "calibration": "power correction fitted on one random half of WorldMove trajectories and evaluated on the other half", "metrics": {k: {"mean": float(np.mean([float(x[k]) for x in records])), "median": float(np.median([float(x[k]) for x in records]))} for k in numeric}}
    with open(os.path.join(OUT, "worldmove_od_validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
