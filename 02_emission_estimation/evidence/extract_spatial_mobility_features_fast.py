from __future__ import annotations

import argparse
import math
import os
import re
import time
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


YEARS = (2017, 2024)
FINE_BOUNDS_KM = np.asarray([0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 100, np.inf])
N_FINE = len(FINE_BOUNDS_KM) - 1
N_COLS = 3600
N_ROWS = 1800
ROOTS = {
    (2017, "low_income"): "${SERVER_HOME}/result/trace_genetare_low_income_2017",
    (2024, "low_income"): "${SERVER_HOME}/result/trace_genetare_low_income_2024",
    (2017, "lower_middle"): "${SERVER_HOME}/result/trace_genetare_lower_middle_2017",
    (2024, "lower_middle"): "${SERVER_HOME}/result/trace_genetare_lower_middle_2024",
    (2017, "upper_middle"): "${SERVER_HOME}/result/trace_genetare_upper_middle_2017",
    (2024, "upper_middle"): "${SERVER_HOME}/result/trace_genetare_upper_middle_2024",
    (2017, "high_income"): "${SERVER_HOME}/result/trace_genetare_hgih_income_2017",
    (2024, "high_income"): "${SERVER_HOME}/result/trace_genetare_hgih_income_2024",
}
ROAD_ROOT = "${SERVER_HOME}/result/road_{year}"
FILE_RE = re.compile(r"od_paths_flows_(\d+)_chunk_\d+\.parquet$")


def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def parse_linestring(text: str | None, fallback: np.ndarray) -> np.ndarray:
    if not text or not text.startswith("LINESTRING"):
        return fallback
    body = text[text.find("(") + 1 : text.rfind(")")]
    try:
        coords = np.asarray([[float(v) for v in token.strip().split()[:2]] for token in body.split(",")])
        return coords if len(coords) >= 2 else fallback
    except Exception:
        return fallback


def segment_edge(coords: np.ndarray, edge_length_km: float, max_segment_km: float = 2.0):
    raw_lengths = haversine_km(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1])
    raw_total = float(raw_lengths.sum())
    if raw_total <= 0:
        return np.empty(0, np.int64), np.empty(0, float)
    cell_parts: defaultdict[int, float] = defaultdict(float)
    for a, b, raw_length in zip(coords[:-1], coords[1:], raw_lengths):
        n = max(1, int(math.ceil(float(raw_length) / max_segment_km)))
        fractions = (np.arange(n, dtype=float) + 0.5) / n
        mids = a[None, :] + fractions[:, None] * (b - a)[None, :]
        cols = np.floor((mids[:, 0] + 180.0) * 10.0).astype(np.int64)
        rows = np.floor((90.0 - mids[:, 1]) * 10.0).astype(np.int64)
        valid = (cols >= 0) & (cols < N_COLS) & (rows >= 0) & (rows < N_ROWS)
        if not valid.any():
            continue
        piece_length = edge_length_km * float(raw_length) / raw_total / n
        for cell_id in (rows[valid] * N_COLS + cols[valid]).tolist():
            cell_parts[int(cell_id)] += piece_length
    if not cell_parts:
        return np.empty(0, np.int64), np.empty(0, float)
    return np.fromiter(cell_parts.keys(), np.int64), np.fromiter(cell_parts.values(), float)


def parse_graph(path: Path):
    key_names = {}
    node_index = {}
    node_xy = []
    selected = {}
    parallel_count = 0
    for _, element in ET.iterparse(path, events=("end",)):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "key":
            key_names[element.attrib["id"]] = element.attrib.get("attr.name", "")
            element.clear()
        elif tag == "node":
            values = {key_names.get(d.attrib.get("key", ""), ""): d.text for d in element if d.tag.rsplit("}", 1)[-1] == "data"}
            node_id = element.attrib["id"]
            node_index[node_id] = len(node_xy)
            node_xy.append((float(values["x"]), float(values["y"])))
            element.clear()
        elif tag == "edge":
            source, target = element.attrib["source"], element.attrib["target"]
            if source in node_index and target in node_index:
                values = {key_names.get(d.attrib.get("key", ""), ""): d.text for d in element if d.tag.rsplit("}", 1)[-1] == "data"}
                code = (node_index[source] << 32) | node_index[target]
                length_km = float(values.get("length") or 0) / 1000.0
                impedance = float(values.get("travel_time") or length_km or np.inf)
                coords = np.asarray([node_xy[node_index[source]], node_xy[node_index[target]]], dtype=float)
                if code in selected:
                    parallel_count += 1
                if code not in selected or impedance < selected[code][0]:
                    selected[code] = (impedance, length_km, coords)
            element.clear()

    codes = np.asarray(sorted(selected), dtype=np.int64)
    lengths = np.empty(len(codes), dtype=float)
    edge_cells, edge_cell_lengths = [], []
    for i, code in enumerate(codes):
        _, length_km, coords = selected[int(code)]
        if not np.isfinite(length_km) or length_km <= 0:
            length_km = float(haversine_km(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1]).sum())
        lengths[i] = length_km
        cells, cell_lengths = segment_edge(coords, length_km)
        edge_cells.append(cells)
        edge_cell_lengths.append(cell_lengths)
    return codes, lengths, edge_cells, edge_cell_lengths, parallel_count


def accumulate_chunk(frame: pd.DataFrame, sorted_codes: np.ndarray, edge_lengths: np.ndarray, edge_class_activity: np.ndarray):
    sequences = frame["edge_sequence"].to_numpy()
    flow = frame["flow"].to_numpy(dtype=float)
    seq_lengths = np.fromiter((len(x) for x in sequences), dtype=np.int64, count=len(sequences))
    nonempty = seq_lengths > 0
    if not nonempty.any():
        return 0, 0.0, 0, np.zeros(N_FINE), np.zeros(N_FINE)
    sequences = sequences[nonempty]
    flow = flow[nonempty]
    seq_lengths = seq_lengths[nonempty]
    flat_codes = np.concatenate(sequences).astype(np.int64, copy=False)
    positions = np.searchsorted(sorted_codes, flat_codes)
    matched = (positions < len(sorted_codes)) & (sorted_codes[np.minimum(positions, len(sorted_codes) - 1)] == flat_codes)
    flat_edge_lengths = np.zeros(len(flat_codes), dtype=float)
    flat_edge_lengths[matched] = edge_lengths[positions[matched]]
    starts = np.r_[0, np.cumsum(seq_lengths)[:-1]]
    route_lengths = np.add.reduceat(flat_edge_lengths, starts)
    route_class = np.searchsorted(FINE_BOUNDS_KM[1:], route_lengths, side="right")
    route_class = np.minimum(route_class, N_FINE - 1)
    repeated_flow = np.repeat(flow, seq_lengths)
    repeated_class = np.repeat(route_class, seq_lengths)
    valid = matched & np.isfinite(repeated_flow) & (repeated_flow > 0)
    keys = positions[valid] * N_FINE + repeated_class[valid]
    edge_class_activity += np.bincount(keys, weights=repeated_flow[valid], minlength=edge_class_activity.size).reshape(edge_class_activity.shape)
    class_flow = np.bincount(route_class, weights=flow, minlength=N_FINE)
    class_routes = np.bincount(route_class, minlength=N_FINE)
    return len(flow), float(flow.sum()), int((~matched).sum()), class_flow, class_routes


def process_city(task):
    year, income, city_id, chunks, output_root = task
    start_time = time.time()
    output = Path(output_root) / "city_cache" / str(year) / f"city_{city_id}.parquet"
    diag_path = Path(output_root) / "city_cache" / str(year) / f"city_{city_id}.diag.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and diag_path.exists():
        return {"year": year, "city_id": city_id, "status": "cached"}
    road = Path(ROAD_ROOT.format(year=year)) / f"road_network_drive_{city_id}.graphml"
    try:
        if not road.exists():
            raise FileNotFoundError(str(road))
        codes, lengths, edge_cells, edge_cell_lengths, parallel_count = parse_graph(road)
        edge_class_activity = np.zeros((len(codes), N_FINE), dtype=float)
        route_count = 0
        total_flow = 0.0
        unmatched_codes = 0
        class_flow = np.zeros(N_FINE)
        class_routes = np.zeros(N_FINE)
        for chunk in chunks:
            parquet = pq.ParquetFile(chunk)
            for batch in parquet.iter_batches(batch_size=50_000, columns=["flow", "edge_sequence"]):
                frame = batch.to_pandas()
                rc, tf, unmatched, cf, cr = accumulate_chunk(frame, codes, lengths, edge_class_activity)
                route_count += rc
                total_flow += tf
                unmatched_codes += unmatched
                class_flow += cf
                class_routes += cr

        cell_values: defaultdict[int, np.ndarray] = defaultdict(lambda: np.zeros(N_FINE, dtype=float))
        for edge_idx in np.flatnonzero(edge_class_activity.sum(axis=1) > 0):
            cells = edge_cells[edge_idx]
            portions = edge_cell_lengths[edge_idx]
            for cell_id, length_km in zip(cells, portions):
                cell_values[int(cell_id)] += edge_class_activity[edge_idx] * length_km
        cell_ids = np.asarray(sorted(cell_values), dtype=np.int64)
        values = np.vstack([cell_values[int(c)] for c in cell_ids]) if len(cell_ids) else np.empty((0, N_FINE))
        result = pd.DataFrame({"cell_id": cell_ids, "year": year, "city_id": city_id})
        for b in range(N_FINE):
            result[f"pkm_b{b:02d}"] = values[:, b]
        result.to_parquet(output, index=False, compression="snappy")
        diag = {"year": year, "income": income, "city_id": city_id, "chunks": len(chunks), "route_count": route_count,
                "total_flow": total_flow, "active_cells": len(cell_ids), "graph_edges": len(codes),
                "parallel_edges_discarded": parallel_count, "unmatched_edge_occurrences": unmatched_codes,
                "runtime_seconds": time.time() - start_time, "status": "ok", "error": ""}
        for b in range(N_FINE):
            diag[f"flow_b{b:02d}"] = class_flow[b]
            diag[f"routes_b{b:02d}"] = class_routes[b]
        pd.DataFrame([diag]).to_csv(diag_path, index=False)
        return diag
    except Exception as exc:
        return {"year": year, "income": income, "city_id": city_id, "status": "failed", "error": repr(exc),
                "traceback": traceback.format_exc(), "runtime_seconds": time.time() - start_time}


def discover_tasks(output_root: Path, years, limit: int | None):
    tasks = []
    for year in years:
        for (root_year, income), root_text in ROOTS.items():
            if root_year != year:
                continue
            grouped: defaultdict[int, list[str]] = defaultdict(list)
            for path in Path(root_text).glob("od_paths_flows_*_chunk_*.parquet"):
                match = FILE_RE.search(path.name)
                if match:
                    grouped[int(match.group(1))].append(str(path))
            for city_id, chunks in grouped.items():
                tasks.append((year, income, city_id, sorted(chunks), str(output_root)))
    tasks.sort(key=lambda x: (x[0], x[2]))
    return tasks[:limit] if limit else tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--city-ids", nargs="+", type=int)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = discover_tasks(args.output_root, args.years, args.limit)
    if args.city_ids:
        selected = set(args.city_ids)
        tasks = [task for task in tasks if task[2] in selected]
    print(f"tasks={len(tasks)} workers={args.workers}", flush=True)
    log_path = args.output_root / "extraction_results.jsonl"
    import json
    with ProcessPoolExecutor(max_workers=args.workers) as executor, log_path.open("a", encoding="utf-8") as log:
        futures = {executor.submit(process_city, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            log.write(json.dumps(result, ensure_ascii=True) + "\n")
            log.flush()
            print(f"[{i}/{len(tasks)}] {result.get('year')} {result.get('city_id')} {result.get('status')} "
                  f"{result.get('runtime_seconds', 0):.1f}s {result.get('error', '')}", flush=True)


if __name__ == "__main__":
    main()
