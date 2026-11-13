import os
import re
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import infer_six_cities_v5 as base


MAX_ROUTE_TOKENS = int(os.environ.get("MODE5_FAST_TOKENS", "16"))
EDGE_PATCH_BATCH = int(os.environ.get("MODE5_EDGE_PATCH_BATCH", "8192"))
ROUTE_BATCH = int(os.environ.get("MODE5_ROUTE_BATCH", "16384"))


def load_model() -> base.UrbanMultiModalClassifier:
    model = base.UrbanMultiModalClassifier(
        num_classes=base.NUM_CLASSES,
        input_channels=base.INPUT_CHANNELS_INFERENCE,
        use_env=True,
        use_landuse=False,
        use_traffic=False,
        use_center=False,
    ).to(base.DEVICE)
    model.load_state_dict(torch.load(base.MODEL_CHECKPOINT, map_location=base.DEVICE))
    model.eval()
    return model


def encoded_forward(
    model: base.UrbanMultiModalClassifier,
    env_embeddings: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        combined = torch.from_numpy(env_embeddings).float().to(base.DEVICE)
        combined = model.final_norm(combined)
        batch_size = combined.shape[0]
        cls_tokens = model.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, combined], dim=1)
        embeddings = model.pos_encoder(embeddings)

        route_lengths = torch.from_numpy(lengths.astype(np.int64, copy=False)).to(base.DEVICE)
        positions = torch.arange(combined.shape[1], device=base.DEVICE).unsqueeze(0)
        pad_mask = positions >= route_lengths.unsqueeze(1)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=base.DEVICE)
        encoded = model.transformer(
            embeddings,
            src_key_padding_mask=torch.cat([cls_mask, pad_mask], dim=1),
        )
        logits = model.classifier(encoded[:, 0, :])
        return torch.softmax(logits, dim=1).cpu().numpy()


def edge_midpoint(geom):
    try:
        return geom.interpolate(0.5, normalized=True)
    except Exception:
        return geom.centroid


def prepare_edge_embeddings(model, city_id: str):
    edge_geometries = base.load_road_network_cached(city_id)
    encoded_codes = np.fromiter(
        ((int(u) << 32) | int(v) for u, v in edge_geometries.keys()),
        dtype=np.int64,
        count=len(edge_geometries),
    )
    order = np.argsort(encoded_codes)
    encoded_codes = encoded_codes[order]
    geometries = np.asarray(list(edge_geometries.values()), dtype=object)[order]

    midpoints = [edge_midpoint(geom) for geom in geometries]
    lons = np.fromiter((point.x for point in midpoints), dtype=np.float64, count=len(midpoints))
    lats = np.fromiter((point.y for point in midpoints), dtype=np.float64, count=len(midpoints))
    edge_lengths = np.fromiter(
        (
            max(float(geom.length) * 111320.0 * np.cos(np.radians(point.y)), 1.0)
            for geom, point in zip(geometries, midpoints)
        ),
        dtype=np.float32,
        count=len(midpoints),
    )

    datasets, _, bounds = base.get_tiff_datasets(city_id)
    embeddings = np.empty((len(encoded_codes), model.embed_dim), dtype=np.float16)
    start_time = time.time()
    for start in range(0, len(encoded_codes), EDGE_PATCH_BATCH):
        stop = min(start + EDGE_PATCH_BATCH, len(encoded_codes))
        patches = base.batch_extract_ae_features(
            lons[start:stop],
            lats[start:stop],
            datasets,
            bounds,
            selected_bands=base.SELECTED_BANDS_INFERENCE,
        )
        patch_array = np.stack(patches).astype(np.float32, copy=False)
        with torch.no_grad():
            tensor = torch.from_numpy(patch_array).to(base.DEVICE)
            encoded = model.embed_norm(model.patch_encoder(tensor))
            embeddings[start:stop] = encoded.cpu().numpy().astype(np.float16)
        if start == 0 or stop == len(encoded_codes) or stop % (EDGE_PATCH_BATCH * 8) == 0:
            print(
                f"Edge embeddings {stop:,}/{len(encoded_codes):,} "
                f"({time.time() - start_time:.1f}s)",
                flush=True,
            )
    return encoded_codes, embeddings, edge_lengths


def gather_route_tokens(list_array, edge_codes, edge_embeddings, edge_lengths):
    offsets = list_array.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    flat_codes = list_array.values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    route_edge_counts = np.diff(offsets)
    valid_routes = route_edge_counts > 0
    model_lengths = np.minimum(route_edge_counts, MAX_ROUTE_TOKENS).astype(np.int64)
    safe_lengths = np.maximum(model_lengths, 1)

    token_index = np.arange(MAX_ROUTE_TOKENS, dtype=np.int64)[None, :]
    clipped_index = np.minimum(token_index, safe_lengths[:, None] - 1)
    denominator = np.maximum(safe_lengths - 1, 1)[:, None]
    relative = np.floor(
        (np.maximum(route_edge_counts - 1, 0)[:, None] * clipped_index) / denominator
    ).astype(np.int64)
    flat_positions = offsets[:-1, None] + relative
    flat_positions = np.minimum(flat_positions, np.maximum(offsets[1:, None] - 1, offsets[:-1, None]))
    sampled_codes = flat_codes[np.clip(flat_positions, 0, max(len(flat_codes) - 1, 0))]

    edge_positions = np.searchsorted(edge_codes, sampled_codes)
    in_range = edge_positions < len(edge_codes)
    safe_positions = np.clip(edge_positions, 0, len(edge_codes) - 1)
    matched = in_range & (edge_codes[safe_positions] == sampled_codes)
    tokens = edge_embeddings[safe_positions].astype(np.float32)
    tokens[~matched] = 0.0

    flat_edge_positions = np.searchsorted(edge_codes, flat_codes)
    flat_in_range = flat_edge_positions < len(edge_codes)
    flat_safe = np.clip(flat_edge_positions, 0, len(edge_codes) - 1)
    flat_matched = flat_in_range & (edge_codes[flat_safe] == flat_codes)
    flat_lengths = np.where(flat_matched, edge_lengths[flat_safe], 0.0)
    total_lengths = np.add.reduceat(flat_lengths, offsets[:-1])
    total_lengths[~valid_routes] = 0.0
    return tokens, model_lengths, total_lengths.astype(np.float32), valid_routes


def process_parquet(model, path: Path, edge_codes, edge_embeddings, edge_lengths):
    match = re.search(r"od_paths_flows_(\d+)_chunk_(\d+)\.parquet$", path.name)
    if not match:
        raise ValueError(f"Unexpected route filename: {path.name}")
    city_id, chunk_id = match.groups()
    destination = Path(base.OUTPUT_PRED_DIR) / f"{city_id}_chunk_{chunk_id}_predictions.parquet"
    if destination.exists():
        print(f"Skip existing {destination.name}", flush=True)
        return

    outputs = {f"prob_mode_{mode}": [] for mode in range(1, base.NUM_CLASSES + 1)}
    outputs.update({"pred_label": [], "confidence": [], "point_count": [], "total_length_m": []})
    row_offset = 0
    started = time.time()
    parquet = pq.ParquetFile(path)
    for record_batch in parquet.iter_batches(batch_size=ROUTE_BATCH, columns=["edge_sequence"]):
        list_array = record_batch.column(0)
        tokens, lengths, total_lengths, valid = gather_route_tokens(
            list_array, edge_codes, edge_embeddings, edge_lengths
        )
        probabilities = encoded_forward(model, tokens, np.maximum(lengths, 1))
        predictions = probabilities.argmax(axis=1).astype(np.int8)
        predictions[~valid] = -1
        confidence = probabilities.max(axis=1).astype(np.float32)
        confidence[~valid] = 0.0

        outputs["pred_label"].append(predictions)
        outputs["confidence"].append(confidence)
        outputs["point_count"].append(lengths.astype(np.int16))
        outputs["total_length_m"].append(total_lengths)
        for mode in range(1, base.NUM_CLASSES + 1):
            values = probabilities[:, mode - 1].astype(np.float32)
            values[~valid] = 0.0
            outputs[f"prob_mode_{mode}"].append(values)
        row_offset += len(record_batch)
        if row_offset % (ROUTE_BATCH * 8) == 0 or row_offset == parquet.metadata.num_rows:
            rate = row_offset / max(time.time() - started, 1e-6)
            print(f"{path.name}: {row_offset:,}/{parquet.metadata.num_rows:,} ({rate:,.0f} routes/s)", flush=True)

    columns = {key: np.concatenate(parts) for key, parts in outputs.items()}
    n_rows = len(columns["pred_label"])
    columns["seg_id"] = pa.array(
        [f"{city_id}_chunk_{chunk_id}_{index}" for index in range(n_rows)]
    )
    columns["chunk_id"] = pa.array([chunk_id] * n_rows)
    table = pa.table(columns)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="snappy")
    print(f"Saved {destination} ({n_rows:,} routes)", flush=True)


def main():
    print(
        f"Fast edge-reuse inference: city={base.CITY_ID}, year={base.YEAR}, "
        f"tokens={MAX_ROUTE_TOKENS}, device={base.DEVICE}",
        flush=True,
    )
    route_files = sorted(Path(base.INPUT_PARQUET_DIR).glob(f"od_paths_flows_{base.CITY_ID}_*.parquet"))
    if not route_files:
        raise FileNotFoundError(f"No route files for {base.CITY_ID} in {base.INPUT_PARQUET_DIR}")
    expected_outputs = []
    for path in route_files:
        match = re.search(r"od_paths_flows_(\d+)_chunk_(\d+)\.parquet$", path.name)
        if not match:
            raise ValueError(f"Unexpected route filename: {path.name}")
        city_id, chunk_id = match.groups()
        expected_outputs.append(
            Path(base.OUTPUT_PRED_DIR) / f"{city_id}_chunk_{chunk_id}_predictions.parquet"
        )
    if all(path.exists() for path in expected_outputs):
        print(f"All {len(expected_outputs)} prediction chunks already exist; skip city-year.", flush=True)
        return

    model = load_model()
    edge_codes, edge_embeddings, edge_lengths = prepare_edge_embeddings(model, base.CITY_ID)
    for path in route_files:
        process_parquet(model, path, edge_codes, edge_embeddings, edge_lengths)


if __name__ == "__main__":
    main()
