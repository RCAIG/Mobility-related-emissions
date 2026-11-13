"""Native 100 m external validation against Hestia road-transport products.

The model's road-edge emissions are allocated along their actual LineString
geometry to the *identical* 100 m EPSG:6404 grid used by Hestia Phoenix-Mesa.
Hestia values remain external: neither Hestia grid nor Arizona road segments
are used to fit, select, rescale, or tune the mobility-emission model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.stats import spearmanr
from shapely import from_wkt
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path("${SERVER_HOME}/result")
OUT = ROOT / "Experiments/external_validation_v1/hestia_native_100m_2017"
HST = ROOT / "Experiments/external_validation_v1/inputs/hestia_phoenix/Hestia_Phoenix_FFCO2_2017.tif"
ROAD_DIR = ROOT / "Experiments/external_validation_v1/phoenix_model_reconstruction/results"
AZ_DIR = ROOT / "Experiments/external_validation_v1/inputs/hestia_arizona_onroad"


def add_line_to_grid(coords: np.ndarray, amount: float, transform, shape, out: np.ndarray) -> None:
    """Allocate one line's amount by physical segment length to 100 m cells."""
    if not np.isfinite(amount) or amount <= 0 or len(coords) < 2:
        return
    dx, dy = np.diff(coords[:, 0]), np.diff(coords[:, 1])
    lengths = np.hypot(dx, dy)
    total = lengths.sum()
    if not np.isfinite(total) or total <= 0:
        return
    x0, y0 = transform.c, transform.f
    sx, sy = transform.a, -transform.e
    height, width = shape
    for i, length in enumerate(lengths):
        if length <= 0:
            continue
        # 50 m sampling avoids skipping a 100 m pixel while remaining tractable.
        n = max(1, int(math.ceil(length / 50.0)))
        frac = (np.arange(n) + .5) / n
        x = coords[i, 0] + dx[i] * frac
        y = coords[i, 1] + dy[i] * frac
        col = ((x - x0) / sx).astype(np.int64)
        row = ((y0 - y) / sy).astype(np.int64)
        valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        if valid.any():
            np.add.at(out, (row[valid], col[valid]), amount * (length / total) / n)


def model_grid(transform, shape) -> np.ndarray:
    """Rasterize directed model edges from Phoenix and Mesa without rescaling."""
    tx = Transformer.from_crs("EPSG:4326", "EPSG:6404", always_xy=True)
    grid = np.zeros(shape, dtype=np.float64)
    rows = 0
    for city_id in (4699, 4736):
        d = pd.read_parquet(ROAD_DIR / f"road_emission_{city_id}_2017.parquet",
                            columns=["geometry_wkt", "reconstructed_emission"])
        d = d.loc[d.reconstructed_emission.gt(0)]
        for geom, amount in zip(d.geometry_wkt, d.reconstructed_emission):
            line = from_wkt(geom)
            coords = np.asarray(line.coords)
            x, y = tx.transform(coords[:, 0], coords[:, 1])
            add_line_to_grid(np.column_stack([x, y]), float(amount), transform, shape, grid)
            rows += 1
    print(f"Rasterized model road edges: {rows}", flush=True)
    return grid


def hestia_grid() -> tuple[np.ndarray, object, list[str]]:
    with rasterio.open(HST) as src:
        desc = list(src.descriptions)
        onr_band = desc.index("ONR") + 1
        onr = src.read(onr_band, masked=True).filled(0).astype(np.float64)
        return onr, src.transform, desc


def arizona_segments_grid(transform, shape) -> np.ndarray:
    """Rasterize Arizona Hestia local plus non-local FFCO2 road segments."""
    bounds = rasterio.transform.array_bounds(shape[0], shape[1], transform)
    bbox = (bounds[0], bounds[1], bounds[2], bounds[3])
    grid = np.zeros(shape, dtype=np.float64)
    count = 0
    for name, layer in (("NonLocal_AZ_v1_1.gpkg", "NonLocal_AZ_v1.1"),
                        ("Local_AZ_v1_1.gpkg", "Local_AZ_v1.1")):
        d = gpd.read_file(AZ_DIR / name, layer=layer, bbox=bbox,
                          columns=["FFCO2tC2017", "geometry"])
        d = d.loc[d.FFCO2tC2017.gt(0) & d.geometry.notna()]
        if d.crs.to_epsg() != 6404:
            d = d.to_crs("EPSG:6404")
        for geom, amount in zip(d.geometry, d.FFCO2tC2017):
            if geom.geom_type == "LineString":
                add_line_to_grid(np.asarray(geom.coords), float(amount), transform, shape, grid)
            elif geom.geom_type == "MultiLineString":
                lengths = np.asarray([part.length for part in geom.geoms], float)
                for part, length in zip(geom.geoms, lengths):
                    add_line_to_grid(np.asarray(part.coords), float(amount) * length / lengths.sum(), transform, shape, grid)
            count += 1
    print(f"Rasterized Arizona Hestia road segments: {count}", flush=True)
    return grid


def metrics(reference: np.ndarray, model: np.ndarray, label: str) -> dict:
    # All pixels are reported only as a coverage diagnostic: roughly a million
    # road-free cells can otherwise dominate. Primary 100 m assessment is the
    # union of positive road-emission pixels, with no zero-zero inflation.
    ref = reference.ravel()
    pred = model.ravel()
    union = (ref > 0) | (pred > 0)
    y, p = ref[union], pred[union]
    ly, lp = np.log1p(y), np.log1p(p)
    qy, qp = np.quantile(y, .9), np.quantile(p, .9)
    top_y, top_p = y >= qy, p >= qp
    inter = int((top_y & top_p).sum())
    return {
        "reference": label,
        "native_grid": "100 m x 100 m, EPSG:6404",
        "reference_unit": "tC per grid cell per year",
        "model_unit": "EDGAR-target emission units; not directly unit-harmonized",
        "all_cells": int(ref.size),
        "reference_positive_cells": int((ref > 0).sum()),
        "model_positive_cells": int((pred > 0).sum()),
        "positive_union_cells": int(union.sum()),
        "positive_union_spearman": float(spearmanr(y, p).statistic),
        "positive_union_pearson_log": float(np.corrcoef(ly, lp)[0, 1]),
        "positive_union_mae_log_unscaled": float(mean_absolute_error(ly, lp)),
        "positive_union_rmse_log_unscaled": float(math.sqrt(mean_squared_error(ly, lp))),
        "positive_union_top_decile_jaccard": float(inter / (top_y | top_p).sum()),
        "positive_union_top_decile_recall": float(inter / top_y.sum()),
        "note": "Metrics are spatial-pattern evidence.  Absolute log errors are descriptive only because model and reference units/boundaries differ.",
    }


def plot_pair(reference: np.ndarray, model: np.ndarray, title: str, filename: str) -> None:
    union = (reference > 0) | (model > 0)
    x, y = np.log1p(reference[union]), np.log1p(model[union])
    fig, ax = plt.subplots(figsize=(5.1, 4.3), dpi=220)
    hb = ax.hexbin(x, y, gridsize=55, mincnt=1, cmap="viridis", linewidths=0)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color="0.25", ls="--", lw=.8)
    ax.set_xlabel("ln(1 + external road CO2)")
    ax.set_ylabel("ln(1 + reconstructed road emission)")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(hb, ax=ax, label="100 m cells")
    fig.tight_layout()
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reference, transform, descriptions = hestia_grid()
    model = model_grid(transform, reference.shape)
    az = arizona_segments_grid(transform, reference.shape)
    with rasterio.open(HST) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="float64", compress="lzw", nodata=0)
    for name, grid in (("model_road_emission_100m.tif", model),
                       ("hestia_onr_100m.tif", reference),
                       ("hestia_arizona_onroad_100m.tif", az)):
        with rasterio.open(OUT / name, "w", **profile) as dst:
            dst.write(grid, 1)
    results = {
        "hestia_phoenix_mesa_onr": metrics(reference, model, "Hestia Phoenix-Mesa ONR 2017"),
        "hestia_arizona_road_segments": metrics(az, model, "Hestia SW-IFL Arizona road segments 2017"),
        "band_descriptions": descriptions,
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_pair(reference, model, "Hestia Phoenix-Mesa ONR", "model_vs_hestia_phoenix_mesa_100m.png")
    plot_pair(az, model, "Hestia SW-IFL Arizona roads", "model_vs_hestia_arizona_100m.png")


if __name__ == "__main__":
    main()
