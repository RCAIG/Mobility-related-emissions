"""External Phoenix-Mesa grid comparison against the Hestia ONR product.

Hestia is never used for fitting or tuning.  It is aggregated from its native
100 m Arizona Central grid to 0.1 degree cells, then compared with frozen
full-fit mobility-model spatial predictions.  The comparison is explicitly a
spatial-pattern external check, not a held-out EDGAR accuracy estimate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path("${SERVER_HOME}/result")
HST = ROOT / "Experiments/external_validation_v1/inputs/hestia_phoenix/Hestia_Phoenix_FFCO2_2017.tif"
OUT = ROOT / "Experiments/external_validation_v1/hestia_phoenix_grid_2017"
GLOBAL = ROOT / "spatial_mobility_carbon_validation_v2/analysis/global_cell_features_v2.parquet"


def cell_key(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the model's 0.1 degree lower-left cell convention."""
    return np.floor(lon * 10).astype(int), np.floor(lat * 10).astype(int)


def hestia_onroad_01deg() -> pd.DataFrame:
    with rasterio.open(HST) as src:
        descriptions = list(src.descriptions)
        if "ONR" not in descriptions:
            raise ValueError(f"ONR road band absent: {descriptions}")
        band = descriptions.index("ONR") + 1
        onroad = src.read(band, masked=True).filled(0.0).astype(np.float64)
        rows, cols = np.indices(onroad.shape)
        x, y = rasterio.transform.xy(src.transform, rows, cols, offset="center")
        to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        lon, lat = to_wgs84.transform(np.asarray(x).ravel(), np.asarray(y).ravel())
    ix, iy = cell_key(np.asarray(lon), np.asarray(lat))
    frame = pd.DataFrame({"ix": ix, "iy": iy, "hestia_onr_tc": onroad.ravel()})
    # Native values are tC / 100 m cell / year.  Retain tC for association
    # statistics; conversion by 44/12 is constant and does not affect them.
    return frame.groupby(["ix", "iy"], as_index=False).hestia_onr_tc.sum()


def model_cells() -> pd.DataFrame:
    # The reconstruction script writes the full-fit model prediction only after
    # road activity has been extracted.  For this independent grid check, use
    # its frozen target-scale spatial predictions when available.
    pred = ROOT / "Experiments/external_validation_v1/phoenix_model_reconstruction/results/full_fit_spatial_predictions.parquet"
    if not pred.exists():
        raise FileNotFoundError(f"Expected reconstruction output not yet available: {pred}")
    model = pd.read_parquet(pred)
    model = model.loc[model.year.eq(2017)].copy()
    model["ix"], model["iy"] = cell_key(model.lon.to_numpy(), model.lat.to_numpy())
    return model.groupby(["ix", "iy"], as_index=False).spatial_predicted_emission.sum()


def summarize(d: pd.DataFrame) -> dict:
    y = np.log1p(d.hestia_onr_tc.to_numpy(float))
    p = np.log1p(d.spatial_predicted_emission.to_numpy(float))
    top_y = d.hestia_onr_tc >= d.hestia_onr_tc.quantile(.9)
    top_p = d.spatial_predicted_emission >= d.spatial_predicted_emission.quantile(.9)
    union = (top_y | top_p).sum()
    return {
        "comparison": "Hestia Phoenix-Mesa 2017 ONR versus mobility-model full-fit 0.1-degree prediction",
        "n_common_0p1_degree_cells": int(len(d)),
        "hestia_unit": "tC per aggregated 0.1-degree cell per year",
        "model_unit": "EDGAR-target emission units; absolute magnitudes are not directly harmonized",
        "spearman_raw": float(spearmanr(d.hestia_onr_tc, d.spatial_predicted_emission).statistic),
        "pearson_log": float(np.corrcoef(y, p)[0, 1]),
        "mae_log_unscaled": float(mean_absolute_error(y, p)),
        "rmse_log_unscaled": float(math.sqrt(mean_squared_error(y, p))),
        "top_decile_jaccard": float((top_y & top_p).sum() / union) if union else float("nan"),
        "top_decile_recall": float((top_y & top_p).sum() / top_y.sum()) if top_y.sum() else float("nan"),
        "interpretation": "External spatial-pattern comparison only. Hestia ONR is road motor-vehicle FFCO2 while the model target is EDGAR transport-labelled emission; common sector, boundary and unit differences preclude a direct absolute-error claim.",
    }


def plot(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=220)
    x, y = np.log1p(d.hestia_onr_tc), np.log1p(d.spatial_predicted_emission)
    ax.hexbin(x, y, gridsize=26, mincnt=1, cmap="viridis", linewidths=0)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color="0.25", lw=.8, ls="--")
    ax.set_xlabel("ln(1 + Hestia ONR, aggregated tC yr$^{-1}$)")
    ax.set_ylabel("ln(1 + model prediction, target units)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "hestia_onr_vs_model_0p1deg.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hestia = hestia_onroad_01deg()
    model = model_cells()
    common = hestia.merge(model, on=["ix", "iy"], how="inner")
    common.to_csv(OUT / "hestia_onr_model_common_0p1deg_cells.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(summarize(common), indent=2), encoding="utf-8")
    plot(common)


if __name__ == "__main__":
    main()
