from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V25")
OUT = ROOT / "supplementary_fig"
CACHE = OUT / "data_cache"
PORTO_ZIP = CACHE / "porto_taxi_uci_full.zip"
CHICAGO_CSV = CACHE / "chicago_taxi_2024_sample.csv"
CHICAGO_SERVER_METRICS = CACHE / "chicago_od_validation_metrics_server.csv"
CHICAGO_2017_METRICS = CACHE / "chicago_od_2017_same_year_metrics.csv"
METRICS = Path(
    r"${USER_HOME}\\Documents\Codex\2026-07-09\1\nature_cities_figure1"
    r"\validation_incremental_v2\three_stage_validation_metrics_long.csv"
)

INK = "#253238"
MUTED = "#6B777C"
GRID = "#DCE3E5"
LAND = "#EEF1F0"
WATER = "#FFFFFF"
CPC_COLOR = "#168F7A"
JS_COLOR = "#D17A57"
FLOW_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "flow", ["#DCE9E5", "#75B5A4", "#167F73", "#174D4A"]
)
DIST_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "distance", ["#E7EEF3", "#87ACC1", "#367895", "#173F5F"]
)


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def add_scalebar(ax, km: float, lon: float, lat: float) -> None:
    degrees = km / (111.32 * math.cos(math.radians(lat)))
    ax.plot([lon, lon + degrees], [lat, lat], color=INK, lw=1.1, solid_capstyle="butt", zorder=20)
    ax.plot([lon, lon], [lat - 0.002, lat + 0.002], color=INK, lw=0.8, zorder=20)
    ax.plot([lon + degrees, lon + degrees], [lat - 0.002, lat + 0.002], color=INK, lw=0.8, zorder=20)
    ax.text(lon + degrees / 2, lat + 0.004, f"{km:g} km", ha="center", va="bottom", color=INK, fontsize=7)


def map_axes(ax) -> None:
    ax.set_facecolor(WATER)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def locate_train_zip(path: Path) -> tuple[zipfile.ZipFile, str, zipfile.ZipFile | None]:
    outer = zipfile.ZipFile(path)
    direct = next((n for n in outer.namelist() if n.lower().endswith("train.csv")), None)
    if direct:
        return outer, direct, None
    nested_name = next((n for n in outer.namelist() if n.lower().endswith("train.csv.zip")), None)
    if nested_name is None:
        nested_name = next((n for n in outer.namelist() if n.lower().endswith(".zip") and "train" in n.lower()), None)
    if nested_name is None:
        raise FileNotFoundError(f"No train.csv or nested train archive in {path}")
    nested = zipfile.ZipFile(io.BytesIO(outer.read(nested_name)))
    train = next(n for n in nested.namelist() if n.lower().endswith("train.csv"))
    return nested, train, outer


def load_porto_trajectories(max_rows: int = 1800) -> list[np.ndarray]:
    if not PORTO_ZIP.exists() or PORTO_ZIP.stat().st_size < 100_000_000:
        raise FileNotFoundError(
            f"Porto archive is missing or incomplete: {PORTO_ZIP}. "
            "Download UCI dataset 339 before running."
        )
    archive, train_name, outer = locate_train_zip(PORTO_ZIP)
    selected: list[np.ndarray] = []
    try:
        with archive.open(train_name) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.DictReader(text)
            for row_number, row in enumerate(reader):
                # Stable systematic sample; avoids selecting a visually convenient subset.
                if row_number % 113 != 0 or row.get("MISSING_DATA", "FALSE").upper() == "TRUE":
                    continue
                try:
                    points = np.asarray(json.loads(row["POLYLINE"]), dtype=float)
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                if points.ndim != 2 or len(points) < 5:
                    continue
                valid = (
                    np.isfinite(points).all(axis=1)
                    & (points[:, 0] > -8.75)
                    & (points[:, 0] < -8.45)
                    & (points[:, 1] > 40.98)
                    & (points[:, 1] < 41.28)
                )
                points = points[valid]
                if len(points) < 5:
                    continue
                selected.append(points)
                if len(selected) >= max_rows:
                    break
    finally:
        archive.close()
        if outer is not None:
            outer.close()
    return selected


def draw_porto_dataset() -> dict:
    trajectories = load_porto_trajectories()
    lengths = np.array(
        [haversine_km(p[:-1, 0], p[:-1, 1], p[1:, 0], p[1:, 1]).sum() for p in trajectories]
    )
    upper = float(np.quantile(lengths, 0.95))
    norm = mpl.colors.Normalize(0, upper, clip=True)
    order = np.argsort(lengths)

    fig, ax = plt.subplots(figsize=(7.15, 4.35))
    map_axes(ax)
    ax.set_facecolor(LAND)
    for idx in order:
        p = trajectories[idx]
        ax.plot(p[:, 0], p[:, 1], color=DIST_CMAP(norm(lengths[idx])), lw=0.34, alpha=0.32, zorder=2)
    all_points = np.concatenate(trajectories)
    lo = np.quantile(all_points, 0.003, axis=0)
    hi = np.quantile(all_points, 0.997, axis=0)
    pad = (hi - lo) * 0.045
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect(1 / math.cos(math.radians(float(np.mean(all_points[:, 1])))))

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=DIST_CMAP)
    cax = ax.inset_axes([0.045, 0.075, 0.018, 0.29])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.outline.set_linewidth(0.55)
    cb.ax.tick_params(length=2.2, pad=2)
    cb.set_label("Observed trajectory length (km)", labelpad=4, fontsize=7.5)
    add_scalebar(ax, 5, ax.get_xlim()[1] - 0.100, ax.get_ylim()[0] + 0.015)
    ax.legend(
        handles=[Line2D([0], [0], color=DIST_CMAP(0.72), lw=1.5, alpha=0.75, label="GPS trajectory (15-s sampling)")],
        loc="upper right",
        frameon=False,
        handlelength=2.4,
    )
    fig.savefig(OUT / "FigS_OD1_Porto_Taxi_observed_trajectories.png", dpi=360, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / "FigS_OD1_Porto_Taxi_observed_trajectories.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return {"trajectories_plotted": len(trajectories), "length_color_limit_km_p95": upper}


def load_chicago() -> pd.DataFrame:
    d = pd.read_csv(CHICAGO_CSV)
    cols = [
        "pickup_centroid_longitude", "pickup_centroid_latitude",
        "dropoff_centroid_longitude", "dropoff_centroid_latitude",
    ]
    for col in cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=cols)
    return d[
        d.pickup_centroid_longitude.between(-88.05, -87.50)
        & d.dropoff_centroid_longitude.between(-88.05, -87.50)
        & d.pickup_centroid_latitude.between(41.62, 42.05)
        & d.dropoff_centroid_latitude.between(41.62, 42.05)
    ].copy()


def draw_chicago_dataset() -> dict:
    d = load_chicago()
    grid = 0.0125
    for prefix in ("pickup", "dropoff"):
        d[f"{prefix}_x"] = np.floor(d[f"{prefix}_centroid_longitude"] / grid).astype(int)
        d[f"{prefix}_y"] = np.floor(d[f"{prefix}_centroid_latitude"] / grid).astype(int)
    od = (
        d.groupby(["pickup_x", "pickup_y", "dropoff_x", "dropoff_y"], observed=True)
        .size().rename("flow").reset_index()
    )
    od = od[(od.pickup_x != od.dropoff_x) | (od.pickup_y != od.dropoff_y)]
    od = od.nlargest(min(500, len(od)), "flow").copy()
    od["o_lon"] = (od.pickup_x + 0.5) * grid
    od["o_lat"] = (od.pickup_y + 0.5) * grid
    od["d_lon"] = (od.dropoff_x + 0.5) * grid
    od["d_lat"] = (od.dropoff_y + 0.5) * grid
    norm = mpl.colors.LogNorm(max(1, float(od.flow.min())), float(od.flow.max()))
    segments = np.stack(
        [od[["o_lon", "o_lat"]].to_numpy(), od[["d_lon", "d_lat"]].to_numpy()], axis=1
    )

    fig, ax = plt.subplots(figsize=(7.15, 4.35))
    map_axes(ax)
    ax.set_facecolor(LAND)
    lc = LineCollection(
        segments,
        colors=FLOW_CMAP(norm(od.flow.to_numpy())),
        linewidths=0.22 + 1.35 * norm(od.flow.to_numpy()),
        alpha=0.38,
        zorder=2,
    )
    ax.add_collection(lc)
    nodes = pd.concat(
        [od[["o_lon", "o_lat", "flow"]].rename(columns={"o_lon": "lon", "o_lat": "lat"}),
         od[["d_lon", "d_lat", "flow"]].rename(columns={"d_lon": "lon", "d_lat": "lat"})]
    ).groupby(["lon", "lat"], as_index=False).flow.sum()
    ax.scatter(nodes.lon, nodes.lat, s=1.8 + 8 * norm(nodes.flow), c=nodes.flow, cmap=FLOW_CMAP,
               norm=norm, edgecolors="white", linewidths=0.18, alpha=0.88, zorder=5)
    ax.set_xlim(-87.96, -87.51)
    ax.set_ylim(41.63, 42.04)
    ax.set_aspect(1 / math.cos(math.radians(41.84)))
    cax = ax.inset_axes([0.045, 0.075, 0.018, 0.29])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=FLOW_CMAP), cax=cax, orientation="vertical")
    cb.outline.set_linewidth(0.55)
    cb.ax.tick_params(length=2.2, pad=2)
    cb.set_label("Observed OD frequency", labelpad=4, fontsize=7.5)
    add_scalebar(ax, 10, -87.66, 41.655)
    ax.legend(
        handles=[
            Line2D([0], [0], color=FLOW_CMAP(0.75), lw=1.2, label="Aggregated directed OD flow"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=FLOW_CMAP(0.62),
                   markeredgecolor="white", markersize=5, label="Pickup or drop-off zone"),
        ],
        loc="upper right", frameon=False, handlelength=2.3,
    )
    fig.savefig(OUT / "FigS_OD2_Chicago_Taxi_observed_OD.png", dpi=360, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / "FigS_OD2_Chicago_Taxi_observed_OD.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return {"taxi_records_used": len(d), "top_od_flows_plotted": len(od), "aggregation_grid_degrees": grid}


def validation_values(dataset: str) -> pd.DataFrame:
    if dataset == "Chicago Taxi 2017":
        if CHICAGO_2017_METRICS.exists():
            d = pd.read_csv(CHICAGO_2017_METRICS)
            return d.set_index("target").loc[["origin_grid", "destination_grid", "distance_bin"]].reset_index()
        d = pd.read_csv(CHICAGO_SERVER_METRICS if CHICAGO_SERVER_METRICS.exists() else METRICS)
        targets = ["origin_grid", "destination_grid", "distance_bin"]
        if "stage" in d.columns:
            keep = d[d.year.astype(int).eq(2017) & d.stage.eq("calibrated") & d.target.isin(targets)].copy()
        else:
            keep = d[d.dataset.eq("Chicago Taxi") & d.year.astype(int).eq(2017)
                     & d.variant.eq("calibrated") & d.target.isin(targets)].copy()
        if set(keep.target) != set(targets):
            raise ValueError(f"Missing 2017 Chicago metrics: {keep.target.tolist()}")
        return keep.set_index("target").loc[targets].reset_index()

    # Chicago's current temporal-transfer run is kept separately because the
    # server-side rerun calibrates origin, destination, and distance jointly.
    if dataset == "Chicago Taxi" and CHICAGO_SERVER_METRICS.exists():
        d = pd.read_csv(CHICAGO_SERVER_METRICS)
        targets = ["origin_grid", "destination_grid", "distance_bin"]
        keep = d[
            d.year.astype(int).eq(2024)
            & d.stage.eq("temporal transfer")
            & d.target.isin(targets)
        ].copy()
        if set(keep.target) != set(targets):
            raise ValueError(f"Missing server-side Chicago metrics: {keep.target.tolist()}")
        return keep.set_index("target").loc[targets].reset_index()

    d = pd.read_csv(METRICS)
    if dataset == "Porto Taxi":
        od = d[
            d.dataset.eq(dataset)
            & d.year.astype(int).eq(2024)
            & d.variant.eq("calibrated_heldout")
            & d.target.isin(["origin_grid", "destination_grid", "distance_bin", "midpoint_grid"])
        ].copy()
        occupancy = d[
            d.dataset.eq(dataset)
            & d.year.astype(int).eq(2024)
            & d.variant.eq("postprocessed")
            & d.target.eq("full-path occupancy")
        ].copy()
        keep = pd.concat([od, occupancy], ignore_index=True)
        targets = ["origin_grid", "destination_grid", "distance_bin", "midpoint_grid", "full-path occupancy"]
    else:
        targets = ["origin_grid", "destination_grid", "distance_bin"]
        keep = d[
            d.dataset.eq(dataset)
            & d.year.astype(int).eq(2024)
            & d.variant.eq("temporal transfer")
            & d.target.isin(targets)
        ].copy()
    if set(keep.target) != set(targets):
        raise ValueError(f"Missing validation metrics for {dataset}: {keep.target.tolist()}")
    return keep.set_index("target").loc[targets].reset_index()


def draw_validation(dataset: str, filename: str) -> dict:
    d = validation_values(dataset)
    labels = (["Origin", "Destination", "Trip distance", "Path midpoint", "Full-path\noccupancy"]
              if dataset == "Porto Taxi" else ["Origin", "Destination", "Trip distance"])
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=((10.4 if n == 5 else 7.15), 2.25), sharey=True)
    for i, (ax, row, label) in enumerate(zip(axes, d.itertuples(), labels)):
        values = [row.cpc, row.js_divergence]
        bars = ax.bar([0, 1], values, width=0.58, color=[CPC_COLOR, JS_COLOR], alpha=0.86,
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=7.2, color=INK)
        ax.set_xticks([0, 1], ["CPC", "JS"])
        ax.set_xlabel(label, labelpad=5, color=INK)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.55, 1.55)
        ax.grid(axis="y", color=GRID, lw=0.55, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if i > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.text(-0.16, 1.04, chr(97 + i), transform=ax.transAxes, fontsize=10, fontweight="bold",
                ha="left", va="bottom", color=INK)
    axes[0].set_ylabel("Metric value")
    note = ""
    fig.legend(
        handles=[Patch(facecolor=CPC_COLOR, alpha=0.86, label="CPC (higher is better)"),
                 Patch(facecolor=JS_COLOR, alpha=0.86, label="JS divergence (lower is better)"),
                 Line2D([], [], color="none", label=note)],
        loc="lower left", bbox_to_anchor=(0.055, 0.995), ncol=3, frameon=False,
        handlelength=1.2, columnspacing=1.5, borderaxespad=0,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.81, wspace=0.17)
    fig.savefig(OUT / f"{filename}.png", dpi=360, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return d[["target", "cpc", "js_divergence", "spearman"]].to_dict("records")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    style()
    audit = {
        "Porto Taxi dataset": draw_porto_dataset(),
        "Chicago Taxi dataset": draw_chicago_dataset(),
        "Porto Taxi validation": draw_validation("Porto Taxi", "FigS_OD3_Porto_Taxi_validation_2024"),
        "Chicago Taxi validation": draw_validation("Chicago Taxi", "FigS_OD4_Chicago_Taxi_validation_2024"),
        "Chicago Taxi 2017 validation": draw_validation("Chicago Taxi 2017", "FigS_OD4_Chicago_Taxi_validation_2017"),
        "notes": {
            "CPC": "Higher is better; overlap of normalized observed and simulated distributions.",
            "JS divergence": "Base-2 Jensen-Shannon divergence; lower is better.",
            "Chicago": "The 2017 calibration rules jointly adjust origin, destination, and trip-distance distributions; the 2024 results are an independent temporal transfer.",
        },
    }
    (OUT / "FigS_OD_validation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
