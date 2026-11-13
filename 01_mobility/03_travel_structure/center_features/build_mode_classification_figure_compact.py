import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "trace_generate"))

from build_travel_simulation_figures_v3 import (
    CITY_META,
    DataLocator,
    apply_city_axes,
    city_bounds,
    draw_geojson,
    parse_graphml,
    set_style,
)


YEARS = (2017, 2024)
MODE_PALETTES = {
    4: ("#2A6F97", "#4D908E", "#E3A832", "#C8553D"),
    5: ("#2A6F97", "#4D908E", "#72A85A", "#E3A832", "#C8553D"),
}
ROAD_COLOR = "#D9DEDD"
INK = "#1F292B"
MUTED = "#687476"


def prediction_file(pred_root: Path, city_id: int, year: int, chunk: str) -> Path:
    return pred_root / f"{city_id}_{year}" / f"{city_id}_chunk_{chunk}_predictions.parquet"


def chunk_from_route(path: Path) -> str:
    match = re.search(r"_(\d{4})\.parquet$", path.name)
    if not match:
        raise ValueError(f"Cannot identify route chunk: {path}")
    return match.group(1)


def aggregate_mode_edge_flows(
    locator: DataLocator,
    pred_root: Path,
    city_id: int,
    year: int,
    cache_dir: Path,
    num_classes: int,
) -> tuple[list[dict[int, float]], np.ndarray, float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"mode_edge_flow_m{num_classes}_{city_id}_{year}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        edge_flows = []
        for mode in range(num_classes):
            edge_flows.append(dict(zip(cached[f"code_{mode}"].tolist(), cached[f"flow_{mode}"].tolist())))
        return edge_flows, cached["mode_total"], float(cached["total_flow"][0])

    edge_flows = [defaultdict(float) for _ in range(num_classes)]
    mode_totals = np.zeros(num_classes, dtype=float)
    total_flow = 0.0
    sample_size = 30000
    candidates = []
    for route_path in locator.routes(city_id, year):
        chunk = chunk_from_route(route_path)
        pred_path = prediction_file(pred_root, city_id, year, chunk)
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions: {pred_path}")

        predictions = pd.read_parquet(
            pred_path,
            columns=[
                "seg_id",
                "flow",
                *[f"prob_mode_{mode}" for mode in range(1, num_classes + 1)],
            ],
        ).reset_index(drop=True)
        probability_columns = [f"prob_mode_{mode}" for mode in range(1, num_classes + 1)]
        probabilities = predictions[probability_columns].to_numpy(dtype=float)
        probabilities = np.nan_to_num(
            probabilities,
            nan=1.0 / num_classes,
            posinf=1.0 / num_classes,
            neginf=1.0 / num_classes,
        )
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
        source_rows = pq.ParquetFile(route_path).metadata.num_rows
        inclusion_probability = min(1.0, len(predictions) / max(source_rows, 1))
        flows = np.nan_to_num(predictions["flow"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        flows /= max(inclusion_probability, 1e-12)
        flows = np.maximum(flows, 0.0)
        total_flow += float(flows.sum())
        mode_totals += np.sum(flows[:, None] * probabilities, axis=0)

        # Priority sampling selects a fixed-size, flow-weighted route sample for
        # map rendering. Full-sample mode totals above remain unaffected.
        seed = city_id * 10_000_019 + year * 1_009 + int(chunk)
        rng = np.random.default_rng(seed)
        priority = np.log(np.maximum(rng.random(len(flows)), np.finfo(float).tiny)) / np.maximum(flows, 1e-12)
        local_n = min(sample_size, len(flows))
        local = np.argpartition(priority, -local_n)[-local_n:]
        row_indices = predictions["seg_id"].str.rsplit("_", n=1).str[-1].astype(np.int64).to_numpy()
        candidates.append(
            pd.DataFrame(
                {
                    "priority": priority[local],
                    "chunk": chunk,
                    "row_index": row_indices[local],
                    "flow": flows[local],
                    **{f"prob_{mode}": probabilities[local, mode] for mode in range(num_classes)},
                }
            )
        )

    sampled = pd.concat(candidates, ignore_index=True)
    sampled = sampled.nlargest(min(sample_size, len(sampled)), "priority")
    sampled_flow = float(sampled["flow"].sum())
    route_lookup = {chunk_from_route(path): path for path in locator.routes(city_id, year)}
    for chunk, selected in sampled.groupby("chunk", sort=False):
        route_path = route_lookup[str(chunk)]
        sequences = pd.read_parquet(route_path, columns=["edge_sequence"])["edge_sequence"]
        for row in selected.itertuples(index=False):
            weighted = float(row.flow) * np.asarray(
                [getattr(row, f"prob_{mode}") for mode in range(num_classes)], dtype=float
            )
            for code in sequences.iloc[int(row.row_index)]:
                code = int(code)
                for mode in range(num_classes):
                    edge_flows[mode][code] += float(weighted[mode])

    payload = {
        "mode_total": mode_totals,
        "total_flow": np.asarray([total_flow]),
        "sample_size": np.asarray([len(sampled)]),
        "sampled_flow": np.asarray([sampled_flow]),
    }
    for mode, edge_flow in enumerate(edge_flows):
        payload[f"code_{mode}"] = np.fromiter(edge_flow.keys(), dtype=np.int64)
        payload[f"flow_{mode}"] = np.fromiter(edge_flow.values(), dtype=float)
    np.savez_compressed(cache_path, **payload)
    return [dict(item) for item in edge_flows], mode_totals, total_flow


def draw_metric_image(ax: plt.Axes, path: Path, title: str) -> None:
    image = np.asarray(Image.open(path).convert("RGB")).copy()
    # Hide only the source title; preserve the full plotting area and axes.
    image[: int(image.shape[0] * 0.052), :, :] = 255
    nonwhite = np.any(image < 248, axis=2)
    rows, cols = np.where(nonwhite)
    if len(rows):
        pad = max(8, int(min(image.shape[:2]) * 0.008))
        top = max(0, int(rows.min()) - pad)
        bottom = min(image.shape[0], int(rows.max()) + pad + 1)
        left = max(0, int(cols.min()) - pad)
        right = min(image.shape[1], int(cols.max()) + pad + 1)
        image = image[top:bottom, left:right]
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=5)


def mode_names(num_classes: int) -> tuple[str, ...]:
    return tuple(
        "Mode 1\nslowest" if mode == 0 else
        f"Mode {mode + 1}\nfastest" if mode == num_classes - 1 else
        f"Mode {mode + 1}"
        for mode in range(num_classes)
    )


def draw_world_compact(
    ax: plt.Axes,
    cities: pd.DataFrame,
    world_path: Path,
    city_order: tuple[int, ...],
) -> None:
    draw_geojson(ax, world_path)
    ax.scatter(cities["lon"], cities["lat"], s=2.0, color="#879397", alpha=0.30, linewidth=0)
    selected = cities[cities["city_id"].isin(city_order)].set_index("city_id")
    for letter, city_id in zip("def", city_order):
        row = selected.loc[city_id]
        ax.scatter(row["lon"], row["lat"], s=34, color="#C8553D", edgecolor="white", linewidth=0.9, zorder=4)
        ax.annotate(
            f"{letter}  {CITY_META[city_id]['name']}",
            (row["lon"], row["lat"]),
            xytext=(6, 4 if row["lat"] >= 0 else -11),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color=INK,
            zorder=5,
        )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 86)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title("a  Global coverage and selected examples", loc="left", fontsize=10.5, fontweight="bold", pad=5)
    ax.text(0.01, 0.025, f"{len(cities):,} urban centres", transform=ax.transAxes, fontsize=8.0, color=MUTED)


def draw_mode_map(
    ax: plt.Axes,
    roads: list[np.ndarray],
    geometries: dict[int, np.ndarray],
    edge_flow: dict[int, float],
    color: str,
    bounds: tuple[float, float, float, float],
    share: float,
) -> None:
    ax.add_collection(LineCollection(roads, colors=ROAD_COLOR, linewidths=0.11, alpha=0.42, rasterized=True))
    valid = [(code, flow) for code, flow in edge_flow.items() if code in geometries and flow > 0]
    if valid:
        values = np.asarray([flow for _, flow in valid], dtype=float)
        lines = [geometries[code] for code, _ in valid]
        threshold = float(np.nanpercentile(values, 90))
        high = values >= threshold
        low_lines = [line for line, flag in zip(lines, high) if not flag]
        if low_lines:
            ax.add_collection(LineCollection(low_lines, colors=color, linewidths=0.16, alpha=0.16, rasterized=True))
        high_values = np.log1p(values[high])
        high_lines = [line for line, flag in zip(lines, high) if flag]
        lo, hi = np.nanpercentile(high_values, [2, 99.5])
        if math.isclose(float(lo), float(hi)):
            hi = lo + 1.0
        scaled = np.clip((high_values - lo) / (hi - lo), 0, 1)
        ax.add_collection(
            LineCollection(
                high_lines,
                colors=color,
                linewidths=0.30 + 1.20 * scaled,
                alpha=0.84,
                rasterized=True,
            )
        )
    apply_city_axes(ax, bounds)
    share_label = "<0.1%" if 0.0 < share < 0.1 else f"{share:.1f}%"
    ax.text(
        0.98,
        0.03,
        share_label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.2},
    )


def build_figure(args: argparse.Namespace) -> None:
    set_style()
    locator = DataLocator(local_data_root=args.local_data_root)
    cities = pd.read_csv(args.city_points)
    city_order = tuple(args.city_ids)
    if len(city_order) != 3:
        raise ValueError("The compact publication figure requires exactly three city IDs.")
    num_classes = args.num_classes
    colors = MODE_PALETTES[num_classes]
    names = mode_names(num_classes)
    fig = plt.figure(figsize=(16.8, 12.4), facecolor="white")
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.82, 2.05, 2.05, 2.05],
        hspace=0.18,
        left=0.028,
        right=0.99,
        top=0.985,
        bottom=0.052,
    )

    top = outer[0].subgridspec(1, 3, width_ratios=[1.50, 0.93, 1.10], wspace=0.025)
    world_ax = fig.add_subplot(top[0, 0])
    draw_world_compact(world_ax, cities, args.world, city_order)
    confusion_ax = fig.add_subplot(top[0, 1])
    draw_metric_image(confusion_ax, args.metrics / "test_confusion_matrix_v5.png", "b  Held-out classification performance")
    latent_ax = fig.add_subplot(top[0, 2])
    draw_metric_image(latent_ax, args.metrics / "test_latent_space_v5.png", "c  Latent representation of held-out trajectories")

    cache_dir = args.cache_dir or (args.output.parent / "mode_figure_cache")
    share_records = []
    for panel_index, city_id in enumerate(city_order):
        city_grid = outer[panel_index + 1].subgridspec(2, num_classes, hspace=0.035, wspace=0.03)
        bounds = city_bounds(locator, city_id)
        first_ax = None
        for row, year in enumerate(YEARS):
            mode_edges, mode_totals, total_flow = aggregate_mode_edge_flows(
                locator, args.predictions, city_id, year, cache_dir, num_classes
            )
            roads, geometries = parse_graphml(locator.road(city_id, year))
            road_stride = max(1, math.ceil(len(roads) / 60000))
            display_roads = roads[::road_stride]
            share_records.append(
                {
                    "city_id": city_id,
                    "city": CITY_META[city_id]["name"],
                    "region": CITY_META[city_id]["region"],
                    "year": year,
                    "total_flow": total_flow,
                    **{
                        f"mode_{mode + 1}_share": mode_totals[mode] / max(total_flow, 1e-12)
                        for mode in range(num_classes)
                    },
                }
            )
            for mode in range(num_classes):
                ax = fig.add_subplot(city_grid[row, mode])
                if first_ax is None:
                    first_ax = ax
                share = mode_totals[mode] / max(total_flow, 1e-12) * 100.0
                draw_mode_map(ax, display_roads, geometries, mode_edges[mode], colors[mode], bounds, share)
                if row == 0:
                    ax.set_title(names[mode], fontsize=8.4, color=INK, pad=3)
                if mode == 0:
                    ax.text(
                        -0.025,
                        0.5,
                        str(year),
                        transform=ax.transAxes,
                        rotation=90,
                        ha="right",
                        va="center",
                        fontsize=8.4,
                        fontweight="bold",
                        color=MUTED,
                    )
            del roads, display_roads, geometries

        box = first_ax.get_position()
        fig.text(
            box.x0,
            box.y1 + 0.017,
            f"{'def'[panel_index]}  {CITY_META[city_id]['name']}",
            ha="left",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
        )
        fig.text(
            0.992,
            box.y1 + 0.017,
            CITY_META[city_id]["region"],
            ha="right",
            va="bottom",
            fontsize=8.4,
            color=MUTED,
        )

    legend = [Line2D([0], [0], color=ROAD_COLOR, lw=1.2, label="OSM road network")]
    legend.extend(Line2D([0], [0], color=color, lw=2.0, label=name.replace("\n", " ")) for color, name in zip(colors, names))
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=num_classes + 1,
        frameon=False,
        fontsize=8.5,
        title=(
            "Percentages are flow-weighted estimates from a deterministic random sample of each route chunk; "
            "maps show sampled routes (bright lines: edge flow >= P90)"
        ),
        title_fontsize=7.8,
        handlelength=2.5,
        columnspacing=1.6,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(share_records).to_csv(
        args.output.with_name(f"{args.output.name}_mode_shares.csv"), index=False
    )
    fig.savefig(args.output.with_suffix(".png"), dpi=350, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(
        args.output.with_suffix(".jpg"),
        dpi=350,
        bbox_inches="tight",
        pad_inches=0.05,
        pil_kwargs={"quality": 96, "subsampling": 0},
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-data-root", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, choices=(4, 5), required=True)
    parser.add_argument("--city-ids", type=int, nargs=3, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--city-points", type=Path, default=ROOT / "global_city_points_2024.csv")
    parser.add_argument("--world", type=Path, default=ROOT / "ne_110m_admin_0_countries.geojson")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "Figure_4_five_mode_inference_v5")
    return parser.parse_args()


if __name__ == "__main__":
    build_figure(parse_args())
