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
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "trace_generate"))

from build_travel_simulation_figures_v3 import (
    CITY_META,
    CITY_ORDER,
    DataLocator,
    apply_city_axes,
    city_bounds,
    draw_world_panel,
    parse_graphml,
    set_style,
)


YEARS = (2017, 2024)
MODE_NAMES = ("Mode 1\nslowest", "Mode 2", "Mode 3", "Mode 4", "Mode 5\nfastest")
MODE_COLORS = ("#2A6F97", "#4D908E", "#72A85A", "#E3A832", "#C8553D")
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
) -> tuple[list[dict[int, float]], np.ndarray, float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"mode_edge_flow_{city_id}_{year}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        edge_flows = []
        for mode in range(5):
            edge_flows.append(dict(zip(cached[f"code_{mode}"].tolist(), cached[f"flow_{mode}"].tolist())))
        return edge_flows, cached["mode_total"], float(cached["total_flow"][0])

    edge_flows = [defaultdict(float) for _ in range(5)]
    mode_totals = np.zeros(5, dtype=float)
    total_flow = 0.0
    sample_size = 30000
    candidates = []
    for route_path in locator.routes(city_id, year):
        chunk = chunk_from_route(route_path)
        pred_path = prediction_file(pred_root, city_id, year, chunk)
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions: {pred_path}")

        routes = pd.read_parquet(route_path, columns=["flow"]).reset_index(drop=True)
        predictions = pd.read_parquet(
            pred_path,
            columns=[f"prob_mode_{mode}" for mode in range(1, 6)],
        ).reset_index(drop=True)
        if len(routes) != len(predictions):
            raise ValueError(f"Row mismatch: {route_path.name} ({len(routes)}) vs {pred_path.name} ({len(predictions)})")

        probabilities = predictions.to_numpy(dtype=float)
        probabilities = np.nan_to_num(probabilities, nan=0.2, posinf=0.2, neginf=0.2)
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
        flows = np.nan_to_num(routes["flow"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
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
        candidates.append(
            pd.DataFrame(
                {
                    "priority": priority[local],
                    "chunk": chunk,
                    "row_index": local,
                    "flow": flows[local],
                    **{f"prob_{mode}": probabilities[local, mode] for mode in range(5)},
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
                [getattr(row, f"prob_{mode}") for mode in range(5)], dtype=float
            )
            for code in sequences.iloc[int(row.row_index)]:
                code = int(code)
                for mode in range(5):
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
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=5)


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
    fig = plt.figure(figsize=(17.5, 23.5), facecolor="white")
    outer = fig.add_gridspec(
        7,
        1,
        height_ratios=[2.30, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05],
        hspace=0.22,
        left=0.025,
        right=0.992,
        top=0.988,
        bottom=0.026,
    )

    top = outer[0].subgridspec(1, 3, width_ratios=[1.55, 1.0, 1.08], wspace=0.14)
    world_ax = fig.add_subplot(top[0, 0])
    draw_world_panel(world_ax, cities, args.world)
    confusion_ax = fig.add_subplot(top[0, 1])
    draw_metric_image(confusion_ax, args.metrics / "test_confusion_matrix_v5.png", "b  Held-out classification performance")
    latent_ax = fig.add_subplot(top[0, 2])
    draw_metric_image(latent_ax, args.metrics / "test_latent_space_v5.png", "c  Latent representation of held-out trajectories")

    cache_dir = args.output.parent / "mode_figure_cache"
    share_records = []
    for panel_index, city_id in enumerate(CITY_ORDER):
        city_grid = outer[panel_index + 1].subgridspec(2, 5, hspace=0.035, wspace=0.035)
        bounds = city_bounds(locator, city_id)
        first_ax = None
        for row, year in enumerate(YEARS):
            mode_edges, mode_totals, total_flow = aggregate_mode_edge_flows(
                locator, args.predictions, city_id, year, cache_dir
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
                        for mode in range(5)
                    },
                }
            )
            for mode in range(5):
                ax = fig.add_subplot(city_grid[row, mode])
                if first_ax is None:
                    first_ax = ax
                share = mode_totals[mode] / max(total_flow, 1e-12) * 100.0
                draw_mode_map(ax, display_roads, geometries, mode_edges[mode], MODE_COLORS[mode], bounds, share)
                if row == 0:
                    ax.set_title(MODE_NAMES[mode], fontsize=8.4, color=INK, pad=3)
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
            f"{'defghi'[panel_index]}  {CITY_META[city_id]['name']}",
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
    legend.extend(Line2D([0], [0], color=color, lw=2.0, label=name.replace("\n", " ")) for color, name in zip(MODE_COLORS, MODE_NAMES))
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=6,
        frameon=False,
        fontsize=8.5,
        title=(
            "Percentages use all simulated routes; maps use 30,000 flow-weighted routes per city-year "
            "(bright lines: edge flow >= P90)"
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
    parser.add_argument("--city-points", type=Path, default=ROOT / "global_city_points_2024.csv")
    parser.add_argument("--world", type=Path, default=ROOT / "ne_110m_admin_0_countries.geojson")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "Figure_4_five_mode_inference_v5")
    return parser.parse_args()


if __name__ == "__main__":
    build_figure(parse_args())
