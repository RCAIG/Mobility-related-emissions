from pathlib import Path
import json
import sys

sys.path.insert(0, r"${WORKSPACE_ROOT}\\.pydeps_analysis")

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError


ROOT = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V25")
OUT = ROOT / "supplementary_fig"
DATA = OUT / "worldmove_validation_all_income"
WORLD_SHP = Path(
    r"${WORKSPACE_ROOT}\\.pydeps_analysis\pyogrio\tests\fixtures\naturalearth_lowres\naturalearth_lowres.shp"
)

INK = "#253238"
GRID = "#DCE3E5"
LAND = "#E8ECEB"
CPC_COLOR = "#168F7A"
JS_COLOR = "#D17A57"
POINT_COLOR = "#277F73"
OD_BACKGROUND = "#AFCBC5"
OD_NODE = "#D79A32"
OD_FLOW = mpl.colors.LinearSegmentedColormap.from_list(
    "worldmove_od", ["#6FA9A3", "#247A78", "#155653"]
)


def set_style():
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=360, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def draw_world_map(locations):
    world = gpd.read_file(WORLD_SHP)
    fig, ax = plt.subplots(figsize=(7.15, 3.55))
    world.plot(ax=ax, facecolor=LAND, edgecolor="#B9C4C1", linewidth=0.28, zorder=1)
    ax.scatter(
        locations.lon,
        locations.lat,
        s=13,
        color=POINT_COLOR,
        edgecolors="white",
        linewidths=0.32,
        alpha=0.82,
        zorder=3,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 82)
    ax.set_aspect(1.35)
    ax.set_axis_off()
    countries = locations.iso3.nunique()
    legend = Line2D(
        [], [], marker="o", linestyle="none", markersize=4.5,
        markerfacecolor=POINT_COLOR, markeredgecolor="white", markeredgewidth=0.35,
        label=f"Country-balanced matched cities (n={len(locations)}; {countries} countries)",
    )
    ax.legend(handles=[legend], loc="lower left", bbox_to_anchor=(0.012, 0.025),
              frameon=False, handletextpad=0.45, borderaxespad=0)
    save(fig, "FigS_WorldMove_country_balanced_cities_world_map")


def draw_od_examples(matrices, labels):
    keys = list(labels)
    fig, axes = plt.subplots(3, 3, figsize=(7.15, 6.25))
    for i, (ax, key) in enumerate(zip(axes.flat, keys)):
        matrix = matrices[key].astype(float)
        xy = matrices[key + "__xy"].astype(float)
        oi, di = np.nonzero(matrix > 0)
        valid = (oi != di) & np.isfinite(xy[oi]).all(axis=1) & np.isfinite(xy[di]).all(axis=1)
        oi, di = oi[valid], di[valid]
        flows = matrix[oi, di]
        lat0 = float(np.nanmean(xy[:, 1]))
        projected = xy.copy()
        projected[:, 0] *= np.cos(np.radians(lat0))
        finite = np.isfinite(projected).all(axis=1)
        segments = np.stack([projected[oi], projected[di]], axis=1)
        high = flows >= np.percentile(flows, 97)
        low = np.flatnonzero(~high)
        if len(low) > 60000:
            low = low[np.linspace(0, len(low) - 1, 60000, dtype=int)]
        ax.add_collection(LineCollection(
            segments[low], colors=OD_BACKGROUND, linewidths=0.10,
            alpha=0.055, rasterized=True, zorder=1,
        ))
        log_flows = np.log1p(flows[high])
        v0, v1 = np.percentile(log_flows, [1, 99.5])
        norm = mpl.colors.Normalize(v0, max(v1, v0 + 1e-9))
        lines = LineCollection(
            segments[high], cmap=OD_FLOW, norm=norm,
            linewidths=0.20 + 0.65 * np.clip(norm(log_flows), 0, 1),
            alpha=0.82, rasterized=True, zorder=2,
        )
        lines.set_array(log_flows)
        ax.add_collection(lines)
        strength = matrix.sum(axis=0) + matrix.sum(axis=1)
        active = np.flatnonzero((strength > 0) & finite)
        if len(active):
            cutoff = np.quantile(strength[active], 0.95)
            nodes = active[strength[active] >= cutoff]
            ax.scatter(projected[nodes, 0], projected[nodes, 1], s=3,
                       color=OD_NODE, edgecolors="white", linewidths=0.13, zorder=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(0.035)
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.5, -0.075, labels[key], transform=ax.transAxes, ha="center",
                va="top", fontsize=7, color=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.11, top=0.985, wspace=0.13, hspace=0.26)
    # Place letters relative to the fixed 3 x 3 GridSpec slots. The map axes
    # themselves shrink differently under equal aspect ratios, which otherwise
    # makes axes-relative panel letters appear misaligned.
    for i, ax in enumerate(axes.flat):
        slot = ax.get_subplotspec().get_position(fig)
        fig.text(slot.x0 - 0.010, slot.y1 + 0.004, chr(97 + i),
                 fontsize=9.5, fontweight="bold", ha="left", va="bottom",
                 color=INK)
    handles = [
        Line2D([], [], color=OD_BACKGROUND, lw=1, label="Other OD flows"),
        Line2D([], [], color="#155653", lw=1.5, label="Top 3% OD flows"),
        Line2D([], [], linestyle="", marker="o", markersize=3,
               markerfacecolor=OD_NODE, markeredgecolor="white", markeredgewidth=0.13,
               label="High-activity nodes"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=3, frameon=False, fontsize=6.5, handlelength=1.7,
               columnspacing=1.2)
    save(fig, "FigS_WorldMove_nine_city_OD_examples")


def bootstrap_ci(values, seed=20261023, n_boot=10000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, 500):
        stop = min(start + 500, n_boot)
        sample = rng.choice(values, size=(stop - start, len(values)), replace=True)
        means[start:stop] = sample.mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def draw_validation(metrics):
    dimensions = [
        ("Origin", "origin"),
        ("Destination", "destination"),
        ("Trip distance", "distance"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25), sharey=True)
    audit = {}
    for i, (ax, (label, prefix)) in enumerate(zip(axes, dimensions)):
        cpc_values = metrics[f"{prefix}_cpc_calibrated_holdout"].to_numpy(float)
        js_values = metrics[f"{prefix}_js_calibrated_holdout"].to_numpy(float)
        means = np.array([np.nanmean(cpc_values), np.nanmean(js_values)])
        intervals = np.array([bootstrap_ci(cpc_values, seed=20261023 + i),
                              bootstrap_ci(js_values, seed=20261123 + i)])
        yerr = np.vstack([means - intervals[:, 0], intervals[:, 1] - means])
        bars = ax.bar(
            [0, 1], means, width=0.58, color=[CPC_COLOR, JS_COLOR], alpha=0.86,
            edgecolor="white", linewidth=0.5, yerr=yerr, capsize=2.2,
            error_kw={"elinewidth": 0.65, "capthick": 0.65, "ecolor": INK}, zorder=3,
        )
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=7.2, color=INK)
        ax.set_xticks([0, 1], ["CPC", "JS"])
        ax.set_xlabel(label, labelpad=5, color=INK)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.55, 1.55)
        ax.grid(axis="y", color=GRID, lw=0.55, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if i:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.text(-0.16, 1.04, chr(97 + i), transform=ax.transAxes, fontsize=10,
                fontweight="bold", ha="left", va="bottom", color=INK)
        audit[prefix] = {
            "n_cities": int(np.isfinite(cpc_values).sum()),
            "mean_cpc": float(means[0]), "mean_js": float(means[1]),
            "cpc_95pct_bootstrap_ci": intervals[0].tolist(),
            "js_95pct_bootstrap_ci": intervals[1].tolist(),
        }
    axes[0].set_ylabel("Metric value")
    fig.legend(
        handles=[Patch(facecolor=CPC_COLOR, alpha=0.86, label="CPC (higher is better)"),
                 Patch(facecolor=JS_COLOR, alpha=0.86, label="JS divergence (lower is better)")],
        loc="lower left", bbox_to_anchor=(0.055, 0.995), ncol=2, frameon=False,
        handlelength=1.2, columnspacing=1.5, borderaxespad=0,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.81, wspace=0.17)
    save(fig, "FigS_WorldMove_calibrated_OD_validation")
    return audit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    set_style()
    locations = pd.read_csv(DATA / "worldmove_matched_city_locations.csv")
    metrics = pd.read_csv(DATA / "WorldMove_OD_validation_city_level.csv")
    labels = json.loads((DATA / "worldmove_nine_city_labels.json").read_text(encoding="utf-8"))
    matrices = np.load(DATA / "worldmove_nine_city_od_matrices.npz")
    draw_world_map(locations)
    draw_od_examples(matrices, labels)
    audit = {
        "matched_city_count": int(len(locations)),
        "covered_country_count": int(locations.iso3.nunique()),
        "example_cities": list(labels.values()),
        "calibrated_heldout_metrics": draw_validation(metrics),
        "calibration_note": (
            "A monotone power correction was fitted to one random half of WorldMove trajectories "
            "and evaluated against the other half; intervals are city-level bootstrap 95% CIs."
        ),
    }
    (DATA / "WorldMove_supplementary_figure_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
