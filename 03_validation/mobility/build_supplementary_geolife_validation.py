from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V25")
OUT = ROOT / "supplementary_fig"
CACHE = OUT / "geolife_cache"
INK, GRID, LAND = "#253238", "#DCE3E5", "#F3F5F4"
PALETTE = ["#3B6EA8", "#D77A61", "#5D9C80", "#C49A42", "#8D6A9F",
           "#4F8C9E", "#B65D62", "#7E9B54", "#9A7358", "#546174"]


def style():
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": .65,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def load():
    trips = pd.read_parquet(CACHE / "geolife_beijing_labelled_trip_proxy_features_v2.parquet")
    metrics = json.loads((CACHE / "geolife_proxy_validity_metrics.json").read_text(encoding="utf-8"))
    return trips, metrics


def dataset_overview(trips):
    modes = sorted(trips.source_mode.dropna().unique())
    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(modes)}
    fig, (ax_map, ax_bar) = plt.subplots(
        1, 2, figsize=(7.15, 3.9), gridspec_kw={"width_ratios": [2.2, 1]}
    )
    ax_map.set_facecolor(LAND)
    # Draw sampled trajectory segments to keep the overview legible at publication size.
    rng = np.random.default_rng(24)
    draw = trips.sample(min(len(trips), 4500), random_state=24)
    for mode, d in draw.groupby("source_mode", sort=True):
        for row in d.itertuples(index=False):
            ax_map.plot([row.start_lon, row.end_lon], [row.start_lat, row.end_lat],
                        color=colors[mode], lw=.28, alpha=.16, solid_capstyle="round")
    ax_map.scatter(trips.start_lon, trips.start_lat, s=.35, color="#44545A", alpha=.16, linewidths=0)
    all_lon = pd.concat([trips.start_lon, trips.end_lon], ignore_index=True)
    all_lat = pd.concat([trips.start_lat, trips.end_lat], ignore_index=True)
    lon_lo, lon_hi = all_lon.quantile([.01, .99])
    lat_lo, lat_hi = all_lat.quantile([.01, .99])
    ax_map.set_xlim(lon_lo - .03, lon_hi + .03)
    ax_map.set_ylim(lat_lo - .03, lat_hi + .03)
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.text(-.02, 1.04, "GeoLife GPS trajectory segments", transform=ax_map.transAxes,
                fontsize=8, color=INK, va="bottom")
    ax_map.text(-.10, 1.04, "a", transform=ax_map.transAxes, fontsize=10,
                fontweight="bold", color=INK, va="top", ha="right")
    ax_map.grid(color="white", lw=.7)
    ax_map.set_axisbelow(True)
    for s in ax_map.spines.values():
        s.set_visible(False)

    counts = trips.source_mode.value_counts().reindex(modes)
    ax_bar.barh(np.arange(len(modes)), counts.values,
                color=[colors[m] for m in modes], alpha=.82, height=.72)
    ax_bar.set_yticks(np.arange(len(modes)), [m.title() for m in modes])
    ax_bar.set_xlabel("Trajectory segments")
    ax_bar.grid(axis="x", color=GRID, lw=.55)
    ax_bar.set_axisbelow(True)
    ax_bar.invert_yaxis()
    for s in ax_bar.spines.values():
        s.set_visible(False)
    ax_bar.text(-.08, 1.04, "Source modes", transform=ax_bar.transAxes,
                fontsize=8, color=INK, va="bottom")
    ax_bar.text(-.18, 1.04, "b", transform=ax_bar.transAxes, fontsize=10,
                fontweight="bold", color=INK, va="top", ha="right")
    fig.subplots_adjust(left=.08, right=.98, bottom=.13, top=.88, wspace=.38)
    fig.savefig(OUT / "FigS_GeoLife_dataset_overview.png", dpi=360, bbox_inches="tight", pad_inches=.025)
    fig.savefig(OUT / "FigS_GeoLife_dataset_overview.pdf", bbox_inches="tight", pad_inches=.025)
    plt.close(fig)


def evaluation_plot(metrics):
    values = [metrics["normalized_mutual_information"], metrics["cramers_v"]]
    labels = ["NMI", "Cramér’s V"]
    colors = ["#3B6EA8", "#D77A61"]
    fig, ax = plt.subplots(figsize=(4.6, 2.35))
    y = np.arange(2)
    ax.barh(y, values, color=colors, height=.48, alpha=.84)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Association with source-mode labels")
    ax.grid(axis="x", color=GRID, lw=.55)
    ax.set_axisbelow(True)
    for yi, value in zip(y, values):
        ax.text(value + .025, yi, f"{value:.3f}", va="center", ha="left", color=INK, fontsize=7)
    ax.text(-.10, 1.08, "GeoLife proxy validation", transform=ax.transAxes,
            fontsize=8, color=INK, va="bottom")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.subplots_adjust(left=.22, right=.97, bottom=.22, top=.84)
    fig.savefig(OUT / "FigS_GeoLife_NMI_CramersV_validation.png", dpi=360, bbox_inches="tight", pad_inches=.025)
    fig.savefig(OUT / "FigS_GeoLife_NMI_CramersV_validation.pdf", bbox_inches="tight", pad_inches=.025)
    plt.close(fig)


def main():
    style()
    trips, metrics = load()
    dataset_overview(trips)
    evaluation_plot(metrics)
    audit = {
        "n_segments": int(len(trips)),
        "n_source_modes": int(trips.source_mode.nunique()),
        "nmi": float(metrics["normalized_mutual_information"]),
        "cramers_v": float(metrics["cramers_v"]),
        "proxy_shares": metrics.get("proxy_shares", {}),
    }
    (OUT / "FigS_GeoLife_validation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
