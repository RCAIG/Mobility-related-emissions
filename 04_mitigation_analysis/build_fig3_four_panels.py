"""Build the revised V25 Figure 3 and standalone panels a-d.

The data and portfolio definitions are inherited from the V22/V25 analysis
tables. This script changes only the visual composition of Figure 3.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

if sys.version_info < (3, 10):
    if os.environ.get("SCHEME_B_PY312_RELAUNCHED") == "1":
        raise RuntimeError("Python 3.10 or newer is required for the plotting libraries.")
    env = os.environ.copy()
    env["SCHEME_B_PY312_RELAUNCHED"] = "1"
    try:
        raise SystemExit(subprocess.call(["py", "-3.12", str(Path(__file__).resolve()), *sys.argv[1:]], env=env))
    except FileNotFoundError as exc:
        raise RuntimeError("Run this script with: py -3.12 plot_scheme_b_v5_v2_11.py") from exc

sys.path[:0] = [r"${WORKSPACE_ROOT}", r"${WORKSPACE_ROOT}\\.pydeps_analysis"]


from pathlib import Path
import sys

sys.path.insert(0, r"${WORKSPACE_ROOT}\\.pydeps_analysis")

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import shapefile


V25 = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V25")
V22 = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V22")
TABLES = V22 / "Tables"
FIGURES = V25 / "Figures"
FIGURES.mkdir(parents=True, exist_ok=True)

WORLD_BASE = Path(r"${URBAN_DATA_ROOT}\\全球城市边界\建筑环境和移动碳排\DrivingFactorAnalysis\data\Nature_Panoramic_Travel_Analysis")
WORLD = next((WORLD_BASE / "Scheme_B_Complete_Extensions_V7" / "Public_Accessibility_Data" / "ne_110m_admin_0_countries").glob("*.shp"))

DPI = 360
INK = "#263238"
GRID = "#D7DEE2"
LAND = "#F4F6F6"
BORDER = "#C7CED2"

# A balanced, colour-vision-friendly categorical palette. The colours are
# intentionally separated in hue and lightness so Joint and Technology-only
# remain distinguishable in dense maps and small legends.
COLORS = {
    "Joint improvement": "#007C78",          # deep teal
    "Demand improvement only": "#D89B18",    # ochre
    "Technology improvement only": "#3F6F9F",# slate blue
    "Dual challenge": "#C6534A",             # vermilion
}
# Softer variants are used in the analytical panels so the dense marks do not
# compete with the map while category identity remains unchanged.
LIGHT_COLORS = {
    "Joint improvement": "#5CAAA6",
    "Demand improvement only": "#E7BA59",
    "Technology improvement only": "#7D9FC1",
    "Dual challenge": "#D9867E",
}
METRIC_COLORS = {
    "Mean Δln(D/P)": "#69AAA4",
    "Mean ΔTTI": "#7E9FBE",
}
PANEL_B_SIZE = (6.7, 5.4)
PANEL_RIGHT_WIDTH = 11.6 - PANEL_B_SIZE[0] - 1
PANEL_RIGHT_HEIGHT = PANEL_B_SIZE[1] / 2
ORDER = [
    "Joint improvement",
    "Demand improvement only",
    "Technology improvement only",
    "Dual challenge",
]
SHORT = {
    "Joint improvement": "Joint",
    "Demand improvement only": "Demand",
    "Technology improvement only": "TTI",
    "Dual challenge": "Dual challenge",
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#87939A",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def panel_label(ax, letter, x=-0.055, y=1.025):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="bottom", ha="left")


def save(fig, stem):
    fig.savefig(FIGURES / f"{stem}.png", dpi=DPI, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)


def world(ax):
    with shapefile.Reader(str(WORLD)) as reader:
        for shape in reader.shapes():
            parts = list(shape.parts) + [len(shape.points)]
            polygons = [shape.points[a:b] for a, b in zip(parts[:-1], parts[1:])]
            ax.add_collection(PolyCollection(
                polygons, facecolor=LAND, edgecolor=BORDER, linewidth=0.28, zorder=0
            ))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 85)
    ax.set_aspect("equal")
    ax.axis("off")


def load_data():
    city = pd.read_csv(TABLES / "city_analysis_v22_full.csv")
    city = city.loc[city["sample_V19"].astype(bool)].copy()
    city = city[city["portfolio_cd_calibrated"].isin(ORDER)].copy()
    summary = pd.read_csv(TABLES / "portfolio_summary_v22.csv")
    summary = summary[
        summary["scheme"].eq("C/D-calibrated city TMI")
        & summary["state"].isin(ORDER)
    ].set_index("state").reindex(ORDER)
    return city, summary


def representative_labels(q):
    # These are selected from the actual table and are kept sparse to avoid
    # turning a dense global scatter into a label cloud.
    wanted = [
        ("Montreal", "Canada"),
        ("Munich", "Germany"),
        ("Phoenix", "United States"),
        ("Wuhan", "China"),
        ("Chengdu", "China"),
        ("Singapore", "Singapore"),
        ("Shimla", "India"),
    ]
    rows = []
    for city_name, country in wanted:
        hit = q[(q.city_name.eq(city_name)) & (q.country.eq(country))]
        if not hit.empty:
            rows.append(hit.iloc[0])
    return rows


def panel_a(q):
    fig, ax = plt.subplots(figsize=(11.6, 6.0))
    world(ax)
    for state in ORDER:
        s = q[q.portfolio_cd_calibrated.eq(state)]
        ax.scatter(s.longitude, s.latitude, s=13, alpha=0.42,
                   linewidths=0.18, edgecolors="white", color=COLORS[state],
                   rasterized=True, label=SHORT[state], zorder=2)
    handles = [Line2D([0], [0], marker="o", ls="", color=COLORS[s],
                      markeredgecolor="white", markeredgewidth=.35,
                      label=SHORT[s], markersize=7) for s in ORDER]
    ax.legend(handles=handles, ncol=4, frameon=False, loc="lower center",
              bbox_to_anchor=(.5, -.095), fontsize=9.5, handletextpad=.45,
              columnspacing=1.25)
    panel_label(ax, "a", x=0.015, y=.94)
    save(fig, "Fig3a_global_demand_technology_portfolios")


def panel_b(q):
    fig, ax = plt.subplots(figsize=PANEL_B_SIZE)
    for state in ORDER:
        s = q[q.portfolio_cd_calibrated.eq(state)]
        ax.scatter(s.demand_change, s.tmi_city_cd_calibrated, s=18,
                   alpha=0.28, linewidths=0.22, edgecolors="white",
                   color=LIGHT_COLORS[state], rasterized=True, label=SHORT[state])
    for row in representative_labels(q):
        ax.scatter(row.demand_change, row.tmi_city_cd_calibrated, s=70,
                   facecolors="none", edgecolors="#263238", linewidths=1.15,
                   zorder=5)
    ax.axhline(0, color=INK, lw=.75, zorder=1)
    ax.axvline(0, color=INK, lw=.75, zorder=1)
    # The limits show the central cloud while retaining the full-sample
    # statistics; no observations are removed from the analysis.
    ax.set_xlim(-.82, 1.55)
    ax.set_ylim(-.65, 1.95)
    rho = q[["demand_change", "tmi_city_cd_calibrated"]].corr(method="spearman").iloc[0, 1]
    # ax.text(.035, .955, f"Spearman ρ = {rho:.3f}\nn = {len(q):,}",
    #         transform=ax.transAxes, va="top", fontsize=9.2,
    #         bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=3))
    for row in representative_labels(q):
        offsets = {
            "Montreal": (-64, 22), "Munich": (-60, 0), "Phoenix": (-64, -22),
            "Wuhan": (22, 24), "Chengdu": (42, 0), "Singapore": (22, -24),
            "Shimla": (-42, 9),
        }
        ax.annotate(row.city_name, (row.demand_change, row.tmi_city_cd_calibrated),
                    xytext=offsets.get(row.city_name, (5, 5)),
                    textcoords="offset points", fontsize=8.2, color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.78, pad=.5),
                    arrowprops=dict(arrowstyle="-", color="#7D878C", lw=.45),
                    annotation_clip=True)
    ax.set_xlabel("Change in ln(D/P)")
    ax.set_ylabel("City-calibrated TTI")
    panel_label(ax, "b")
    fig.subplots_adjust(left=.16, right=.96, bottom=.16, top=.91)
    save(fig, "Fig3b_observed_demand_technology_dimensions")


def panel_c(summary):
    fig, ax = plt.subplots(figsize=(PANEL_RIGHT_WIDTH, PANEL_RIGHT_HEIGHT))
    x = np.arange(len(ORDER))
    ax.bar(x, summary["mean_emission_change"], color=[LIGHT_COLORS[s] for s in ORDER],
           width=.58, edgecolor="white", linewidth=.4)
    ax.axhline(0, color=INK, lw=.75)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[s] for s in ORDER], rotation=0, ha="center", fontsize=8.2)
    ax.set_ylabel("Mean change in ln(C/P)")
    ax.set_ylim(min(-.03, summary.mean_emission_change.min() * 1.35),
                max(.22, summary.mean_emission_change.max() * 1.15))
    panel_label(ax, "c")
    fig.subplots_adjust(left=.19, right=.96, bottom=.24, top=.88)
    save(fig, "Fig3c_per_capita_emission_outcomes")


def draw_panel_d(ax, q, summary, letter_x=-.075):
    values = pd.DataFrame({
        "Mean Δln(D/P)": summary["mean_demand_change"],
        "Mean ΔTTI": q.groupby("portfolio_cd_calibrated")["tmi_city_cd_calibrated"].mean().reindex(ORDER),
    })
    x = np.arange(len(ORDER))
    width = .3
    for i, metric in enumerate(values.columns):
        ax.bar(x + (i - .5) * width, values[metric], width=width,
               color=METRIC_COLORS[metric], edgecolor="white", linewidth=.35,
               label=metric)
    ax.axhline(0, color=INK, lw=.75)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[state] for state in ORDER], rotation=0,
                       ha="center", fontsize=8.2)
    ax.set_ylabel("Mean change")
    ax.grid(axis="y", color=GRID, lw=.55, alpha=.65)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=1, loc="upper right", fontsize=8.7,
              handlelength=1.2, labelspacing=.45)
    panel_label(ax, "d", x=letter_x)


def panel_d(q, summary):
    fig, ax = plt.subplots(figsize=(PANEL_RIGHT_WIDTH, PANEL_RIGHT_HEIGHT))
    draw_panel_d(ax, q, summary, letter_x=-.065)
    fig.subplots_adjust(left=.19, right=.96, top=.88, bottom=.24)
    save(fig, "Fig3d_portfolio_decomposition")


def main():
    q, summary = load_data()
    panel_a(q)
    panel_b(q)
    panel_c(summary)
    panel_d(q, summary)

    fig = plt.figure(figsize=(11.6, 10.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[2.4, 1.65, 1.65],
                          hspace=.48, wspace=.55, left=.085, right=.985,
                          top=.96, bottom=.075)
    # Recreate panels in the group without saving temporary figures.
    ax_a = fig.add_subplot(gs[0, :])
    world(ax_a)
    for state in ORDER:
        s = q[q.portfolio_cd_calibrated.eq(state)]
        ax_a.scatter(s.longitude, s.latitude, s=11.5, alpha=.42,
                     linewidths=.15, edgecolors="white", color=COLORS[state],
                     rasterized=True)
    ax_a.legend(handles=[Line2D([0], [0], marker="o", ls="", color=COLORS[s],
                                markeredgecolor="white", markeredgewidth=.3,
                                label=SHORT[s], markersize=6.5) for s in ORDER],
                ncol=4, frameon=False, loc="lower center",
                bbox_to_anchor=(.5, -.08), fontsize=9, columnspacing=1.2)
    panel_label(ax_a, "a", x=.015, y=.94)

    ax_b = fig.add_subplot(gs[1:, :3])
    for state in ORDER:
        s = q[q.portfolio_cd_calibrated.eq(state)]
        ax_b.scatter(s.demand_change, s.tmi_city_cd_calibrated, s=16,
                     alpha=.27, linewidths=.18, edgecolors="white",
                     color=LIGHT_COLORS[state], rasterized=True)
    for row in representative_labels(q):
        ax_b.scatter(row.demand_change, row.tmi_city_cd_calibrated, s=52,
                     facecolors="none", edgecolors="#263238", linewidths=.95,
                     zorder=5)
    ax_b.axhline(0, color=INK, lw=.75); ax_b.axvline(0, color=INK, lw=.75)
    ax_b.set_xlim(-.82, 1.55); ax_b.set_ylim(-.65, 1.95)
    rho = q[["demand_change", "tmi_city_cd_calibrated"]].corr(method="spearman").iloc[0, 1]
    ax_b.text(.035, .95, f"Spearman ρ = {rho:.3f}\nn = {len(q):,}",
              transform=ax_b.transAxes, va="top", fontsize=8.8,
              bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=3))
    for row in representative_labels(q):
        offsets = {
            "Montreal": (-51, 19), "Munich": (-49, 0), "Phoenix": (-51, -19),
            "Wuhan": (17, 21), "Chengdu": (31, 0), "Singapore": (17, -21),
            "Shimla": (-38, 8),
        }
        ax_b.annotate(row.city_name, (row.demand_change, row.tmi_city_cd_calibrated),
                      xytext=offsets.get(row.city_name, (4, 4)),
                      textcoords="offset points", fontsize=7.6,
                      bbox=dict(facecolor="white", edgecolor="none", alpha=.78, pad=.4),
                      arrowprops=dict(arrowstyle="-", color="#7D878C", lw=.4))
    ax_b.set_xlabel("Change in ln(D/P)"); ax_b.set_ylabel("City-calibrated TTI")
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 3:])
    x = np.arange(4)
    ax_c.bar(x, summary.mean_emission_change, color=[LIGHT_COLORS[s] for s in ORDER],
             width=.54, edgecolor="white", linewidth=.35)
    ax_c.axhline(0, color=INK, lw=.7)
    ax_c.set_xticks(x); ax_c.set_xticklabels([SHORT[s] for s in ORDER], rotation=0,
                                             ha="center", fontsize=7.2)
    ax_c.set_ylabel("Mean change in ln(C/P)", fontsize=9)
    ax_c.set_ylim(min(-.03, summary.mean_emission_change.min()*1.35), .22)
    panel_label(ax_c, "c", x=-.11)

    ax_d = fig.add_subplot(gs[2, 3:])
    draw_panel_d(ax_d, q, summary, letter_x=-.055)
    save(fig, "Fig3_joint_transition_advantage")


if __name__ == "__main__":
    main()
