from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info < (3, 10):
    if os.environ.get("SCHEME_B_PY312_RELAUNCHED") == "1":
        raise RuntimeError("Python 3.10 or newer is required for the plotting libraries.")
    env = os.environ.copy()
    env["SCHEME_B_PY312_RELAUNCHED"] = "1"
    try:
        raise SystemExit(subprocess.call(["py", "-3.12", str(Path(__file__).resolve()), *sys.argv[1:]], env=env))
    except FileNotFoundError as exc:
        raise RuntimeError("Run this script with: py -3.12 plot_scheme_b_v5_v2_1.py") from exc

sys.path[:0] = [r"${WORKSPACE_ROOT}", r"${WORKSPACE_ROOT}\\.pydeps_analysis"]



import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), r"${WORKSPACE_ROOT}", r"${WORKSPACE_ROOT}\\.pydeps_analysis"]

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

import build_fig1_mobility_maps_2024 as reference


COLORS = {
    "Joint improvement": "#006B58",
    "Demand improvement": "#D89B18",
    "Emission-intensity improvement": "#5672BC",
    "Dual challenge": "#C6534A",
}


def draw(data, technology_col, improvement, stem, legend_labels=None):
    q = data.dropna(subset=["longitude", "latitude", "demand_change", technology_col]).copy()
    demand = q.demand_change < 0
    technical = improvement(q[technology_col])
    q["category"] = "Dual challenge"
    q.loc[demand & technical, "category"] = "Joint improvement"
    q.loc[demand & ~technical, "category"] = "Demand improvement"
    q.loc[~demand & technical, "category"] = "Emission-intensity improvement"
    counts = q.category.value_counts()

    fig, ax = plt.subplots(figsize=(5.0, 3.55))
    reference.world(ax)
    # Great-circle neighbourhoods avoid longitude distortions at high latitudes.
    coordinates = np.radians(q[['latitude', 'longitude']].to_numpy())
    neighbours = BallTree(coordinates, metric='haversine').query_radius(
        coordinates, r=150 / 6371.0088)
    categories = q.category.to_numpy()
    density = np.array([len(indices) for indices in neighbours])
    q['_local_category_count'] = [
        np.count_nonzero(categories[indices] == categories[index])
        for index, indices in enumerate(neighbours)]
    # Sparse areas keep the reference area; crowded areas approach 1.5 pt².
    q['_marker_area'] = np.clip(
        reference.POINT_SIZE / np.sqrt(np.maximum(density, 1)),
        1.5, reference.POINT_SIZE)
    q = q.sort_values('_local_category_count', ascending=False, kind='stable')
    ax.scatter(q.longitude, q.latitude, s=q['_marker_area'],
        c=q.category.map(COLORS), alpha=.7, edgecolors="white",
        linewidths=.15, rasterized=True, zorder=2)
    ax.text(.018, .982, "e", transform=ax.transAxes, ha="left", va="top",
        fontsize=14, fontweight="bold", color=reference.INK, zorder=4)
    if legend_labels is None:
        legend_labels = {
            "Joint improvement": "Joint",
            "Demand improvement": "Demand-led",
            "Emission-intensity improvement": "Emission-intensity-led",
            "Dual challenge": "Dual challenge",
        }
    handles = [Line2D([], [], linestyle="", marker="o", markersize=4,
        markerfacecolor=color, markeredgewidth=0, alpha=.75,
        label=legend_labels[category])
        for category, color in COLORS.items()]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.020),
        ncol=4, frameon=False, fontsize=8.3, handletextpad=.35,
        columnspacing=.35, borderaxespad=0)
    fig.subplots_adjust(left=.015, right=.985, top=.985, bottom=.015)
    reference.save(fig, stem)
    print(stem, counts.to_dict())
    print('Local density radius: 150 km; marker area:',
          float(q['_marker_area'].min()), float(q['_marker_area'].max()))


def main():
    columns = ["longitude", "latitude", "sample_V19", "demand_change",
        "intensity_change", "tmi_city_cd_calibrated"]
    data = pd.read_csv(reference.CITY_TABLE, usecols=columns + ['country'], low_memory=False)
    data = data[data.sample_V19.astype(str).str.strip().str.lower().isin({"true", "1"})].copy()
    for col in columns[0:2] + columns[3:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    draw(data, "tmi_city_cd_calibrated", lambda values: values > 0,
        "Fig3e_demand_TTI_portfolios")
    draw(data, "intensity_change", lambda values: values < 0,
        "Fig3e_demand_intensity_portfolios", legend_labels={
            "Joint improvement": "Joint",
            "Demand improvement": "Demand-driven bot.",
            "Emission-intensity improvement": "Emission-intensity bot.",
            "Dual challenge": "Dual challenge",
        })


if __name__ == "__main__":
    main()
