"""Render Figure 3a-c observed-change maps in the Figure 1 mobility style."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    r"${WORKSPACE_ROOT}",
    r"${WORKSPACE_ROOT}\\.pydeps_analysis",
]

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd

import build_fig1_mobility_maps_2024 as reference


SOURCE = reference.CITY_TABLE
OUT = reference.OUT
PANELS = [
    ("a", "emission_change", "Change in ln(C/P)", "Fig3a_per_capita_emission_change"),
    ("b", "demand_change", "Change in ln(D/P)", "Fig3b_per_capita_distance_change"),
    ("c", "intensity_change", "Change in ln(C/D)", "Fig3c_emission_intensity_change"),
]


def main():
    columns = ["longitude", "latitude", "sample_V19"] + [row[1] for row in PANELS]
    data = pd.read_csv(SOURCE, usecols=columns, low_memory=False)
    data = data[data.sample_V19.astype(str).str.strip().str.lower().isin({"true", "1"})].copy()
    for letter, key, label, stem in PANELS:
        q = data.copy()
        for col in ("longitude", "latitude", key):
            q[col] = pd.to_numeric(q[col], errors="coerce")
        q = q.dropna(subset=["longitude", "latitude", key])
        values = q[key].to_numpy(float)
        # Preserve the original divergent colour semantics around zero.
        limit = 0.5 if key == "emission_change" else max(0.5, float(np.percentile(np.abs(values), 99)))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        q["_absolute_change"] = q[key].abs()
        q = q.sort_values("_absolute_change")

        fig, ax = plt.subplots(figsize=(5.0, 3.55))
        reference.world(ax)
        points = ax.scatter(q.longitude, q.latitude, c=q[key],
            s=reference.POINT_SIZE, cmap="RdYlBu_r", norm=norm,
            alpha=0.5, linewidths=0, rasterized=True, zorder=2)
        ax.text(.018, .982, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=14, fontweight="bold", color=reference.INK, zorder=4)
        cax = inset_axes(ax, width="2.0%", height="23%", loc="lower left",
            bbox_to_anchor=(.050, .075, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cb = fig.colorbar(points, cax=cax, orientation="vertical")
        cb.ax.set_xlabel(label, fontsize=8.3, labelpad=5)
        cb.ax.xaxis.label.set_horizontalalignment("left")
        cb.ax.xaxis.label.set_x(0)
        cb.ax.tick_params(labelsize=7.0, length=1.5, width=.5, pad=2)
        cb.outline.set_linewidth(.45)
        fig.subplots_adjust(left=.015, right=.985, top=.985, bottom=.015)
        reference.save(fig, stem)
        print(stem, len(q), "cities", "color range", (-limit, limit))


if __name__ == "__main__":
    main()
