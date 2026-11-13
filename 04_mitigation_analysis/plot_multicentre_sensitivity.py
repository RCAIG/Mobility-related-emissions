"""Plot paired single-centre versus multicentre sensitivity results for V25."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def relaunch_with_python_312() -> None:
    if sys.version_info >= (3, 10):
        return
    if os.environ.get("V25_MULTICENTRE_PLOT_PY312") == "1":
        raise RuntimeError("Python 3.10 or newer is required.")
    env = os.environ.copy()
    env["V25_MULTICENTRE_PLOT_PY312"] = "1"
    try:
        code = subprocess.call(["py", "-3.12", str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    except FileNotFoundError as exc:
        raise RuntimeError("Python 3.12 was not found.") from exc
    raise SystemExit(code)


relaunch_with_python_312()
sys.path[:0] = [r"${WORKSPACE_ROOT}", r"${WORKSPACE_ROOT}\\.pydeps_analysis"]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V25\Figures\Fig4_function_accessibility")
DI = [
    ("services_and_business", "Services and business"),
    ("shopping", "Shopping"),
    ("food_and_drink", "Food and drink"),
    ("health_care", "Health care"),
    ("lifestyle_services", "Lifestyle services"),
    ("education", "Education"),
    ("travel_and_transportation", "Travel and transportation"),
    ("sports_and_recreation", "Sports and recreation"),
    ("cultural_and_historic", "Cultural and historic"),
    ("community_and_government", "Community and government"),
    ("lodging", "Lodging"),
    ("arts_and_entertainment", "Arts and entertainment"),
]


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9DEE3", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def main():
    comp = pd.read_csv(OUT / "multicentre_single_comparison.csv")
    stats = pd.read_csv(OUT / "multicentre_vs_single_statistics.csv")
    labels = [label for _, label in DI]
    y = np.arange(len(DI))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.7), gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})
    ax = axes[0]
    ax.errorbar(comp.multicentre_mean, y, xerr=None, fmt="o", color="#168F7A", ms=6, label="Multicentre")
    ax.scatter(comp.single_mean, y, color="#7656A8", s=34, label="Single-centre", zorder=3)
    for i, row in comp.iterrows():
        ax.plot([row.single_mean, row.multicentre_mean], [i, i], color="#AEB7C0", lw=1.1, zorder=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean normalized mismatch")
    ax.set_title("Paired means", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)

    ax = axes[1]
    delta = comp.mean_difference_multi_minus_single.to_numpy()
    ax.barh(y, delta, color="#B45F4D", alpha=0.82)
    ax.axvline(0, color="#27313A", lw=0.9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Multicentre minus single-centre")
    ax.set_title("Absolute change", fontsize=12)
    style(ax)

    ax = axes[2]
    for j, scale in enumerate(["temporal_2017_2024", "spatial_2024"]):
        sub = stats[stats.scale.eq(scale)]
        vals = sub.groupby("method").rho.mean()
        ax.scatter([j - 0.09], [vals.get("single", np.nan)], color="#7656A8", s=45, label="Single-centre" if j == 0 else None)
        ax.scatter([j + 0.09], [vals.get("multicentre", np.nan)], color="#168F7A", s=45, label="Multicentre" if j == 0 else None)
        ax.plot([j - 0.09, j + 0.09], [vals.get("single", np.nan), vals.get("multicentre", np.nan)], color="#AEB7C0", lw=1)
    ax.axhline(0, color="#27313A", lw=0.9)
    ax.set_xticks([0, 1], ["Temporal", "Spatial 2024"])
    ax.set_ylabel("Mean Spearman correlation")
    ax.set_title("Association sensitivity", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    style(ax)

    fig.suptitle("Single-centre versus multicentre population–function mismatch", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "multicentre_sensitivity_comparison.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "multicentre_sensitivity_comparison.pdf", facecolor="white")
    plt.close(fig)

    # A separate full-size paired scatter is easier to inspect for outliers.
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for domain, label in DI:
        x = comp.loc[comp.domain.eq(domain), "single_mean"].iloc[0]
        yy = comp.loc[comp.domain.eq(domain), "multicentre_mean"].iloc[0]
        ax.scatter(x, yy, s=55, color="#168F7A", edgecolor="white", linewidth=0.6)
        ax.annotate(label, (x, yy), xytext=(4, 3), textcoords="offset points", fontsize=8)
    lo = min(comp.single_mean.min(), comp.multicentre_mean.min()) - 0.01
    hi = max(comp.single_mean.max(), comp.multicentre_mean.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], ls="--", color="#8A949D", lw=1)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Single-centre mean mismatch", ylabel="Multicentre mean mismatch")
    ax.set_title("Function-specific paired comparison", fontsize=12)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "multicentre_single_paired_scatter.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "multicentre_single_paired_scatter.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
