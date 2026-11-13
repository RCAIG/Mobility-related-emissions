"""Rebuild the original V20/V25 Figure 4 layout with harmonized labels.

All four panels retain the original estimates and layout. Only the five
panel-a labels are harmonized with the formal indicator names. The separate
18-indicator table documents the full library.
"""
from pathlib import Path
import sys

sys.path[:0] = [r"${WORKSPACE_ROOT}", r"${WORKSPACE_ROOT}\\.pydeps_analysis"]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties

import build_fig4_function_specific_and_accessibility as base

ROOT = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V25")
V20 = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V20")
V19 = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V19")
OUT = ROOT / "Figures"
OUT.mkdir(parents=True, exist_ok=True)
TEAL, GOLD, BLUE, RED, GRAY = "#168F7A", "#D5AB51", "#477C9D", "#B6604D", "#A6AAAD"


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=360, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def title(ax, letter, label):
    ax.set_title(label, loc="left", pad=9, fontweight="normal")
    ax.text(-0.12, 1.055, letter, transform=ax.transAxes, fontweight="bold", fontsize=12)


def forest(ax, q, labels, color, fontsize=7.2):
    yy = np.arange(len(q))
    ax.axvline(0, color=GRAY, lw=.7)
    ax.errorbar(q.beta, yy, xerr=[q.beta - q.low, q.high - q.beta], fmt="o", color=color, capsize=2.0, ms=3.4)
    ax.set_yticks(yy, labels, fontsize=fontsize)
    ax.set_ylim(len(q) - .5, -.5)
    ax.grid(axis="x", color="#E7E8E8", lw=.5)
    ax.set_axisbelow(True)


def render_table():
    rows = [
        ("Population-weighted density", r"$D_{pop}=\frac{\sum_i p_i d_i}{\sum_i p_i}$", "Population-weighted mean density experienced by residents."),
        ("Population Gini", r"$G=\frac{\sum_i\sum_j |d_i-d_j|}{2n\sum_i d_i}$", "Inequality in the distribution of population density across grids."),
        ("Top-20% concentration", r"$C_{20}=\frac{\sum_{i\in top\,20\%}p_i}{\sum_i p_i}$", "Share of residents living in the densest 20% of grids."),
        ("Normalized population entropy", r"$H=-\frac{\sum_i p_i\ln p_i}{\ln n}$", "How evenly population is distributed across the urban area."),
        ("Density heterogeneity", r"$CV_d=\frac{SD(d_i)}{\overline{d}}$", "Relative variation in density between urban locations."),
        ("Compact proximity", r"$CP=\sum_i w_i\,prox_i$", "Population-weighted proximity to high-density cells."),
        ("Central population share", r"$S_c=\frac{P_{central}}{P_{total}}$", "Fraction of residents within the central-area mask."),
        ("Population-centre offset", r"$O=dist(\mathrm{c}_{pop},\mathrm{c}_{geom})$", "Distance between the population-weighted and geometric centres."),
        ("Population Moran's I", r"$I=\frac{n}{W}\frac{\sum_{ij}w_{ij}(d_i-\bar d)(d_j-\bar d)}{\sum_i(d_i-\bar d)^2}$", "Spatial autocorrelation of population density."),
        ("Dense-patch cohesion", r"$COH=\frac{A_{connected\ dense}}{A_{dense}}$", "Whether dense residential patches form connected areas."),
        ("Resident local exposure", r"$E_{3km}=\sum_i p_i\sum_j p_j\mathrm{1}(dist_{ij}\leq3\,km)$", "Population encountered within a 3-km neighbourhood, weighted by residents."),
        ("Morphological polycentricity", r"$P=f(\mathrm{centre\ balance},\mathrm{centre\ spacing})$", "Balance and spacing of identified population centres."),
        ("Number of population centres", r"$N_c=\#\{\mathrm{identified\ density\ peaks}\}$", "Count of effective population centres."),
        ("Largest-centre dominance", r"$D_{max}=\frac{P_{largest\ centre}}{P_{all\ centres}}$", "Population share controlled by the largest centre."),
        ("Peripheral population share", r"$S_p=\frac{P_{periphery}}{P_{total}}$", "Fraction of residents living in peripheral areas."),
        ("Population elongation", r"$E=\frac{L_{major}}{L_{minor}}$", "Whether the population footprint is stretched along one axis."),
        ("Local function mix", r"$M_{local}=-\frac{\sum_k s_k\ln s_k}{\ln K}$", "Diversity of functions available in local grids."),
        ("City-wide function mix", r"$M_{city}=-\frac{\sum_k S_k\ln S_k}{\ln K}$", "Diversity of function categories across the whole city."),
    ]
    d = OUT / "Fig4_18_indicator_table"
    d.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows, columns=["Indicator", "Formula", "Plain-language explanation"])
    frame.to_csv(d / "Fig4_18_indicators_formula_explanation.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(17, 14.2)); ax.axis("off")
    tab = ax.table(cellText=frame.values, colLabels=frame.columns, colWidths=[.23, .40, .37], cellLoc="left", loc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(9.2); tab.scale(1, 2.05)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#C7D0D9"); cell.set_linewidth(.5); cell.PAD = .01
        cell.get_text().set_wrap(True)
        if r == 0:
            cell.set_facecolor("#477C9D"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        elif r % 2 == 0: cell.set_facecolor("#F3F6F8")
    save(fig, "Fig4_18_indicator_table")
    # Keep table copies beside the figure for convenient manuscript insertion.
    fig.savefig(d / "Fig4_18_indicators_formula_explanation.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(d / "Fig4_18_indicators_formula_explanation.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    original = pd.read_csv(V20 / "Tables" / "structure_models.csv")
    q = original[(original["sample"] == "V19") & (original["family"] == "Baseline_or_access")]
    fun = pd.read_csv(V19 / "Tables" / "function_source_sensitivity.csv")
    spatial = original[(original["sample"] == "V19") & (original["family"] == "Spatial_2024")].copy()
    access = fun.iloc[4:].copy()
    fig, ax = plt.subplots(2, 2, figsize=(6.65, 6.05))
    fig.subplots_adjust(left=.235, right=.98, bottom=.10, top=.94, hspace=.58, wspace=1.05)
    a = q[q["term"].isin(["resident_compactness_2017", "central_share_2017", "dense_patch_cohesion_2017", "population_elongation_2017", "adaptive_polycentricity_2017"])].copy()
    a = a.set_index("term").loc[["resident_compactness_2017", "central_share_2017", "dense_patch_cohesion_2017", "population_elongation_2017", "adaptive_polycentricity_2017"]].reset_index()
    forest(ax[0, 0], a, ["Compact\nproximity", "Central population", "Dense-patch\ncohesion", "Population\nelongation", "Morphological\npolycentricity"], TEAL, 7.2); title(ax[0, 0], "a", "Baseline population form"); ax[0, 0].set_xlabel("Slope for distance change (SD)")
    f = fun[fun.metric.isin(["local_mix_2017", "city_mix_2017"])].reset_index(drop=True)
    forest(ax[0, 1], f, ["Local mix\nAll cities", "Local mix\nObserved only", "City mix\nAll cities", "City mix\nObserved only"], GOLD); title(ax[0, 1], "b", "Function data sensitivity"); ax[0, 1].set_xlabel("Slope for distance change (SD)")
    labs = ["Resident\ncompactness", "Central\npopulation", "Dense-patch\ncohesion", "Population\nelongation", "Adaptive\npolycentricity"]
    forest(ax[1, 0], spatial.iloc[:5], labs, BLUE); title(ax[1, 0], "c", "Spatial levels in 2024"); ax[1, 0].set_xlabel("Slope for distance level (SD)")
    forest(ax[1, 1], access, ["Population-\nfunction mismatch", "Accessible\ndiversity", "Sony walking\ntime"], RED); title(ax[1, 1], "d", "External accessibility"); ax[1, 1].set_xlabel("Slope for distance change (SD)")
    for aa in ax.flat: aa.xaxis.label.set_fontsize(8)
    save(fig, "Fig4_urban_form_and_accessibility")
    render_table()
    print("Wrote original-layout Figure 4 and English indicator table", flush=True)


if __name__ == "__main__": main()
