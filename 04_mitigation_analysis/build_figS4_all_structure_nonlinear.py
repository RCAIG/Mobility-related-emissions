"""Build the V25 supplementary nonlinear structure--demand figure.

The figure uses the V19 primary sample and the direct observed change in
per-capita travel distance, rather than the earlier MGMI residual outcome.
It covers the complete population-form library plus both mixed-function
indicators.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def relaunch_with_python_312() -> None:
    if sys.version_info >= (3, 10):
        return
    if os.environ.get("V25_PY312_RELAUNCHED") == "1":
        raise RuntimeError("Python 3.10 or newer is required.")
    env = os.environ.copy()
    env["V25_PY312_RELAUNCHED"] = "1"
    try:
        code = subprocess.call(
            ["py", "-3.12", str(Path(__file__).resolve()), *sys.argv[1:]], env=env
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Python 3.12 was not found. Run this script with py -3.12.") from exc
    raise SystemExit(code)


relaunch_with_python_312()

# Use the same bundled analysis environment as the other V25 figure scripts.
sys.path[:0] = [r"${WORKSPACE_ROOT}", r"${WORKSPACE_ROOT}\\.pydeps_analysis"]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess


ROOT = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V25")
TABLE = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V21\Tables\city_analysis_data.csv")
OUT = ROOT / "Figures"
OUT.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUT / "FigS4_all_urban_form_nonlinear_demand.png"
OUTPUT_PDF = OUT / "FigS4_all_urban_form_nonlinear_demand.pdf"
OUTPUT_CSV = OUT / "FigS4_all_urban_form_nonlinear_demand_statistics.csv"

OUTCOMES = [
    ("d_log_total_distance", "Total travel distance change", "FigS4a_total_distance"),
    ("d_log_percap_travel_distance", "Per-capita travel distance change", "FigS4b_percap_distance"),
    ("d_long_share", "Long-distance share change", "FigS4c_long_share"),
    ("d_short_share", "Short-distance share change", "FigS4d_short_share"),
]

METRICS = [
    ("Agglomeration_2017", "Population-weighted density", "Intensity"),
    ("Gini_2017", "Population Gini", "Concentration and evenness"),
    ("Concentration_2017", "Top-20% concentration", "Concentration and evenness"),
    ("Entropy_2017", "Normalized population entropy", "Concentration and evenness"),
    ("Density_CV_2017", "Density heterogeneity", "Concentration and evenness"),
    ("CompactProximity_2017", "Compact proximity", "Centrality and compactness"),
    ("CentralShare50_2017", "Central population share", "Centrality and compactness"),
    ("CentroidOffset_2017", "Population-centre offset", "Centrality and compactness"),
    ("MoranI_2017", "Population Moran's I", "Clustering and continuity"),
    ("DensePatchCohesion_2017", "Dense-patch cohesion", "Clustering and continuity"),
    ("LocalExposure3km_2017", "Resident local exposure", "Clustering and continuity"),
    ("Polycentricity_2017", "Morphological polycentricity", "Nuclearity and polycentricity"),
    ("LogCenterCount_2017", "Number of population centres", "Nuclearity and polycentricity"),
    ("Largest_Center_Share_2017", "Largest-centre dominance", "Nuclearity and polycentricity"),
    ("PeripheralShare_2017", "Peripheral population share", "Peripheralization"),
    ("PopulationElongation_2017", "Population elongation", "Spatial shape"),
    ("local_mix_2017", "Local function mix", "Mixed urban functions"),
    ("city_mix_2017", "City-wide function mix", "Mixed urban functions"),
]

# The spatial analysis uses the same complete indicator library, measured in
# 2024. It is deliberately separate from the temporal analysis, which uses
# 2017 baseline indicators to explain 2017-2024 changes.
SPATIAL_METRICS = [
    (column.rsplit("_", 1)[0] + "_2024", label, dimension)
    for column, label, dimension in METRICS
]

COLORS = {
    "Intensity": "#2F5D8C",
    "Concentration and evenness": "#B7791F",
    "Centrality and compactness": "#168F7A",
    "Clustering and continuity": "#3D91A6",
    "Nuclearity and polycentricity": "#7656A8",
    "Peripheralization": "#C84630",
    "Spatial shape": "#6B7280",
    "Mixed urban functions": "#168F7A",
}


def curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return LOWESS curve and a reproducible bootstrap confidence band.

    Large city samples are reduced to an evenly spaced slice in x before the
    repeated LOWESS fits. This preserves the empirical support of the curve
    while keeping the supplementary figure practical to regenerate.
    """
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(x) > 700:
        selected = np.linspace(0, len(x) - 1, 700).astype(int)
        x, y = x[selected], y[selected]
    fitted = lowess(y, x, frac=0.38, it=1, return_sorted=True)
    estimate = np.interp(grid, fitted[:, 0], fitted[:, 1])
    rng = np.random.default_rng(seed)
    boot = []
    n = len(x)
    for _ in range(40):
        ids = rng.integers(0, n, n)
        sample = lowess(y[ids], x[ids], frac=0.38, it=1, return_sorted=True)
        boot.append(np.interp(grid, sample[:, 0], sample[:, 1]))
    band = np.asarray(boot)
    return estimate, np.nanquantile(band, 0.025, axis=0), np.nanquantile(band, 0.975, axis=0)


def spearman_interval(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float, float]:
    """Return Spearman rho and a city-level bootstrap 95% interval."""
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    n = len(x)
    if n > 2400:
        selected = np.linspace(0, n - 1, 2400).astype(int)
        x, y = x[selected], y[selected]
        n = len(x)
    values = []
    for _ in range(250):
        ids = rng.integers(0, n, n)
        value = spearmanr(x[ids], y[ids]).statistic
        if np.isfinite(value):
            values.append(value)
    if not values:
        return rho, np.nan, np.nan
    return rho, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def render_nonlinear(data: pd.DataFrame, outcome: str, outcome_label: str, stem: str,
                     summary: list[dict], outcome_index: int,
                     metrics: list[tuple[str, str, str]] = METRICS,
                     panel_label: str = "a") -> None:
    """Render one 18-indicator nonlinear association figure."""
    fig, axes = plt.subplots(5, 4, figsize=(14.2, 16.4), constrained_layout=False)
    for metric_index, (ax, (column, label, dimension)) in enumerate(zip(axes.flat, metrics)):
        z = data[[column, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
        color = COLORS[dimension]
        rho = p_value = endpoint_change = np.nan
        if len(z) >= 80 and z[column].nunique() >= 12:
            x = z[column].to_numpy(float)
            y = z[outcome].to_numpy(float)
            lo, hi = np.nanpercentile(x, [1, 99])
            keep = (x >= lo) & (x <= hi)
            x, y = x[keep], y[keep]
            grid = np.linspace(np.nanpercentile(x, 2), np.nanpercentile(x, 98), 100)
            est, lower, upper = curve(x, y, grid, 20260929 + outcome_index * 100 + metric_index)
            ax.fill_between(grid, lower, upper, color=color, alpha=0.16, linewidth=0)
            ax.plot(grid, est, color=color, linewidth=1.8)
            rho, p_value = spearmanr(x, y)
            endpoint_change = float(est[-1] - est[0])
            ax.axhline(0, color="#9AA3AD", linestyle="--", linewidth=0.7)
            ax.text(0.04, 0.94, f"n={len(z):,}", transform=ax.transAxes,
                    va="top", fontsize=8, color="#6B7280")
        ax.set_title(label, fontsize=10.2, fontweight="normal", pad=5)
        ax.set_xlabel("Indicator level")
        ax.set_ylabel(outcome_label)
        ax.grid(color="#D8DEE5", linewidth=0.4)
        ax.tick_params(direction="in")
        summary.append({
            "outcome": outcome,
            "outcome_label": outcome_label,
            "indicator": column,
            "label": label,
            "dimension": dimension,
            "n_complete": len(z),
            "spearman_rho": rho,
            "spearman_p": p_value,
            "lowess_change_from_low_to_high": endpoint_change,
        })
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    fig.text(0.03, 0.985, panel_label, ha="left", va="top", fontsize=18, fontweight="bold", color="#17202A")
    fig.text(0.5, 0.012, f"Nonlinear associations between urban form / mixed function and {outcome_label.lower()}",
             ha="center", fontsize=10.5)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.045, top=0.965, wspace=0.34, hspace=0.48)
    fig.savefig(OUT / f"{stem}_nonlinear.png", dpi=360, facecolor="white")
    fig.savefig(OUT / f"{stem}_nonlinear.pdf", facecolor="white")
    plt.close(fig)


def render_forest(data: pd.DataFrame, outcome: str, outcome_label: str, stem: str,
                  forest_rows: list[dict], outcome_index: int,
                  metrics: list[tuple[str, str, str]] = METRICS,
                  panel_label: str = "a") -> None:
    """Render a forest plot summarizing monotonic association for all metrics."""
    rows = []
    for metric_index, (column, label, dimension) in enumerate(metrics):
        z = data[[column, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < 80 or z[column].nunique() < 12:
            continue
        x = z[column].to_numpy(float)
        y = z[outcome].to_numpy(float)
        lo, hi = np.nanpercentile(x, [1, 99])
        keep = (x >= lo) & (x <= hi)
        x, y = x[keep], y[keep]
        rho, lower, upper = spearman_interval(x, y, 20261001 + outcome_index * 100 + metric_index)
        rows.append((label, dimension, rho, lower, upper))
        forest_rows.append({
            "outcome": outcome,
            "outcome_label": outcome_label,
            "indicator": column,
            "label": label,
            "dimension": dimension,
            "spearman_rho": rho,
            "spearman_ci_low": lower,
            "spearman_ci_high": upper,
            "n_complete": len(z),
        })
    rows.reverse()
    fig, ax = plt.subplots(figsize=(8.4, 10.8))
    yloc = np.arange(len(rows))
    for i, (label, dimension, rho, lower, upper) in enumerate(rows):
        color = COLORS[dimension]
        ax.plot([lower, upper], [i, i], color=color, linewidth=2.0, solid_capstyle="round")
        ax.scatter(rho, i, s=42, color=color, edgecolor="white", linewidth=0.7, zorder=3)
    ax.axvline(0, color="#7D8790", linestyle="--", linewidth=0.8)
    ax.set_yticks(yloc)
    ax.set_yticklabels([row[0] for row in rows], fontsize=9)
    ax.set_xlabel("Spearman association with outcome", fontsize=10.5)
    ax.set_title(f"Urban form / mixed function associations with {outcome_label.lower()}",
                 fontsize=11.5, fontweight="normal", pad=10)
    ax.grid(axis="x", color="#D8DEE5", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(0.03, 0.985, panel_label, ha="left", va="top", fontsize=18, fontweight="bold", color="#17202A")
    fig.subplots_adjust(left=0.39, right=0.96, bottom=0.08, top=0.94)
    fig.savefig(OUT / f"{stem}_forest.png", dpi=360, facecolor="white")
    fig.savefig(OUT / f"{stem}_forest.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    data = pd.read_csv(TABLE, low_memory=False)
    if "primary_sample" in data:
        flag = data["primary_sample"]
        if flag.dtype == object:
            flag = flag.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
        else:
            flag = flag.fillna(False).astype(bool)
        data = data[flag].copy()
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9.5,
        "axes.labelsize": 9.5, "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    summary = []
    forest_rows = []
    for outcome_index, (outcome, outcome_label, stem) in enumerate(OUTCOMES):
        if outcome not in data:
            raise KeyError(f"Required observed outcome column is missing: {outcome}")
        render_nonlinear(data, outcome, outcome_label, stem, summary, outcome_index)
        render_forest(data, outcome, outcome_label, stem, forest_rows, outcome_index)

    # Cross-sectional 2024 analysis: each structural indicator and each
    # mobility outcome is measured in 2024, with no 2017-2024 difference.
    data["ln_D_2024"] = np.log(data["total_travel_distance_m_2024"].where(data["total_travel_distance_m_2024"] > 0))
    data["ln_D_over_P_2024"] = np.log(
        data["total_travel_distance_m_2024"].where(data["total_travel_distance_m_2024"] > 0)
        / data["Pop_2024"].where(data["Pop_2024"] > 0)
    )
    spatial_outcomes = [
        ("ln_D_over_P_2024", r"$\ln(D/P)$ in 2024", "FigS4_spatial_2024_lnD_over_P"),
        ("ln_D_2024", r"$\ln(D)$ in 2024", "FigS4_spatial_2024_lnD"),
        ("short_share_2024", "Short-distance share in 2024", "FigS4_spatial_2024_short_share"),
        ("long_share_2024", "Long-distance share in 2024", "FigS4_spatial_2024_long_share"),
    ]
    spatial_summary = []
    spatial_forest = []
    for outcome_index, (outcome, outcome_label, stem) in enumerate(spatial_outcomes):
        render_nonlinear(data, outcome, outcome_label, stem, spatial_summary,
                         10 + outcome_index, metrics=SPATIAL_METRICS, panel_label="a")
        render_forest(data, outcome, outcome_label, stem, spatial_forest,
                      10 + outcome_index, metrics=SPATIAL_METRICS, panel_label="a")
    pd.DataFrame(spatial_summary).to_csv(
        OUT / "FigS4_spatial_2024_nonlinear_statistics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(spatial_forest).to_csv(
        OUT / "FigS4_spatial_2024_forest_statistics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summary).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(forest_rows).to_csv(OUT / "FigS4_all_outcomes_forest_statistics.csv", index=False, encoding="utf-8-sig")
    print(OUTPUT_CSV)
    print(OUT / "FigS4_all_outcomes_forest_statistics.csv")
    print(OUT / "FigS4_spatial_2024_nonlinear_statistics.csv")
    print(OUT / "FigS4_spatial_2024_forest_statistics.csv")


if __name__ == "__main__":
    main()
