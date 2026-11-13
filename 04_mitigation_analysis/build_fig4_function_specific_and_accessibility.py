"""V25 supplementary analysis for native functions and accessibility.

The Overture/Sony variables in the source table are current snapshots. They
are therefore analysed in two explicitly different ways:

* snapshot-to-change: current function/accessibility versus 2017-2024
  observed mobility changes;
* 2024 cross-section: current function/accessibility versus 2024 outcomes.

The script writes one forest plot and one nonlinear plot for each family and
outcome, plus a CSV containing descriptive, within-country, and controlled
associations. It relaunches itself with Python 3.12 on older interpreters.
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

sys.path[:0] = [r"${WORKSPACE_ROOT}", r"${WORKSPACE_ROOT}\\.pydeps_analysis"]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess


ROOT = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V25")
TABLE = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V21\Tables\city_analysis_data.csv")
OUT = ROOT / "Figures" / "Fig4_function_accessibility"
OUT.mkdir(parents=True, exist_ok=True)

DPI = 360
SEED = 20260929

FUNCTIONS = [
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

ACCESS = [
    ("resident_weighted_hill_15min", "Resident-weighted diversity, 15 min"),
    ("resident_weighted_hill_30min", "Resident-weighted diversity, 30 min"),
    ("resident_weighted_hill_45min", "Resident-weighted diversity, 45 min"),
    ("walk_proximity_direct_hq_min", "Sony walking time"),
    ("population_function_mismatch", "Population-function mismatch"),
]

OUTCOMES_TIME = [
    ("d_log_percap_travel_distance", "Change in ln(D/P)"),
    ("d_log_total_distance", "Change in ln(D)"),
    ("d_short_share", "Change in short-distance share"),
    ("d_long_share", "Change in long-distance share"),
]
OUTCOMES_SPACE = [
    ("log_distance_pc", "2024 ln(D/P)"),
    ("log_distance", "2024 ln(D)"),
    ("short_share", "2024 short-distance share"),
    ("long_share", "2024 long-distance share"),
]

PALETTE = ["#2F5D8C", "#B7791F", "#168F7A", "#7656A8", "#C84630", "#6B7280"]


def finite_frame(data: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    cols = [x, y, "iso3"]
    z = data[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    return z if z[x].nunique() >= 8 else z.iloc[0:0]


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 8 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def bootstrap_rho(x: np.ndarray, y: np.ndarray, seed: int, reps: int = 400) -> tuple[float, float, float]:
    rho = rank_corr(x, y)
    if not np.isfinite(rho):
        return rho, np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = []
    n = len(x)
    for _ in range(reps):
        ids = rng.integers(0, n, n)
        value = rank_corr(x[ids], y[ids])
        if np.isfinite(value):
            values.append(value)
    return rho, float(np.quantile(values, .025)), float(np.quantile(values, .975))


def within_country_rho(z: pd.DataFrame, x: str, y: str) -> float:
    q = z[[x, y, "iso3"]].copy()
    q[x] = q[x] - q.groupby("iso3")[x].transform("mean")
    q[y] = q[y] - q.groupby("iso3")[y].transform("mean")
    return rank_corr(q[x].to_numpy(), q[y].to_numpy())


def controlled_model(data: pd.DataFrame, x: str, y: str) -> dict[str, float]:
    controls = ["log_pop", "log_gdppc", "logArea"]
    cols = [x, y, "iso3"] + controls
    z = data[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(z) < 80 or z[x].nunique() < 8:
        return {"beta_std": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan,
                "n_model": len(z), "countries_model": z.iso3.nunique()}
    # Standardize all continuous terms, then absorb country means. This is a
    # within-country association conditional on size, GDP per capita and area.
    terms = [x] + controls
    X = z[terms].astype(float)
    X = (X - X.mean()) / X.std(ddof=0)
    yy = (z[y] - z[y].mean()) / z[y].std(ddof=0)
    X = X - X.groupby(z["iso3"].to_numpy()).transform("mean")
    yy = yy - yy.groupby(z["iso3"].to_numpy()).transform("mean")
    model = sm.OLS(yy, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": z.iso3})
    interval = model.conf_int().loc[x]
    return {"beta_std": float(model.params[x]), "ci_low": float(interval[0]),
            "ci_high": float(interval[1]), "p": float(model.pvalues[x]),
            "n_model": len(z), "countries_model": z.iso3.nunique()}


def statistics(data: pd.DataFrame, features: list[tuple[str, str]], outcomes: list[tuple[str, str]],
               scale: str, family: str) -> pd.DataFrame:
    rows: list[dict] = []
    for feature, label in features:
        for outcome, outcome_label in outcomes:
            z = finite_frame(data, feature, outcome)
            if len(z) < 8:
                continue
            x, y = z[feature].to_numpy(float), z[outcome].to_numpy(float)
            rho, lo, hi = bootstrap_rho(x, y, SEED + len(rows))
            adjusted = controlled_model(data, feature, outcome)
            rows.append({
                "scale": scale, "family": family, "feature": feature,
                "feature_label": label, "outcome": outcome,
                "outcome_label": outcome_label, "n": len(z),
                "countries": z.iso3.nunique(), "spearman_rho": rho,
                "spearman_ci_low": lo, "spearman_ci_high": hi,
                "within_country_spearman": within_country_rho(z, feature, outcome),
                **adjusted,
            })
    return pd.DataFrame(rows)


def lowess_curve(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x, y = x[order], y[order]
    lo, hi = np.nanpercentile(x, [1, 99])
    keep = (x >= lo) & (x <= hi)
    x, y = x[keep], y[keep]
    if len(x) > 900:
        ids = np.linspace(0, len(x) - 1, 900).astype(int)
        x, y = x[ids], y[ids]
    if len(x) < 12 or np.unique(x).size < 8 or np.unique(y).size < 4:
        raise ValueError("Insufficient variation for a stable LOWESS curve")
    grid = np.linspace(np.nanpercentile(x, 2), np.nanpercentile(x, 98), 120)
    fit = lowess(y, x, frac=.38, it=1, return_sorted=True)
    estimate = np.interp(grid, fit[:, 0], fit[:, 1])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(60):
        ids = rng.integers(0, len(x), len(x))
        if np.unique(x[ids]).size < 8:
            continue
        fitted = lowess(y[ids], x[ids], frac=.38, it=1, return_sorted=True)
        boots.append(np.interp(grid, fitted[:, 0], fitted[:, 1]))
    if not boots:
        band = np.repeat(estimate[None, :], 2, axis=0)
    else:
        band = np.asarray(boots)
    return grid, estimate, np.quantile(band, .025, axis=0), np.quantile(band, .975, axis=0)


def render_nonlinear(data: pd.DataFrame, features: list[tuple[str, str]], outcome: str,
                     outcome_label: str, scale: str, family: str, stem: str) -> None:
    n = len(features)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12.6, 3.15 * rows), squeeze=False)
    for i, (feature, label) in enumerate(features):
        ax = axes.flat[i]
        z = finite_frame(data, feature, outcome)
        if len(z) >= 30:
            x, y = z[feature].to_numpy(float), z[outcome].to_numpy(float)
            try:
                grid, est, low, high = lowess_curve(x, y, SEED + i)
            except ValueError:
                grid = est = low = high = None
            if grid is not None:
                color = PALETTE[i % len(PALETTE)]
                ax.fill_between(grid, low, high, color=color, alpha=.16, linewidth=0)
                ax.plot(grid, est, color=color, linewidth=1.8)
            ax.axhline(0, color="#9AA3AD", linestyle="--", linewidth=.7)
            ax.text(.04, .94, f"n={len(z):,}", transform=ax.transAxes, va="top", fontsize=8, color="#6B7280")
        ax.set_title(label, fontsize=10.2, fontweight="normal", pad=5)
        ax.set_xlabel("Indicator value", fontsize=9)
        ax.set_ylabel(outcome_label, fontsize=9)
        ax.tick_params(labelsize=8, direction="in")
        ax.grid(color="#D8DEE5", linewidth=.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.text(.03, .985, "a", ha="left", va="top", fontsize=18, fontweight="bold", color="#17202A")
    fig.suptitle(f"{family}: {outcome_label}", fontsize=12, fontweight="normal", y=.995)
    fig.subplots_adjust(left=.07, right=.985, bottom=.06, top=.94, wspace=.34, hspace=.58)
    fig.savefig(OUT / f"{stem}_nonlinear.png", dpi=DPI, facecolor="white")
    fig.savefig(OUT / f"{stem}_nonlinear.pdf", facecolor="white")
    plt.close(fig)


def render_forest(stats: pd.DataFrame, scale: str, family: str, outcome: str,
                  outcome_label: str, stem: str) -> None:
    q = stats[(stats.scale == scale) & (stats.family == family) & (stats.outcome == outcome)].copy()
    q = q.sort_values("spearman_rho")
    fig, ax = plt.subplots(figsize=(8.8, max(5.0, .38 * len(q) + 1.8)))
    for i, row in enumerate(q.itertuples()):
        color = PALETTE[i % len(PALETTE)]
        ax.plot([row.spearman_ci_low, row.spearman_ci_high], [i, i], color=color, linewidth=2.0,
                solid_capstyle="round")
        ax.scatter(row.spearman_rho, i, s=42, color=color, edgecolor="white", linewidth=.7, zorder=3)
    ax.axvline(0, color="#7D8790", linestyle="--", linewidth=.8)
    ax.set_yticks(range(len(q)))
    ax.set_yticklabels(q.feature_label, fontsize=9)
    ax.set_xlabel("Spearman association", fontsize=10.5)
    ax.set_title(f"{family}: {outcome_label}", fontsize=11.5, fontweight="normal", pad=10)
    ax.grid(axis="x", color="#D8DEE5", linewidth=.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(.03, .985, "a", ha="left", va="top", fontsize=18, fontweight="bold", color="#17202A")
    fig.subplots_adjust(left=.39, right=.96, bottom=.09, top=.94)
    fig.savefig(OUT / f"{stem}_forest.png", dpi=DPI, facecolor="white")
    fig.savefig(OUT / f"{stem}_forest.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    data = pd.read_csv(TABLE, low_memory=False)
    # The source table stores the components separately. Create the explicit
    # controls used by the spatial comparison: 2024 population, 2024 GDP per
    # capita, and urbanized area. They are controls, not the focal indicators.
    data["log_pop"] = pd.to_numeric(data["logPop2024"], errors="coerce")
    data["log_gdppc"] = pd.to_numeric(data["logGDP2024"], errors="coerce") - data["log_pop"]
    # The source table stores 2024 distance and population separately. These
    # explicit aliases make the spatial outcomes transparent and reproducible.
    distance_2024 = pd.to_numeric(data["total_travel_distance_m_2024"], errors="coerce")
    population_2024 = pd.to_numeric(data["Pop_2024"], errors="coerce")
    data["log_distance"] = np.where(distance_2024 > 0, np.log(distance_2024), np.nan)
    data["log_distance_pc"] = np.where(
        (distance_2024 > 0) & (population_2024 > 0),
        np.log(distance_2024 / population_2024),
        np.nan,
    )
    data["short_share"] = pd.to_numeric(data["short_share_2024"], errors="coerce")
    data["long_share"] = pd.to_numeric(data["long_share_2024"], errors="coerce")
    # Keep the V25 primary sample where it is available; otherwise retain all
    # rows with complete function/accessibility observations.
    if "primary_sample" in data:
        flag = data["primary_sample"]
        if flag.dtype == object:
            flag = flag.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
        else:
            flag = flag.fillna(False).astype(bool)
        if flag.sum() >= 100:
            data = data[flag].copy()

    panel = data.copy()
    all_stats = []
    # The city table has 2017/2024 outcome columns; accessibility remains
    # the current snapshot and is not relabelled as a historical measure.
    for family, features in [("Native function mismatch", [
            (f"population_function_mismatch_{name}", label) for name, label in FUNCTIONS
        ]), ("Accessibility", ACCESS)]:
        valid_features = [(x, y) for x, y in features if x in panel.columns]
        if not valid_features:
            continue
        for outcome, outcome_label in OUTCOMES_TIME:
            stats = statistics(panel, valid_features, [(outcome, outcome_label)],
                               "snapshot_vs_2017_2024_change", family)
            if stats.empty:
                continue
            all_stats.append(stats)
            stats.to_csv(OUT / f"{family.replace(' ', '_').lower()}_temporal_{outcome}_statistics.csv",
                         index=False, encoding="utf-8-sig")
            render_forest(stats, "snapshot_vs_2017_2024_change", family, outcome, outcome_label,
                          f"{family.replace(' ', '_').lower()}_temporal_{outcome}")
            render_nonlinear(panel, valid_features, outcome, outcome_label,
                             "snapshot_vs_2017_2024_change", family,
                             f"{family.replace(' ', '_').lower()}_temporal_{outcome}")

    panel = data.copy()
    # city_year_observed_panel is not required: all four 2024 outcomes are
    # already retained in the city table used by V25.
    for family, features in [("Native function mismatch", [
        (f"population_function_mismatch_{name}", label) for name, label in FUNCTIONS
    ]), ("Accessibility", ACCESS)]:
        valid_features = [(x, y) for x, y in features if x in panel.columns]
        for outcome, outcome_label in OUTCOMES_SPACE:
            stats = statistics(panel, valid_features, [(outcome, outcome_label)],
                               "2024_cross_section", family)
            if stats.empty:
                continue
            all_stats.append(stats)
            render_forest(stats, "2024_cross_section", family, outcome, outcome_label,
                          f"{family.replace(' ', '_').lower()}_spatial2024_{outcome}")
            render_nonlinear(panel, valid_features, outcome, outcome_label,
                             "2024_cross_section", family,
                             f"{family.replace(' ', '_').lower()}_spatial2024_{outcome}")

    combined = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    combined.to_csv(OUT / "function_accessibility_all_statistics.csv", index=False, encoding="utf-8-sig")
    (OUT / "README_method.txt").write_text(
        "The function/accessibility variables are current snapshots in the V21 city table.\n"
        "Temporal figures relate that snapshot to observed 2017-2024 changes.\n"
        "Spatial figures compare the same snapshot with 2024 city outcomes.\n"
        "Forest points are raw Spearman rho with city bootstrap 95% intervals.\n"
        "CSV also reports within-country Spearman and standardized country-absorbed OLS\n"
        "controlling for log population, log GDP per capita and log area.\n",
        encoding="utf-8",
    )
    print(f"Wrote results to {OUT}")


if __name__ == "__main__":
    main()
