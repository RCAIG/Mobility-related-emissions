"""Build a 3 x 3 sensitivity audit for TDI versus change in ln(C/D).

The historical files call this construct TMI/TTI. New outputs use the more
specific name Technological Decarbonization Index (TDI).
"""
from pathlib import Path
import sys

sys.path.insert(0, r"${WORKSPACE_ROOT}\\.pydeps_analysis")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr


ROOT = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V25")
SOURCE = Path(
    r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V22"
    r"\Tables\city_analysis_v22_full.csv"
)
OUT = ROOT / "Figures_paper"
OUT.mkdir(parents=True, exist_ok=True)

STEM = "FigS_TDI_intensity_weight_sensitivity_3x3"
ENERGY_WEIGHTS = [0.20, 0.35, 0.50]
CHARGING_WEIGHTS = [0.10, 0.20, 0.30]
CITY_CALIBRATION_WEIGHT = 0.50
DOMAIN_COLUMNS = [
    "tmi_energy_absolute",
    "tmi_fleet_absolute",
    "tmi_charging_absolute",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "savefig.dpi": 360,
})


def as_bool(series):
    """Read boolean audit fields robustly from either bool or text CSV data."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def weighted_available(frame, columns, weights, required=()):
    """Match the V22 available-domain renormalization exactly."""
    values = frame[columns].to_numpy(float)
    weight_array = np.asarray(weights, float)[None, :]
    valid = np.isfinite(values)
    numerator = np.nansum(values * weight_array, axis=1)
    denominator = np.sum(valid * weight_array, axis=1)
    result = np.divide(
        numerator,
        denominator,
        out=np.full(len(frame), np.nan),
        where=denominator > 0,
    )
    if required:
        complete = frame[list(required)].notna().all(axis=1).to_numpy()
        result[~complete] = np.nan
    return result


def load_analysis_sample():
    columns = [
        "ID_UC_G0", "iso3", "country", "city_name", "sample_V19",
        "intensity_change", "EIMI_orthogonal_signal", "tmi_absolute",
        "tmi_city_cd_calibrated", *DOMAIN_COLUMNS,
    ]
    data = pd.read_csv(SOURCE, usecols=columns, low_memory=False)
    data = data.loc[as_bool(data["sample_V19"])].copy()

    numeric_columns = [
        "intensity_change", "EIMI_orthogonal_signal", "tmi_absolute",
        "tmi_city_cd_calibrated", *DOMAIN_COLUMNS,
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Keep one common complete sample across all nine scenarios. Energy and
    # fleet are mandatory in the published definition; charging is optional.
    required = [
        "intensity_change", "EIMI_orthogonal_signal",
        "tmi_energy_absolute", "tmi_fleet_absolute",
    ]
    return data.dropna(subset=required).copy()


def build_scenarios(data):
    rows = []
    city_values = data[["ID_UC_G0", "iso3", "country", "city_name", "intensity_change"]].copy()

    for energy_weight in ENERGY_WEIGHTS:
        for charging_weight in CHARGING_WEIGHTS:
            fleet_weight = 1.0 - energy_weight - charging_weight
            weights = (energy_weight, fleet_weight, charging_weight)
            country_tdi = weighted_available(
                data,
                DOMAIN_COLUMNS,
                weights,
                required=("tmi_energy_absolute", "tmi_fleet_absolute"),
            )
            scenario = (
                f"E/F/G={energy_weight:.2f}/{fleet_weight:.2f}/{charging_weight:.2f}; "
                f"lambda={CITY_CALIBRATION_WEIGHT:.2f}"
            )
            column = (
                f"tdi_e{str(energy_weight).replace('.', 'p')}"
                f"_f{str(fleet_weight).replace('.', 'p')}"
                f"_g{str(charging_weight).replace('.', 'p')}"
                f"_lambda_{str(CITY_CALIBRATION_WEIGHT).replace('.', 'p')}"
            )
            tdi = country_tdi + CITY_CALIBRATION_WEIGHT * data["EIMI_orthogonal_signal"].to_numpy(float)
            city_values[column] = tdi

            x = data["intensity_change"].to_numpy(float)
            valid = np.isfinite(x) & np.isfinite(tdi)
            x_valid, y_valid = x[valid], tdi[valid]
            pearson = pearsonr(x_valid, y_valid)
            spearman = spearmanr(x_valid, y_valid)
            fit = linregress(x_valid, y_valid)
            rows.append({
                "scenario": scenario,
                "energy_weight": energy_weight,
                "fleet_weight": fleet_weight,
                "charging_weight": charging_weight,
                "city_calibration_lambda": CITY_CALIBRATION_WEIGHT,
                "n_cities": int(valid.sum()),
                "n_countries": int(data.loc[valid, "iso3"].nunique()),
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
                "linear_slope": fit.slope,
                "linear_intercept": fit.intercept,
                "linear_r_squared": fit.rvalue ** 2,
            })

    statistics = pd.DataFrame(rows)

    # Recalculation checks protect the audit against silent formula drift.
    baseline_country = weighted_available(
        data, DOMAIN_COLUMNS, (0.35, 0.45, 0.20),
        required=("tmi_energy_absolute", "tmi_fleet_absolute"),
    )
    baseline_city = baseline_country + 0.50 * data["EIMI_orthogonal_signal"].to_numpy(float)
    country_error = np.nanmax(np.abs(baseline_country - data["tmi_absolute"].to_numpy(float)))
    city_error = np.nanmax(np.abs(baseline_city - data["tmi_city_cd_calibrated"].to_numpy(float)))
    if country_error > 1e-10 or city_error > 1e-10:
        raise RuntimeError(
            "TDI reconstruction does not match V22: "
            f"country error={country_error}, city error={city_error}"
        )
    statistics["max_country_reconstruction_error"] = country_error
    statistics["max_city_reconstruction_error"] = city_error
    return statistics, city_values


def draw(data, statistics, city_values):
    x = data["intensity_change"].to_numpy(float)
    tdi_columns = [column for column in city_values if column.startswith("tdi_")]
    all_y = city_values[tdi_columns].to_numpy(float).ravel()
    x_limits = np.nanquantile(x, [0.01, 0.99])
    y_limits = np.nanquantile(all_y, [0.01, 0.99])
    x_pad = 0.035 * np.diff(x_limits)[0]
    y_pad = 0.035 * np.diff(y_limits)[0]
    x_limits = (x_limits[0] - x_pad, x_limits[1] + x_pad)
    y_limits = (y_limits[0] - y_pad, y_limits[1] + y_pad)

    fig, axes = plt.subplots(3, 3, figsize=(11.2, 10.0), sharex=True, sharey=True)
    panel_letters = iter("abcdefghi")
    plot_index = 0
    hexbins = []

    for row_index, energy_weight in enumerate(ENERGY_WEIGHTS):
        for column_index, charging_weight in enumerate(CHARGING_WEIGHTS):
            ax = axes[row_index, column_index]
            tdi_column = tdi_columns[plot_index]
            y = city_values[tdi_column].to_numpy(float)
            valid = np.isfinite(x) & np.isfinite(y)
            fleet_weight = 1.0 - energy_weight - charging_weight

            hb = ax.hexbin(
                x[valid], y[valid], gridsize=48, mincnt=1,
                cmap="viridis", norm=LogNorm(), linewidths=0,
                extent=(x_limits[0], x_limits[1], y_limits[0], y_limits[1]),
                rasterized=True,
            )
            hexbins.append(hb)

            in_view = valid & (x >= x_limits[0]) & (x <= x_limits[1])
            fit = linregress(x[in_view], y[in_view])
            fit_x = np.linspace(*x_limits, 200)
            ax.plot(
                fit_x, fit.intercept + fit.slope * fit_x,
                color="#B35B55", lw=1.15, linestyle="--", zorder=3,
            )
            ax.axvline(0, color="#727272", lw=0.55, alpha=0.55, zorder=2)
            ax.axhline(0, color="#727272", lw=0.55, alpha=0.55, zorder=2)
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)

            stat = statistics.iloc[plot_index]
            ax.text(
                0.04, 0.95,
                f"Pearson $r$ = {stat.pearson_r:.3f}\n"
                f"Spearman $\\rho$ = {stat.spearman_rho:.3f}",
                transform=ax.transAxes, va="top", fontsize=8.5,
                bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none", "pad": 2.2},
            )
            ax.text(
                -0.16, 1.04, next(panel_letters), transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom", ha="left",
            )
            ax.set_title(
                f"E/F/G = {energy_weight:.2f}/{fleet_weight:.2f}/{charging_weight:.2f}",
                fontsize=10.2, pad=8,
            )
            if column_index == 0:
                ax.text(
                    -0.29, 0.50,
                    f"Energy weight = {energy_weight:.2f}",
                    transform=ax.transAxes, rotation=90, ha="center", va="center",
                    fontsize=9.2, color="#303030",
                )
            plot_index += 1

    fig.supxlabel(r"Change in $\ln(C/D)$", fontsize=11, y=0.04)
    fig.supylabel("Technological Decarbonization Index (TDI)", fontsize=11, x=0.025)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.085, top=0.94, wspace=0.12, hspace=0.15)

    # Keep a common density scale across all panels without adding a bulky
    # colorbar; this matches the feature-ablation reference style.
    max_count = max(float(np.nanmax(hb.get_array())) for hb in hexbins)
    common_norm = LogNorm(vmin=1, vmax=max(2, max_count))
    for hb in hexbins:
        hb.set_norm(common_norm)

    fig.savefig(OUT / f"{STEM}.png", dpi=360, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{STEM}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main():
    data = load_analysis_sample()
    statistics, city_values = build_scenarios(data)
    statistics.to_csv(OUT / f"{STEM}_statistics.csv", index=False, encoding="utf-8-sig")
    city_values.to_csv(OUT / f"{STEM}_city_values.csv", index=False, encoding="utf-8-sig")
    draw(data, statistics, city_values)
    print(statistics[
        ["energy_weight", "fleet_weight", "charging_weight",
         "city_calibration_lambda", "n_cities", "n_countries", "pearson_r", "spearman_rho"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
