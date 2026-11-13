"""Build the proposed V25 main Figure 4 and its four standalone panels.

Panel a: population organization and temporal demand change.
Panel b: mixed functions/accessibility and temporal demand change.
Panel c: multicentre function mismatch and 2024 per-capita distance.
Panel d: observed demand--intensity portfolios and the route beyond a
technical-transition-only strategy.
"""

from __future__ import annotations

import os
import subprocess
import sys
import argparse
from pathlib import Path


def relaunch_with_python_312() -> None:
    if sys.version_info >= (3, 10):
        return
    if os.environ.get("V25_FIG4_MAIN_PY312") == "1":
        raise RuntimeError("Python 3.10 or newer is required.")
    env = os.environ.copy()
    env["V25_FIG4_MAIN_PY312"] = "1"
    raise SystemExit(subprocess.call(["py", "-3.12", str(Path(__file__).resolve()), *sys.argv[1:]], env=env))


relaunch_with_python_312()
sys.path[:0] = [r"${WORKSPACE_ROOT}", r"${WORKSPACE_ROOT}\\.pydeps_analysis"]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

import build_fig4_function_specific_and_accessibility as base


ROOT = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V25")
TABLE = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V21\Tables\city_analysis_data.csv")
OUT = ROOT / "Figures" / "Fig4_main_multicentre"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 360
INK = "#17202A"
GRID = "#D8DEE5"
COLORS = ["#2F5D8C", "#B7791F", "#168F7A", "#7656A8", "#C84630", "#6B7280"]


def prep_data() -> pd.DataFrame:
    raw = pd.read_csv(TABLE, low_memory=False)
    data = raw.copy()
    data["log_pop"] = pd.to_numeric(data["logPop2024"], errors="coerce")
    data["log_gdppc"] = pd.to_numeric(data["logGDP2024"], errors="coerce") - data["log_pop"]
    distance = pd.to_numeric(data["total_travel_distance_m_2024"], errors="coerce")
    pop = pd.to_numeric(data["Pop_2024"], errors="coerce")
    data["log_distance_pc"] = np.log((distance / pop).where((distance > 0) & (pop > 0)))
    flag = data["primary_sample"]
    if flag.dtype == object:
        flag = flag.astype(str).str.lower().str.strip().isin(["1", "true", "yes", "y"])
    else:
        flag = flag.fillna(False).astype(bool)
    if flag.sum() >= 100:
        data = data[flag].copy()
    metrics = pd.read_csv(ROOT / "Figures" / "Fig4_function_accessibility" / "multicentre_function_metrics_all_cities.csv")
    mismatch_columns = [f"multicentre_mismatch_{x}" for x, _ in base.FUNCTIONS]
    metrics["multicentre_mismatch_mean"] = metrics[mismatch_columns].mean(axis=1)
    return data.merge(metrics[["ID_UC_G0", "multicentre_mismatch_mean"] + mismatch_columns],
                      on="ID_UC_G0", how="left", validate="one_to_one")


def styled(ax):
    ax.axvline(0, color="#818B94", ls="--", lw=.8)
    ax.grid(axis="x", color=GRID, lw=.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def adjusted_forest(data, features, outcome, xlabel, title, label, filename,
                    show_title=True, show_label=True):
    stats = base.statistics(data, features, [(outcome, xlabel)], "main", title)
    q = stats.dropna(subset=["beta_std", "ci_low", "ci_high"]).sort_values("beta_std")
    tall = len(features) >= 12
    fig, ax = plt.subplots(figsize=(9.4, 10.4) if tall else (7.2, 5.7))
    for i, row in enumerate(q.itertuples()):
        c = COLORS[i % len(COLORS)]
        ax.plot([row.ci_low, row.ci_high], [i, i], color=c, lw=2.2, solid_capstyle="round")
        ax.scatter(row.beta_std, i, color=c, s=48, edgecolor="white", linewidth=.8, zorder=3)
    styled(ax)
    ax.set_yticks(range(len(q)), q.feature_label, fontsize=8.4 if tall else 9.5)
    ax.set_xlabel(xlabel, fontsize=10.5)
    if show_title:
        ax.set_title(title, fontsize=12, fontweight="normal", pad=9)
    if show_label:
        fig.text(.045, .96, label, fontsize=20, fontweight="bold", color=INK, va="top")
    fig.subplots_adjust(left=.48 if tall else .42, right=.97, bottom=.09 if tall else .12, top=.92 if tall else .89)
    fig.savefig(
        OUT / filename, dpi=DPI, facecolor="white",
        bbox_inches="tight", pad_inches=.035,
    )
    fig.savefig(
        OUT / filename.replace(".png", ".pdf"), facecolor="white",
        bbox_inches="tight", pad_inches=.035,
    )
    plt.close(fig)
    return stats


INDICATOR_TABLE = [
    ("Population-weighted density", "D_pop = sum_i(p_i * d_i) / sum_i(p_i)", "居民平均所在网格的人口密度；人口越集中在高密度网格，数值越高。"),
    ("Population Gini", "G = sum_i sum_j |d_i - d_j| / (2 * n * sum_i d_i)", "人口密度在网格之间分配得是否不均，越高表示越集中。"),
    ("Top-20% concentration", "C20 = sum of population in densest 20% of grids / total population", "人口最多的五分之一网格吸纳了多少居民。"),
    ("Normalized population entropy", "H = -sum_i p_i ln(p_i) / ln(n)", "人口是否均匀铺开；越高表示分布越均匀。"),
    ("Density heterogeneity", "CV = sd(d_i) / mean(d_i)", "城市不同区域的人口密度差异有多大。"),
    ("Compact proximity", "CP = population-weighted mean proximity to high-density cells", "居民总体上是否靠近连续的高密度居住区域。"),
    ("Central population share", "S_c = population within the central-area mask / total population", "中心区域覆盖的居民比例。"),
    ("Population-centre offset", "O = distance(population-weighted centre, geometric centre)", "人口重心偏离城市几何中心的程度。"),
    ("Population Moran's I", "I = (n / W) * sum_ij w_ij (d_i-bar_d)(d_j-bar_d) / sum_i(d_i-bar_d)^2", "相邻网格的密度是否相似；越高表示高密度更成片。"),
    ("Dense-patch cohesion", "COH = connected high-density patch area / total high-density area", "高密度斑块是连成一片还是彼此分散。"),
    ("Resident local exposure", "E_3km = sum_j p_j * 1(distance(i,j) <= 3 km), population-weighted", "居民在3公里邻域内接触到的本地人口规模。"),
    ("Morphological polycentricity", "P = balance and spacing of identified population centres", "城市由单一中心还是多个规模较均衡的中心共同支撑。"),
    ("Number of population centres", "N_c = count of identified population-density peaks", "识别出的有效人口中心数量。"),
    ("Largest-centre dominance", "D_max = population in largest centre / population in all centres", "最大中心对全部中心人口的支配程度。"),
    ("Peripheral population share", "S_p = peripheral-area population / total population", "生活在城市外围区域的居民比例。"),
    ("Population elongation", "E = major-axis length / minor-axis length of population ellipse", "人口空间形态是紧凑还是沿某一方向拉长。"),
    ("Local function mix", "M_local = normalized entropy of function shares within each local grid", "居民附近是否同时存在多种功能。"),
    ("City-wide function mix", "M_city = normalized entropy of city-wide function shares", "整个城市的功能类别是否多元。"),
]


def render_indicator_table() -> None:
    table_dir = OUT / "Fig4a_indicator_table"
    table_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(INDICATOR_TABLE, columns=["Indicator", "Formula", "Plain-language explanation"])
    frame.to_csv(table_dir / "Fig4a_18_indicators_formula_explanation.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(18, 15.5))
    ax.axis("off")
    cjk_font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=10)
    tab = ax.table(cellText=frame.values, colLabels=frame.columns, colWidths=[.22, .39, .39],
                   cellLoc="left", colLoc="left", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 2.25)
    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor("#C7D0D9")
        cell.set_linewidth(.55)
        cell.PAD = .012
        if row == 0:
            cell.set_facecolor("#2F5D8C")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F3F6F8")
        cell.get_text().set_wrap(True)
        cell.get_text().set_fontproperties(cjk_font)
    fig.savefig(table_dir / "Fig4a_18_indicators_formula_explanation.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(table_dir / "Fig4a_18_indicators_formula_explanation.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def portfolio_panel(data):
    q = data[["delta_log_distance_pc", "intensity_change", "delta_log_emissions_pc", "translation_quadrant"]].replace([np.inf, -np.inf], np.nan).dropna()
    names = {
        "Joint mitigation": "Joint improvement",
        "Technology or mode led": "TTI-led",
        "Carbon translation gap": "Demand-led",
        "Dual challenge": "Dual challenge",
    }
    colors = {"Joint improvement": "#168F7A", "TTI-led": "#2F5D8C", "Demand-led": "#B7791F", "Dual challenge": "#C84630"}
    q["type"] = q.translation_quadrant.map(names).fillna(q.translation_quadrant)
    fig, ax = plt.subplots(figsize=(7.2, 5.7))
    rng = np.random.default_rng(20260929)
    for typ, col in colors.items():
        z = q[q.type.eq(typ)]
        ax.scatter(z.delta_log_distance_pc, z.intensity_change, s=9, alpha=.28, color=col, linewidths=0, label=f"{typ} (n={len(z):,})")
    ax.axvline(0, color="#818B94", lw=.8)
    ax.axhline(0, color="#818B94", lw=.8)
    ax.text(.02, .04, "Lower demand and lower intensity", transform=ax.transAxes, fontsize=8.5, color="#44515C", va="bottom")
    ax.text(.98, .05, "Demand expansion\nwith intensity improvement", transform=ax.transAxes, fontsize=8.5, color="#44515C", ha="right")
    ax.set_xlabel(r"Observed change in $\ln(D/P)$")
    ax.set_ylabel(r"Observed change in $\ln(C/D)$")
    ax.set_xlim(np.nanpercentile(q.delta_log_distance_pc, .5), np.nanpercentile(q.delta_log_distance_pc, 99.5))
    ax.set_ylim(np.nanpercentile(q.intensity_change, .5), np.nanpercentile(q.intensity_change, 99.5))
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=.86,
              fontsize=8, loc="upper left", markerscale=2)
    ax.grid(color=GRID, lw=.4); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.text(.045, .96, "d", fontsize=20, fontweight="bold", color=INK, va="top")
    ax.set_title("Observed demand–intensity portfolios", fontsize=12, fontweight="normal", pad=9)
    fig.subplots_adjust(left=.14, right=.98, bottom=.13, top=.89)
    fig.savefig(OUT / "Fig4d_observed_portfolios.png", dpi=DPI, facecolor="white")
    fig.savefig(OUT / "Fig4d_observed_portfolios.pdf", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--abc-only", action="store_true",
        help="Render only standalone panels a-c; leave panel d and the assembled figure unchanged.",
    )
    parser.add_argument(
        "--mismatch-change-only", action="store_true",
        help="Render multicentre mismatch associations with 2017-2024 change in ln(D/P).",
    )
    args = parser.parse_args()
    data = prep_data()
    a_features = [
        ("Agglomeration_2017", "Population-weighted density"),
        ("Gini_2017", "Population Gini"),
        ("Concentration_2017", "Top-20% concentration"),
        ("Entropy_2017", "Normalized population entropy"),
        ("Density_CV_2017", "Density heterogeneity"),
        ("CompactProximity_2017", "Compact proximity"),
        ("CentralShare50_2017", "Central population share"),
        ("CentroidOffset_2017", "Population-centre offset"),
        ("MoranI_2017", "Population Moran's I"),
        ("DensePatchCohesion_2017", "Dense-patch cohesion"),
        ("LocalExposure3km_2017", "Resident local exposure"),
        ("Polycentricity_2017", "Morphological polycentricity"),
        ("LogCenterCount_2017", "Number of population centres"),
        ("Largest_Center_Share_2017", "Largest-centre dominance"),
        ("PeripheralShare_2017", "Peripheral population share"),
        ("PopulationElongation_2017", "Population elongation"),
    ]
    b_features = [
        ("local_mix_2017", "Local function mix"),
        ("city_mix_2017", "City-wide function mix"),
        ("resident_weighted_hill_30min", "30-min accessible diversity"),
        ("resident_weighted_hill_45min", "45-min accessible diversity"),
        ("walk_proximity_direct_hq_min", "Walking time to nearest function"),
        ("multicentre_mismatch_mean", "Mean multicentre mismatch"),
    ]
    c_features = [("multicentre_mismatch_" + name, label) for name, label in base.FUNCTIONS]
    if args.mismatch_change_only:
        adjusted_forest(
            data, c_features, "d_log_percap_travel_distance",
            "Adjusted standardized association with Δln(D/P)",
            "Multicentre function mismatch and demand change", "",
            "Fig4c_multicentre_mismatch_demand_change.png",
            show_title=False, show_label=False,
        )
        print("Wrote multicentre mismatch-demand change panel to " + str(OUT), flush=True)
        return
    a = adjusted_forest(data, a_features, "d_log_percap_travel_distance", "Adjusted standardized association with Δln(D/P)", "Population organization", "a", "Fig4a_population_organization.png", show_title=False, show_label=False)
    b = adjusted_forest(data, b_features, "d_log_percap_travel_distance", "Adjusted standardized association with Δln(D/P)", "Functions, access and demand", "b", "Fig4b_functions_access_demand.png", show_title=False, show_label=False)
    c = adjusted_forest(data, c_features, "log_distance_pc", "Adjusted standardized association with 2024 ln(D/P)", "Multicentre function mismatch", "c", "Fig4c_multicentre_mismatch.png", show_title=False, show_label=False)
    if args.abc_only:
        pd.concat([a, b, c], ignore_index=True).to_csv(
            OUT / "Fig4_main_forest_statistics.csv", index=False, encoding="utf-8-sig"
        )
        print("Wrote standalone panels a-c to " + str(OUT), flush=True)
        return
    portfolio_panel(data)
    render_indicator_table()
    # Assemble from already rendered panels without changing standalone files.
    paths = [OUT / "Fig4a_population_organization.png", OUT / "Fig4b_functions_access_demand.png", OUT / "Fig4c_multicentre_mismatch.png", OUT / "Fig4d_observed_portfolios.png"]
    images = [plt.imread(p) for p in paths]
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 11.4))
    for ax, image in zip(axes.flat, images):
        ax.imshow(image); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1, wspace=.015, hspace=.02)
    fig.savefig(OUT / "Fig4_main_multicentre.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "Fig4_main_multicentre.pdf", facecolor="white")
    pd.concat([a, b, c], ignore_index=True).to_csv(OUT / "Fig4_main_forest_statistics.csv", index=False, encoding="utf-8-sig")
    (OUT / "README_main_figure.txt").write_text(
        "Panel a: baseline population organization and observed 2017-2024 per-capita travel-distance change.\n"
        "Panel b: mixed functions, resident-weighted accessible diversity, walking proximity and mean multicentre mismatch.\n"
        "Panel c: all 12 native Overture function mismatches, using multiple population and function centres, against 2024 ln(D/P).\n"
        "Panel d: observed demand-intensity portfolios. Joint improvement is the lower-left quadrant; TTI-led is higher demand with lower intensity; demand-led is lower demand with higher intensity; dual challenge is higher demand and higher intensity.\n"
        "Panels a and b use standardized country-absorbed OLS with log population, log GDP per capita and log area controls. Panel c uses descriptive Spearman associations.\n"
        "The figure supports an integrated demand-side route beyond a technology-only transition: compact and connected population organization, mixed functions, accessible diversity, and reduced population-function mismatch are candidate demand-side levers. These are observational associations, not causal policy effects.\n", encoding="utf-8")
    print("Wrote " + str(OUT), flush=True)


if __name__ == "__main__":
    main()
