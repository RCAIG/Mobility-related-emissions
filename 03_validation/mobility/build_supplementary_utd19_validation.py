from pathlib import Path
import json
import math
import sys
sys.path.insert(0, r"${WORKSPACE_ROOT}\\.pydeps_analysis")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

ROOT = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V25")
OUT = ROOT / "supplementary_fig"
CACHE = OUT / "utd19_cache"
INK, GRID, LAND = "#253238", "#DCE3E5", "#EEF1F0"
FLOW = mpl.colors.LinearSegmentedColormap.from_list("flow", ["#DCE9E5", "#75B5A4", "#167F73", "#174D4A"])
WORLD_SHP = Path(r"${WORKSPACE_ROOT}\\.pydeps_analysis\pyogrio\tests\fixtures\naturalearth_lowres\naturalearth_lowres.shp")

def style():
    mpl.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": .65,
                         "pdf.fonttype": 42, "ps.fonttype": 42})

def load():
    links = pd.read_parquet(CACHE / "utd19_observed_links.parquet")
    metrics = pd.read_csv(CACHE / "utd19_loco_metrics.csv")
    return links, metrics

def world_map(links, metrics):
    city = links.groupby("citycode", as_index=False).agg(lon=("lon", "mean"), lat=("lat", "mean"))
    fig, ax = plt.subplots(figsize=(7.15, 3.55))
    ax.set_facecolor(LAND)
    world = gpd.read_file(WORLD_SHP)
    world.plot(ax=ax, facecolor="#D7DEDC", edgecolor="#AEBBB7", linewidth=.28, zorder=1)
    ax.scatter(city.lon, city.lat, s=24, c="#167F73", edgecolors="white", linewidths=.45, alpha=.88, zorder=3)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_aspect(1.35)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.text(.015, .045, f"UTD19 monitored urban road links ({city.citycode.nunique()} cities)", transform=ax.transAxes, color=INK, fontsize=7)
    for s in ax.spines.values(): s.set_visible(False)
    fig.savefig(OUT / "FigS_UTD19_cities_world_map.png", dpi=360, bbox_inches="tight", pad_inches=.025)
    fig.savefig(OUT / "FigS_UTD19_cities_world_map.pdf", bbox_inches="tight", pad_inches=.025)
    plt.close(fig)
    return int(city.citycode.nunique())

def city_examples(links):
    chosen = ["birmingham", "frankfurt", "hamburg", "madrid", "manchester",
              "melbourne", "munich", "paris", "toronto"]
    available = [c for c in chosen if c in set(links.citycode)]
    if len(available) < 9:
        available = links.citycode.drop_duplicates().head(9).tolist()
    fig, axes = plt.subplots(3, 3, figsize=(7.15, 7.15))
    selected_links = links[links.citycode.isin(available[:9])]
    global_norm = mpl.colors.Normalize(
        vmin=max(1, selected_links.observed_flow.quantile(.05)),
        vmax=selected_links.observed_flow.quantile(.95),
        clip=True,
    )
    for i, (ax, code) in enumerate(zip(axes.flat, available[:9])):
        d = links[links.citycode.eq(code)].copy()
        ax.scatter(d.lon, d.lat, c="#D8E0DE", s=2.5, alpha=.65,
                   linewidths=0, zorder=1)
        ax.scatter(d.lon, d.lat, c=d.observed_flow, cmap=FLOW, norm=global_norm,
                   s=2.0, alpha=.72, linewidths=0, zorder=2)
        ax.text(.5, -.055, code.replace("_", " ").title(), transform=ax.transAxes,
                ha="center", va="top", fontsize=8, color=INK, clip_on=False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("white")
        for s in ax.spines.values(): s.set_visible(False)
        ax.text(.02, .92, chr(97+i), transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK)
    sm = mpl.cm.ScalarMappable(norm=global_norm, cmap=FLOW)
    sm.set_array([])
    fig.subplots_adjust(wspace=.08, hspace=.24, left=.02, right=.98, bottom=.12, top=.96)
    cax = fig.add_axes([.12, .065, .76, .018])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Observed road-link flow", fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2, width=.5)
    cbar.outline.set_linewidth(.45)
    fig.savefig(OUT / "FigS_UTD19_nine_city_examples.png", dpi=360, bbox_inches="tight", pad_inches=.025)
    fig.savefig(OUT / "FigS_UTD19_nine_city_examples.pdf", bbox_inches="tight", pad_inches=.025)
    plt.close(fig)
    return available[:9]

def metrics_plot(metrics, year):
    d = metrics[metrics.simulation_year.eq(year) & metrics.stage.eq("LOCO power calibration")].copy()
    d = d.sort_values("cpc")
    y = np.arange(len(d)); fig, ax1 = plt.subplots(figsize=(7.15, 4.2))
    ax1.barh(y, d.cpc, color="#167F73", alpha=.82, label="CPC")
    labels = d.citycode.str.replace("taipeh", "taipei", case=False).str.title()
    ax1.set_yticks(y, labels); ax1.set_xlim(0, 1); ax1.set_xlabel("CPC")
    ax1.set_ylabel("City"); ax1.grid(axis="x", color=GRID, lw=.55); ax1.set_axisbelow(True)
    ax2 = ax1.twiny(); ax2.plot(d.js_divergence, y, "o", color="#D17A57", ms=3.5, label="JS divergence")
    ax2.set_xlim(0, max(.35, d.js_divergence.max()*1.15)); ax2.set_xlabel("JS divergence")
    ax1.legend(loc="lower right", frameon=False); ax2.legend(loc="upper right", frameon=False)
    ax1.spines["top"].set_visible(False); ax2.spines["bottom"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"FigS_UTD19_LOCO_power_validation_{year}.png", dpi=360, bbox_inches="tight", pad_inches=.025)
    fig.savefig(OUT / f"FigS_UTD19_LOCO_power_validation_{year}.pdf", bbox_inches="tight", pad_inches=.025)
    plt.close(fig)
    return d[["citycode", "cpc", "js_divergence"]].to_dict("records")

def main():
    style(); links, metrics = load()
    audit = {"unique_cities_in_links": int(links.citycode.nunique()), "metric_city_year_rows": int(len(metrics)),
             "world_map_cities": world_map(links, metrics), "example_cities": city_examples(links),
             "LOCO_power_2017": metrics_plot(metrics, 2017),
             "LOCO_power_2024": metrics_plot(metrics, 2024)}
    (OUT / "FigS_UTD19_validation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))

if __name__ == "__main__": main()
