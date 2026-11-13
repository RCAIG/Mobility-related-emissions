"""Create eight compact transparent boxplots accompanying Figure 3d."""
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent), r'${WORKSPACE_ROOT}\\.pydeps_analysis']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import build_fig3_four_panels as original


ROOT = Path(r'${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V25')
OUT = ROOT / 'Figures_paper'
DPI = 360


def classify(data, y_col, favorable_positive):
    demand_good = data['demand_change'].lt(0)
    tech_good = data[y_col].gt(0) if favorable_positive else data[y_col].lt(0)
    return np.select(
        [demand_good & tech_good, demand_good & ~tech_good,
         ~demand_good & tech_good],
        original.ORDER[:3], default=original.ORDER[3])


def compact_boxplot(data, x_col, y_col, labels, category, color, output,
                    width_px, height_px):
    fig, ax = plt.subplots(figsize=(width_px / DPI, height_px / DPI), dpi=DPI)
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    values = [data[x_col].dropna().to_numpy(), data[y_col].dropna().to_numpy()]
    box = ax.boxplot(values, positions=[1, 2], widths=.55, patch_artist=True,
        whis=1.5, showfliers=True,
        medianprops={'color': original.INK, 'linewidth': .75},
        boxprops={'facecolor': color, 'edgecolor': color, 'linewidth': .65,
                  'alpha': .72},
        whiskerprops={'color': color, 'linewidth': .65},
        capprops={'color': color, 'linewidth': .65},
        flierprops={'marker': 'o', 'markersize': .85, 'markerfacecolor': color,
                    'markeredgewidth': 0, 'alpha': .28})
    ax.axhline(0, color=original.INK, linewidth=.45, alpha=.75, zorder=0)
    if category in original.ORDER[1:3]:
        combined = np.concatenate(values)
        lo, hi = np.nanpercentile(combined, [1, 99])
        lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
        padding = max((hi - lo) * .08, .025)
        ax.set_ylim(lo - padding, hi + padding)
    ax.set_xticks([1, 2], labels, fontsize=6.7)
    ax.tick_params(axis='x', length=0, pad=1.2)
    ax.tick_params(axis='y', labelsize=6.2, width=.45, length=1.6, pad=1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#87939A')
        ax.spines[side].set_linewidth(.5)
    ax.margins(x=.12)
    fig.subplots_adjust(left=.30, right=.97, top=.97, bottom=.18)
    fig.savefig(OUT / f'{output}.png', dpi=DPI, transparent=True)
    fig.savefig(OUT / f'{output}.pdf', transparent=True)
    plt.close(fig)
    return len(data)


def main():
    q, _ = original.load_data()
    versions = [
        ('TTI', 'tmi_city_cd_calibrated', True, ('D/P', 'TTI')),
        ('intensity', 'intensity_change', False, ('D/P', 'C/D')),
    ]
    audit = []
    for version, y_col, favorable_positive, labels in versions:
        reference_path = OUT / f'Fig3d_observed_demand_{version}_dimensions.png'
        with Image.open(reference_path) as image:
            width_px = round(image.width / 8 * 1.13)
            height_px = round(image.height / 4)
        data = q.dropna(subset=['demand_change', y_col]).copy()
        data['category'] = classify(data, y_col, favorable_positive)
        for index, category in enumerate(original.ORDER, 1):
            subset = data[data.category.eq(category)]
            stem = f'Fig3d_{version}_box_{index}_{category.lower().replace(" ", "_")}'
            n = compact_boxplot(subset, 'demand_change', y_col, labels, category,
                                original.COLORS[category], stem,
                                width_px, height_px)
            audit.append({'version': version, 'category': category, 'n': n,
                          'width_px': width_px, 'height_px': height_px,
                          'file': stem + '.png'})
    pd.DataFrame(audit).to_csv(OUT / 'Fig3d_category_boxplots_audit.csv', index=False)
    print(pd.DataFrame(audit).to_string(index=False))


if __name__ == '__main__':
    main()
