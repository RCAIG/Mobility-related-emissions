"""Re-render the original dimensions scatter at the map-based export size."""
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parent), r'${WORKSPACE_ROOT}\\.pydeps_analysis']
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import build_fig3_four_panels as original

ROOT = Path(r'${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V25')
OUT = ROOT / 'Figures_paper'
with Image.open(OUT / 'Fig3a_per_capita_emission_change.png') as image:
    WIDTH, HEIGHT = image.size
DPI = 360
VERSION = 'TTI'
PANEL_X = -.13  # Axes-relative position: negative values move left of the y-axis.
PANEL_Y = 1.02
PANEL_FONT = 14

INTENSITY_LABELS = [
    ("Chengdu", "China", 1, 1),       # upper right
    ("Shimla", "India", 1, 1),
    ("Florencia", "Colombia", -1, 1), # upper left
    ("Manavgat", "Turkey", -1, 1),
    ("Montreal", "Canada", -1, -1),   # lower left
    ("Munich", "Germany", -1, -1),
    ("Wuhan", "China", 1, -1),        # lower right
    ("Singapore", "Singapore", 1, -1),
]


def intensity_representative_labels(q):
    """Return exactly two labelled cities from each intensity quadrant."""
    rows = []
    for city_name, country, x_sign, y_sign in INTENSITY_LABELS:
        hit = q[(q.city_name.eq(city_name)) & (q.country.eq(country))]
        hit = hit[(hit.demand_change * x_sign > 0)
                  & (hit.tmi_city_cd_calibrated * y_sign > 0)]
        if not hit.empty:
            rows.append(hit.iloc[0])
    if len(rows) != 8:
        missing = [name for name, _, _, _ in INTENSITY_LABELS
                   if not any(row.city_name == name for row in rows)]
        raise ValueError(f"Missing intensity-panel label cities: {missing}")
    counts = {}
    for row in rows:
        quadrant = (row.demand_change > 0, row.tmi_city_cd_calibrated > 0)
        counts[quadrant] = counts.get(quadrant, 0) + 1
    if sorted(counts.values()) != [2, 2, 2, 2]:
        raise ValueError(f"Expected two labels per quadrant; got {counts}")
    return rows


def panel_label(ax, letter, **kwargs):
    ax.text(PANEL_X, PANEL_Y, 'd', transform=ax.transAxes, ha='left', va='top',
            fontsize=PANEL_FONT, fontweight='bold', color=original.INK,
            zorder=10, clip_on=False)


def save(fig, stem):
    ax = fig.axes[0]
    for collection in ax.collections:
        if collection.get_zorder() == 5:
            collection.set_sizes([18])
            collection.set_linewidths([.45])
    if VERSION == 'intensity':
        ax.set_ylabel('Change in ln(C/D)')
        ax.set_ylim(-1.2, .65)
        offsets = {
            'Chengdu': (8, 6), 'Shimla': (8, 25),
            'Florencia': (-8, -8), 'Manavgat': (-8, -27),
            'Montreal': (-8, -8), 'Munich': (-8, -25),
            'Wuhan': (8, -8), 'Singapore': (8, -27),
        }
        for annotation in ax.texts:
            if annotation.get_text() in offsets:
                annotation.set_position(offsets[annotation.get_text()])
                x, y = annotation.xy
                annotation.set_ha('left' if x >= 0 else 'right')
                annotation.set_va('bottom' if y >= 0 else 'top')
                if annotation.get_text() in {'Florencia', 'Manavgat'}:
                    annotation.set_va('top')
    target_width, target_height = WIDTH / DPI, 2 * HEIGHT / DPI
    fig.set_size_inches(target_width, target_height)
    for _ in range(40):
        fig.canvas.draw()
        box = fig.get_tightbbox(fig.canvas.get_renderer())
        fig.set_size_inches(fig.get_figwidth() + target_width - box.width - .08,
                            fig.get_figheight() + target_height - box.height - .08)
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / f'Fig3d_observed_demand_{VERSION}_dimensions.{extension}',
                    dpi=DPI, bbox_inches='tight', pad_inches=.04, facecolor='white')
    plt.close(fig)


def main():
    global VERSION
    q, _ = original.load_data()
    original.save = save
    original.panel_label = panel_label
    original.panel_b(q)
    VERSION = 'intensity'
    # Reuse the original scatter/annotation layout with the observed C/D axis.
    q = q.dropna(subset=['intensity_change']).copy()
    demand = q.demand_change.lt(0)
    technical = q.intensity_change.lt(0)
    q['portfolio_cd_calibrated'] = np.select(
        [demand & technical, demand & ~technical, ~demand & technical],
        original.ORDER[:3], default='Dual challenge')
    q['tmi_city_cd_calibrated'] = q.intensity_change
    original.representative_labels = intensity_representative_labels
    original.panel_b(q)
    print(f'Exported both variants: {WIDTH} x {2 * HEIGHT} px target')


if __name__ == '__main__':
    main()
