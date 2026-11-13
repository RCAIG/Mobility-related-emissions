"""Summarize distance-bin experiments without selecting a favorable split."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('${SERVER_HOME}/result/distance_bin_audit_v1')
d = pd.read_csv(ROOT / 'results_repeated.csv')
if 'model' not in d:
    d['model'] = 'HGBR'
f = pd.read_csv(ROOT / 'results_followup.csv')
combined = pd.concat([d.assign(experiment='bin_ablation'), f.assign(experiment='model_followup')], ignore_index=True)
combined.to_csv(ROOT / 'all_results.csv', index=False)
summary = combined.groupby(['experiment', 'scope', 'split', 'bins', 'model']).agg(
    r2_mean=('r2_log', 'mean'), r2_min=('r2_log', 'min'), r2_max=('r2_log', 'max'),
    n_repeats=('r2_log', 'size'), mae_mean=('mae_log', 'mean'),
    train_r2_mean=('r2_train_log', 'mean')).reset_index()
summary.to_csv(ROOT / 'summary.csv', index=False)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 220})
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
panels = [('city_2017', 'heldout_city', '2017 cities'), ('city_2024', 'heldout_city', '2024 cities'),
          ('grid', 'random_cell', 'Random held-out grid cells'), ('grid', 'block_5deg', 'Held-out 5-degree blocks')]
for ax, (scope, split, title) in zip(axes.flat, panels):
    part = d[(d.scope == scope) & (d.split == split)]
    for expanded, color, label in [(False, '#3A7C9B', 'Distance totals + shares'),
                                    (True, '#BB7255', '+ composition features')]:
        names = ['5_expanded', '10_expanded', '13_expanded'] if expanded else ['5', '8', '10', '13']
        s = part[part.bins.isin(names)].groupby('bins').r2_log.agg(['mean', 'min', 'max']).reindex(names)
        x = [int(n.split('_')[0]) for n in names]
        ax.errorbar(x, s['mean'], yerr=[s['mean']-s['min'], s['max']-s['mean']], color=color,
                    marker='o', capsize=3, label=label)
    ax.axhline(.8 if scope.startswith('city') else .7, color='#999999', ls=':', lw=1)
    ax.set(title=title, xlabel='Number of absolute-distance bins', ylabel='Held-out R-squared (log target)')
    ax.set_xticks([5, 8, 10, 13])
axes[0, 0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / 'distance_bin_comparison.png')
plt.close(fig)
print(summary.round(4).to_string(index=False))
