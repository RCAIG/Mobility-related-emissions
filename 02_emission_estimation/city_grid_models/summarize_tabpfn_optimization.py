"""Summarize repeated held-out experiments without test-driven selection."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

ROOT = Path('${SERVER_HOME}/result/tabpfn_emission_optimization_v1')
files = list(ROOT.glob('*/*/metrics.csv'))
d = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
assert len(files) == 12, f'Expected 12 completed split files, found {len(files)}'
assert len(d) == 48 and d.status.eq('ok').all(), 'Some model evaluations are incomplete or failed'
assert not d.duplicated(['scope', 'seed', 'model']).any()
assert np.isfinite(d.r2_test_log).all()
for p in ROOT.glob('*/*/splits.csv'):
    split = pd.read_csv(p)
    assert not split.row.duplicated().any()
    groups = {name: set(part.group) for name, part in split.groupby('subset')}
    assert groups['train'].isdisjoint(groups['test'])
    assert groups['validation'].isdisjoint(groups['test'])
    assert groups['train'].isdisjoint(groups['validation'])
d.to_csv(ROOT / 'all_metrics.csv', index=False)
linear_rows = []
for p in ROOT.glob('*/*/test_predictions.csv'):
    scope, seed = p.parent.parent.name, int(p.parent.name)
    pred = pd.read_csv(p)
    transform = np.expm1
    observed = transform(pred.observed_log.to_numpy(dtype=float))
    for model in pred.columns.difference(['row', 'observed_log']):
        estimate = transform(pred[model].to_numpy(dtype=float))
        linear_rows.append({'scope': scope, 'seed': seed, 'model': model,
                            'r2_test_original_scale': r2_score(observed, estimate)})
pd.DataFrame(linear_rows).to_csv(ROOT / 'original_scale_metrics.csv', index=False)
good = d[d.status.eq('ok')].copy()
fine = good[good.model.str.startswith('TabPFN_finetuned')]
selected = fine.loc[fine.groupby(['scope', 'seed']).r2_validation_log.idxmax()].copy()
selected['selected_configuration'] = selected.model
selected['model'] = 'TabPFN_validation_selected'
selected.to_csv(ROOT / 'validation_selected.csv', index=False)
linear = pd.DataFrame(linear_rows)
linear_selected = selected[['scope', 'seed', 'selected_configuration']].merge(
    linear, left_on=['scope', 'seed', 'selected_configuration'], right_on=['scope', 'seed', 'model'], validate='one_to_one')
linear_selected['model'] = 'TabPFN_validation_selected'
linear = pd.concat([linear, linear_selected[linear.columns]], ignore_index=True)
linear.to_csv(ROOT / 'original_scale_metrics.csv', index=False)
linear.groupby(['scope', 'model']).r2_test_original_scale.agg(['mean', 'std', 'min', 'max']).to_csv(ROOT / 'original_scale_summary.csv')
combined = pd.concat([good, selected], ignore_index=True)
summary = combined.groupby(['scope', 'model']).agg(
    repeats=('seed', 'nunique'), r2_mean=('r2_test_log', 'mean'),
    r2_sd=('r2_test_log', 'std'), r2_min=('r2_test_log', 'min'),
    r2_max=('r2_test_log', 'max'), mae_mean=('mae_test_log', 'mean')).reset_index()
summary.to_csv(ROOT / 'summary.csv', index=False)
paired = combined.pivot(index=['scope', 'seed'], columns='model', values='r2_test_log')
for name in ['TabPFN_finetuned', 'TabPFN_finetuned_longer', 'TabPFN_validation_selected']:
    paired[name + '_gain_over_pretrained'] = paired[name] - paired.TabPFN_pretrained
paired.to_csv(ROOT / 'paired_comparisons.csv')
scopes = ['city_2017', 'city_2024', 'grid_random', 'grid_block']
labels = ['Cities: 2017', 'Cities: 2024', 'Grid: grouped random', 'Grid: spatial blocks']
models = ['HGBR_regularized', 'TabPFN_pretrained', 'TabPFN_validation_selected']
names = ['Regularized boosting', 'Pretrained TabPFN', 'Fine-tuned TabPFN (validation-selected)']
colors = ['#78949D', '#DCA56A', '#698B72']
fig, ax = plt.subplots(figsize=(10, 5))
for j, (model, label, color) in enumerate(zip(models, names, colors)):
    sub = summary[summary.model.eq(model)].set_index('scope').reindex(scopes)
    pos = np.arange(4)+(j-1)*.23
    ax.bar(pos, sub.r2_mean, width=.21, color=color, label=label)
    ax.errorbar(pos, sub.r2_mean, yerr=sub.r2_sd, fmt='none', color='#333333', capsize=3, linewidth=1)
    for x, value, sd in zip(pos, sub.r2_mean, sub.r2_sd):
        ax.text(x, value+sd+.01, f'{value:.3f}', ha='center', fontsize=9)
ax.set_xticks(np.arange(4), labels)
ax.set_ylabel('Held-out R-squared (log emission target)')
ax.set_ylim(0, .86)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc='upper right')
ax.text(0, -.17, 'Error bars: SD across three fixed splits. Fine-tuning selected using validation data only.',
        transform=ax.transAxes, fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / 'heldout_comparison.png', dpi=240, bbox_inches='tight')
print(summary.to_string(index=False))
