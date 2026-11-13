from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
EXP = ROOT if (ROOT/'grid_random').exists() else ROOT/'Experiments'
FIG = ROOT / 'Figures'
TAB = ROOT / 'Tables'
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
SCOPES = ['city_2017', 'city_2024', 'grid_random', 'grid_block']
SEEDS = [17, 42, 83, 20261006]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})

def draw(ax, data, regime, positive=False, raw=False, letter=None):
    q = data[data.regime.eq(regime)].copy()
    if positive:
        q = q[q.reference_raw.gt(0)]
    x = q.observed_log.to_numpy()
    y = q.Validation_blend.to_numpy()
    scope = q.scope.iloc[0] if 'scope' in q else ''
    if raw:
        x, y = q.reference_raw.to_numpy(), (np.expm1(y) if scope.startswith('grid') else np.exp(y))
        # Display raw numerical values on asinh axes without dropping extremes or zeros.
        ax.scatter(x, y, s=5, alpha=.22, color='#387D92', linewidths=0, rasterized=True)
        ax.set_xscale('asinh', linear_width=100)
        ax.set_yscale('asinh', linear_width=100)
        label = 'emission (reference units; asinh axis)'
    else:
        ax.hexbin(x, y, gridsize=65, mincnt=1, cmap='viridis', norm=LogNorm(), linewidths=0)
        label = 'ln(1 + emission in reference units)' if scope.startswith('grid') else 'ln(emission in reference units)'
    lo, hi = min(0, x.min(), y.min()), max(x.max(), y.max())*1.025
    ax.plot([lo, hi], [lo, hi], color='#B35B55', lw=1, linestyle='--')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('EDGAR reference\n'+label)
    ax.set_ylabel('Held-out prediction\n'+label)
    title = {'city_2017': 'Cities 2017', 'city_2024': 'Cities 2024',
             'grid_random': '0.1-degree cells: cell holdout',
             'grid_block': '0.1-degree cells: 5-degree block holdout'}.get(scope, scope)
    ax.set_title(title+'\n'+('All-reference training' if regime == 'All_reference' else 'Positive-reference training'), fontsize=10)
    ax.text(.04, .95, f'n = {len(q):,}\n$R^2$ = {r2_score(x,y):.3f}\nMAE = {mean_absolute_error(x,y):.3g}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox={'facecolor':'white', 'alpha':.85, 'edgecolor':'none', 'pad':2})
    if letter:
        ax.text(-.13, 1.08, letter, transform=ax.transAxes, fontweight='bold', fontsize=13)

frames = []
audits = {}
for scope in SCOPES:
    audits[scope] = json.loads((EXP/scope/'input_audit.json').read_text())
    for seed in SEEDS:
        f = pd.read_csv(EXP/scope/str(seed)/'metrics.csv')
        assert {'All_reference','Positive_reference_only'} <= set(f.regime)
        frames.append(f)
allm = pd.concat(frames, ignore_index=True)
allm.to_csv(TAB/'all_model_metrics.csv', index=False)
numeric = ['r2_log','r2_raw','mae_log','rmse_log','mae_raw','spearman','n_test']
fixed = allm[allm.seed.isin([17,42,83])]
summary = fixed.groupby(['scope','regime','model','evaluation'])[numeric].agg(['mean','std'])
summary.columns = ['_'.join(c) for c in summary.columns]
summary.reset_index().to_csv(TAB/'fixed_seed_summary.csv', index=False)
allm[allm.seed.eq(20261006)].to_csv(TAB/'new_split_metrics.csv', index=False)
(TAB/'input_audits.json').write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding='utf-8')

blend = fixed[fixed.model.eq('Validation_blend')]
paired = []
for scope in SCOPES:
    q = blend[blend.scope.eq(scope)&blend.evaluation.eq('Positive_test')]
    p = q.pivot(index='seed', columns='regime', values=['r2_log','mae_log','r2_raw'])
    for seed, row in p.iterrows():
        paired.append({'scope':scope,'seed':seed,
            'positive_only_minus_all_r2_log':row[('r2_log','Positive_reference_only')]-row[('r2_log','All_reference')],
            'positive_only_minus_all_mae_log':row[('mae_log','Positive_reference_only')]-row[('mae_log','All_reference')],
            'positive_only_minus_all_r2_raw':row[('r2_raw','Positive_reference_only')]-row[('r2_raw','All_reference')]})
pd.DataFrame(paired).to_csv(TAB/'paired_positive_test_comparison.csv', index=False)

new = {}
for scope in SCOPES:
    q = pd.read_csv(EXP/scope/'20261006'/'test_predictions.csv')
    q['scope'] = scope
    new[scope] = q
    for regime in ['All_reference','Positive_reference_only']:
        for raw in [False, True]:
            fig, ax = plt.subplots(figsize=(5.3,5.2), layout='constrained')
            draw(ax,q,regime,raw=raw)
            name = f'fit_{scope}_{regime}_'+('raw' if raw else 'log')
            fig.savefig(FIG/(name+'.png'))
            fig.savefig(FIG/(name+'.pdf'))
            plt.close(fig)

for family, scopes in [('city',['city_2017','city_2024']),('grid',['grid_random','grid_block'])]:
    for positive in [False,True]:
        fig, axes = plt.subplots(2,2,figsize=(10.2,9.4),layout='constrained')
        for i, scope in enumerate(scopes):
            for j, regime in enumerate(['All_reference','Positive_reference_only']):
                draw(axes[i,j],new[scope],regime,positive=positive,letter='abcd'[i*2+j])
        name = f'fit_{family}_'+('matched_positive_test' if positive else 'complete_test')
        fig.savefig(FIG/(name+'.png'))
        fig.savefig(FIG/(name+'.pdf'))
        plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
for ax, metric, label in zip(axes,['r2_log','mae_log'],['Held-out log-scale R-squared','Held-out log-scale MAE']):
    for j,(regime,evaluation,color) in enumerate([
        ('All_reference','All_test','#387D92'),
        ('All_reference','Positive_test','#B16B53'),
        ('Positive_reference_only','Positive_test','#65976B'),
        ('Positive_reference_only','All_test','#8D79A7')]):
        q=blend[blend.regime.eq(regime)&blend.evaluation.eq(evaluation)].groupby('scope')[metric].agg(['mean','std']).reindex(SCOPES)
        ax.errorbar(np.arange(4)+(j-1.5)*.13,q['mean'],yerr=q['std'],fmt='o',markersize=5,capsize=3,color=color,
            label=('All train' if regime=='All_reference' else 'Positive train')+' / '+('all test' if evaluation=='All_test' else 'positive test'))
    ax.set_xticks(range(4),['Cities\n2017','Cities\n2024','Grid\ncell holdout','Grid\nblock holdout'])
    ax.set_ylabel(label)
    ax.axhline(0,color='0.7',lw=.6)
axes[0].legend(fontsize=8)
fig.savefig(FIG/'zero_regime_comparison.png')
fig.savefig(FIG/'zero_regime_comparison.pdf')
plt.close(fig)
print(summary.loc[(slice(None),slice(None),'Validation_blend',slice(None)),:].to_string())
