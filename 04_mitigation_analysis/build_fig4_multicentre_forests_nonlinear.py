"""V25 multicentre forests and LOWESS, using the original analysis settings."""
import os
import subprocess
import sys
from pathlib import Path

if sys.version_info < (3, 10):
    if os.environ.get('MULTICENTRE_FIG_PY312') == '1':
        raise RuntimeError('Python 3.10 or newer is required.')
    env = os.environ.copy()
    env['MULTICENTRE_FIG_PY312'] = '1'
    raise SystemExit(subprocess.call(['py', '-3.12', str(Path(__file__).resolve()), *sys.argv[1:]], env=env))

sys.path[:0] = [r'${WORKSPACE_ROOT}', r'${WORKSPACE_ROOT}\\.pydeps_analysis']
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import build_fig4_function_specific_and_accessibility as base

OUTPUT = base.OUT / 'Multicentre_Forests_Nonlinear'
OUTPUT.mkdir(parents=True, exist_ok=True)


def adjusted_forest(stats, label, stem):
    q = stats.dropna(subset=['beta_std', 'ci_low', 'ci_high']).sort_values('beta_std')
    fig, ax = plt.subplots(figsize=(8.8, max(5, .38 * len(q) + 1.8)))
    for i, row in enumerate(q.itertuples()):
        color = base.PALETTE[i % len(base.PALETTE)]
        ax.plot([row.ci_low, row.ci_high], [i, i], color=color, lw=2)
        ax.scatter(row.beta_std, i, s=42, color=color, edgecolor='white', linewidth=.7, zorder=3)
    ax.axvline(0, color='#7D8790', ls='--', lw=.8)
    ax.set_yticks(range(len(q)), q.feature_label, fontsize=9)
    ax.set_xlabel('Adjusted standardized association (95% CI)', fontsize=10.5)
    ax.set_title('Multicentre mismatch: ' + label, fontsize=11.5, fontweight='normal')
    ax.grid(axis='x', color='#D8DEE5', lw=.45)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.subplots_adjust(left=.39, right=.96, bottom=.09, top=.94)
    for ext in ['png', 'pdf']:
        fig.savefig(OUTPUT / (stem + '_adjusted_forest.' + ext), dpi=base.DPI, facecolor='white')
    plt.close(fig)


def main():
    raw = pd.read_csv(base.TABLE, low_memory=False)
    metrics = pd.read_csv(base.OUT / 'multicentre_function_metrics_all_cities.csv')
    if metrics.ID_UC_G0.duplicated().any():
        raise ValueError('Duplicate multicentre city IDs')
    data = raw.merge(metrics, on='ID_UC_G0', how='inner', validate='one_to_one')
    flag = data.primary_sample
    if flag.dtype == object:
        flag = flag.astype(str).str.lower().str.strip().isin(['1', 'true', 'yes', 'y'])
    else:
        flag = flag.fillna(False).astype(bool)
    if flag.sum() >= 100:
        data = data[flag].copy()
    data['log_pop'] = pd.to_numeric(data.logPop2024, errors='coerce')
    data['log_gdppc'] = pd.to_numeric(data.logGDP2024, errors='coerce') - data.log_pop
    distance = pd.to_numeric(data.total_travel_distance_m_2024, errors='coerce')
    pop = pd.to_numeric(data.Pop_2024, errors='coerce')
    data['log_distance'] = np.log(distance.where(distance > 0))
    data['log_distance_pc'] = np.log((distance / pop).where((distance > 0) & (pop > 0)))
    data['short_share'] = data.short_share_2024
    data['long_share'] = data.long_share_2024
    data.to_csv(OUTPUT / 'multicentre_analysis_sample.csv', index=False, encoding='utf-8-sig')
    base.OUT = OUTPUT
    features = [('multicentre_mismatch_' + name, label) for name, label in base.FUNCTIONS]
    all_stats = []
    for scale, prefix, outcomes in [
        ('snapshot_vs_2017_2024_change', 'temporal', base.OUTCOMES_TIME),
        ('2024_cross_section', 'spatial2024', base.OUTCOMES_SPACE),
    ]:
        for outcome, label in outcomes:
            stem = 'multicentre_' + prefix + '_' + outcome
            print('Processing ' + stem, flush=True)
            stats = base.statistics(data, features, [(outcome, label)], scale, 'Multicentre function mismatch')
            stats.to_csv(OUTPUT / (stem + '_statistics.csv'), index=False, encoding='utf-8-sig')
            all_stats.append(stats)
            base.render_forest(stats, scale, 'Multicentre function mismatch', outcome, label, stem)
            base.render_nonlinear(data, features, outcome, label, scale, 'Multicentre function mismatch', stem)
            adjusted_forest(stats, label, stem)
    pd.concat(all_stats, ignore_index=True).to_csv(OUTPUT / 'multicentre_all_statistics.csv', index=False, encoding='utf-8-sig')
    (OUTPUT / 'README_method.txt').write_text(
        'Same primary_sample filter and modelling settings as the original single-centre script.\n'
        'Population: 2024 grid. Functional centres: Overture 2026-08-19 snapshot.\n'
        'Temporal plots relate a current snapshot to 2017-2024 observed changes; they are not historical change effects.\n'
        'Spatial plots relate the same functional snapshot to 2024 mobility outcomes.\n'
        'Raw forest: Spearman rho, city bootstrap 95% intervals (400 replicates).\n'
        'Adjusted forest: standardized OLS, country fixed effects, log population, log GDP per capita, log area; country-clustered 95% intervals.\n'
        'LOWESS: unadjusted descriptive curve, frac=0.38; 60 city bootstrap replicates; x tails trimmed at 1/99 percentiles, fitting sample capped at 900 as in original script.\n'
        'LOWESS bands are pointwise descriptive intervals, not causal effects or simultaneous intervals.\n'
        'No original figures are overwritten. Accessibility based on all population cells and POIs is unchanged by centre detection.\n', encoding='utf-8')
    print('Completed: ' + str(OUTPUT), flush=True)


if __name__ == '__main__':
    main()
