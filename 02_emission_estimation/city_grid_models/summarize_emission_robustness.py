"""Audit training-only trimming and validation-selected spatial ensembles."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('${SERVER_HOME}/result')
OUT=ROOT/'emission_robustness_summary'
OUT.mkdir(exist_ok=True)
trim=pd.concat([pd.read_csv(p) for p in (ROOT/'emission_outlier_robustness_v1').glob('*/*/metrics.csv')],ignore_index=True)
assert len(trim)==132 and not trim.duplicated(['scope','seed','model']).any()
assert np.isfinite(trim.r2_test_log).all()
for p in (ROOT/'emission_outlier_robustness_v1').glob('*/*/splits.csv'):
    d=pd.read_csv(p)
    groups={k:set(v.group) for k,v in d.groupby('subset')}
    assert groups['train'].isdisjoint(groups['test']) and groups['validation'].isdisjoint(groups['test'])
    assert groups['train'].isdisjoint(groups['validation'])
trim.to_csv(OUT/'outlier_all_metrics.csv',index=False)
trim.groupby(['scope','model']).agg(r2_log_mean=('r2_test_log','mean'),r2_log_sd=('r2_test_log','std'),
    r2_raw_mean=('r2_test_original_scale','mean'),mae_log_mean=('mae_test_log','mean'),
    removed_mean=('removed_fraction','mean'),top1pct_sse_log=('top1pct_test_sse_share_log','mean'),
    top1pct_sse_raw=('top1pct_test_sse_share_original','mean')).to_csv(OUT/'outlier_summary.csv')
ensemble=pd.concat([pd.read_csv(p) for p in (ROOT/'emission_spatial_ensemble_v1').glob('*/*/metrics.csv')],ignore_index=True)
assert len(ensemble)==30 and not ensemble.duplicated(['scope','seed','model']).any()
ensemble.to_csv(OUT/'ensemble_all_metrics.csv',index=False)
ensemble.groupby(['scope','model']).agg(r2_log_mean=('r2_test_log','mean'),r2_log_sd=('r2_test_log','std'),
    r2_log_min=('r2_test_log','min'),r2_log_max=('r2_test_log','max'),
    r2_raw_mean=('r2_test_original_scale','mean')).to_csv(OUT/'ensemble_summary.csv')
calibration=pd.concat([pd.read_csv(p) for p in (ROOT/'emission_spatial_calibration_v1').glob('*/*/metrics.csv')],ignore_index=True)
assert len(calibration)==18 and not calibration.duplicated(['scope','seed','model']).any()
calibration.to_csv(OUT/'calibration_all_metrics.csv',index=False)
assert np.isfinite(calibration[['r2_test_log','r2_test_original_scale']]).all().all()
for p in (ROOT/'emission_spatial_calibration_v1').glob('*/*/test_predictions.csv'):
    predictions=pd.read_csv(p)
    original=pd.read_csv(ROOT/'emission_outlier_robustness_v1'/p.parent.parent.name/p.parent.name/'test_predictions.csv')
    assert np.array_equal(predictions.row.to_numpy(),original.row.to_numpy())
    assert np.allclose(predictions.observed_log,original.observed_log)
calibration.groupby(['scope','model']).agg(r2_log_mean=('r2_test_log','mean'),r2_log_sd=('r2_test_log','std'),
    r2_log_min=('r2_test_log','min'),r2_log_max=('r2_test_log','max'),
    r2_raw_mean=('r2_test_original_scale','mean')).to_csv(OUT/'calibration_summary.csv')
# Validate policy: choose only by validation, keeping complete test samples.
candidates=trim[trim.model.str.startswith('HGB_')].copy()
chosen=candidates.loc[candidates.groupby(['scope','seed']).r2_validation_log.idxmax()].copy()
chosen.to_csv(OUT/'validation_selected_robust_tree.csv',index=False)
errors=[]
for p in (ROOT/'emission_outlier_robustness_v1').glob('*/*/test_predictions.csv'):
    d=pd.read_csv(p);scope=p.parent.parent.name;seed=int(p.parent.name)
    e=(d.observed_log-d.HGB_full)**2
    k=int(np.ceil(len(d)*.01))
    z=d.loc[e.nlargest(k).index,['row','observed_log','HGB_full']].copy()
    z['scope']=scope;z['seed']=seed;z['squared_error_log']=e.loc[z.index]
    errors.append(z)
pd.concat(errors,ignore_index=True).to_csv(OUT/'largest_test_errors_diagnostic_only.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(12,5),sharey=True)
for ax,scope in zip(axs,['grid_random','grid_block']):
    subsets=[('HGB full',trim[trim.model.eq('HGB_full')]),
        ('HGB trim 0.5%',trim[trim.model.eq('HGB_Trim_0.5pct')]),
        ('HGB trim 2%',trim[trim.model.eq('HGB_Trim_2pct')]),
        ('TabPFN fine-tuned',trim[trim.model.eq('TabPFN_finetuned_validation_rule')]),
        ('Validation blend',ensemble[ensemble.model.eq('Validation_selected_blend')]),
        ('Blend + spatial calibration',calibration[calibration.model.eq('Validation_spatial_calibration_blend')])]
    for i,(name,d) in enumerate(subsets):
        scores=d[d.scope.eq(scope)].r2_test_log.to_numpy()
        ax.errorbar(i,scores.mean(),yerr=scores.std(ddof=1),fmt='o',capsize=4,color='#477E91')
        ax.scatter(np.full(len(scores),i)+np.array([-.08,0,.08]),scores,s=12,color='#B28362')
    ax.set_xticks(range(len(subsets)),[a for a,b in subsets],rotation=30,ha='right',fontsize=9)
    ax.axhline(.6,color='#888888',linestyle='--',linewidth=1)
    ax.set_title('Grouped random holdout' if scope=='grid_random' else 'Spatial-block holdout',fontsize=11)
    ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('Held-out R-squared: log emission')
fig.tight_layout();fig.savefig(OUT/'spatial_robustness_comparison.png',dpi=240,bbox_inches='tight')
fig,axs=plt.subplots(1,2,figsize=(9,4.5),sharex=True,sharey=True)
for ax,scope in zip(axs,['grid_random','grid_block']):
    p=pd.read_csv(ROOT/'emission_spatial_calibration_v1'/scope/'42'/'test_predictions.csv')
    density=ax.hexbin(p.observed_log,p.calibrated_blend_log,gridsize=55,mincnt=1,bins='log',cmap='viridis')
    fig.colorbar(density,ax=ax,fraction=.035,pad=.025,label='Grid-year count')
    lo=min(p.observed_log.min(),p.calibrated_blend_log.min())
    hi=max(p.observed_log.max(),p.calibrated_blend_log.max())
    ax.plot([lo,hi],[lo,hi],color='#555555',linewidth=1,linestyle='--')
    value=calibration[(calibration.scope.eq(scope))&(calibration.seed.eq(42))&
        (calibration.model.eq('Validation_spatial_calibration_blend'))].r2_test_log.iloc[0]
    ax.set_title(('Grouped random holdout' if scope=='grid_random' else 'Spatial-block holdout')+f'\nSeed 42, R-squared = {value:.3f}',fontsize=11)
    ax.set_xlabel('Reference ln(1 + emission)')
    ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('Predicted ln(1 + emission)')
fig.tight_layout();fig.savefig(OUT/'spatial_predictions_seed42.png',dpi=240,bbox_inches='tight')
print(pd.read_csv(OUT/'ensemble_summary.csv').to_string(index=False))
print(pd.read_csv(OUT/'outlier_summary.csv').to_string(index=False))
print(pd.read_csv(OUT/'calibration_summary.csv').to_string(index=False))
