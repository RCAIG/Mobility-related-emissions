"""Summarize all fixed and new split evaluations without selecting on test."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_absolute_error
ROOT=Path('${SERVER_HOME}/result')
OUT=ROOT/'emission_structural_optimization_v2'
metrics=pd.concat([pd.read_csv(p) for p in OUT.glob('grid_*/*/metrics.csv')],ignore_index=True)
assert len(metrics)==64 and not metrics.duplicated(['scope','seed','model']).any()
assert np.isfinite(metrics[['r2_test_log','r2_test_original_scale']]).all().all()
for p in OUT.glob('grid_*/*/splits.csv'):
    split=pd.read_csv(p);groups={k:set(v.group) for k,v in split.groupby('subset')}
    assert all(groups[a].isdisjoint(groups[b]) for a,b in [('train','test'),('train','validation'),('validation','test')])
    pred=pd.read_csv(p.parent/'test_predictions.csv')
    assert np.array_equal(pred.row,split[split.subset.eq('test')].row)
    for name in pred.columns[2:]:
        score=r2_score(pred.observed_log,pred[name])
        record=metrics[(metrics.scope.eq(p.parent.parent.name))&(metrics.seed.eq(int(p.parent.name)))&(metrics.model.eq(name))]
        assert len(record)==1 and np.isclose(score,record.r2_test_log.iloc[0],atol=1e-6)
metrics.to_csv(OUT/'all_metrics.csv',index=False)
old=metrics[metrics.seed.ne(20261006)]
old.groupby(['scope','model']).agg(r2_log_mean=('r2_test_log','mean'),r2_log_sd=('r2_test_log','std'),
    r2_raw_mean=('r2_test_original_scale','mean'),mae_log_mean=('mae_test_log','mean')).to_csv(OUT/'fixed_split_summary.csv')
metrics[metrics.seed.eq(20261006)].to_csv(OUT/'new_split_confirmation.csv',index=False)
clas=pd.concat([pd.read_csv(p).assign(scope=p.parent.parent.name,seed=int(p.parent.name)) for p in OUT.glob('grid_*/*/zero_classification.csv')])
clas.to_csv(OUT/'zero_classification_summary.csv',index=False)
sub=[]
for p in OUT.glob('grid_*/*/test_predictions.csv'):
    d=pd.read_csv(p)
    for name in d.columns[2:]:
        for label,mask in [('zero',d.observed_log.eq(0)),('positive',d.observed_log.gt(0))]:
            sub.append({'scope':p.parent.parent.name,'seed':int(p.parent.name),'model':name,'reference_subset':label,
                'n':int(mask.sum()),'mae_log':mean_absolute_error(d.loc[mask,'observed_log'],d.loc[mask,name]),
                'r2_log':r2_score(d.loc[mask,'observed_log'],d.loc[mask,name]) if label=='positive' else np.nan})
pd.DataFrame(sub).to_csv(OUT/'zero_positive_diagnostics.csv',index=False)
diagnostic=pd.DataFrame(sub)
fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
zero_order=['Baseline_matched','Mobility_restructured','Hurdle_expected_log','Hurdle_hard_validation','Regional_partial_pooling']
for ax,scope in zip(axes,['grid_random','grid_block']):
    for i,name in enumerate(zero_order):
        for offset,subset,color in [(-.15,'zero','#36788A'),(.15,'positive','#B98562')]:
            scores=diagnostic[(diagnostic.scope.eq(scope))&(diagnostic.seed.ne(20261006))&
                (diagnostic.model.eq(name))&(diagnostic.reference_subset.eq(subset))].mae_log.to_numpy()
            ax.bar(i+offset,scores.mean(),width=.28,color=color,alpha=.8,label=subset if i==0 else None)
    ax.set_xticks(range(5),['Baseline','Mobility','Soft hurdle','Hard hurdle','Regional'],rotation=25,ha='right',fontsize=9)
    ax.set_title('Grouped grid holdout' if scope=='grid_random' else 'Spatial-block holdout',fontsize=11)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Mean absolute error: ln(1 + emission)')
axes[1].legend(frameon=False)
fig.tight_layout();fig.savefig(OUT/'zero_positive_error_comparison.png',dpi=260,bbox_inches='tight')
order=['Baseline_matched','Mobility_restructured','Hurdle_expected_log','Hurdle_hard_validation',
    'Hurdle_expected_raw','Regional_partial_pooling','Validation_selected_combination','Combination_plus_spatial_calibration']
labels=['Matched baseline','Mobility descriptors','Hurdle: expected log','Hurdle: hard zero','Hurdle: expected raw',
    'Regional pooling','Validation blend','Blend + calibration']
fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
for ax,scope in zip(axes,['grid_random','grid_block']):
    for i,name in enumerate(order):
        a=old[(old.scope.eq(scope))&(old.model.eq(name))].r2_test_log.to_numpy()
        ax.errorbar(i,a.mean(),yerr=a.std(ddof=1),fmt='o',capsize=3,color='#36788A',ms=5)
        ax.scatter(i+np.array([-.08,0,.08]),a,s=12,color='#B98562')
        new=metrics[(metrics.scope.eq(scope))&(metrics.seed.eq(20261006))&(metrics.model.eq(name))].r2_test_log.iloc[0]
        ax.scatter(i,new,marker='D',facecolors='none',edgecolors='#965875',s=32,zorder=4)
    ax.axhline(.6,ls='--',color='#777777',lw=1)
    ax.set_xticks(range(len(order)),labels,rotation=35,ha='right',fontsize=8)
    ax.set_title('Grouped grid holdout' if scope=='grid_random' else '5-degree spatial-block holdout',fontsize=11)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Held-out R-squared: ln(1 + emission)')
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([],[],color='#36788A',marker='o',label='Mean and SD: seeds 17 / 42 / 83'),
    Line2D([],[],color='#965875',marker='D',markerfacecolor='none',ls='none',label='New split: seed 20261006')],
    loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.04))
fig.tight_layout();fig.savefig(OUT/'structural_strategy_comparison.png',dpi=260,bbox_inches='tight')
fig,axes=plt.subplots(1,2,figsize=(10,4.5))
for ax,scope in zip(axes,['grid_random','grid_block']):
    d=pd.read_csv(OUT/scope/'20261006/test_predictions.csv')
    h=ax.hexbin(d.observed_log,d.Combination_plus_spatial_calibration,gridsize=60,mincnt=1,bins='log',cmap='viridis')
    ax.plot([0,16],[0,16],ls='--',lw=1,color='#555555')
    fig.colorbar(h,ax=ax,label='Grid-year count',fraction=.04,pad=.025)
    score=r2_score(d.observed_log,d.Combination_plus_spatial_calibration)
    ax.set_title(('Grid holdout' if scope=='grid_random' else 'Spatial-block holdout')+f'\nNew split, R-squared = {score:.3f}',fontsize=11)
    ax.set_xlabel('Reference ln(1 + emission)');ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Predicted ln(1 + emission)')
fig.tight_layout();fig.savefig(OUT/'new_split_predictions.png',dpi=260,bbox_inches='tight')
print(pd.read_csv(OUT/'fixed_split_summary.csv').to_string(index=False))
print(metrics[metrics.seed.eq(20261006)][['scope','model','r2_test_log','r2_test_original_scale']].to_string(index=False))
cross=pd.read_csv(OUT/'cross_family_metrics.csv')
assert len(cross)==16 and not cross.duplicated(['scope','seed','model']).any()
cross[cross.seed.ne(20261006)].groupby(['scope','model']).agg(r2_log_mean=('r2_test_log','mean'),
    r2_log_sd=('r2_test_log','std'),r2_raw_mean=('r2_test_original_scale','mean'),mae_log_mean=('mae_test_log','mean')).to_csv(OUT/'cross_family_fixed_summary.csv')
fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
prior=pd.read_csv(ROOT/'emission_robustness_summary/calibration_all_metrics.csv')
for ax,scope in zip(axes,['grid_random','grid_block']):
    series=[('Prior blend + calibration',prior[(prior.scope.eq(scope))&(prior.model.eq('Validation_spatial_calibration_blend'))].r2_test_log.to_numpy()),
        ('Structural blend',old[(old.scope.eq(scope))&(old.model.eq('Validation_selected_combination'))].r2_test_log.to_numpy()),
        ('Cross-family blend',cross[(cross.scope.eq(scope))&(cross.seed.ne(20261006))&(cross.model.eq('Cross_family_validation_blend'))].r2_test_log.to_numpy()),
        ('Cross-family + calibration',cross[(cross.scope.eq(scope))&(cross.seed.ne(20261006))&(cross.model.eq('Cross_family_plus_calibration'))].r2_test_log.to_numpy())]
    for i,(label,scores) in enumerate(series):
        ax.errorbar(i,scores.mean(),yerr=scores.std(ddof=1),fmt='o',capsize=4,color='#36788A')
        ax.scatter(i+np.array([-.08,0,.08]),scores,s=14,color='#B98562')
    for i,name in [(2,'Cross_family_validation_blend'),(3,'Cross_family_plus_calibration')]:
        value=cross[(cross.scope.eq(scope))&(cross.seed.eq(20261006))&(cross.model.eq(name))].r2_test_log.iloc[0]
        ax.scatter(i,value,s=36,marker='D',facecolors='none',edgecolors='#965875')
    ax.axhline(.6,ls='--',color='#777777',lw=1)
    ax.set_xticks(range(4),[a for a,b in series],rotation=25,ha='right',fontsize=9)
    ax.set_title('Grouped grid holdout' if scope=='grid_random' else '5-degree spatial-block holdout',fontsize=11)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Held-out R-squared: ln(1 + emission)')
fig.tight_layout();fig.savefig(OUT/'previous_and_new_strategy_comparison.png',dpi=260,bbox_inches='tight')
print(pd.read_csv(OUT/'cross_family_fixed_summary.csv').to_string(index=False))
print(cross[cross.seed.eq(20261006)].to_string(index=False))
