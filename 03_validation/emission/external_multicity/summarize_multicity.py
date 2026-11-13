from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

O = Path('${SERVER_HOME}/result/external_validation_multicity_v1')
names = ['india_all_city_metrics', 'india_coarse_all_city_metrics', 'brazil_all_city_metrics', 'kenya_all_city_coarse_metrics', 'arizona_all_city_metrics', 'france_idf_all_city_metrics']
parts = []
summary = []
for name in names:
    path = O / (name + '.csv')
    if not path.exists() or path.stat().st_size < 5:
        continue
    frame = pd.read_csv(path)
    if 'status' in frame:
        frame = frame[frame.status.eq('completed')].copy()
    frame['dataset_group'] = name
    parts.append(frame)
    subsets = [(name, frame)]
    if name == 'india_all_city_metrics':
        subsets = [(f'India {int(res)} m', group[group.support.eq('all_reference_valid')]) for res, group in frame.groupby('nominal_resolution_m')]
    for label, group in subsets:
        for minimum in [3, 10, 20]:
            q = group[group.n.ge(minimum)]
            summary.append({'group': label, 'minimum_spatial_units': minimum, 'cities': len(q), 'median_rho': q.spearman_rho.median(), 'min_rho': q.spearman_rho.min(), 'max_rho': q.spearman_rho.max(), 'cities_rho_ge_0_5': int(q.spearman_rho.ge(.5).sum())})
if parts:
    pd.concat(parts, ignore_index=True).to_csv(O/'all_city_external_validation_metrics.csv', index=False)
pd.DataFrame(summary).to_csv(O/'dataset_summary_by_sample_size.csv', index=False)
coverage_rows = []
for path in sorted(O.glob('*city*status.csv')):
    if path.stat().st_size < 5:
        continue
    status = pd.read_csv(path)
    for state, group in status.groupby('status'):
        coverage_rows.append({'file':path.name,'status':state,'cities':len(group)})
pd.DataFrame(coverage_rows).to_csv(O/'city_coverage_and_failures_summary.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False), flush=True)
india = O/'india_all_city_metrics.csv'
if india.exists():
    data = pd.read_csv(india)
    data = data[data.support.eq('all_reference_valid')]
    pivot = data.pivot(index='city', columns='nominal_resolution_m', values='spearman_rho').sort_index()
    pivot.rename(columns={500:'rho_approx_500m',1000:'rho_approx_1km',2000:'rho_approx_2km'}).to_csv(O/'india_city_scale_comparison_wide.csv')
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    y = np.arange(len(pivot))
    for offset, (resolution, color) in zip([-.2, 0, .2], [(500, '#488F91'), (1000, '#CE875D'), (2000, '#8277A5')]):
        if resolution in pivot:
            ax.scatter(pivot[resolution], y+offset, color=color, label=f'~{resolution} m', s=35)
            uncertainty_path = O/'india_city_spatial_block_uncertainty.csv'
            if uncertainty_path.exists():
                u = pd.read_csv(uncertainty_path)
                u = u[u.nominal_resolution_m.eq(resolution)].set_index('city').reindex(pivot.index)
                ax.hlines(y+offset,u.spatial_block_bootstrap_ci_low,u.spatial_block_bootstrap_ci_high,color=color,lw=1,alpha=.7)
    ax.set_yticks(y, pivot.index)
    ax.axvline(0, color='.6', linewidth=.7)
    ax.set_xlim(-1, 1)
    ax.set_xlabel('Spearman spatial rank correlation')
    ax.legend(loc='lower left', frameon=False)
    fig.savefig(O/'India_multicity_scale_comparison.png', dpi=300)
    plt.close(fig)
    print(pivot.to_string(), flush=True)
    coarse = pd.concat([x for x in parts if not x.dataset_group.iloc[0].startswith('india')], ignore_index=True)
    coarse = coarse[coarse.n.ge(10)].copy()
    coarse['label'] = coarse.city.astype(str)+' ('+coarse.n.astype(str)+')'
    coarse = coarse.sort_values(['dataset_group','city'])
    fig, axes = plt.subplots(1, 2, figsize=(12, max(6,len(coarse)*.28)), layout='constrained')
    for resolution, color in [(500, '#488F91'), (1000, '#CE875D')]:
        if resolution in pivot:
            axes[0].scatter(pivot[resolution], np.arange(len(pivot))+(0 if resolution==500 else .2), color=color, s=30, label=f'~{resolution} m')
    axes[0].set_yticks(np.arange(len(pivot)),pivot.index)
    axes[0].legend(loc='lower left', frameon=False)
    axes[0].set_title('India: road reconstruction at fine scales', fontsize=10)
    colors = {'brazil_all_city_metrics':'#488F91','arizona_all_city_metrics':'#CE875D','france_idf_all_city_metrics':'#8277A5','kenya_all_city_coarse_metrics':'#A59B55'}
    for key, group in coarse.groupby('dataset_group', sort=False):
        positions = [coarse.index.get_loc(i) for i in group.index]
        axes[1].scatter(group.spearman_rho, positions, c=colors.get(key,'.5'), s=30, label=key.split('_')[0])
    axes[1].set_yticks(np.arange(len(coarse)),coarse.label)
    axes[1].legend(loc='lower left', frameon=False, fontsize=8)
    axes[1].set_title('Coarse external comparisons: n >= 10 units', fontsize=10)
    for ax in axes:
        ax.set_xlim(-1,1); ax.axvline(0,color='.6',lw=.7)
        ax.set_xlabel('Spearman spatial rank correlation')
        ax.tick_params(axis='y',labelsize=8)
    fig.savefig(O/'Global_multicity_comparison.png',dpi=300)
    plt.close(fig)

lines = ['# 多城市外部排放验证', '', '本次按数据集覆盖范围逐城匹配，不按相关系数筛选城市。完整逐城指标见 all_city_external_validation_metrics.csv，未完成或空间单元不足的城市见各数据集 status.csv。', '', '## 比较尺度与解释', '', '- 印度 CHETNA-Road：2021 年道路 CO2，原生 0.005°，约500米；与2024年模拟道路排放按道路实际相交长度匹配，并汇总至约1公里和2公里。全有效参考网格作为主结果，双正值子集作为敏感性结果。原生经纬度网格的米制宽度随纬度变化，500米是近似称呼。', '- 印度粗网格补充：所有15城市另尝试0.1°参考保守汇总，模型按有效参考覆盖面积分配。面积分配可能贡献较高相关，只作粗尺度诊断，不能替代原生道路验证。德里、加尔各答缺少原始模拟轨迹，单独记录；加尔各答只有1个可匹配粗网格，无法计算逐城空间相关。', '- 巴西 BRAVES：2017年道路 CO2；本次共同尺度为0.1°。原生约5公里网格按球面重叠面积保守重映射，已检查汇总排放守恒。', '- 美国 Arizona Hestia：2017年道路化石燃料CO2；新增城市本次为0.1°共同网格对照，不能称为100米验证。此前Phoenix/Mesa细尺度验证保留。', '- 法国 AIRPARIF：2019年交通CO2，IRIS统计区。新增比较将0.1°模型按面积投影到IRIS，是粗模型的行政区对照，不是独立的道路级重建验证。', '- 肯尼亚 Climate TRACE：2024年道路CO2，选取与城市相交的完整选区；新增比较是粗网格按面积汇总至选区，属于城市及周边区域对照。此前Nairobi道路重建的17区验证单独保留，不能与本次26个相交区的比较混用。', '', '不同年份比较主要检验空间格局。模型与参考的单位未统一，不将空间排序相关写成排放绝对量R²。对数Pearson相关受原始单位和log1p影响，仅作辅助指标；Spearman排序相关是主指标。粗模型按面积投影时，行政区面积差异可能影响总量相关，因此不能把相关完全归因于城市内部道路预测能力。', '', '## 样本数量', '', '3个空间单元仅是能够计算相关的最低门槛，并不意味着可靠的单城市验证。因此同时报告至少10、20个空间单元的汇总；小样本的±1相关不应解释为完美验证。相邻城市可能共用网格或行政区，不属于完全独立重复。', '', pd.DataFrame(summary).to_csv(index=False), '', '## 与论文证据的关系', '', '应将印度逐城细尺度验证、已有Phoenix/Mesa及Nairobi道路验证作为细尺度证据，将其余新增城市列为尺度更粗的外部空间对照。不能据此宣称所有大洲都已完成500米道路排放验证。负相关、低相关和缺失原始轨迹的城市均需保留并说明。']
lines.extend(['', '## 城市覆盖与无法计算的情况', '', pd.DataFrame(coverage_rows).to_csv(index=False), '', '## 印度细尺度处理流程', '', '1. 读取每座城市2024年原始交通分配轨迹，按flow累计经过道路的活动，并结合道路长度形成道路活动；使用原有绝对距离五分类和已经估计的类别响应系数，不使用CHETNA重新训练或调参。', '2. 将已训练全球模型的0.1°预测按城市活动份额和道路活动下分到道路段；检查城市网格排放与道路排放总和相等。', '3. 把道路几何投影到当地UTM坐标系，计算每条道路与参考网格实际相交的长度；按相交长度/道路长度分配该段预测排放。', '4. 参考数据只保留有有效日值的网格，缺失范围不补成零。对参考年有效日值求和得到年度量。', '5. 在相同参考空间范围内比较模型和参考，计算Spearman相关及高排放前10%重叠率；同时报告全有效网格、任一正值和双方正值三个口径。', '6. 原生网格按2×2和4×4相邻格求和，得到约1公里和2公里对照。尺度变粗后的相关提升不能反推原生尺度同样精确。', '', '## 道路重建覆盖审计', '', 'reconstruction/tables保存轨迹数量、道路匹配比例、缺少全球预测网格数量、道路/网格排放守恒结果及实际轨迹目录。若某些网格缺少全球预测，当前重建用该城市已匹配网格的预测排放/道路活动中位比值填补；该数量必须随结果公开。这属于模型空间覆盖补充，不使用外部参考排放。', '', '原始CHETNA参考网格与GHSL城市边界可能不完全一致；本次在完整有效参考范围评估，城市模拟活动缺失的参考位置保留为模型零预测。因此边缘覆盖不一致也可能降低相关。', '', '## 下载与复现', '', '数据出处见multicity_source_manifest.csv。脚本保存在scripts目录；大体积原始轨迹、道路重建parquet和参考数组保存在服务器external_validation_multicity_v1目录，不重复复制到本地。CSV与PNG结果已同步到本地。'])
(O/'多城市验证说明.md').write_text('\n'.join(lines), encoding='utf-8')
with (O/'多城市验证说明.md').open('a',encoding='utf-8') as file:
    file.write('\n\n空间单元数n表示参与比较的网格或行政区数量，不等于统计独立样本数。法国数千个IRIS区由更少的0.1°模型网格面积分配得到，不能视为数千次独立细尺度预测。巴西和亚利桑那州选取与城市相交的完整共同网格，因此城市边缘网格可能含有城市外活动；本次检验的是城市相关范围的空间格局，而非严格裁剪后城市总量。\n')
    file.write('\n印度额外按原生4×4网格（约2公里）形成空间块，进行1000次块重采样，报告Spearman相关的2.5%—97.5%区间。该区间反映有限空间样本的不确定性，不包含参考清单误差、模型参数误差或2021—2024年份差异。详见india_city_spatial_block_uncertainty.csv。\n')
