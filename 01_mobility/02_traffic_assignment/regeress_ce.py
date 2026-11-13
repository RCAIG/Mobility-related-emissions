import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
import warnings
warnings.filterwarnings("ignore")

# ==================== 1. 加载数据 ====================
print("=" * 80)
print("加载数据集...")
print("=" * 80)

# 特征数据：城市出行模式距离
features_path = "${SERVER_DATA_ROOT}/prediction_travel_mode/trace_generate/code/city_mode_distances.csv"
df_features = pd.read_csv(features_path)

# 标签数据：碳排放量
labels_path = "${SERVER_DATA_ROOT}/prediction_travel_mode/trace_generate/code/ce_result.csv"
df_labels = pd.read_csv(labels_path)

print(f"特征数据形状: {df_features.shape} (列: {list(df_features.columns)})")
print(f"标签数据形状: {df_labels.shape} (列: {list(df_labels.columns)})")

# ==================== 2. 数据预处理与匹配 ====================
print("\n" + "=" * 80)
print("数据预处理与匹配...")
print("=" * 80)

# 2.1 标准化ID字段（转换为字符串并去除空格）
df_features['city_id_clean'] = df_features['city_id'].astype(str).str.strip()
df_labels['ID_UC_G0_clean'] = df_labels['ID_UC_G0'].astype(str).str.strip()

# 2.2 检查ID重叠情况
common_ids = set(df_features['city_id_clean']) & set(df_labels['ID_UC_G0_clean'])
print(f"特征数据中唯一 city_id 数量: {df_features['city_id_clean'].nunique()}")
print(f"标签数据中唯一 ID_UC_G0 数量: {df_labels['ID_UC_G0_clean'].nunique()}")
print(f"匹配成功的样本数量: {len(common_ids)}")

if len(common_ids) == 0:
    raise ValueError("错误：两个数据集之间没有匹配的ID！请检查ID字段格式")

# 2.3 合并数据集（仅保留匹配的样本）
df_merged = pd.merge(
    df_features[['city_id_clean', 'mode_0_km', 'mode_1_km', 'mode_2_km', 'mode_3_km']],
    df_labels[['ID_UC_G0_clean', 'carbon_2024']],
    left_on='city_id_clean',
    right_on='ID_UC_G0_clean',
    how='inner'
)

print(f"\n合并后数据集形状: {df_merged.shape}")
print(f"前5行预览:\n{df_merged.head().to_string(index=False)}")

# 2.4 处理缺失值和异常值
print("\n缺失值检查:")
print(df_merged.isnull().sum())

# 删除缺失值
df_clean = df_merged.dropna(subset=['carbon_2024', 'mode_0_km', 'mode_1_km', 'mode_2_km', 'mode_3_km']).copy()

# 过滤异常值（碳排放量必须为正）
df_clean = df_clean[df_clean['carbon_2024'] > 0].copy()
df_clean = df_clean[(df_clean['mode_0_km'] >= 0) & (df_clean['mode_1_km'] >= 0) & 
                    (df_clean['mode_2_km'] >= 0) & (df_clean['mode_3_km'] >= 0)].copy()

print(f"\n清洗后有效样本数量: {len(df_clean)}")

# if len(df_clean) < 10:
#     raise ValueError(f"有效样本数量过少 ({len(df_clean)})，无法进行训练")

# ==================== 3. 准备特征和目标变量 ====================
print("\n" + "=" * 80)
print("准备特征和目标变量...")
print("=" * 80)

# 特征：4种出行模式的距离（千米·流量）
feature_cols = ['mode_0_km', 'mode_1_km', 'mode_2_km', 'mode_3_km']
X = df_clean[feature_cols].values.astype(np.float32)

# 目标：碳排放量（吨）
y = df_clean['carbon_2024'].values.astype(np.float32)

print(f"特征维度: {X.shape}")
print(f"目标维度: {y.shape}")
print(f"碳排放统计: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}, std={y.std():.2f}")

# 特征标准化（TabPFN对尺度敏感，标准化可提升性能）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================== 4. 划分训练集和测试集 ====================
print("\n" + "=" * 80)
print("划分训练集和测试集...")
print("=" * 80)

# 按碳排放量分层抽样（确保训练/测试分布一致）
# 由于样本量可能较小，使用分层抽样需要谨慎，这里使用简单随机划分
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.3, 
    random_state=42
)

print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")

# ==================== 5. 训练 TabPFN 回归模型 ====================
print("\n" + "=" * 80)
print("训练 TabPFN 回归模型...")
print("=" * 80)

regressor = TabPFNRegressor()
regressor.fit(X_train, y_train)

# ==================== 6. 模型评估 ====================
print("\n" + "=" * 80)
print("模型评估...")
print("=" * 80)

# 预测
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

# 训练集指标
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

# 测试集指标
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n训练集性能:")
print(f"  MSE  : {mse_train:>10.4f}")
print(f"  RMSE : {rmse_train:>10.4f} 吨")
print(f"  MAE  : {mae_train:>10.4f} 吨")
print(f"  R²   : {r2_train:>10.4f}")

print("\n测试集性能:")
print(f"  MSE  : {mse_test:>10.4f}")
print(f"  RMSE : {rmse_test:>10.4f} 吨")
print(f"  MAE  : {mae_test:>10.4f} 吨")
print(f"  R²   : {r2_test:>10.4f}")

# ==================== 7. 结果可视化（可选）====================
print("\n" + "=" * 80)
print("预测结果示例（测试集前10个样本）:")
print("=" * 80)

# 创建结果DataFrame
results = pd.DataFrame({
    'city_id': df_clean.iloc[X_test.shape[0]:]['city_id_clean'].values[:10],  # 注意：需要从原始df_clean中获取对应ID
    'true_carbon': y_test[:10],
    'pred_carbon': y_pred_test[:10],
    'abs_error': np.abs(y_test[:10] - y_pred_test[:10])
})

# 由于train_test_split打乱了顺序，我们需要重新匹配ID
# 更准确的做法：在划分前保留索引
df_clean_reset = df_clean.reset_index(drop=True)
indices = np.arange(len(df_clean_reset))
train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)

# 重新创建带ID的测试集结果
test_results = pd.DataFrame({
    'city_id': df_clean_reset.loc[test_idx[:10], 'city_id_clean'].values,
    'true_carbon': y_test[:10],
    'pred_carbon': y_pred_test[:10],
    'abs_error': np.abs(y_test[:10] - y_pred_test[:10])
})

print(test_results.to_string(index=False))

# ==================== 8. 特征重要性分析（基于排列）====================
print("\n" + "=" * 80)
print("特征重要性分析（排列重要性）...")
print("=" * 80)

# 使用测试集计算排列重要性
baseline_score = r2_score(y_test, y_pred_test)
importances = []

for i, col in enumerate(feature_cols):
    X_test_permuted = X_test.copy()
    np.random.shuffle(X_test_permuted[:, i])  # 打乱第i列
    y_pred_permuted = regressor.predict(X_test_permuted)
    permuted_score = r2_score(y_test, y_pred_permuted)
    importance = baseline_score - permuted_score
    importances.append((col, importance))
    print(f"  {col:15s}: {importance:>6.4f} (R²下降)")

# 按重要性排序
importances_sorted = sorted(importances, key=lambda x: x[1], reverse=True)
print("\n特征重要性排序:")
for col, imp in importances_sorted:
    bar = '█' * int(imp * 50) if imp > 0 else ''
    print(f"  {col:15s}: {imp:>6.4f} {bar}")

print("\n" + "=" * 80)
print("任务完成！")
print("=" * 80)