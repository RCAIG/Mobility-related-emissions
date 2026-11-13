import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
import re
from collections import defaultdict

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

# 配置目录路径
PRED_DIR = Path(
    "${SERVER_HOME}/result/mode_predict/dynamic_predictions_with_center/"
)
PATH_DIR = Path("${SERVER_HOME}/result/trace_genetate/")
OUTPUT_CSV = Path(
    "${SERVER_DATA_ROOT}/prediction_travel_mode/trace_generate/code/city_mode_distances.csv"
)


def extract_city_and_chunk(filename: str):
    """
    从预测文件名提取 (city_id, chunk_id)
    示例: '2176_chunk_0000_predictions.parquet' -> ('2176', '0000')
    返回: (city_id: str, chunk_id: str) 或 (None, None) 表示无效文件
    """
    stem = Path(filename).stem  # 移除 .parquet
    # 匹配模式: <数字>_chunk_<数字>_predictions
    match = re.match(r'^(\d+)_chunk_(\d+)_predictions$', stem)
    if match:
        return match.group(1), match.group(2)
    return None, None


def build_path_filename(city_id: str, chunk_id: str) -> str:
    """构建路径文件名: od_paths_flows_<city_id>_chunk_<chunk_id>.parquet"""
    return f"od_paths_flows_{city_id}_chunk_{chunk_id}.parquet"


def process_chunk(city_id: str, chunk_id: str, pred_path: Path, path_path: Path) -> dict:
    """
    处理单个 chunk：计算该 chunk 的加权距离
    Returns:
        dict: {mode_0_km, mode_1_km, mode_2_km, mode_3_km} 或 None（失败时）
    """
    # 1. 读取预测文件
    try:
        pred_df = pd.read_parquet(pred_path)
    except Exception as e:
        print(f"  跳过 {city_id}_chunk_{chunk_id}：预测文件读取失败 - {e}")
        return None

    if 'pred_label' not in pred_df.columns:
        print(f"  跳过 {city_id}_chunk_{chunk_id}：预测文件缺少 'pred_label' 字段")
        return None
    if 'total_length_m' not in pred_df.columns:
        print(f"  跳过 {city_id}_chunk_{chunk_id}：预测文件缺少 'total_length_m' 字段")
        return None

    # 2. 读取路径文件（普通表格 Parquet）
    try:
        path_df = pd.read_parquet(path_path)
    except Exception as e:
        print(f"  跳过 {city_id}_chunk_{chunk_id}：路径文件读取失败 - {e}")
        return None

    if 'flow' not in path_df.columns:
        print(f"  跳过 {city_id}_chunk_{chunk_id}：路径文件缺少 'flow' 字段")
        return None

    # 3. 严格检查行数一致性
    if len(pred_df) != len(path_df):
        print(f"  跳过 {city_id}_chunk_{chunk_id}：行数不一致 (预测:{len(pred_df)} vs 路径:{len(path_df)})")
        return None

    # 4. 标准化 pred_label 为整数
    pred_df['pred_label'] = pd.to_numeric(pred_df['pred_label'], errors='coerce').fillna(-1).astype(int)

    # 5. 处理 total_length_m：米 → 千米
    pred_df['length_km'] = pd.to_numeric(pred_df['total_length_m'], errors='coerce').fillna(0.0) / 1000.0

    # 6. 处理 flow：转为数值
    path_df['flow'] = pd.to_numeric(path_df['flow'], errors='coerce').fillna(0.0)

    # 7. 按行顺序合并
    combined = pd.DataFrame({
        'pred_label': pred_df['pred_label'].values,
        'length_km': pred_df['length_km'].values,
        'flow': path_df['flow'].values
    })

    # 8. 计算加权距离
    combined['weighted_km'] = combined['length_km'] * combined['flow']

    # 9. 仅保留有效模式 (0-3)
    valid_mask = combined['pred_label'].between(0, 3)
    combined_valid = combined[valid_mask]

    # 10. 按出行模式分组求和
    mode_totals = [0.0] * 4
    if not combined_valid.empty:
        grouped = combined_valid.groupby('pred_label')['weighted_km'].sum()
        for mode in range(4):
            mode_totals[mode] = grouped.get(mode, 0.0)

    # 11. 诊断：mode_2 有记录但距离≈0
    mode2_count = ((combined['pred_label'] == 2) & (combined['flow'] > 0)).sum()
    if mode2_count > 0 and mode_totals[2] < 1e-6:
        print(f"  警告 {city_id}_chunk_{chunk_id}：mode_2 有 {mode2_count} 条有效记录，但加权距离≈0")

    return {
        'mode_0_km': mode_totals[0],
        'mode_1_km': mode_totals[1],
        'mode_2_km': mode_totals[2],
        'mode_3_km': mode_totals[3]
    }


def main():
    # 查找所有预测文件
    pred_files = list(PRED_DIR.glob("*_chunk_*_predictions.parquet"))
    if not pred_files:
        print(f"错误：在 {PRED_DIR} 下未找到任何 *_chunk_*_predictions.parquet 文件")
        return

    print(f"发现 {len(pred_files)} 个预测文件（含 chunk）")

    # 按 city_id 分组：{city_id: [(chunk_id, pred_file_path), ...]}
    city_chunks = defaultdict(list)
    invalid_files = 0

    for pred_file in pred_files:
        city_id, chunk_id = extract_city_and_chunk(pred_file.name)
        if city_id is None or chunk_id is None:
            print(f"  跳过无效文件名: {pred_file.name}")
            invalid_files += 1
            continue
        city_chunks[city_id].append((chunk_id, pred_file))

    print(f"有效城市分组: {len(city_chunks)} 个（跳过 {invalid_files} 个无效文件）")

    # 处理每个城市（聚合所有 chunk）
    city_records = []
    total_chunks_processed = 0
    total_chunks_skipped = 0
    cities_processed = 0
    cities_skipped = 0

    # 按城市ID数值排序处理（更直观）
    sorted_cities = sorted(city_chunks.keys(), key=lambda x: int(x))

    for city_id in tqdm(sorted_cities, desc="处理城市"):
        chunks = city_chunks[city_id]
        # 初始化累加器
        city_totals = [0.0, 0.0, 0.0, 0.0]  # mode_0 ~ mode_3
        chunks_successful = 0
        chunks_failed = 0

        # 处理该城市的所有 chunk
        for chunk_id, pred_file in chunks:
            path_filename = build_path_filename(city_id, chunk_id)
            path_file = PATH_DIR / path_filename

            if not path_file.exists():
                print(f"\n  跳过 {city_id}_chunk_{chunk_id}：路径文件不存在 → {path_file}")
                chunks_failed += 1
                total_chunks_skipped += 1
                continue

            # 处理单个 chunk
            chunk_result = process_chunk(city_id, chunk_id, pred_file, path_file)
            if chunk_result:
                # 累加四个模式的距离
                for i in range(4):
                    city_totals[i] += chunk_result[f'mode_{i}_km']
                chunks_successful += 1
                total_chunks_processed += 1
            else:
                chunks_failed += 1
                total_chunks_skipped += 1

        # 判断城市是否成功（至少一个 chunk 成功）
        if chunks_successful > 0:
            city_records.append({
                'city_id': city_id,
                'mode_0_km': round(city_totals[0], 4),
                'mode_1_km': round(city_totals[1], 4),
                'mode_2_km': round(city_totals[2], 4),
                'mode_3_km': round(city_totals[3], 4)
            })
            cities_processed += 1
            # 可选：打印城市级汇总
            # print(f"  城市 {city_id}: 成功 {chunks_successful}/{len(chunks)} 个 chunk")
        else:
            print(f"\n跳过城市 {city_id}：所有 {len(chunks)} 个 chunk 均处理失败")
            cities_skipped += 1

    # 生成结果 DataFrame
    if not city_records:
        print("错误：未成功处理任何城市")
        return

    result_df = pd.DataFrame(city_records)

    # 按 city_id 数值排序（已按排序处理，此处确保）
    result_df = result_df.sort_values(
        by='city_id',
        key=lambda x: x.astype(int)
    ).reset_index(drop=True)

    # 保存为 CSV
    result_df.to_csv(OUTPUT_CSV, index=False, float_format='%.4f')

    # 打印统计摘要
    print("\n" + "=" * 80)
    print("处理完成！")
    print(f"  成功处理城市: {cities_processed} 个")
    print(f"  跳过城市:     {cities_skipped} 个")
    print(f"  成功处理 chunk: {total_chunks_processed} 个")
    print(f"  跳过 chunk:     {total_chunks_skipped} 个")
    print(f"  输出文件: {OUTPUT_CSV}")
    print("\n结果预览 (前15行):")
    print(result_df.head(15).to_string(index=False))
    print("=" * 80)

    # 全局统计
    print("\n全局出行距离统计 (千米 × 流量):")
    total_modes = result_df[['mode_0_km', 'mode_1_km', 'mode_2_km', 'mode_3_km']].sum()
    total_all = total_modes.sum()
    print(f"  模式0: {total_modes['mode_0_km']:>15,.2f} km·flow ({total_modes['mode_0_km'] / total_all * 100:>5.2f}%)")
    print(f"  模式1: {total_modes['mode_1_km']:>15,.2f} km·flow ({total_modes['mode_1_km'] / total_all * 100:>5.2f}%)")
    print(f"  模式2: {total_modes['mode_2_km']:>15,.2f} km·flow ({total_modes['mode_2_km'] / total_all * 100:>5.2f}%)")
    print(f"  模式3: {total_modes['mode_3_km']:>15,.2f} km·flow ({total_modes['mode_3_km'] / total_all * 100:>5.2f}%)")
    print(f"  总计:  {total_all:>15,.2f} km·flow")
    print("=" * 80)

    # mode_2 专项检查
    mode2_positive = result_df[result_df['mode_2_km'] > 0.0001].sort_values('mode_2_km', ascending=False)
    if not mode2_positive.empty:
        print(f"\nmode_2 有贡献的城市数量: {len(mode2_positive)}")
        print("前10个 mode_2 贡献最大的城市:")
        print(mode2_positive[['city_id', 'mode_2_km']].head(10).to_string(index=False))
    else:
        print("\n⚠️  警告：没有任何城市的 mode_2 有非零贡献！")
        print("   请检查：")
        print("   1. 预测文件中是否存在 pred_label=2 的记录")
        print("   2. 对应记录的 total_length_m 和 flow 是否 >0")
    print("=" * 80)


if __name__ == "__main__":
    main()