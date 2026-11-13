import os
import glob
import pandas as pd

# 配置路径
INPUT_AE_DIR = "${SERVER_DATA_ROOT}/prediction_travel_mode/trace_generate/code/output_ae_test/"

# 查找 parquet 文件
parquet_files = glob.glob(os.path.join(INPUT_AE_DIR, "*.parquet"))

if not parquet_files:
    print(f"❌ No .parquet files found in {INPUT_AE_DIR}")
else:
    # 读取第一个文件
    first_file = parquet_files[0]
    print(f"🔍 Reading first file: {first_file}")

    try:
        df = pd.read_parquet(first_file)
        print("\n✅ Columns in the parquet file:")
        for i, col in enumerate(df.columns):
            print(f"  {i + 1}. '{col}'")

        print(f"\n📊 Total columns: {len(df.columns)}")
        print(f"🧮 Number of rows: {len(df)}")

        # 模型推理实际需要的字段（根据你提供的特征生成代码）
        required_fields = ['seg_id', 'ae_patch_11x11x64_b64']
        print(f"\n🧠 Fields required by inference script:")
        for field in required_fields:
            print(f"  - '{field}'")

        # 检查缺失
        missing = [f for f in required_fields if f not in df.columns]
        if missing:
            print(f"\n❌ Missing required fields: {missing}")
            print("💡 Possible causes:")
            print("   - The original trajectory file didn't have 'seg_id'")
            print("   - The feature generation code used a different ID column (e.g., 'id', 'segment_id')")
            print("   - You need to check how 'seg_id' was defined in the upstream process.")
        else:
            print("\n✅ All required fields are present!")

    except Exception as e:
        print(f"❌ Error reading {first_file}: {e}")