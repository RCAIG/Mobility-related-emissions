from pathlib import Path

# 路径
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
# OUT_DIR = Path("${SERVER_DATA_ROOT}/prediction_travel_mode/trace_generate/code/output")

OUT_DIR = Path("${SERVER_HOME}/result/trace_genetare_lower_middle_2024")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# 生成的od数据
OD_CSV = DATA_DIR / "od_flows_wgs84.csv"
# 用build_wuhan_gragh生成的网络//可以使walk也可以是drive
GRAPHML = DATA_DIR / "city_drive.graphml"

# 路网阻抗字段：osmnx add_edge_travel_times 后是 travel_time（秒）
WEIGHT_COL = "travel_time"   # 或 "length"

# Monte Carlo 参数
N_SIM = 30                  # 先30验证；最终可50/100
SIGMA = 0.15                # lognormal 扰动强度：exp(sigma*Z)
DETOUR_RATIO = 1.30         # 绕路过滤阈值：C_noise <= ratio * C0

# 大规模建议：对非常小的OD流量直接丢弃（可选）
MIN_FLOW = 1.0              # 例如 1.0

# 随机种子（保证可复现）
RANDOM_SEED = 42
