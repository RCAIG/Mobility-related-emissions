from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
EXP = ROOT / "Experiments" / "emission_feature_ablation_year_v1"
OUT = ROOT / "Tables"
OUT.mkdir(parents=True, exist_ok=True)

frames = []
for source in sorted(EXP.glob("*/20261006/metrics.csv")):
    frames.append(pd.read_csv(source))

summary = pd.concat(frames, ignore_index=True).sort_values(
    ["scope", "year", "r2_log"], ascending=[True, True, False]
)
summary.to_csv(OUT / "feature_ablation_year_v1_metrics.csv", index=False)

pivot = summary.pivot_table(
    index="feature_set", columns=["scope", "year"],
    values=["r2_log", "mae_log", "rmse_log"], aggfunc="first",
)
pivot.to_csv(OUT / "feature_ablation_year_v1_metrics_wide.csv")
