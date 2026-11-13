"""Run the 2017 Chicago same-year OD calibration on the migrated server data."""
import json
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, "${SERVER_DATA_ROOT}/prediction_travel_mode/large_city_v2")
from validate_chicago_od_transfer import features, fit, score, TARGETS

observed_path = Path("${SERVER_HOME}/public_mobility_validation_data_v1/chicago_taxi/chicago_taxi_2017_01_sample.csv")
simulated_path = Path("${SERVER_DATA_ROOT}/prediction_travel_mode/generate-od-pubtools-main/generate-od-pubtools-main/assets/example_data/my_test/result/od_flows_wgs84_high_income_2017/7532_od_flows_wgs84.parquet")
out = Path("${SERVER_DATA_ROOT}/prediction_travel_mode/validation_incremental_v2/chicago_od_2017_same_year")
out.mkdir(parents=True, exist_ok=True)
columns = ["pickup_centroid_latitude", "pickup_centroid_longitude", "dropoff_centroid_latitude", "dropoff_centroid_longitude"]
obs = features(pd.read_csv(observed_path, usecols=columns), True, 0.05)
sim = features(pd.read_parquet(simulated_path), False, 0.05)
calibrated, rules = fit(obs, sim, 100, 1.0, 0.001, 100.0, calibration_targets=["origin_grid", "destination_grid", "distance_bin"])
rows = [score(obs, calibrated, target, "adjusted_weight", "calibrated", 2017) for target in TARGETS[:3]]
pd.DataFrame(rows).to_csv(out / "chicago_od_2017_same_year_metrics.csv", index=False)
(out / "chicago_od_2017_same_year_model.json").write_text(json.dumps({"year": 2017, "grid_degrees": 0.05, "bounds": [0.001, 100.0], "iterations": 100, "damping": 1.0, "targets": TARGETS[:3], "rules": rules}, indent=2))
print(pd.DataFrame(rows).to_string(index=False))
