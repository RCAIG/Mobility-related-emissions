from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


CODE_DIR = Path("${SERVER_DATA_ROOT}/prediction_travel_mode/model/code")
CITIES = {
    1262: ("Calgary", "North America", "high_income"),
    2335: ("Cordoba", "South America", "upper_middle"),
    1186: ("Cologne", "Europe", "high_income"),
    1301: ("Lubango", "Africa", "lower_middle"),
    906: ("Taichung", "Asia", "high_income"),
    353: ("Christchurch", "Oceania", "high_income"),
}
RUNS = {
    "v5_center_matched_with_center": 5,
    "v4_center_matched_with_center": 4,
}


def source_dir(income: str, year: int) -> Path:
    if income == "high_income":
        return Path(f"${SERVER_HOME}/result/trace_genetare_hgih_income_{year}")
    return Path(f"${SERVER_HOME}/result/trace_genetare_{income}_{year}")


for run_name, classes in RUNS.items():
    run_dir = CODE_DIR / "mode5_experiments" / run_name
    records = []
    for city_id, (city, region, income) in CITIES.items():
        for year in (2017, 2024):
            weighted_counts = np.zeros(classes, dtype=np.float64)
            sampled_rows = 0
            source_rows = 0
            for prediction_path in sorted(
                (run_dir / "sampled_city_inference" / f"{city_id}_{year}").glob("*.parquet")
            ):
                match = re.search(r"_chunk_(\d+)_predictions", prediction_path.name)
                if not match:
                    continue
                chunk = match.group(1)
                route_path = source_dir(income, year) / f"od_paths_flows_{city_id}_chunk_{chunk}.parquet"
                total = pq.ParquetFile(route_path).metadata.num_rows
                predictions = pd.read_parquet(prediction_path)
                predictions = predictions[predictions["pred_label"] >= 0].copy()
                inclusion_probability = min(1.0, len(predictions) / total)
                weights = predictions["flow"].to_numpy(dtype=float) / inclusion_probability
                for label in range(classes):
                    weighted_counts[label] += weights[predictions["pred_label"].to_numpy() == label].sum()
                sampled_rows += len(predictions)
                source_rows += total
            shares = weighted_counts / weighted_counts.sum()
            positive = shares[shares > 0]
            entropy = float(-(positive * np.log(positive)).sum() / math.log(classes))
            record = {
                "run_name": run_name,
                "num_classes": classes,
                "city_id": city_id,
                "city": city,
                "region": region,
                "year": year,
                "sampled_rows": sampled_rows,
                "source_rows": source_rows,
                "normalized_entropy": entropy,
                "minimum_class_share": float(shares.min()),
            }
            record.update({f"mode_{index + 1}_share": value for index, value in enumerate(shares)})
            records.append(record)
    frame = pd.DataFrame(records)
    output = run_dir / f"sampled_mode_shares_{classes}class.csv"
    frame.to_csv(output, index=False)
    ranking = (
        frame.groupby(["city_id", "city", "region"], as_index=False)
        .agg(
            mean_entropy=("normalized_entropy", "mean"),
            worst_year_entropy=("normalized_entropy", "min"),
            minimum_class_share=("minimum_class_share", "min"),
        )
        .sort_values(["mean_entropy", "worst_year_entropy"], ascending=False)
    )
    ranking.to_csv(run_dir / f"sampled_city_balance_ranking_{classes}class.csv", index=False)
    print(f"\n{run_name}")
    print(ranking.to_string(index=False))
    print(frame.to_string(index=False))
