import json
import re
from pathlib import Path

import pyarrow.parquet as pq


RUN_DIR = Path(
    "${SERVER_DATA_ROOT}/prediction_travel_mode/model/code/"
    "mode5_experiments/v5_alphaearth_only"
)

CITY_YEARS = (
    (1262, 2017, "high_income"),
    (1262, 2024, "high_income"),
    (2335, 2017, "upper_middle"),
    (2335, 2024, "upper_middle"),
    (1186, 2017, "high_income"),
    (1186, 2024, "high_income"),
    (1301, 2017, "lower_middle"),
    (1301, 2024, "lower_middle"),
    (906, 2017, "high_income"),
    (906, 2024, "high_income"),
    (353, 2017, "high_income"),
    (353, 2024, "high_income"),
)


def route_root(year: int, income: str) -> Path:
    if income == "high_income":
        return Path(f"${SERVER_HOME}/result/trace_genetare_hgih_income_{year}")
    return Path(f"${SERVER_HOME}/result/trace_genetare_{income}_{year}")


def chunk_id(path: Path) -> str:
    match = re.search(r"_chunk_(\d+)\.parquet$", path.name)
    if not match:
        raise ValueError(path)
    return match.group(1)


def main() -> None:
    records = []
    all_valid = True
    for city_id, year, income in CITY_YEARS:
        routes = sorted(route_root(year, income).glob(f"od_paths_flows_{city_id}_chunk_*.parquet"))
        pred_dir = RUN_DIR / "six_city_inference" / f"{city_id}_{year}"
        input_rows = 0
        prediction_rows = 0
        chunks = []
        for route in routes:
            chunk = chunk_id(route)
            prediction = pred_dir / f"{city_id}_chunk_{chunk}_predictions.parquet"
            route_rows = pq.ParquetFile(route).metadata.num_rows
            pred_rows = pq.ParquetFile(prediction).metadata.num_rows if prediction.exists() else None
            valid = pred_rows == route_rows
            all_valid &= valid
            input_rows += route_rows
            prediction_rows += pred_rows or 0
            chunks.append(
                {
                    "chunk": chunk,
                    "input_rows": route_rows,
                    "prediction_rows": pred_rows,
                    "valid": valid,
                }
            )
        records.append(
            {
                "city_id": city_id,
                "year": year,
                "input_rows": input_rows,
                "prediction_rows": prediction_rows,
                "chunk_count": len(routes),
                "valid": bool(routes) and all(item["valid"] for item in chunks),
                "chunks": chunks,
            }
        )
    result = {
        "all_valid": all_valid and all(record["valid"] for record in records),
        "total_input_rows": sum(record["input_rows"] for record in records),
        "total_prediction_rows": sum(record["prediction_rows"] for record in records),
        "city_years": records,
    }
    destination = RUN_DIR / "six_city_inference_audit_v5.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_valid"] else 1)


if __name__ == "__main__":
    main()
