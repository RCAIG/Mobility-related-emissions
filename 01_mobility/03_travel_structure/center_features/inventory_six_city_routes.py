from pathlib import Path

import pyarrow.parquet as pq


CITIES = {
    1262: "high_income",
    2335: "upper_middle",
    1186: "high_income",
    1301: "lower_middle",
    906: "high_income",
    353: "high_income",
}


def route_root(income: str, year: int) -> Path:
    if income == "high_income":
        return Path(f"${SERVER_HOME}/result/trace_genetare_hgih_income_{year}")
    return Path(f"${SERVER_HOME}/result/trace_genetare_{income}_{year}")


for city_id, income in CITIES.items():
    for year in (2017, 2024):
        files = sorted(route_root(income, year).glob(f"od_paths_flows_{city_id}_*.parquet"))
        rows = [pq.ParquetFile(path).metadata.num_rows for path in files]
        sizes = [path.stat().st_size for path in files]
        print(
            city_id,
            year,
            f"chunks={len(files)}",
            f"rows={sum(rows)}",
            f"min_rows={min(rows) if rows else 0}",
            f"max_rows={max(rows) if rows else 0}",
            f"gb={sum(sizes) / 1024**3:.2f}",
        )
