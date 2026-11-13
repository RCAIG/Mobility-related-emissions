from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import infer_six_cities_v5 as base


N = 5000
SOURCE = Path(base.INPUT_PARQUET_DIR) / f"od_paths_flows_{base.CITY_ID}_chunk_0000.parquet"
TEMP = SOURCE.parent / f"od_paths_flows_{base.CITY_ID}_chunk_9999.parquet"
EXACT = Path(base.OUTPUT_PRED_DIR) / f"{base.CITY_ID}_chunk_9999_predictions.parquet"
FAST = Path(base.OUTPUT_PRED_DIR) / f"{base.CITY_ID}_chunk_0000_predictions.parquet"


def main():
    table = pq.read_table(SOURCE).slice(0, N)
    pq.write_table(table, TEMP, compression="snappy")
    try:
        base.init_worker()
        base.process_city_file(str(TEMP), {}, {})
        exact = pd.read_parquet(EXACT).iloc[:N]
        fast = pd.read_parquet(FAST).iloc[:N]
        exact_probs = exact[[f"prob_mode_{i}" for i in range(1, 6)]].to_numpy()
        fast_probs = fast[[f"prob_mode_{i}" for i in range(1, 6)]].to_numpy()
        agreement = np.mean(exact.pred_label.to_numpy() == fast.pred_label.to_numpy())
        mean_abs = np.mean(np.abs(exact_probs - fast_probs))
        print(f"n={N}")
        print(f"label_agreement={agreement:.4f}")
        print(f"mean_absolute_probability_difference={mean_abs:.4f}")
        print("exact_share=", np.bincount(exact.pred_label.clip(lower=0), minlength=5) / N)
        print("fast_share=", np.bincount(fast.pred_label.clip(lower=0), minlength=5) / N)
    finally:
        TEMP.unlink(missing_ok=True)
        EXACT.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
