"""External validation for mobility-emission estimates.

This script never trains or tunes an emission model.  It compares frozen,
out-of-fold city predictions with independently published inventories and
records every boundary, sector and time limitation alongside the statistics.

The first implemented adapter is Carbon Monitor Cities.  Carbon Monitor starts
in 2019 whereas the current mobility model has 2017 and 2024 snapshots, so
the adapter intentionally reports a 2024 cross-sectional comparison only.  It
does not relabel a 2019--2024 external change as a 2017--2024 validation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(os.environ.get(
    "EXTERNAL_VALIDATION_ROOT",
    r"${TRANSPORT_WORKSPACE}\\Emission_Estimation_Validation_Study_V3",
))
WORK = ROOT / "Experiments" / "external_validation_v1"
INPUT = WORK / "inputs"
OUT = WORK / "output"

# Explicit aliases only. Ambiguous city names are deliberately not fuzzy-matched.
CITY_ALIASES = {
    "Bogota": "Bogota",
    "Mexico City": "Mexico City",
    "New York": "New York",
    "Los Angeles": "Los Angeles",
    "San Francisco": "San Francisco",
    "Washington": "Washington",
    "Cape Town": "Cape Town",
    "New Delhi": "New Delhi",
}


def norm_city(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("-", " ").replace(".", "")
    return " ".join(text.split())


def metric(frame: pd.DataFrame) -> dict:
    y = frame["carbon_monitor_2024_mt"].to_numpy(float)
    p = frame["model_prediction_2024"].to_numpy(float)
    ly, lp = np.log1p(y), np.log1p(p)
    # A fitted scale factor is descriptive only: it is not predictive accuracy.
    calibration = LinearRegression().fit(lp.reshape(-1, 1), ly)
    calibrated = calibration.predict(lp.reshape(-1, 1))
    return {
        "n_matched_cities": int(len(frame)),
        "spearman_raw_totals": float(spearmanr(y, p).statistic),
        "pearson_log_totals": float(np.corrcoef(ly, lp)[0, 1]),
        "descriptive_log_scale_calibration_r2": float(r2_score(ly, calibrated)),
        "descriptive_log_scale_calibration_mae": float(mean_absolute_error(ly, calibrated)),
        "descriptive_log_scale_calibration_rmse": float(np.sqrt(mean_squared_error(ly, calibrated))),
        "calibration_intercept": float(calibration.intercept_),
        "calibration_slope": float(calibration.coef_[0]),
    }


def carbon_monitor_city_2024() -> tuple[pd.DataFrame, dict]:
    cm = pd.read_csv(INPUT / "carbon_monitor_cities.csv")
    cm["date"] = pd.to_datetime(cm["date"], dayfirst=True)
    cm = cm.loc[(cm["sector"] == "Ground Transport") & (cm["date"].dt.year == 2024)].copy()
    cm = cm.groupby("city", as_index=False)["value"].sum().rename(columns={"value": "carbon_monitor_2024_mt"})
    cm["city_key"] = cm["city"].map(norm_city)

    pred = pd.read_parquet(INPUT / "city_oof_predictions.parquet")
    pred = pred.loc[(pred["year"] == 2024) & (pred["feature_set"] == "Full model")].copy()
    cities = pd.read_csv(INPUT / "city_year_validation_dataset.csv", usecols=["ID_UC_G0", "GC_UCN_MAI_2025", "year"])
    cities = cities.loc[cities["year"] == 2024].drop_duplicates("ID_UC_G0")
    pred = pred.merge(cities[["ID_UC_G0", "GC_UCN_MAI_2025"]], on="ID_UC_G0", how="left", validate="one_to_one")
    pred["city_key"] = pred["GC_UCN_MAI_2025"].map(norm_city)
    pred["model_prediction_2024"] = np.exp(pred["predicted_log"])

    # Match exact normalized names only, then inspect aliases explicitly.
    match = pred.merge(cm[["city", "city_key", "carbon_monitor_2024_mt"]], on="city_key", how="inner", validate="many_to_one")
    match = match[["ID_UC_G0", "GC_UCN_MAI_2025", "city", "model_prediction_2024", "carbon_monitor_2024_mt"]]
    match = match.sort_values("city")
    metadata = {
        "external_source": "Carbon Monitor Cities",
        "external_sector": "Ground Transport",
        "external_year": 2024,
        "model_output": "Frozen out-of-fold Full model prediction for 2024",
        "comparison_type": "Cross-sectional city-total ordering and descriptive scale calibration",
        "not_a_time_change_validation": True,
        "reason": "Carbon Monitor Cities begins in 2019; the current global mobility model has only 2017 and 2024 snapshots.",
        "boundary_warning": "Carbon Monitor city boundaries and inventory accounting may differ from GHSL urban-centre boundaries and the EDGAR-labelled training target.",
        **metric(match),
    }
    return match, metadata


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matched, metadata = carbon_monitor_city_2024()
    matched.to_csv(OUT / "carbon_monitor_cities_2024_matched_predictions.csv", index=False)
    (OUT / "carbon_monitor_cities_2024_metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    registry = pd.DataFrame([
        ["CAMS-GLOB-ANT", "global grid", "not retained", "0.1 degree", "n/a", "excluded", "Not used: it is EDGAR-related and would not provide independent transport-sector validation."],
        ["Carbon Monitor Cities", "city", "Ground Transport", "city", "2024", "complete cross-section", "2019 onward only; cannot validate the 2017-2024 model change without a 2019 model run."],
        ["GRACED", "global grid", "not retained", "0.1 degree", "n/a", "excluded", "Not used: authoritative retrieval is restricted and this study retains currently available transport-specific benchmarks."],
        ["preACO2 RFR-SR01", "global grid", "total anthropogenic CO2", "0.1 degree", "2021/2022", "downloaded; excluded", "Not used because it is not a transport-only inventory and is explicitly EDGAR-informed."],
        ["Hestia Phoenix-Mesa", "urban grid", "separable on-road FFCO2", "100 m", "2017/2022", "downloaded; adapter in progress", "Downloaded from Zenodo record 18340784. Compare the ONR band only; values are tC per grid cell per year and require conversion to tCO2 only when absolute units are reported."],
        ["Hestia SW-IFL Arizona", "road segment", "on-road FFCO2", "road segment", "2017-2022", "downloaded; adapter in progress", "Downloaded from ESS-DIVE DOI 10.15485/2335384. Local and non-local road layers are EPSG:6404; FFCO2tC requires a 44/12 conversion for tCO2."],
        ["DARTE", "United States road/grid", "on-road CO2", "road/grid", "matching year", "pending", "Requires acquisition of the published road/grid release and harmonization to road links or 1 km cells."],
        ["Vulcan", "selected US cities", "urban fossil CO2", "city/grid/road depending release", "matching year", "pending", "Use only releases with separable on-road transport and documented boundaries."],
        ["LAEI/AIRPARIF/Netherlands", "local city/region", "road transport", "road/grid", "matching year", "pending", "Manual source acquisition and boundary/units audit required."],
    ], columns=["source", "validation_level", "target_sector", "spatial_unit", "required_year", "status", "implementation_note"])
    registry.to_csv(OUT / "external_validation_registry.csv", index=False)


if __name__ == "__main__":
    main()
