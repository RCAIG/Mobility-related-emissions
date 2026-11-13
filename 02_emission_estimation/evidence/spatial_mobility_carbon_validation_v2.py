from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EPS = 1e-12
FINE_COLS = [f"pkm_b{i:02d}" for i in range(13)]
FINE_BOUNDS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 10), (10, 15),
               (15, 20), (20, 30), (30, 40), (40, 60), (60, 100), (100, np.inf)]
CLASS_COLS = [f"activity_c{i}" for i in range(1, 6)]
DEV_LEVELS = ["Low income", "Lower Middle", "Upper Middle", "High income"]
REGION_LEVELS = [
    "Eastern and South-Eastern Asia", "Central and Southern Asia", "Sub-Saharan Africa",
    "Europe", "Latin America and the Caribbean", "Northern Africa and Western Asia",
    "Northern America", "Australia and New Zealand", "Oceania",
]


def to_five_classes(frame: pd.DataFrame) -> np.ndarray:
    fine = frame[FINE_COLS].to_numpy(float)
    out = np.zeros((len(frame), 5), dtype=float)
    cuts = [2, 5, 15, 40]
    for i, (lo, hi) in enumerate(FINE_BOUNDS):
        midpoint = (lo + hi) / 2 if np.isfinite(hi) else 150
        out[:, np.searchsorted(cuts, midpoint, side="right")] += fine[:, i]
    return out


def read_city_cells(cache_root: Path) -> pd.DataFrame:
    frames = []
    for year in [2017, 2024]:
        paths = glob.glob(str(cache_root / str(year) / "city_*.parquet"))
        table = ds.dataset(paths, format="parquet").to_table(
            columns=["cell_id", "year", "city_id"] + FINE_COLS
        )
        frames.append(table.to_pandas())
    d = pd.concat(frames, ignore_index=True)
    d = d.groupby(["cell_id", "year", "city_id"], as_index=False)[FINE_COLS].sum()
    classes = to_five_classes(d)
    for i, col in enumerate(CLASS_COLS):
        d[col] = classes[:, i]
    d["activity_total"] = classes.sum(axis=1)
    return d[d.activity_total.gt(0)].copy()


def prepare(cache_root: Path, reference_path: Path, city_context_path: Path, out_path: Path):
    city_cells = read_city_cells(cache_root)
    context = pd.read_csv(city_context_path)
    keep = ["ID_UC_G0", "year", "log_population", "log_area", "log_gdp",
            "development_group", "urban_type", "Global_NS"]
    context = context[keep].drop_duplicates(["ID_UC_G0", "year"])
    city_cells = city_cells.merge(
        context, left_on=["city_id", "year"], right_on=["ID_UC_G0", "year"], how="left"
    )
    reference = pd.read_parquet(reference_path, columns=["cell_id", "year", "reference_emission", "lon", "lat"])
    city_cells = city_cells.merge(reference, on=["cell_id", "year"], how="left")
    city_cells["reference_emission"] = city_cells.reference_emission.fillna(0).clip(lower=0)

    city_group = city_cells.groupby(["city_id", "year"])
    city_cells["city_n_cells"] = city_group.cell_id.transform("nunique")
    city_cells["city_activity_total"] = city_group.activity_total.transform("sum")
    city_cells["city_reference_total"] = city_group.reference_emission.transform("sum")
    city_cells["activity_share_city"] = city_cells.activity_total / np.maximum(city_cells.city_activity_total, EPS)
    city_cells["reference_share_city"] = city_cells.reference_emission / np.maximum(city_cells.city_reference_total, EPS)
    for col in CLASS_COLS:
        denom = city_group[col].transform("sum")
        city_cells[f"{col}_share_city"] = city_cells[col] / np.maximum(denom, EPS)

    weight = city_cells.activity_total.to_numpy(float)
    city_cells["weighted_log_population"] = weight * city_cells.log_population.fillna(city_cells.log_population.median())
    city_cells["weighted_log_area"] = weight * city_cells.log_area.fillna(city_cells.log_area.median())
    city_cells["weighted_log_gdp"] = weight * city_cells.log_gdp.fillna(city_cells.log_gdp.median())
    for level in DEV_LEVELS:
        city_cells[f"dev_{level}"] = weight * city_cells.development_group.eq(level).to_numpy()
    for level in REGION_LEVELS:
        city_cells[f"region_{level}"] = weight * city_cells.urban_type.eq(level).to_numpy()
    city_cells["weight_sq"] = weight ** 2

    sum_cols = CLASS_COLS + ["activity_total", "weighted_log_population", "weighted_log_area",
                             "weighted_log_gdp", "weight_sq"] + [f"dev_{x}" for x in DEV_LEVELS] + [f"region_{x}" for x in REGION_LEVELS]
    global_cells = city_cells.groupby(["cell_id", "year"], as_index=False)[sum_cols].sum()
    counts = city_cells.groupby(["cell_id", "year"]).city_id.nunique().rename("contributing_cities").reset_index()
    global_cells = global_cells.merge(counts, on=["cell_id", "year"])
    global_cells = global_cells.merge(reference, on=["cell_id", "year"], how="left")
    global_cells.reference_emission = global_cells.reference_emission.fillna(0).clip(lower=0)
    for col in ["weighted_log_population", "weighted_log_area", "weighted_log_gdp"]:
        global_cells[col] = global_cells[col] / np.maximum(global_cells.activity_total, EPS)
    for col in [f"dev_{x}" for x in DEV_LEVELS] + [f"region_{x}" for x in REGION_LEVELS]:
        global_cells[col] = global_cells[col] / np.maximum(global_cells.activity_total, EPS)
    global_cells["city_contribution_hhi"] = global_cells.weight_sq / np.maximum(global_cells.activity_total ** 2, EPS)
    global_cells = global_cells.drop(columns="weight_sq")
    global_cells["dominant_development"] = global_cells[[f"dev_{x}" for x in DEV_LEVELS]].idxmax(axis=1).str.replace("dev_", "", regex=False)
    global_cells["dominant_region"] = global_cells[[f"region_{x}" for x in REGION_LEVELS]].idxmax(axis=1).str.replace("region_", "", regex=False)
    global_cells.to_parquet(out_path, index=False)
    city_cells.to_parquet(out_path.with_name("city_cell_activity_reference_v2.parquet"), index=False)
    print(json.dumps({"global_rows": len(global_cells), "city_cell_rows": len(city_cells),
                      "cities": int(city_cells.city_id.nunique()),
                      "city_cells_median": float(city_cells.groupby(["city_id", "year"]).size().median()),
                      "cities_ge3_cells": int((city_cells.groupby(["city_id", "year"]).size() >= 3).sum())}), flush=True)


def engineer_global(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    for col in CLASS_COLS:
        x[f"log_{col}"] = np.log1p(x[col].clip(lower=0))
        x[f"share_{col}"] = x[col] / np.maximum(x.activity_total, EPS)
    x["log_activity_total"] = np.log1p(x.activity_total)
    x["log_contributing_cities"] = np.log1p(x.contributing_cities)
    x["sin_lon"] = np.sin(np.radians(x.lon)); x["cos_lon"] = np.cos(np.radians(x.lon))
    x["sin_lat"] = np.sin(np.radians(x.lat)); x["cos_lat"] = np.cos(np.radians(x.lat))
    x["year_2024"] = x.year.eq(2024).astype(float)
    x["target"] = np.log1p(x.reference_emission.clip(lower=0))
    return x.replace([np.inf, -np.inf], np.nan)


MOBILITY = ([f"log_{x}" for x in CLASS_COLS] + [f"share_{x}" for x in CLASS_COLS]
            + ["log_activity_total", "city_contribution_hhi", "log_contributing_cities"])
CONTEXT = (["weighted_log_population", "weighted_log_area", "weighted_log_gdp",
            "sin_lon", "cos_lon", "sin_lat", "cos_lat", "year_2024"]
           + [f"dev_{x}" for x in DEV_LEVELS] + [f"region_{x}" for x in REGION_LEVELS])
FEATURE_SETS = {
    "Mobility only": MOBILITY,
    "City context only": CONTEXT,
    "Mobility + city context": MOBILITY + CONTEXT,
}


def model(name: str, seed: int):
    if name == "HGBR":
        estimator = HistGradientBoostingRegressor(
            learning_rate=.05, max_iter=220, max_leaf_nodes=28,
            min_samples_leaf=35, l2_regularization=1.2, random_state=seed,
        )
    elif name == "ExtraTrees":
        estimator = ExtraTreesRegressor(
            n_estimators=350, min_samples_leaf=5, max_features=.85,
            n_jobs=-1, random_state=seed,
        )
    else:
        raise ValueError(name)
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", estimator)])


def metrics(y, pred):
    return {"r2_log": float(r2_score(y, pred)),
            "rmse_log": float(np.sqrt(mean_squared_error(y, pred))),
            "mae_log": float(mean_absolute_error(y, pred)),
            "spearman": float(spearmanr(y, pred).statistic)}


def split_indices(d: pd.DataFrame, split_name: str, seed: int):
    if split_name == "Random held-out cells":
        groups = d.cell_id
    elif split_name == "Held-out 5-degree blocks":
        groups = (np.floor((d.lon + 180) / 5).astype(int) + 100 * np.floor((d.lat + 90) / 5).astype(int))
    else:
        raise ValueError(split_name)
    return next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(d, groups=groups))


def fit_tabpfn(train_x, train_y, test_x, seed, gpu):
    from tabpfn import TabPFNRegressor
    prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    x = prep.fit_transform(train_x).astype(np.float32)
    z = prep.transform(test_x).astype(np.float32)
    if len(x) > 12000:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(x), 12000, replace=False)
        x, train_y = x[keep], np.asarray(train_y)[keep]
    reg = TabPFNRegressor(n_estimators=4, device=gpu, ignore_pretraining_limits=True,
                          fit_mode="fit_preprocessors", random_state=seed)
    reg.fit(x, np.asarray(train_y, np.float32))
    return reg.predict(z)


def fit_stratified(train, test, features, seed):
    global_model = model("HGBR", seed).fit(train[features], train.target)
    pred = global_model.predict(test[features])
    pop_cuts = train.weighted_log_population.quantile([0, 1/3, 2/3, 1]).to_numpy()
    pop_cuts[0], pop_cuts[-1] = -np.inf, np.inf
    tr_size = pd.cut(train.weighted_log_population, pop_cuts, labels=False, include_lowest=True)
    te_size = pd.cut(test.weighted_log_population, pop_cuts, labels=False, include_lowest=True)
    tr_key = train.dominant_development.astype(str) + "|" + tr_size.astype(str)
    te_key = test.dominant_development.astype(str) + "|" + te_size.astype(str)
    for key in tr_key.unique():
        a, b = tr_key.eq(key), te_key.eq(key)
        if a.sum() < 400 or b.sum() == 0:
            continue
        local = model("HGBR", seed).fit(train.loc[a, features], train.loc[a, "target"])
        pred[b.to_numpy()] = local.predict(test.loc[b, features])
    return pred


def validate_global(table_path: Path, out_dir: Path, pilot: bool, gpu: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    d = engineer_global(pd.read_parquet(table_path))
    seeds = [42] if pilot else [17, 42, 83]
    models = ["HGBR", "TabPFN"]
    rows, preds = [], []
    for split_name in ["Random held-out cells", "Held-out 5-degree blocks"]:
        for seed in seeds:
            tr, te = split_indices(d, split_name, seed)
            train, test = d.iloc[tr].copy(), d.iloc[te].copy()
            for feature_name, features in FEATURE_SETS.items():
                for model_name in models:
                    if model_name == "Stratified HGBR" and feature_name != "Mobility + city context":
                        continue
                    if model_name == "HGBR":
                        reg = model("HGBR", seed).fit(train[features], train.target)
                        pred = reg.predict(test[features])
                    elif model_name == "TabPFN":
                        pred = fit_tabpfn(train[features], train.target, test[features], seed, gpu)
                    else:
                        pred = fit_stratified(train, test, features, seed)
                    row = metrics(test.target, pred)
                    row.update({"validation": split_name, "seed": seed, "model": model_name,
                                "feature_set": feature_name, "n_train": len(train), "n_test": len(test)})
                    rows.append(row); print(json.dumps(row), flush=True)
                    p = test[["cell_id", "year", "lon", "lat", "reference_emission", "dominant_development", "dominant_region"]].copy()
                    p["observed_log"], p["predicted_log"] = test.target.to_numpy(), pred
                    p["validation"], p["seed"], p["model"], p["feature_set"] = split_name, seed, model_name, feature_name
                    preds.append(p)
    pd.DataFrame(rows).to_csv(out_dir / "global_validation_metrics.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(out_dir / "global_heldout_predictions.parquet", index=False)


def validate_temporal(table_path: Path, out_dir: Path):
    d = engineer_global(pd.read_parquet(table_path))
    a = d[d.year.eq(2017)].set_index("cell_id"); b = d[d.year.eq(2024)].set_index("cell_id")
    common = a.index.intersection(b.index)
    old, new = a.loc[common].copy(), b.loc[common].copy()
    t = pd.DataFrame(index=common)
    t["target"] = new.target; t["previous_emission"] = old.target
    t["lon"], t["lat"] = new.lon, new.lat
    for col in MOBILITY + CONTEXT:
        t[f"new_{col}"] = new[col]
        t[f"delta_{col}"] = new[col] - old[col]
    rows = []
    block = np.floor((t.lon + 180) / 5).astype(int) + 100 * np.floor((t.lat + 90) / 5).astype(int)
    for seed in [17, 42, 83]:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(t, groups=block))
        train, test = t.iloc[tr], t.iloc[te]
        sets = {
            "Previous inventory only": ["previous_emission"],
            "Previous inventory + mobility change": ["previous_emission"] + [f"new_{x}" for x in MOBILITY] + [f"delta_{x}" for x in MOBILITY],
            "Previous inventory + full update": ["previous_emission"] + [f"new_{x}" for x in MOBILITY + CONTEXT] + [f"delta_{x}" for x in MOBILITY + CONTEXT],
        }
        for label, features in sets.items():
            reg = model("HGBR", seed).fit(train[features], train.target)
            pred = reg.predict(test[features])
            row = metrics(test.target, pred); row.update({"validation": "2017 to 2024 spatial update", "seed": seed, "feature_set": label, "model": "HGBR", "n_train": len(train), "n_test": len(test)})
            rows.append(row); print(json.dumps(row), flush=True)
    pd.DataFrame(rows).to_csv(out_dir / "temporal_update_metrics.csv", index=False)


def validate_allocation(city_table_path: Path, out_dir: Path):
    d = pd.read_parquet(city_table_path)
    d = d[(d.city_n_cells >= 3) & (d.city_reference_total > 0)].copy()
    features = (["activity_share_city", "city_n_cells", "log_population", "log_area", "log_gdp"]
                + [f"{x}_share_city" for x in CLASS_COLS])
    d["target_share_log"] = np.log(d.reference_share_city.clip(lower=0) + 1e-6)
    rows, prediction_rows = [], []
    for seed in [17, 42, 83]:
        group = d.city_id.astype(str) + "_" + d.year.astype(str)
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(d, groups=group))
        train, test = d.iloc[tr].copy(), d.iloc[te].copy()
        learned = model("HGBR", seed).fit(train[features], train.target_share_log)
        score = learned.predict(test[features])
        test["learned_weight"] = np.exp(score)
        test["learned_share"] = test.learned_weight / np.maximum(test.groupby(["city_id", "year"]).learned_weight.transform("sum"), EPS)
        methods = {
            "Mobility-proportional allocation": test.activity_share_city.to_numpy(),
            "Learned mobility allocation": test.learned_share.to_numpy(),
        }
        for label, share in methods.items():
            pred_emission = share * test.city_reference_total.to_numpy()
            row = metrics(np.log1p(test.reference_emission), np.log1p(np.maximum(pred_emission, 0)))
            row.update({"validation": "Held-out-city conditional allocation", "seed": seed,
                        "feature_set": label, "model": "Deterministic" if label.startswith("Mobility-proportional") else "HGBR",
                        "n_train": len(train), "n_test": len(test), "heldout_cities": int(test.city_id.nunique())})
            rows.append(row); print(json.dumps(row), flush=True)
            p = test[["city_id", "year", "cell_id", "reference_emission", "city_reference_total", "city_n_cells"]].copy()
            p["predicted_emission"], p["method"], p["seed"] = pred_emission, label, seed
            prediction_rows.append(p)
    pd.DataFrame(rows).to_csv(out_dir / "conditional_allocation_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(out_dir / "conditional_allocation_predictions.parquet", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--city-context", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--gpu", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    global_path = args.out / "global_cell_features_v2.parquet"
    city_path = args.out / "city_cell_activity_reference_v2.parquet"
    if args.prepare:
        prepare(args.cache_root, args.reference, args.city_context, global_path)
    validate_global(global_path, args.out, args.pilot, args.gpu)
    if not args.pilot:
        validate_temporal(global_path, args.out)
        validate_allocation(city_path, args.out)


if __name__ == "__main__":
    main()
