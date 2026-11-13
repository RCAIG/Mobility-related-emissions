"""Year-specific three-block ablation for city and 0.1-degree emission models."""
from pathlib import Path
import argparse
import gc
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from tabpfn import TabPFNRegressor

from tabpfn_emission_optimization import WEIGHT, predict_batches
from mobility_distance_bin_audit import FINE, ROAD, mobility_features


ROOT = Path("${SERVER_HOME}/result")
OUT = Path("${SERVER_HOME}/result/emission_feature_ablation_year_v1")
SEED = 20261006
MIN_GRID_REFERENCE = 100.0
SIZE = ["log_population", "log_area", "log_gdp"]
TECHNICAL = [
    "logistics_index", "vehicle_co2_intensity", "log_bev_ownership",
    "public_transport_efficiency", "train_service_efficiency", "transport_tax",
    "rise_index", "oil_share", "electricity_share",
]
SETS = {
    "Scale only": ["Scale"],
    "Mobility only": ["Mobility"],
    "Socioeconomic only": ["Socioeconomic"],
    "Scale + mobility": ["Scale", "Mobility"],
    "Scale + socioeconomic": ["Scale", "Socioeconomic"],
    "Mobility + socioeconomic": ["Mobility", "Socioeconomic"],
    "Full features": ["Scale", "Mobility", "Socioeconomic"],
}


def hgb(seed):
    return HistGradientBoostingRegressor(
        max_iter=500, max_leaf_nodes=31, min_samples_leaf=40,
        learning_rate=.04, l2_regularization=15, random_state=seed,
    )


def clean(values):
    values = np.asarray(values, dtype=np.float32)
    values[~np.isfinite(values)] = np.nan
    return values


def city_inputs(year):
    path = ROOT / "city_mobility_carbon_validation_v2/input/city_year_validation_dataset.csv"
    d = pd.read_csv(path, low_memory=False)
    d = d[d.year.eq(year)].dropna(subset=["log_transport_ce"] + SIZE).reset_index(drop=True)
    categories = pd.get_dummies(
        d[["development_group", "urban_type", "Global_NS"]].astype(str), dtype=float,
    )
    scale = np.column_stack([d[SIZE], categories])
    values = d[[f"mode_{i}_km" for i in range(5)]].to_numpy(float)
    total = np.nansum(values, axis=1)
    mobility = np.column_stack([
        mobility_features(values, expanded=True), d[ROAD],
        np.log1p(total / np.maximum(d.population.to_numpy(float), 1)),
        np.log1p(total / np.maximum(d.area.to_numpy(float), 1e-3)),
    ])
    blocks = {
        "Scale": clean(scale),
        "Mobility": clean(mobility),
        "Socioeconomic": clean(d[TECHNICAL]),
    }
    raw = d.transport_ce.to_numpy(float)
    y = d.log_transport_ce.to_numpy(np.float32)
    assert np.all(raw > 0) and np.allclose(y, np.log(raw), atol=2e-5)
    names = {
        "Scale": SIZE + categories.columns.tolist(),
        "Mobility": ["five-class activity transformations"] + ROAD
                    + ["log activity per capita", "log activity per area"],
        "Socioeconomic": TECHNICAL,
    }
    return d, blocks, y, raw, d.ID_UC_G0.to_numpy(), None, names


def technical_grid_features(data):
    cells = pd.read_parquet(
        ROOT / "spatial_mobility_carbon_validation_v2/analysis/city_cell_activity_reference_v2.parquet",
        columns=["cell_id", "year", "city_id", "activity_total"],
    )
    city = pd.read_csv(
        ROOT / "city_mobility_carbon_validation_v2/input/city_year_validation_dataset.csv",
        low_memory=False,
    )[["ID_UC_G0", "year"] + TECHNICAL].drop_duplicates(["ID_UC_G0", "year"])
    cells = cells.merge(
        city, left_on=["city_id", "year"], right_on=["ID_UC_G0", "year"],
        how="left", validate="many_to_one",
    )
    weight = cells.activity_total.clip(lower=0).fillna(0)
    total = cells.assign(_weight=weight).groupby(["cell_id", "year"])["_weight"].sum()
    out = pd.DataFrame(index=total.index)
    names = []
    for col in TECHNICAL:
        value = pd.to_numeric(cells[col], errors="coerce")
        numerator = (value.fillna(0) * weight).groupby([cells.cell_id, cells.year]).sum()
        valid_weight = (weight * value.notna()).groupby([cells.cell_id, cells.year]).sum()
        out[f"tech_{col}"] = numerator / valid_weight.replace(0, np.nan)
        out[f"coverage_{col}"] = valid_weight / total.replace(0, np.nan)
        names.extend([f"tech_{col}", f"coverage_{col}"])
    aligned = data[["cell_id", "year"]].merge(
        out.reset_index(), on=["cell_id", "year"], how="left", validate="one_to_one",
    )
    return clean(aligned[names]), names


def grid_inputs(scope, year):
    d = pd.read_parquet(
        ROOT / "spatial_mobility_carbon_validation_v2/analysis/global_cell_features_v2.parquet"
    ).reset_index(drop=True)
    extra = pd.read_parquet(
        ROOT / "emission_structural_optimization_v2/target_free_mobility_features.parquet"
    )
    assert np.array_equal(d[["cell_id", "year"]].to_numpy(), extra[["cell_id", "year"]].to_numpy())
    for direction in ["lon", "lat"]:
        d[f"sin_{direction}"] = np.sin(np.radians(d[direction]))
        d[f"cos_{direction}"] = np.cos(np.radians(d[direction]))

    scale_names = [
        "weighted_log_population", "weighted_log_area", "weighted_log_gdp",
        "sin_lon", "cos_lon", "sin_lat", "cos_lat", "city_contribution_hhi",
        "contributing_cities",
    ] + [c for c in d if c.startswith(("dev_", "region_"))]
    fine = extra[FINE].to_numpy(float)
    share = fine / np.maximum(fine.sum(axis=1)[:, None], 1e-9)
    midpoint = np.array([.5, 1.5, 2.5, 4, 6.5, 9, 12.5, 17.5, 25, 35, 50, 80, 150])
    shape = np.column_stack([
        share[:, :4].sum(axis=1), share[:, 8:].sum(axis=1),
        share[:, 10:].sum(axis=1), share @ np.log1p(midpoint),
        (share ** 2).sum(axis=1), np.cumsum(share, axis=1)[:, :-1],
    ])
    extra_names = [c for c in extra if c not in ["cell_id", "year", "total"] + FINE]
    mobility = np.column_stack([
        mobility_features(fine, expanded=True), shape, extra[extra_names],
    ])
    technical, technical_names = technical_grid_features(d)
    blocks = {
        "Scale": clean(d[scale_names]),
        "Mobility": clean(mobility),
        "Socioeconomic": technical,
    }
    raw = d.reference_emission.to_numpy(float)
    keep = d.year.eq(year).to_numpy() & (raw > MIN_GRID_REFERENCE)
    d = d.loc[keep].reset_index(drop=True)
    blocks = {key: values[keep] for key, values in blocks.items()}
    raw = raw[keep]
    y = np.log(raw).astype(np.float32)
    if scope == "grid_random":
        groups = d.cell_id.to_numpy()
    else:
        groups = (np.floor((d.lon + 180) / 5).astype(int)
                  + 100 * np.floor((d.lat + 90) / 5).astype(int)).to_numpy()
    region_names = [c for c in d if c.startswith("region_")]
    region_values = d[region_names].to_numpy(float)
    region = np.argmax(region_values, axis=1)
    region[region_values.sum(axis=1) <= 0] = -1
    names = {
        "Scale": scale_names,
        "Mobility": ["13-bin activity transformations", "distance-shape summaries"] + extra_names,
        "Socioeconomic": technical_names,
    }
    return d, blocks, y, raw, groups, region, names


def scores(y, raw, pred):
    pred_raw = np.exp(pred.astype(float))
    return {
        "r2_log": float(r2_score(y, pred)),
        "r2_raw": float(r2_score(raw, pred_raw)),
        "mae_log": float(mean_absolute_error(y, pred)),
        "rmse_log": float(np.sqrt(mean_squared_error(y, pred))),
        "mae_raw": float(mean_absolute_error(raw, pred_raw)),
        "spearman": float(spearmanr(y, pred).statistic),
    }


def run(scope, year, device):
    if scope == "city":
        d, blocks, y, raw, groups, region, feature_names = city_inputs(year)
        test_size = .25
    else:
        d, blocks, y, raw, groups, region, feature_names = grid_inputs(scope, year)
        test_size = .2
    out = OUT / f"{scope}_{year}" / str(SEED)
    out.mkdir(parents=True, exist_ok=True)

    pool, test = next(GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=SEED,
    ).split(y, groups=groups))
    inner, val = next(GroupShuffleSplit(
        n_splits=1, test_size=.15, random_state=731,
    ).split(y[pool], groups=groups[pool]))
    train = pool[inner]
    pd.DataFrame({
        "row": np.r_[train, val, test],
        "subset": ["train"] * len(train) + ["validation"] * len(val) + ["test"] * len(test),
        "group": groups[np.r_[train, val, test]],
    }).to_csv(out / "splits.csv", index=False)

    rows = []
    predictions = d.iloc[test][["ID_UC_G0", "year"] if scope == "city" else ["cell_id", "year", "lon", "lat"]].copy()
    predictions["observed_log"] = y[test]
    predictions["reference_raw"] = raw[test]
    rules = []
    for label, required in SETS.items():
        x = np.column_stack([blocks[name] for name in required])
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        a = imputer.fit_transform(x[train])
        v = imputer.transform(x[val])
        t = imputer.transform(x[test])

        # Grid ablations use one fixed learner so changes in performance are
        # attributable to feature blocks, not model-family or regional-model
        # selection. The city experiment retains the established candidate
        # blend used by the current city-scale model.
        if scope.startswith("grid"):
            model = hgb(SEED).fit(a, y[train])
            selected_val = model.predict(v)
            selected_test = model.predict(t)
            predictions[label] = selected_test
            row = {
                "scope": scope, "year": year, "seed": SEED,
                "feature_set": label, "feature_blocks": " + ".join(required),
                "n_features": int(x.shape[1]), "model": "HGB",
                "n_train": len(train), "n_validation": len(val), "n_test": len(test),
                "r2_validation_log": float(r2_score(y[val], selected_val)),
                **scores(y[test], raw[test], selected_test),
            }
            rows.append(row)
            rules.append({"feature_set": label, "model": "HGB", "fixed_learner": True})
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
            predictions.to_csv(out / "test_predictions.csv", index=False)
            print(json.dumps(row), flush=True)
            continue

        pv, pt = {}, {}
        candidates = {
            "HGB": hgb(SEED),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=300, min_samples_leaf=5, max_features=.8,
                n_jobs=8, random_state=SEED,
            ),
        }
        for name, model in candidates.items():
            model.fit(a, y[train])
            pv[name] = model.predict(v)
            pt[name] = model.predict(t)

        sample = np.random.default_rng(SEED).choice(len(a), min(len(a), 8192), replace=False)
        tabpfn = TabPFNRegressor(
            n_estimators=8, device=device, model_path=WEIGHT,
            ignore_pretraining_limits=True, random_state=SEED,
            fit_mode="fit_preprocessors",
        )
        tabpfn.fit(a[sample], y[train][sample])
        pv["TabPFN"] = predict_batches(tabpfn, v)
        pt["TabPFN"] = predict_batches(tabpfn, t)

        if region is not None:
            global_val, global_test = pv["HGB"], pt["HGB"]
            local_val, local_test = global_val.copy(), global_test.copy()
            for region_id in np.unique(region):
                tr_mask = region[train] == region_id
                va_mask = region[val] == region_id
                te_mask = region[test] == region_id
                support = int(tr_mask.sum())
                if support >= 500:
                    local = hgb(SEED).fit(a[tr_mask], y[train][tr_mask])
                    local_weight = support / (support + 2000)
                    if va_mask.any():
                        local_val[va_mask] = local_weight * local.predict(v[va_mask]) + (1 - local_weight) * global_val[va_mask]
                    if te_mask.any():
                        local_test[te_mask] = local_weight * local.predict(t[te_mask]) + (1 - local_weight) * global_test[te_mask]
            _, regional_weight = max(
                (r2_score(y[val], global_val + weight * (local_val - global_val)), weight)
                for weight in [0, .25, .5, .75, 1]
            )
            pv["Regional"] = global_val + regional_weight * (local_val - global_val)
            pt["Regional"] = global_test + regional_weight * (local_test - global_test)
        else:
            regional_weight = None

        options = []
        names = list(pv)
        for i, left in enumerate(names):
            for right in names[i:]:
                for weight in [0, .25, .5, .75, 1]:
                    options.append((
                        r2_score(y[val], weight * pv[left] + (1 - weight) * pv[right]),
                        left, right, weight,
                    ))
        _, left, right, weight = max(options, key=lambda item: item[0])
        selected_val = weight * pv[left] + (1 - weight) * pv[right]
        selected_test = weight * pt[left] + (1 - weight) * pt[right]
        predictions[label] = selected_test
        row = {
            "scope": scope, "year": year, "seed": SEED,
            "feature_set": label, "feature_blocks": " + ".join(required),
            "n_features": int(x.shape[1]), "model": "Validation_blend",
            "n_train": len(train), "n_validation": len(val), "n_test": len(test),
            "r2_validation_log": float(r2_score(y[val], selected_val)),
            **scores(y[test], raw[test], selected_test),
        }
        rows.append(row)
        rules.append({
            "feature_set": label, "left": left, "right": right,
            "weight_left": weight, "regional_strength": regional_weight,
        })
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        predictions.to_csv(out / "test_predictions.csv", index=False)
        print(json.dumps(row), flush=True)
        del tabpfn
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    (out / "feature_dictionary.json").write_text(json.dumps({
        "scope": scope, "year": year, "target": "ln(C)",
        "minimum_grid_reference_exclusive": MIN_GRID_REFERENCE if scope.startswith("grid") else None,
        "feature_sets": SETS, "feature_names": feature_names,
        "grid_socioeconomic_downscaling": "Route-activity-weighted city fleet, energy and transit values, with coverage indicators.",
        "selection": "City: HGB, ExtraTrees and TabPFN candidates blended using validation R2 only. Grid: one fixed global HGB learner for controlled feature ablation.",
        "rules": rules,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["city", "grid_random", "grid_block"], required=True)
    parser.add_argument("--year", type=int, choices=[2017, 2024], required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.scope, args.year, args.device)
