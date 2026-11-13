# Three-stage human-mobility workflow

This directory is one consolidated implementation of the mobility pipeline used
in the study. It is organized by scientific stage and contains one canonical
code path for each modelling step.

## Workflow

```text
01_od_generation/run_annual_od.py
    -> annual OD matrices and OD-flow tables
02_traffic_assignment/main.py
    -> routed OD trajectories, edge flows and network distances
03_travel_structure/{four_class,five_class,center_features}
    -> movement-proxy classes and city-level class shares
```

The stages exchange files, not Python objects. This is deliberate: global runs
were sharded by city and year and executed on multiple compute nodes.

## 1. OD generation

`01_od_generation/generate_od_annual/` is the complete annual OD model package.
`run_annual_od.py` is the batch entry point used for year-matched processing.
It expects city polygons and annual population rasters under the configured
project data root and reads the satellite-service token from `ESRI_TOKEN`.

Example dry run:

```bash
cd 01_od_generation
python run_annual_od.py --year 2024 --group all --gpu 0 --dry-run
```

Before a real run, replace the data placeholders in `run_annual_od.py` or wire
them to the paths defined in the package-level `config/paths.example.yaml`.

## 2. Traffic assignment

`02_traffic_assignment/main.py` is the primary assignment entry point. The
modules in the same directory build the network representation, snap and group
OD pairs, run stochastic route assignment, and calculate routed distances.
`prepare_inputs/` contains the one-time road, population and POI preparation
scripts. The two `step-3-trajectory-*.py` scripts derive trajectory context for
the subsequent movement-structure model.

Configure `config.py` and the input roots near the top of `main.py`, then run:

```bash
cd 02_traffic_assignment
python main.py
```

Run one city first. Global assignment is memory intensive and is designed to
resume from chunked Parquet outputs.

## 3. Travel-structure inference

- `four_class/`: four-class training, inference and model ablations.
- `five_class/`: five-class training, fast inference, selected-band settings,
  audit and validation.
- `center_features/`: centre-aware five-class inference, aggregation,
  benchmarking and diagnostic utilities.

Training and inference scripts expect routed trajectory Parquet files, image or
land-use representations, centre features and model weights. Weights and raw
data are not redistributed. Run scripts from their own subdirectory so that
their local model imports resolve consistently.

## Reproducibility notes

- Raw mobility records, road graphs, image features and trained weights are not
  included in this code archive.
- `${...}` strings are portable path placeholders and must be configured.
- Preserve metres for routed edge lengths unless a script explicitly converts
  them to kilometres.
- Scripts labelled 2017 or 2024 encode the corresponding input schema; inspect
  year constants before applying them to another year.

