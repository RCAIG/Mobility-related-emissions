# Human Mobility and Road-Transport Emission Workflow

This archive contains the research code supporting the three-stage global human-mobility model, multiscale road-transport CO2 estimation, external validation, and mitigation analyses used in the manuscript *From human mobility to road-network-level mobile emission estimation*.

Raw mobility records, geospatial rasters and vectors, external inventories, trained model weights, cached road networks, and generated figures are intentionally excluded. Paths and connection details have been replaced by portable placeholders.

## Scientific workflow

1. **OD generation (`01_mobility/01_od_generation`)**
   - Generates annual origin-destination flows from gridded population, land-use and satellite-image covariates.
   - The complete annual model is in `generate_od_annual/`; `run_annual_od.py` is the batch entry point used for 2017-2024 processing.
2. **Road-network traffic assignment (`01_mobility/02_traffic_assignment`)**
   - Builds or loads road graphs, snaps OD endpoints to the network, performs Monte Carlo route assignment, and extracts routed trajectory distance and edge flow.
   - Network preparation, OD preprocessing, assignment and trajectory statistics are consolidated into one implementation without machine-specific mirrors.
3. **Travel-structure inference (`01_mobility/03_travel_structure`)**
   - Trains and applies the four- and five-class movement-proxy models.
   - Includes four-class, five-class and centre-aware branches, together with training, inference, selected-band configuration, aggregation and validation utilities.
4. **Mobility-to-emission regression (`02_emission_estimation`)**
   - Constructs city- and 0.1-degree-grid mobility features.
   - Compares TabPFN, histogram gradient boosting and related baselines; includes zero-reference handling, positive-reference models, outlier robustness, distance-bin experiments, spatial calibration, ensembling and feature ablation.
   - Reconstructs road-level emissions by allocating grid/city predictions to routed network activity.
5. **Validation (`03_validation`)**
   - Mobility: Porto Taxi, Chicago Taxi, WorldMove, UTD19 and GeoLife workflows.
   - Emissions: EDGAR-based held-out evaluation and independent local inventories, including Hestia and multicity regional comparisons.
6. **Mitigation analysis (`04_mitigation_analysis`)**
   - Demand-emission-intensity decomposition, urban portfolio classes, Technological Decarbonization Index sensitivity, and intra-urban population/function/multicentre analyses.

## Configuration

Copy `config/paths.example.yaml` to a private location and replace the placeholders with paths on the target system. Legacy scripts were kept close to the versions used for the analysis, so some read paths directly from string constants. Search for `${...}` placeholders before running a script and connect them to the corresponding configuration value or environment variable.

No password, SSH host, personal username, or original machine-specific path is included. Do not commit private configuration files or credentials.

## Recommended execution order

```text
01_mobility/01_od_generation
  -> 01_mobility/02_traffic_assignment
  -> 01_mobility/03_travel_structure
  -> 02_emission_estimation/city_grid_models
  -> 02_emission_estimation/feature_ablation
  -> 02_emission_estimation/road_reconstruction
  -> 03_validation
  -> 04_mitigation_analysis
```

The scripts are research workflows rather than a single command-line application. Run each stage only after its expected inputs have been prepared. Several global operations require substantial RAM, GPU capacity, or a scheduler; test on one city before launching the complete city set.

## Environment

Python 3.10-3.12 is recommended. Install the core packages with:

```bash
python -m pip install -r requirements.txt
```

Heavy or platform-sensitive dependencies such as GDAL, PyTorch, PyTorch Lightning, OpenCLIP, TabPFN and OSMnx are best installed with conda/mamba or the platform-specific instructions of each project. `requirements.txt` distinguishes core, modelling and optional figure/document dependencies.

## Inputs and outputs

Typical inputs include city boundaries, annual population grids, land-use/POI data, satellite representations, generated OD tables, road graphs, trajectory/edge-flow tables, EDGAR road-transport CO2 grids, socioeconomic covariates and independent validation inventories. Typical outputs include annual OD matrices, routed distances, edge flows, movement-class shares, city/grid emission predictions, road-level reconstructions, validation tables and manuscript figures.

Units must be checked at module boundaries. The study treats EDGAR road-transport emissions as annual grid totals and converts distance fields explicitly where required. Do not mix metres and kilometres or CO2 and CO2-equivalent inventories without a documented conversion.
- The package preserves the analysis code used for the manuscript, including scientifically distinct experiment variants; it does not claim that every historical script is a polished general-purpose API.

