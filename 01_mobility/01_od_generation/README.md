# Stage 1: annual OD generation

The `generate_od_annual` package contains the model and all helper modules used
to generate annual origin-destination matrices. `run_annual_od.py` orchestrates
city groups, years, GPU selection, sharding, timeout handling and resumable
outputs.

Required inputs are city polygons, annual gridded population and satellite
representations. Principal outputs are an OD matrix (`.npy`), a tabular OD-flow
file (`.csv`) and an optional arc plot for each city-year.

Use `python run_annual_od.py --help` to inspect the batch options.

