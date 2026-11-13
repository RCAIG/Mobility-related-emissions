# Stage 2: road-network traffic assignment

The files in this directory form one assignment implementation:

- `build_network.py`: load and convert the road graph for routing;
- `preprocess_od.py`: read, filter, snap and aggregate OD pairs;
- `mc_assign.py`: stochastic route assignment;
- `main.py`: resumable city-level batch orchestration;
- `get_distance.py` and `explain_trace.py`: trajectory statistics;
- `prepare_inputs/`: one-time preparation of road and contextual inputs.

Set the graph, OD and output roots before running `python main.py`. Outputs are
chunked Parquet trajectory tables so interrupted global jobs can resume.

