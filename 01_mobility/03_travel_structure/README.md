# Stage 3: movement-structure inference

This stage converts routed trajectories and their spatial context into four- or
five-class movement proxies.

- `four_class/` contains the four-class model family and inference scripts.
- `five_class/` contains the five-class model, training, inference, selected
  feature-band configurations and validation utilities.
- `center_features/` contains the centre-aware inference and aggregation used
  to quantify city-level movement composition.

Model weights and input features are external to this archive. Run each script
from its own directory after replacing path placeholders with the local data
configuration.
