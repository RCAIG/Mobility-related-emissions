# Road-level mobility-carbon reconstruction V1

## Scientific interpretation

The outputs are **reconstructed road-level transport CO2 emissions constrained by simulated
mobility, the validated 0.1-degree spatial model, and the validated city-total model**. They are
not observations of emissions on individual road links.

## Estimation chain

1. Each simulated route is assigned to one of five network-distance classes: <2, 2-5, 5-15,
   15-40, and >=40 km. Route flow is accumulated on every traversed directed road edge.
2. Edge flow is multiplied by the portion of edge length in each 0.1-degree cell, producing
   class-specific road activity. This follows the activity-by-distance logic of link-based road
   emission inventories, while avoiding unsupported assumptions about local fleet and speed cycles.
3. The selected spatial model is HGBR with mobility and city context. Its strict held-out
   5-degree-block R2 reached 0.604 (seed 42), exceeding the corresponding TabPFN result (0.592).
4. A non-negative first- or second-order polynomial in log class activity is selected by
   five-fold held-out 5-degree-block validation. It supplies only the relative class response used
   to allocate a cell total; it is not interpreted as a causal or laboratory emission factor.
5. The selected city-total model is TabPFN with size and socioeconomic/road context. The six
   displayed cities are excluded when their totals are predicted. Published validation files show
   R2=0.733 in 2017 and R2=0.688 in 2024 for this model family.
6. Spatial predictions determine the within-city distribution. A single multiplicative city
   reconciliation factor makes the sum of all road links equal the held-out city-total prediction.
   Within each cell, emissions are allocated to edges by class-specific activity and the empirical
   polynomial response. Conservation is checked in `road_reconstruction_conservation_audit.csv`.

## Why TabPFN is not used directly on road links

There is no link-level emission truth for training. Direct road-link regression would therefore
create pseudo-precision and would not guarantee that link predictions sum to the validated grid and
city totals. Machine learning is used only at the scales where a reference target exists; explicit
mass-preserving reconciliation performs the downscaling.

## Key limitations

- EDGAR at 0.1 degree is a gridded reference, not road-link truth.
- The five categories are network-distance proxies. They must not be renamed as observed transport
  modes unless route-level mode predictions are joined in a later version.
- Flow-weighted distance captures activity but not local fleet composition, congestion, gradients,
  cold starts, or speed trajectories.
- The class response is associational. The grid total is identifiable; the decomposition among five
  classes is not unique without independent mode/fleet observations.
- Validation performance must be quoted from held-out spatial and city tests. The full-data spatial
  fit used for reconstruction is not an independent validation result.

## Methodological anchors

- Smit, Ntziachristos & Boulter (2010), *Atmospheric Environment* 44, 2943-2953,
  https://doi.org/10.1016/j.atmosenv.2010.05.022.
- Barth & Boriboonsomsin (2008), *Transportation Research Record* 2058, 163-171,
  https://doi.org/10.3141/2058-20.
- EMEP/EEA Air Pollutant Emission Inventory Guidebook, Road Transport chapter.
- US EPA MOVES and the Handbook Emission Factors for Road Transport (HBEFA), as reference
  frameworks for future speed-, fleet-, and fuel-resolved extensions.
- Roberts et al. (2017), *Ecography* 40, 913-929, https://doi.org/10.1111/ecog.02881,
  for spatially structured cross-validation.
- Hollmann et al. (2025), *Nature* 637, 319-326,
  https://doi.org/10.1038/s41586-024-08328-6, for TabPFN.

## Main outputs

- `activity/`: edge-level class flow and edge-cell class activity.
- `results/`: calibrated cell totals and reconstructed directed-edge emissions.
- `tables/`: model evidence, polynomial selection, city totals, and conservation audit.
- `figures/`: response curves, 12 individual maps, and the six-city 2017/2024 panel.
