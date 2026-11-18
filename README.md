# FARSITE Full-Fire Simulations

This repository runs [FARSITE](https://www.firelab.org/project/farsite) wildfire-spread simulations for whole fires, not just a single step, and scores them against observed perimeters. FARSITE predicts how a fire perimeter grows over time from terrain, fuels and weather.

One notebook handles every fire. A fire needs only a few settings: ignition point, start and end time, fuel data year, and optionally a satellite overlay file. Everything else (perimeters, landscape, weather, simulation and evaluation) is generic. It's set up for five California fires: **Airport 2024, Zogg 2020, Bobcat 2020, CZU Lightning Complex 2020 and Glass 2020**.

## What it does

The workflow lives in **`FARSITE_NIROP_VIIRS.ipynb`**. For each fire it:

1. **Gets observed perimeters:**
   - **From a satellite overlay** (`overlays/*.html`) when the fire has one: timed perimeter series from NIROPS (nightly airborne infrared) and FEDS (12-hourly, from VIIRS satellite detections).
   - **Otherwise from the [WIFIRE Firemap](https://firemap.sdsc.edu)** layer `WIFIRE:view_historical_fires`.
2. **Builds a landscape** (`landscape.lcp`) from [LANDFIRE](https://lfps.usgs.gov) fuel and terrain data, covering every observed perimeter plus a 25 km buffer. It's cached per fire.
3. **Fetches hourly weather** (wind, temperature, humidity) from the station nearest the ignition, starting 72 h before ignition. Dead fuel moisture for the 1-, 10- and 100-hour fuels is derived from it hour by hour.
4. **Runs FARSITE forward from the ignition point**, in 30-minute steps with that step's weather and fuel moisture. It never resets to an observation.
5. **Scores the simulation at every observed perimeter time** (IoU, Sørensen, area error), against both the main fire and all mapped parts.
6. **Runs restart experiments** that test FARSITE as a short-range forecaster:
   - (i) restart from each NIROPS perimeter and compare with the next VIIRS overpasses;
   - (ii) restart from each VIIRS perimeter and compare with the next NIROPS flight.

   Each forecast is compared with a **persistence** baseline (the starting perimeter, unchanged).

## Requirements

- Linux with **conda** (the install script uses conda for GDAL).
- At least **4 CPUs**. More helps: one fire can use 12 cores, or several fires can run in parallel.
- Jupyter, plus `nbconvert` for headless parallel runs.
- Internet access to WIFIRE (perimeters and weather) and LANDFIRE (landscapes).

## Installation

```bash
chmod +x TestFARSITE lcpmake install_packages.sh
./install_packages.sh          # GDAL via conda + requirements.txt; output in install.log
```

Then select the **Python (gdal_env)** kernel in Jupyter. The figures also use `contextily` for basemaps (optional: maps are drawn without one if it's missing).

## Quick start: the Zogg example (only in parallel version)

Section 12 of the notebook runs one complete example for the Zogg Fire, which is the quickest way to see the whole workflow.

1. Open `FARSITE_run_parallel.ipynb` and set `LANDFIRE_EMAIL` in Section 2 to your own address (LANDFIRE requires one).
2. Restart the kernel and **Run All**. Sections 9–11 only *plan* by default (`DRY_RUN = True`), so the real work happens in Section 12.
3. Section 12 runs:
   - the 3-day forward run,
   - a weather and fuel-moisture check,
   - both restart experiments,
   - their figures.

   With 12 cores and spotting off, that's about 25–35 minutes.

Section 12.1's settings: `ZOGG_DAYS` (`None` = the whole fire), cores, spotting, and switches to skip the main run or the experiments.

## Running other or all fires

- **In the notebook:**
  - Set `FIRES_TO_RUN` and `DRY_RUN = False` (Section 2), then run Section 9.
  - For the experiments, set `EXP_FIRES` and `EXP_DRY_RUN = False` (Section 11).
  - Dry runs print each fire's plan, with its number of FARSITE steps.
- **In parallel (recommended for all five):** `run_parallel.py` runs headless copies of the notebook side by side, one group of fires each:

```bash
python run_parallel.py --max-days 1                       # quick end-to-end check
nohup python run_parallel.py > runs/launcher.log 2>&1 &   # the full run, in the background
tail -f runs/run_A.log                                    # live progress of one job
```

Options:
- `--jobs A C`: run only some jobs.
- `--ncores N`: FARSITE threads per job.
- `--spot 0`: spotting off.
- `--skip-experiments`: main runs only.
- `--dry-run`: plans only.
- `--kernel NAME`: Jupyter kernel to use.

The default split is 4 jobs × 3 cores: Bobcat · CZU · Glass · Zogg + Airport. When all jobs finish, a final pass combines every fire's results into `fires/summary.csv` and `fires/experiments_summary.csv`. If you rename the notebook, update `NOTEBOOK` at the top of `run_parallel.py`.

**Rough run sizes**, in FARSITE steps:

| fire | full run | experiments (i) + (ii) |
|---|---|---|
| Airport | ~1,260 | — (no VIIRS series) |
| Zogg | ~770 | ~590 |
| Bobcat | ~990 (21-day cap) | ~970 |
| CZU | ~1,000 (21-day cap) | ~920 |
| Glass | ~1,110 | ~710 |

Steps take roughly 5–15 s each, more for large fires or with spotting on.

## The fire registry

Each fire is one entry in `FIRES` (Section 2):

```python
'Zogg_2020': dict(
    name='ZOGG', year=2020,
    ignition=(40.5392, -122.5660),                 # lat, lon
    overlay='overlays/CA2020_ZOGG_overlay.html',   # satellite perimeters (optional)
    start='2020-09-27 15:00', end='2020-10-13 17:00',
    start_from='ignition',
    fuel_version='LF2016',                         # LANDFIRE fuels that pre-date the fire
),
```

**Optional per-fire keys**
- `max_sim_days`: cap on how many days are simulated.
- `score_series`: which overlay series to score against.
- `overlay_ignition`: which overlay ignition candidate to use.
- `live_fm`: live (herbaceous, woody) fuel moisture, in %.
- `lcp_buffer_km`, `dist_res`, `perim_res`: landscape buffer and FARSITE resolutions.
- `search_radius_km`: how far from ignition to accept Firemap perimeters.

**With an overlay, the ignition point and time come from the overlay:**
- **Which point:** among its candidates (IRWIN, GOFER, MTBS, CAL FIRE), the one closest to the earliest perimeter. For Bobcat, that correctly picks Cogswell Dam over the IRWIN point, which is 14 km off.
- **Re-timing:** FEDS perimeters are re-timed from their 12-hour bin labels to the actual VIIRS overpass.
- **Cleaning:** same-time pieces are merged, and older burn scars are dropped.

**To add a fire,** add an entry like the one above, put its overlay in `overlays/` if there is one, and run a dry run to check the plan.

## Key settings (Section 2)

| setting | default | meaning |
|---|---|---|
| `LANDFIRE_EMAIL` | — | your email for LANDFIRE downloads |
| `FARSITE_NCORES` | 4 | FARSITE threads per run |
| `SPOT_PROBABILITY` | 0.5 | spotting; 0 is much faster and avoids timeouts |
| `FARSITE_TIMEOUT_MIN` | 20 | limit per FARSITE call; raise to 45–60 with spotting on |
| `DIST_RES`, `PERIM_RES` | 150 m | FARSITE resolutions (coarser = faster) |
| `LCP_BUFFER_KM` | 25 | landscape margin around the observed perimeters |
| `MAX_SIM_DAYS` | None | global cap on simulated days |
| `FUEL_MOISTURE_FROM_WEATHER` | True | dead fuel moisture from the weather; False = fixed `farsite.FUEL_MOISTURES_DATA` |
| `LIVE_FUEL_MOISTURE` | (30, 60) | live herbaceous, woody % |
| `OVERLAY_SCORE_SERIES` | NIROPS, FEDS | overlay series the main run is scored against |
| `EXP_MAX_LEAD_H` | 48 | longest restart forecast (Section 11) |
| `BASEMAP` | Esri.WorldTopoMap | map tiles; None = no basemap |
| `QUIET_GDAL` | True | hide GDAL/PROJ warnings (Section 1) |

## Outputs

```
fires/<fire>/
    landscape.lcp, landscape.json      # cached landscape and its extent
    weather.csv                        # cached hourly weather
    <fire>_metrics.csv                 # scores at each observed perimeter
    <fire>_sim_steps.gpkg              # every 30-min simulated perimeter, with weather and fuel moisture
    <fire>_sim_at_obs.gpkg             # simulated perimeters at the observation times
    <fire>_observed.gpkg               # observed perimeters (layers: largest_part, all_parts)
    <fire>_evaluation.png              # map + area and IoU over time
    <fire>_exp_<experiment>.csv/.gpkg  # restart experiments
    <fire>_exp_*_maps.png, <fire>_exp_summary.png, <fire>_weather.png
fires/summary.csv, fires/experiments_summary.csv, fires/experiments_iou_vs_lead.png   # all fires
runs/                                  # run_parallel.py logs and executed notebooks
```

## Reading the results

- **IoU** runs from 0 to 1. Roughly, above 0.5 is a decent match and below 0.3 is poor.
  - `iou` scores the main observed part: the one nearest the ignition, or overlapping the restart perimeter.
  - `iou_full` scores all mapped parts. That matters for complexes like CZU and merged fires like Glass.
- **Area error:** negative = FARSITE grew the fire too slowly, positive = too fast.
- **Free runs build up error,** because FARSITE has no suppression and nothing pulls it back to observations. The restart experiments are the fairer test of forecast skill. A positive `iou_gain` means FARSITE beat persistence.

## Notebook map

| section | contents |
|---|---|
| 1 | setup, GDAL quiet mode |
| 2 | settings and fire registry; 2b optional cleanup of old logs and results |
| 3 / 3b | perimeters from Firemap / from satellite overlays |
| 4 | LANDFIRE → landscape, cached, rebuilt automatically if too small |
| 5 | hourly weather and fuel moisture, cached |
| 6–7 | scoring and the full-fire driver |
| 8 | Firemap layer check |
| 9–10 | run fires, cross-fire summary |
| 11 | restart experiments and their summary |
| 12 | worked example: Zogg |

## Changes to `farsite.py`

Based on the original single-step module. Each change is marked `PATCH` in the file, and older calls still work:

1. Relative humidity is written in percent (it was written as a fraction, e.g. 0.08 instead of 8).
2. The **final** perimeter is read from FARSITE's output, not the first row. Fronts are turned into polygons as in `geometry.py`.
3. Each step's temporary files are cleaned up; failed steps keep theirs for inspection.
4. Geometry checks ignore non-polygon parts.
5. `forward_pass_farsite_multistep` accepts time-varying weather, and can return every step.
6. FARSITE logs survive timeouts, and failures say why: timeout, exit code, last log line.
7. Short leftover steps are merged into the previous 30-minute step (FARSITE rejects very short runs).
8. Optional smoothing of each step's input perimeter.
9. Per-step fuel moistures.

## Troubleshooting

- **A step "timed out after 20 min":** usually heavy spotting. Set `SPOT_PROBABILITY = 0`, or raise `FARSITE_TIMEOUT_MIN`.
- **"exit code -11":** FARSITE crashed, usually because the simulated fire reached the landscape edge. The message reports the distance to the edge; raise `lcp_buffer_km` if it's small.
- **No perimeters for a Firemap fire:** names are exact and case-sensitive (e.g. `'GLASS'`). The error lists the names found near the ignition point.
- **No weather or a short spin-up:** the weather service sometimes has no data before ignition. The notebook then uses what's cached and prints a warning.
- **Map tiles show "Access blocked" or "API key required":** keep `BASEMAP = 'Esri.WorldTopoMap'`. OpenStreetMap blocks scripts, and CARTO now needs a key.
- **PROJ / `proj.db` warnings:** hidden by `QUIET_GDAL`. To fix them, run `conda install -n gdal_env -c conda-forge proj`.

## Limitations

- **One ignition polygon:** FARSITE starts from a single polygon, the largest part. Fires that begin as several pieces, like CZU, are only partly represented; `iou_full` shows the effect.
- **No suppression, single-station weather:** the weather is hourly means from one station. Expect overgrowth after the initial run, and missed gust-driven runs.
- **FEDS (VIIRS) perimeters run larger than NIROPS,** because of VIIRS's 375 m pixels. NIROPS is the better reference.
- **Fuel moisture uses air temperature and humidity:** sunlit fine fuels can be 1–2 % drier than estimated.

## Repository contents

| file / folder | description |
|---|---|
| `FARSITE_NIROP_VIIRS.ipynb` | the main notebook (all fires, experiments) |
| `FARSITE_run_parallel.ipynb` | parallelized version of the main notebook (all fires, experiments, Zogg example) |
| `farsite.py` | FARSITE wrapper module (see changes above) |
| `run_parallel.py` | runs headless notebook copies in parallel |
| `overlays/` | satellite overlay HTML files (perimeters + ignition data) per fire |
| `TestFARSITE` | FARSITE executable |
| `lcpmake` | landscape (`.lcp`) builder |
| `NoBarrier/` | empty barrier shapefile (FARSITE input) |
| `requirements.txt`, `install_packages.sh` | dependencies and installer |
| `fires/`, `runs/`, `tmp/` | outputs, logs and working files (created during runs; git-ignored) |