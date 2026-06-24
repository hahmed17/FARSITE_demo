# FARSITE Demo
 
This repository is a self-contained demonstration of running a single [FARSITE](https://www.firelab.org/project/farsite) wildfire-spread simulation on a simple landscape. FARSITE is a fire-growth model that predicts how a fire perimeter spreads over time given terrain, fuel, and weather conditions.
 
Everything needed to run one simulation step is included in this repository. The demo takes an observed fire perimeter at one point in time, simulates how the fire should spread over a short interval, and compares the prediction against the next observed perimeter.

 
## What the demo does
 
The entire workflow lives in **`Farsite.ipynb`**. Running the notebook from top to bottom will:
 
1. **Define fire parameters** — name, ignition location, and landscape download radius for a sample fire.
2. **Fetch fire perimeters** from the [WIFIRE Firemap](https://firemap.sdsc.edu) service. The earliest perimeter (t=0) becomes the simulation's starting point; the next (t=1) is held back as ground truth.
3. **Build the landscape file** — download [LANDFIRE](https://lfps.usgs.gov) terrain and fuel rasters for the area and generate `landscape.lcp`. This step is skipped automatically if `landscape.lcp` already exists.
4. **Fetch weather** (wind speed and direction) for the simulation period from WIFIRE.
5. **Run FARSITE** to predict the fire perimeter from t=0 to t=1.
6. **Compare** the predicted perimeter against the observed one, reporting area error and Intersection-over-Union (IoU), and plotting both perimeters on a map.
## Requirements
 
- A Linux environment with **conda** available (the install script uses `conda` to install GDAL).
- At least **4 CPUs**.
- Python with Jupyter (the notebook runs in a standard Jupyter environment).

## Installation
 
Make the bundled executables and install script runnable, then install dependencies:
 
```bash
chmod +x TestFARSITE lcpmake install_packages.sh
./install_packages.sh
```
 
`install_packages.sh` installs GDAL via conda and the remaining Python packages from `requirements.txt`. Installation output is written to `install.log`.
 
> The first notebook cell also runs `./install_packages.sh`, so if you launch the notebook in a fresh environment it will install dependencies for you.
 
## Usage
 
1. Open `Farsite.ipynb` in Jupyter.
2. In the **Define Fire Parameters** cell, set `LANDFIRE_EMAIL` to your own email address — LANDFIRE requires a valid email to process raster download requests.
3. Run all cells from top to bottom.
The final cells print accuracy metrics and display a map comparing the predicted and observed fire perimeters.
 
> **Note:** Steps 2–4 fetch data from external services (WIFIRE and LANDFIRE), so an internet connection is required. Downloading and generating `landscape.lcp` can take a few minutes the first time; subsequent runs reuse the existing file.
 
## Repository contents
 
| File / folder        | Description                                                        |
| -------------------- | ----------------------------------------------------------------- |
| `Farsite.ipynb`      | The demo notebook — the only file you need to run.                |
| `farsite.py`         | Standalone module that prepares inputs and runs the simulator. |
| `TestFARSITE`        | The FARSITE simulation executable.                                |
| `lcpmake`            | Executable for building the landscape (`.lcp`) file.                 |
| `landscape.lcp`      | Pre-built landscape file.                 |
| `NoBarrier/`         | Empty barrier shapefile used as a FARSITE input.                  |
| `requirements.txt`   | Python dependencies.                                              |
| `install_packages.sh`| Dependency installation script.                                   |
| `tmp/`               | Working directory created during simulation runs.    
