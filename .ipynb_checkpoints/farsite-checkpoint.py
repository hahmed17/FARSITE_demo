"""
Standalone FARSITE fire simulation module.
Exact copy of the working farsite.py from the Firemap repo, with config
and geometry imports inlined so no external dependencies are needed.

Directory layout:
    ./
    ├── farsite.py       (this file)
    ├── TestFARSITE
    ├── lcpmake
    ├── landscape.lcp
    ├── NoBarrier/
    │   └── NoBarrier.shp
    └── tmp/             (created automatically)
"""
import datetime
import os
import uuid
import subprocess
import shutil
import glob
import warnings
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely import make_valid


# ============================================================================
# PATHS  (inlined from config.py — all relative to this file)
# ============================================================================
_DIR               = Path(__file__).parent
FARSITE_EXECUTABLE = _DIR / "TestFARSITE"
NO_BARRIER_PATH    = _DIR / "NoBarrier" / "NoBarrier.shp"
FARSITE_TMP_DIR    = _DIR / "tmp"

# ============================================================================
# CONSTANTS  (inlined from config.py — identical values)
# ============================================================================
MAX_FARSITE_TIMESTEP                 = 30

FARSITE_MIN_IGNITION_VERTEX_DISTANCE = 15.0
FARSITE_SPOT_GRID_RESOLUTION         = 60.0
FARSITE_SPOT_PROBABILITY             = 0
FARSITE_SPOT_IGNITION_DELAY          = 0
FARSITE_MINIMUM_SPOT_DISTANCE        = 60
FARSITE_ACCELERATION_ON              = 1
FARSITE_FILL_BARRIERS                = 1
SPOTTING_SEED                        = 253114

FUEL_MOISTURES_DATA         = [[0, 3, 4, 6, 30, 60]]
RAWS_ELEVATION              = 2501
RAWS_UNITS                  = 'English'
DEFAULT_TEMPERATURE         = 66
DEFAULT_HUMIDITY            = 8
DEFAULT_PRECIPITATION       = 0
DEFAULT_CLOUDCOVER          = 0
FOLIAR_MOISTURE_CONTENT     = 100
CROWN_FIRE_METHOD           = 'ScottReinhardt'
WRITE_OUTPUTS_EACH_TIMESTEP = 0

DEFAULT_DIST_RES  = 150
DEFAULT_PERIM_RES = 150


# ============================================================================
# GEOMETRY UTILITIES  (inlined from geometry.py)
# ============================================================================

def validate_geom(poly):
    """Validate and clean a geometry, returning the largest polygon."""
    poly = make_valid(poly)
    if isinstance(poly, (GeometryCollection, MultiPolygon)):
        poly = max(poly.geoms, key=lambda g: g.area)
    if not isinstance(poly, Polygon):
        print(f'Validated geometry is not a Polygon. Type: {type(poly)}')
    return poly


# ============================================================================
# FARSITE CONFIGURATION FILE
# ============================================================================

class Config_File:
    """Generates FARSITE configuration (.cfg) files."""

    def __init__(self,
                 FARSITE_START_TIME: datetime.datetime,
                 FARSITE_END_TIME: datetime.datetime,
                 windspeed: int,
                 winddirection: int,
                 FARSITE_DISTANCE_RES: int,
                 FARSITE_PERIMETER_RES: int):

        self.__set_default()

        self.FARSITE_START_TIME = FARSITE_START_TIME
        self.FARSITE_END_TIME   = FARSITE_END_TIME
        total_minutes = int((FARSITE_END_TIME - FARSITE_START_TIME).total_seconds() / 60)
        self.FARSITE_TIMESTEP      = min(MAX_FARSITE_TIMESTEP, max(1, total_minutes))
        self.FARSITE_DISTANCE_RES  = FARSITE_DISTANCE_RES
        self.FARSITE_PERIMETER_RES = FARSITE_PERIMETER_RES
        self.windspeed             = windspeed
        self.winddirection         = winddirection

    def __set_default(self):
        self.version                              = 1.0
        self.FARSITE_MIN_IGNITION_VERTEX_DISTANCE = FARSITE_MIN_IGNITION_VERTEX_DISTANCE
        self.FARSITE_SPOT_GRID_RESOLUTION         = FARSITE_SPOT_GRID_RESOLUTION
        self.FARSITE_SPOT_PROBABILITY             = FARSITE_SPOT_PROBABILITY
        self.FARSITE_SPOT_IGNITION_DELAY          = FARSITE_SPOT_IGNITION_DELAY
        self.FARSITE_MINIMUM_SPOT_DISTANCE        = FARSITE_MINIMUM_SPOT_DISTANCE
        self.FARSITE_ACCELERATION_ON              = FARSITE_ACCELERATION_ON
        self.FARSITE_FILL_BARRIERS                = FARSITE_FILL_BARRIERS
        self.SPOTTING_SEED                        = SPOTTING_SEED
        self.FUEL_MOISTURES_DATA                  = FUEL_MOISTURES_DATA
        self.RAWS_ELEVATION                       = RAWS_ELEVATION
        self.RAWS_UNITS                           = RAWS_UNITS
        self.FOLIAR_MOISTURE_CONTENT              = FOLIAR_MOISTURE_CONTENT
        self.CROWN_FIRE_METHOD                    = CROWN_FIRE_METHOD
        self.WRITE_OUTPUTS_EACH_TIMESTEP          = WRITE_OUTPUTS_EACH_TIMESTEP
        self.temperature                          = DEFAULT_TEMPERATURE
        self.humidity                             = DEFAULT_HUMIDITY
        self.precipitation                        = DEFAULT_PRECIPITATION
        self.cloudcover                           = DEFAULT_CLOUDCOVER

    def tostring(self):
        config_text  = f'FARSITE INPUTS FILE VERSION {self.version}\n'
        str_start    = f'{self.FARSITE_START_TIME.month} {self.FARSITE_START_TIME.day} {self.FARSITE_START_TIME.hour:02d}{self.FARSITE_START_TIME.minute:02d}'
        config_text += f'FARSITE_START_TIME: {str_start}\n'
        str_end      = f'{self.FARSITE_END_TIME.month} {self.FARSITE_END_TIME.day} {self.FARSITE_END_TIME.hour:02d}{self.FARSITE_END_TIME.minute:02d}'
        config_text += f'FARSITE_END_TIME: {str_end}\n'
        config_text += f'FARSITE_TIMESTEP: {self.FARSITE_TIMESTEP}\n'
        config_text += f'FARSITE_DISTANCE_RES: {self.FARSITE_DISTANCE_RES}\n'
        config_text += f'FARSITE_PERIMETER_RES: {self.FARSITE_PERIMETER_RES}\n'
        config_text += f'FARSITE_MIN_IGNITION_VERTEX_DISTANCE: {self.FARSITE_MIN_IGNITION_VERTEX_DISTANCE}\n'
        config_text += f'FARSITE_SPOT_GRID_RESOLUTION: {self.FARSITE_SPOT_GRID_RESOLUTION}\n'
        config_text += f'FARSITE_SPOT_PROBABILITY: {self.FARSITE_SPOT_PROBABILITY}\n'
        config_text += f'FARSITE_SPOT_IGNITION_DELAY: {self.FARSITE_SPOT_IGNITION_DELAY}\n'
        config_text += f'FARSITE_MINIMUM_SPOT_DISTANCE: {self.FARSITE_MINIMUM_SPOT_DISTANCE}\n'
        config_text += f'FARSITE_ACCELERATION_ON: {self.FARSITE_ACCELERATION_ON}\n'
        config_text += f'FARSITE_FILL_BARRIERS: {self.FARSITE_FILL_BARRIERS}\n'
        config_text += f'SPOTTING_SEED: {self.SPOTTING_SEED}\n'
        config_text += f'FUEL_MOISTURES_DATA: {len(self.FUEL_MOISTURES_DATA)}\n'
        for data in self.FUEL_MOISTURES_DATA:
            config_text += f'{data[0]} {data[1]} {data[2]} {data[3]} {data[4]} {data[5]}\n'
        config_text += f'RAWS_ELEVATION: {self.RAWS_ELEVATION}\n'
        config_text += f'RAWS_UNITS: {self.RAWS_UNITS}\n'
        config_text += 'RAWS: 1\n'
        config_text += (f'{self.FARSITE_START_TIME.year} {self.FARSITE_START_TIME.month} '
                        f'{self.FARSITE_START_TIME.day} {self.FARSITE_START_TIME.hour:02d}'
                        f'{self.FARSITE_START_TIME.minute:02d} {self.temperature} '
                        f'{self.humidity} {self.precipitation} {self.windspeed} '
                        f'{self.winddirection} {self.cloudcover}\n')
        config_text += f'FOLIAR_MOISTURE_CONTENT: {self.FOLIAR_MOISTURE_CONTENT}\n'
        config_text += f'CROWN_FIRE_METHOD: {self.CROWN_FIRE_METHOD}\n'
        config_text += f'WRITE_OUTPUTS_EACH_TIMESTEP: {self.WRITE_OUTPUTS_EACH_TIMESTEP}'
        return config_text

    def to_file(self, filepath: str):
        with open(filepath, mode='w') as file:
            file.write(self.tostring())


# ============================================================================
# FARSITE RUN FILE
# ============================================================================

class Run_File:
    """Generates FARSITE run files."""

    def __init__(self, lcppath, cfgpath, ignitepath, barrierpath, outpath):
        self.lcppath     = lcppath
        self.cfgpath     = cfgpath
        self.ignitepath  = ignitepath
        self.barrierpath = barrierpath
        self.outpath     = outpath

    def tostring(self):
        return f'{self.lcppath} {self.cfgpath} {self.ignitepath} {self.barrierpath} {self.outpath} -1'

    def to_file(self, filepath: str):
        with open(filepath, mode='w') as file:
            file.write(self.tostring())


# ============================================================================
# FARSITE SIMULATION WRAPPER
# ============================================================================

class Farsite:
    """Wrapper for a single FARSITE simulation run."""

    def __init__(self, initial: Polygon, params: dict,
                 start_time: datetime.datetime,
                 lcppath: str = None, barrierpath: str = None,
                 dist_res: int = 30, perim_res: int = 60,
                 debug: bool = False):

        self.farsitepath = str(FARSITE_EXECUTABLE)
        self.id          = uuid.uuid4().hex

        self.tmpfolder = str(FARSITE_TMP_DIR)
        Path(self.tmpfolder).mkdir(parents=True, exist_ok=True)

        self.lcppath = lcppath

        # Parse start time
        if isinstance(start_time, datetime.datetime):
            start_dt = start_time
        else:
            start_dt = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

        end_dt        = start_dt + params['dt']
        windspeed     = params['windspeed']
        winddirection = params['winddirection']

        # Config file
        self.config     = Config_File(start_dt, end_dt, windspeed, winddirection, dist_res, perim_res)
        self.configpath = os.path.join(self.tmpfolder, f'{self.id}_config.cfg')
        self.config.to_file(self.configpath)

        # Barrier
        self.barrierpath = barrierpath if barrierpath else str(NO_BARRIER_PATH)

        # Ignition shapefile
        self.ignitepath = os.path.join(self.tmpfolder, f'{self.id}_ignite.shp')
        ignite_gdf = gpd.GeoDataFrame({'FID': [0], 'geometry': [initial]}, crs="EPSG:5070")
        ignite_gdf.to_file(self.ignitepath)

        # Output path — FARSITE treats this as a file prefix, writing {outpath}_Perimeters.shp
        self.outpath = os.path.join(self.tmpfolder, f'{self.id}_out')

        # Run file — no extension on runpath, matching the working repo exactly
        self.runfile = Run_File(self.lcppath, self.configpath,
                                self.ignitepath, self.barrierpath, self.outpath)
        self.runpath = os.path.join(self.tmpfolder, f'{self.id}_run')
        self.runfile.to_file(self.runpath)

        self.debug = debug

    def run(self, timeout=20, ncores=4):
        """Execute FARSITE. Returns subprocess return code."""
        log_dir = Path(self.tmpfolder) / "farsite_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        out_log = log_dir / f"{self.id}.out"
        err_log = log_dir / f"{self.id}.err"

        cmd = ["timeout", f"{timeout}m", self.farsitepath, self.runpath, str(ncores)]

        with open(out_log, "w") as fout, open(err_log, "w") as ferr:
            p = subprocess.run(cmd, stdout=fout, stderr=ferr)

        return p.returncode

    def output_geom(self):
        """
        Extract output geometry from FARSITE shapefile results.
        FARSITE writes {outpath}_Perimeters.shp as a prefix in tmp/.
        """
        output_path = self.outpath + '_Perimeters.shp'
        if not os.path.exists(output_path):
            return None

        gdf = gpd.read_file(output_path)
        if len(gdf) == 0:
            return None

        geom = gdf['geometry'][0]
        return Polygon(geom.coords)


# ============================================================================
# CLEANUP
# ============================================================================

def cleanup_farsite_outputs(run_id, base_dir):
    """Delete all files/directories starting with run_id in base_dir."""
    base_dir = Path(base_dir)
    for p in base_dir.glob(f"{run_id}_*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


# ============================================================================
# HIGH-LEVEL FORWARD PASS — exact copy of working forward_pass_farsite
# ============================================================================

def forward_pass_farsite(poly, params, start_time, lcppath,
                         dist_res=30, perim_res=60, debug=False):
    """
    Run FARSITE forward simulation for specified time period.
    Exact copy of the working implementation from the Firemap repo.

    Args:
        poly:        Initial fire perimeter (Shapely Polygon, EPSG:5070)
        params:      Dict with 'windspeed' (int), 'winddirection' (int), 'dt' (timedelta)
        start_time:  Start time (datetime or "YYYY-MM-DD HH:MM:SS" string)
        lcppath:     Path to .lcp landscape file
        dist_res:    Distance resolution (meters)
        perim_res:   Perimeter resolution (meters)
        debug:       Keep intermediate files if True

    Returns:
        Final fire perimeter as Shapely Polygon, or None on failure
    """
    dt      = params['dt']
    MAX_SIM = int(dt.total_seconds() / 60)

    if dist_res > 500:
        warnings.warn(f'dist_res ({dist_res}) must be 1-500. Setting to 500')
        dist_res = 500
    if perim_res > 500:
        warnings.warn(f'perim_res ({perim_res}) must be 1-500. Setting to 500')
        perim_res = 500

    run_id = uuid.uuid4().hex

    # Run multiple FARSITE steps if needed
    number_of_farsites = dt.seconds // (MAX_SIM * 60)
    for i in range(number_of_farsites):
        new_params = {
            'windspeed':     params['windspeed'],
            'winddirection': params['winddirection'],
            'dt':            datetime.timedelta(minutes=MAX_SIM),
        }
        farsite = Farsite(poly, new_params, start_time=start_time,
                          lcppath=lcppath, dist_res=dist_res,
                          perim_res=perim_res, debug=debug)
        farsite.run()
        out = farsite.output_geom()
        if out is None:
            print("FARSITE output geometry is None")
            return None
        poly = validate_geom(out)

    # Handle remaining time
    remaining_dt = dt - number_of_farsites * datetime.timedelta(minutes=MAX_SIM)
    if remaining_dt < datetime.timedelta(minutes=10):
        cleanup_farsite_outputs(run_id, str(FARSITE_TMP_DIR))
        return poly

    new_params = {
        'windspeed':     params['windspeed'],
        'winddirection': params['winddirection'],
        'dt':            remaining_dt,
    }
    farsite = Farsite(poly, new_params, start_time=start_time,
                      lcppath=lcppath, dist_res=dist_res,
                      perim_res=perim_res, debug=debug)
    farsite.run()
    out = farsite.output_geom()

    if out is None:
        print("No output perimeter produced; keeping outputs for inspection.")
        return None

    cleanup_farsite_outputs(farsite.id, str(FARSITE_TMP_DIR))
    return out
