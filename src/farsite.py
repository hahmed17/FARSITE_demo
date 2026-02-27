"""
FARSITE fire simulation module.
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
from shapely.geometry import Polygon
from shapely.ops import unary_union


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
SRC_DIR  = Path(__file__).parent          # .../repo/src/
BASE_DIR = SRC_DIR.parent                 # .../repo/
DATA_DIR = BASE_DIR / "data"              # .../repo/data/
TMP_DIR  = BASE_DIR / "tmp"               # .../repo/tmp/

# Executables and static assets
FARSITE_EXECUTABLE = SRC_DIR / "TestFARSITE"
LCPMAKE_EXECUTABLE = SRC_DIR / "lcpmake"
NO_BARRIER_PATH    = SRC_DIR / "NoBarrier" / "NoBarrier.shp"
LCP_PATH           = DATA_DIR / "landscape.lcp"

# Temporary working directory for FARSITE runs
FARSITE_TMP_DIR = TMP_DIR


# FARSITE parameters
FARSITE_MIN_IGNITION_VERTEX_DISTANCE = 15.0
FARSITE_SPOT_GRID_RESOLUTION = 60.0
FARSITE_SPOT_PROBABILITY = 0
FARSITE_SPOT_IGNITION_DELAY = 0
FARSITE_MINIMUM_SPOT_DISTANCE = 60
FARSITE_ACCELERATION_ON = 1
FARSITE_FILL_BARRIERS = 1
SPOTTING_SEED = 253114
FUEL_MOISTURES_DATA = "1"
RAWS_ELEVATION = 2501
RAWS_UNITS = "English"
DEFAULT_TEMPERATURE = 66
DEFAULT_HUMIDITY = 25
DEFAULT_PRECIPITATION = 0
DEFAULT_CLOUDCOVER = 0
FOLIAR_MOISTURE_CONTENT = 100
CROWN_FIRE_METHOD = "ScottReinhardt"
WRITE_OUTPUTS_EACH_TIMESTEP = 0
MAX_FARSITE_TIMESTEP = 30  # minutes


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_geom(geom):
    """Validate and clean geometry."""
    if geom is None:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def cleanup_farsite_outputs(run_id, base_dir):
    """Delete all files/directories starting with run_id."""
    base_dir = Path(base_dir)
    for p in base_dir.glob(f"{run_id}_*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


# ============================================================================
# FARSITE CONFIGURATION FILE
# ============================================================================

class Config_File:
    """Generates FARSITE configuration (.cfg) files."""
    
    def __init__(self, FARSITE_START_TIME, FARSITE_END_TIME,
                 windspeed, winddirection, FARSITE_DISTANCE_RES, FARSITE_PERIMETER_RES,
                 fuel_moistures=None, temperature=None, humidity=None):
        
        self.__set_default()
        
        self.FARSITE_START_TIME = FARSITE_START_TIME
        self.FARSITE_END_TIME = FARSITE_END_TIME
        total_minutes = int((self.FARSITE_END_TIME - self.FARSITE_START_TIME).total_seconds() / 60)
        self.FARSITE_TIMESTEP = min(MAX_FARSITE_TIMESTEP, max(1, total_minutes))
        self.FARSITE_DISTANCE_RES = FARSITE_DISTANCE_RES
        self.FARSITE_PERIMETER_RES = FARSITE_PERIMETER_RES
        self.windspeed = windspeed
        self.winddirection = winddirection
        
        # Override defaults if provided
        if fuel_moistures:
            self.FUEL_MOISTURES_DATA = [fuel_moistures]
        if temperature is not None:
            self.temperature = temperature
        if humidity is not None:
            self.humidity = humidity

    def __set_default(self):
        """Set default FARSITE configuration parameters."""
        self.version = 1.0
        self.FARSITE_MIN_IGNITION_VERTEX_DISTANCE = FARSITE_MIN_IGNITION_VERTEX_DISTANCE
        self.FARSITE_SPOT_GRID_RESOLUTION = FARSITE_SPOT_GRID_RESOLUTION
        self.FARSITE_SPOT_PROBABILITY = FARSITE_SPOT_PROBABILITY
        self.FARSITE_SPOT_IGNITION_DELAY = FARSITE_SPOT_IGNITION_DELAY
        self.FARSITE_MINIMUM_SPOT_DISTANCE = FARSITE_MINIMUM_SPOT_DISTANCE
        self.FARSITE_ACCELERATION_ON = FARSITE_ACCELERATION_ON
        self.FARSITE_FILL_BARRIERS = FARSITE_FILL_BARRIERS
        self.SPOTTING_SEED = SPOTTING_SEED
        self.FUEL_MOISTURES_DATA = [(0, 3, 5, 7, 60, 90)]  # Active wildfire conditions (dry)
        self.RAWS_ELEVATION = RAWS_ELEVATION
        self.RAWS_UNITS = RAWS_UNITS
        self.FOLIAR_MOISTURE_CONTENT = FOLIAR_MOISTURE_CONTENT
        self.CROWN_FIRE_METHOD = CROWN_FIRE_METHOD
        self.WRITE_OUTPUTS_EACH_TIMESTEP = WRITE_OUTPUTS_EACH_TIMESTEP
        self.temperature = DEFAULT_TEMPERATURE
        self.humidity = DEFAULT_HUMIDITY
        self.precipitation = DEFAULT_PRECIPITATION
        self.cloudcover = DEFAULT_CLOUDCOVER

    def tostring(self):
        """Generate FARSITE configuration file content."""
        config_text = f'FARSITE INPUTS FILE VERSION {self.version}\n'
        
        # Times
        str_start = f'{self.FARSITE_START_TIME.month} {self.FARSITE_START_TIME.day} {self.FARSITE_START_TIME.hour:02d}{self.FARSITE_START_TIME.minute:02d}'
        config_text += f'FARSITE_START_TIME: {str_start}\n'
        
        str_end = f'{self.FARSITE_END_TIME.month} {self.FARSITE_END_TIME.day} {self.FARSITE_END_TIME.hour:02d}{self.FARSITE_END_TIME.minute:02d}'
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
        
        # Fuel moistures
        config_text += f'FUEL_MOISTURES_DATA: {len(self.FUEL_MOISTURES_DATA)}\n'
        for data in self.FUEL_MOISTURES_DATA:
            config_text += f'{data[0]} {data[1]} {data[2]} {data[3]} {data[4]} {data[5]}\n'
        
        config_text += f'RAWS_ELEVATION: {self.RAWS_ELEVATION}\n'
        config_text += f'RAWS_UNITS: {self.RAWS_UNITS}\n'
        
        # Weather
        config_text += 'RAWS: 1\n'
        config_text += f'{self.FARSITE_START_TIME.year} {self.FARSITE_START_TIME.month} {self.FARSITE_START_TIME.day} {self.FARSITE_START_TIME.hour:02d}{self.FARSITE_START_TIME.minute:02d} {self.temperature} {self.humidity} {self.precipitation} {self.windspeed} {self.winddirection} {self.cloudcover}\n'
        
        config_text += f'FOLIAR_MOISTURE_CONTENT: {self.FOLIAR_MOISTURE_CONTENT}\n'
        config_text += f'CROWN_FIRE_METHOD: {self.CROWN_FIRE_METHOD}\n'
        config_text += f'WRITE_OUTPUTS_EACH_TIMESTEP: {self.WRITE_OUTPUTS_EACH_TIMESTEP}'
        
        return config_text

    def to_file(self, filepath):
        """Write configuration to file."""
        with open(filepath, mode='w') as file:
            file.write(self.tostring())


# ============================================================================
# FARSITE RUN FILE
# ============================================================================

class Run_File:
    """Generates FARSITE run files."""
    
    def __init__(self, lcppath, cfgpath, ignitepath, barrierpath, outpath):
        self.lcppath = lcppath
        self.cfgpath = cfgpath
        self.ignitepath = ignitepath
        self.barrierpath = barrierpath
        self.outpath = outpath

    def tostring(self):
        """Generate run file content."""
        return f'{self.lcppath} {self.cfgpath} {self.ignitepath} {self.barrierpath} {self.outpath} -1'

    def to_file(self, filepath):
        """Write run file to disk."""
        with open(filepath, mode='w') as file:
            file.write(self.tostring())


# ============================================================================
# FARSITE SIMULATION WRAPPER
# ============================================================================

class Farsite:
    """Wrapper class for running a single FARSITE simulation."""
    
    def __init__(self, initial, params, start_time,
                 lcppath=None, barrierpath=None,
                 dist_res=30, perim_res=60, debug=False,
                 fuel_moistures=None, temperature=None, humidity=None):
        """
        Initialize FARSITE simulation.
        
        Args:
            initial: Initial fire perimeter (Shapely Polygon in EPSG:5070)
            params: Dict with 'windspeed', 'winddirection', 'dt' (timedelta)
            start_time: Simulation start time
            lcppath: Path to landscape (.lcp) file
            barrierpath: Path to barrier shapefile
            dist_res: Distance resolution (meters)
            perim_res: Perimeter resolution (meters)
            debug: Keep intermediate files if True
            fuel_moistures: Tuple (month, fm1, fm10, fm100, herb, wood) or None for defaults
            temperature: Temperature (F) or None for default
            humidity: Relative humidity (%) or None for default
        """
        self.farsitepath = str(FARSITE_EXECUTABLE)
        self.id = uuid.uuid4().hex
        
        self.tmpfolder = str(FARSITE_TMP_DIR)
        Path(self.tmpfolder).mkdir(parents=True, exist_ok=True)
        
        self.lcppath = lcppath
        
        # Parse start time
        if isinstance(start_time, datetime.datetime):
            start_dt = start_time
        else:
            start_dt = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        
        end_dt = start_dt + params['dt']
        windspeed = params['windspeed']
        winddirection = params['winddirection']
        
        # Create config file
        self.config = Config_File(
            FARSITE_START_TIME=start_dt,
            FARSITE_END_TIME=end_dt,
            windspeed=windspeed,
            winddirection=winddirection,
            FARSITE_DISTANCE_RES=dist_res,
            FARSITE_PERIMETER_RES=perim_res,
            fuel_moistures=fuel_moistures,
            temperature=temperature,
            humidity=humidity
        )
        
        self.configpath = os.path.join(self.tmpfolder, f'{self.id}_config.cfg')
        self.config.to_file(self.configpath)
        
        # Create ignition shapefile
        self.ignitepath = os.path.join(self.tmpfolder, f'{self.id}_ignite.shp')
        gdf = gpd.GeoDataFrame(geometry=[initial], crs="EPSG:5070")
        gdf.to_file(self.ignitepath)
        
        # Barrier path
        if barrierpath:
            self.barrierpath = barrierpath
        elif NO_BARRIER_PATH.exists():
            self.barrierpath = str(NO_BARRIER_PATH)
        else:
            self.barrierpath = ""
        
        # Output directory
        self.outpath = os.path.join(self.tmpfolder, f'{self.id}_out')
        Path(self.outpath).mkdir(parents=True, exist_ok=True)
        
        # Create run file
        self.runfile = Run_File(
            lcppath=self.lcppath,
            cfgpath=self.configpath,
            ignitepath=self.ignitepath,
            barrierpath=self.barrierpath,
            outpath=self.outpath
        )
        
        self.runpath = os.path.join(self.tmpfolder, f'{self.id}_run.txt')
        self.runfile.to_file(self.runpath)
        
        self.debug = debug

    def run(self, ncores=4, timeout=60):
        """
        Execute FARSITE simulation.
        
        Args:
            ncores: Number of cores to use
            timeout: Timeout in minutes
            
        Returns:
            Return code from FARSITE
        """
        log_dir = Path(FARSITE_TMP_DIR) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        out_log = log_dir / f"{self.id}.out"
        err_log = log_dir / f"{self.id}.err"
        
        cmd = ["timeout", f"{timeout}m", self.farsitepath, self.runpath, str(ncores)]
        
        with open(out_log, "w") as fout, open(err_log, "w") as ferr:
            p = subprocess.run(cmd, stdout=fout, stderr=ferr)
        
        return p.returncode

    def output_geom(self):
        base = self.outpath
        out_dir = base  # <-- was: os.path.dirname(base)
        
        candidates = sorted(
            glob.glob(os.path.join(out_dir, "**", "*.shp"), recursive=True),
            key=os.path.getmtime,
            reverse=True
        )
        
        for shp in candidates[:25]:
            try:
                gdf = gpd.read_file(shp)
            except Exception:
                continue
            if len(gdf) == 0:
                continue
            
            geom = unary_union(list(gdf.geometry.values))
            return geom
        
        return None


# ============================================================================
# HIGH-LEVEL FARSITE FUNCTIONS
# ============================================================================

def forward_pass_farsite(poly, params, start_time, lcppath, 
                        dist_res=30, perim_res=60, debug=False):
    """
    Run FARSITE forward simulation for specified time period.
    Splits long simulations into multiple MAX_FARSITE_TIMESTEP chunks.
    
    Args:
        poly: Initial fire perimeter (Shapely Polygon)
        params: Dict with 'windspeed', 'winddirection', 'dt' (timedelta)
        start_time: Start time (datetime or string)
        lcppath: Path to landscape file
        dist_res: Distance resolution (meters)
        perim_res: Perimeter resolution (meters)
        debug: Keep intermediate files if True
        
    Returns:
        Final fire perimeter geometry or None
    """
    dt = params['dt']
    MAX_SIM = int(dt.total_seconds() / 60)
    
    if dist_res > 500:
        warnings.warn(f'dist_res ({dist_res}) must be 1-500. Setting to 500')
        dist_res = 500
    
    if perim_res > 500:
        warnings.warn(f'perim_res ({perim_res}) must be 1-500. Setting to 500')
        perim_res = 500
    
    run_id = uuid.uuid4().hex   # Single unique ID for this forward pass
    
    # Run multiple FARSITE steps if needed
    number_of_farsites = dt.seconds // (MAX_SIM * 60)
    for i in range(number_of_farsites):
        new_params = {
            'windspeed': params['windspeed'],
            'winddirection': params['winddirection'],
            'dt': datetime.timedelta(minutes=MAX_SIM)
        }
        
        farsite = Farsite(
            poly, new_params,
            start_time=start_time,
            lcppath=lcppath,            dist_res=dist_res,
            perim_res=perim_res,
            debug=debug
        )
        # farsite.id = run_id   # Share ID so cleanup catches all files
        
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
        # print("FARSITE outputs cleaned")
        return poly
    
    new_params = {
        'windspeed': params['windspeed'],
        'winddirection': params['winddirection'],
        'dt': remaining_dt
    }
    
    farsite = Farsite(
        poly, new_params,
        start_time=start_time,
        lcppath=lcppath,        dist_res=dist_res,
        perim_res=perim_res,
        debug=debug
    )
    # farsite.id = run_id   # Share ID so cleanup catches all files
    
    farsite.run()
    out = farsite.output_geom()
    
    if out is None:
        print("No output perimeter produced; keeping outputs for inspection.")
        return None
    
    cleanup_farsite_outputs(farsite.id, str(FARSITE_TMP_DIR))
    # print("FARSITE outputs cleaned")
    
    return out


def forward_pass_farsite_24h(poly, params, start_time, lcppath,
                             simulation_hours=24,
                             dist_res=30, perim_res=60, debug=False,
                             max_step_minutes=30, min_final_minutes=1):
    """
    Run FARSITE forward simulation for extended periods (e.g., 24 hours).
    Automatically splits into manageable timesteps and fetches fuel moisture data.
    
    Args:
        poly: Initial fire perimeter (Shapely Polygon)
        params: Dict with 'windspeed', 'winddirection', 'lat', 'lon'
        start_time: Start time (datetime or string "YYYY-mm-dd HH:MM:SS")
        lcppath: Path to landscape file
        simulation_hours: Total simulation duration in hours (default: 24)
        dist_res: Distance resolution (meters)
        perim_res: Perimeter resolution (meters)
        debug: Keep intermediate files if True
        max_step_minutes: Maximum timestep per FARSITE run (minutes)
        min_final_minutes: Minimum final timestep to run (skip if smaller)
        
    Returns:
        Final fire perimeter geometry or None
    """
    from data_retrieval import fetch_fuel_moisture
    
    # Use simulation_hours parameter, or fall back to params['dt'] if provided
    if 'dt' in params:
        total_dt = params["dt"]
    else:
        total_dt = datetime.timedelta(hours=simulation_hours)
    
    if not isinstance(total_dt, datetime.timedelta):
        raise TypeError("Must provide simulation_hours or params['dt'] as timedelta")
    
    # Normalize start_time
    if isinstance(start_time, str):
        start_time = start_time.replace("T", " ")
        start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    elif not isinstance(start_time, datetime.datetime):
        raise TypeError("start_time must be datetime or string 'YYYY-mm-dd HH:MM:SS'")
    
    # Fetch fuel moisture for the fire location and date
    if 'lat' in params and 'lon' in params:
        fuel_moistures = fetch_fuel_moisture(params['lat'], params['lon'], start_time)
    else:
        print("⚠ No lat/lon in params, using default fuel moistures")
        fuel_moistures = None
    
    if dist_res > 500:
        warnings.warn(f"dist_res ({dist_res}) must be 1-500. Setting to 500")
        dist_res = 500
    if perim_res > 500:
        warnings.warn(f"perim_res ({perim_res}) must be 1-500. Setting to 500")
        perim_res = 500

    run_id = uuid.uuid4().hex
    
    max_step = datetime.timedelta(minutes=max_step_minutes)
    remaining = total_dt
    step_idx = 0
    
    print(f"Starting simulation: {total_dt.total_seconds()/3600:.1f} hours in {max_step_minutes}-min steps")
    
    while remaining > datetime.timedelta(0):
        step_dt = min(max_step, remaining)
        print(f"Step {step_idx+1}: {remaining.total_seconds()/3600:.1f}h remaining, running {step_dt.total_seconds()/60:.0f}min")
        
        # Skip tiny remainder
        if step_dt < datetime.timedelta(minutes=min_final_minutes):
            break
        
        new_params = {
            "windspeed": params["windspeed"],
            "winddirection": params["winddirection"],
            "dt": step_dt,
        }
        
        farsite = Farsite(
            poly, new_params,
            start_time=start_time,
            lcppath=lcppath,
            dist_res=dist_res,
            perim_res=perim_res,
            debug=debug,
            fuel_moistures=fuel_moistures
        )
        farsite.id = run_id   # Share ID so cleanup catches all files
        farsite.run(ncores=4)
        
        out = farsite.output_geom()
        
        if out is None:
            print("FARSITE output geometry is None. Returning last valid geometry.")
            cleanup_farsite_outputs(run_id, str(FARSITE_TMP_DIR))
            return poly  # returns the IGNITION polygon, not grown fire
        
        poly = validate_geom(out)
        print(f"  → Area: {poly.area/1e6:.3f} km²")
        
        # Advance time
        start_time = start_time + step_dt
        remaining = remaining - step_dt
        step_idx += 1
    
    cleanup_farsite_outputs(run_id, str(FARSITE_TMP_DIR))
    print(f"✓ Completed {step_idx} timesteps")
    
    return poly