"""
Standalone FARSITE fire simulation module.
Based on the working farsite.py from the Firemap repo, with config
and geometry imports inlined so no external dependencies are needed.

PATCHED (see "PATCH" comments):
  1. Relative humidity is written to the RAWS line in percent (FARSITE's unit).
     `humidity` may be given as a fraction (0.08) or as percent (8); both work.
  2. Farsite.output_geom() returns the *final* perimeter (max Elapsed_Mi),
     closes the ring, and unions multiple fronts at the final time.
  3. Each step's temporary files (cfg, run, ignition shp, outputs, logs) are
     deleted after the step unless debug=True. The original multistep cleanup
     used a run_id that no file was ever named with, so nothing was deleted.
  4. validate_geom() follows geometry.py (lines are polygonized, largest polygon
     kept) but ignores non-polygon parts and returns None instead of a
     non-Polygon. Adds lines_to_polygon() from geometry.py and full_geom()
     (all parts merged, for scoring against multi-part observed perimeters).
  5. forward_pass_farsite_multistep() accepts time-varying weather:
     `params` can be a dict (constant, as before) or a callable
     f(step_start, step_end) -> dict. It can also return the per-step history
     and call an on_step callback. Called the old way, it behaves as before.
  6. FARSITE output is line-buffered (stdbuf) so logs survive a timeout, and a
     failed step reports why (timeout / exit code + last log line).
  7. Short leftover pieces are folded into the previous 30-min step instead of
     being run alone, since FARSITE rejects runs whose start/end are too close.
  8. Optional smoothing of each step's input perimeter (simplify_m), and the
     input's vertex count/area in failure messages.
  9. Per-step fuel moistures: a weather dict with fm1/fm10/fm100 (+ fm_herb/fm_woody)
     overrides FUEL_MOISTURES_DATA for that step.

Directory layout:
    ./
    ├── farsite.py       (this file)
    ├── TestFARSITE
    ├── lcpmake
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
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString, MultiLineString
from shapely.ops import unary_union, polygonize
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
FARSITE_SPOT_PROBABILITY             = .5
FARSITE_SPOT_IGNITION_DELAY          = 0
FARSITE_MINIMUM_SPOT_DISTANCE        = 60
FARSITE_ACCELERATION_ON              = 1
FARSITE_FILL_BARRIERS                = 1
SPOTTING_SEED                        = 253114

FUEL_MOISTURES_DATA         = [[0, 7, 4, 7, 30, 60]]
RAWS_ELEVATION              = 2501
RAWS_UNITS                  = 'English'
DEFAULT_TEMPERATURE         = 66      # deg F
DEFAULT_HUMIDITY            = .08     # fraction (written to FARSITE as 8 %)
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

def lines_to_polygon(geom):
    """
    From geometry.py: convert LineString/MultiLineString perimeter boundaries
    (FARSITE's output format) to a Polygon/MultiPolygon via polygonize.
    Polygonal input is returned unchanged. Returns None if nothing polygonizes.
    """
    if geom is None or geom.is_empty:
        return None
    gt = geom.geom_type
    if gt in ("Polygon", "MultiPolygon"):
        return geom
    if gt in ("LineString", "MultiLineString"):
        polys = list(polygonize(unary_union(geom)))
        if not polys:
            return None
        return polys[0] if len(polys) == 1 else MultiPolygon(polys)
    if gt == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if polys:
            return unary_union(polys)
        lines = [g for g in geom.geoms if g.geom_type in ("LineString", "MultiLineString")]
        if lines:
            polys2 = list(polygonize(unary_union(lines)))
            if not polys2:
                return None
            return MultiPolygon(polys2) if len(polys2) > 1 else polys2[0]
    return None


def polygonal_parts(geom):
    """All Polygon parts of a (valid-ified) geometry, flattened."""
    if geom is None or geom.is_empty:
        return []
    geom = make_valid(geom)
    if isinstance(geom, Polygon):
        return [geom]
    parts = []
    if hasattr(geom, 'geoms'):
        for g in geom.geoms:
            parts.extend(polygonal_parts(g))
    return [p for p in parts if not p.is_empty]


def validate_geom(poly):
    """
    Validate and clean a geometry (geometry.py semantics): lines are
    polygonized, the result made valid, and the largest polygon returned.
    PATCH 4: non-polygon parts are ignored and None is returned instead of a
    non-Polygon when nothing polygonal is left.
    """
    poly = lines_to_polygon(poly)
    parts = polygonal_parts(poly)
    if not parts:
        return None
    return max(parts, key=lambda g: g.area)


def fuel_moisture_row(wx):
    """PATCH 9: FUEL_MOISTURES_DATA row [model 0 (all), 1h, 10h, 100h, live herb, live woody] from a
    weather dict with fm1/fm10/fm100 (and optionally fm_herb/fm_woody), as integer percent."""
    default = FUEL_MOISTURES_DATA[0]
    dead = [max(2, int(round(wx[k]))) for k in ('fm1', 'fm10', 'fm100')]
    live = [int(round(wx.get('fm_herb', default[4]))), int(round(wx.get('fm_woody', default[5])))]
    return [0] + dead + live


def prepare_ignition(poly, simplify_m=0):
    """PATCH 8: optional light smoothing of a step's input perimeter (tolerance in m)."""
    if simplify_m and simplify_m > 0:
        smooth = validate_geom(poly.simplify(simplify_m, preserve_topology=True))
        if smooth is not None and smooth.area > 0.98 * poly.area:
            return smooth
    return poly


def full_geom(geom):
    """All polygonal parts merged (MultiPolygon or Polygon), or None. Used for scoring."""
    parts = polygonal_parts(lines_to_polygon(geom))
    return unary_union(parts) if parts else None


def _rh_percent(humidity):
    """PATCH 1: FARSITE RAWS humidity is integer percent. Accept fraction or percent."""
    h = float(humidity)
    if h <= 1.0:
        h *= 100.0
    return int(round(min(max(h, 1.0), 100.0)))


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
                 temperature: int,
                 humidity: float,
                 FARSITE_DISTANCE_RES: int,
                 FARSITE_PERIMETER_RES: int):

        self.__set_default()

        self.FARSITE_START_TIME = FARSITE_START_TIME
        self.FARSITE_END_TIME   = FARSITE_END_TIME
        total_minutes = int((FARSITE_END_TIME - FARSITE_START_TIME).total_seconds() / 60)
        self.FARSITE_TIMESTEP      = min(MAX_FARSITE_TIMESTEP, max(1, total_minutes))
        self.FARSITE_DISTANCE_RES  = FARSITE_DISTANCE_RES
        self.FARSITE_PERIMETER_RES = FARSITE_PERIMETER_RES
        self.windspeed             = int(round(float(windspeed)))
        self.winddirection         = int(round(float(winddirection))) % 360
        self.temperature           = int(round(float(temperature)))
        self.humidity              = humidity

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
                        f'{_rh_percent(self.humidity)} {self.precipitation} {self.windspeed} '
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
                 dist_res: int = DEFAULT_DIST_RES, perim_res: int = DEFAULT_PERIM_RES,
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
        temperature   = params.get('temperature', DEFAULT_TEMPERATURE)
        humidity      = params.get('humidity', DEFAULT_HUMIDITY)

        # Config file
        self.config     = Config_File(start_dt, end_dt, windspeed, winddirection, temperature, humidity, dist_res, perim_res)
        if params.get('fuel_moistures'):                 # PATCH 9: per-step fuel moistures
            self.config.FUEL_MOISTURES_DATA = params['fuel_moistures']
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
        # PATCH 6: line-buffer FARSITE's stdout so the log survives if `timeout` kills it
        if shutil.which("stdbuf"):
            cmd = cmd[:2] + ["stdbuf", "-oL", "-eL"] + cmd[2:]

        t0 = datetime.datetime.now()
        with open(out_log, "w") as fout, open(err_log, "w") as ferr:
            p = subprocess.run(cmd, stdout=fout, stderr=ferr)
        self.returncode = p.returncode
        self.elapsed_min = (datetime.datetime.now() - t0).total_seconds() / 60
        self.timeout_min = timeout
        return p.returncode

    def failure_reason(self):
        """Human-readable reason for a step that produced no perimeter."""
        rc = getattr(self, 'returncode', None)
        if rc == 124:
            return f"timed out after {self.timeout_min} min (FARSITE_TIMEOUT_MIN)"
        log = Path(self.tmpfolder, 'farsite_logs', f'{self.id}.out')
        lines = [l.strip() for l in log.read_text(errors='replace').splitlines() if l.strip()] if log.exists() else []
        last = lines[-1] if lines else '(empty log)'
        return f"exit code {rc} after {getattr(self, 'elapsed_min', float('nan')):.1f} min; last log line: {last}"

    def output_geom(self):
        """
        Extract the final perimeter from FARSITE's {outpath}_Perimeters.shp.

        PATCH 2: the original returned Polygon(gdf['geometry'][0].coords), i.e.
        the FIRST row, which is not guaranteed to be the end-of-run perimeter.
        Now: take all rows at the largest elapsed time, close each ring, union
        them, and keep the largest polygon (consistent with validate_geom).
        """
        output_path = self.outpath + '_Perimeters.shp'
        if not os.path.exists(output_path):
            return None

        gdf = gpd.read_file(output_path)
        if len(gdf) == 0:
            return None

        time_col = next((c for c in ('Elapsed_Mi', 'ElapsedMin', 'Elapsed_Min')
                         if c in gdf.columns), None)
        if time_col is not None:
            gdf = gdf[gdf[time_col] == gdf[time_col].max()]
        else:
            gdf = gdf.iloc[[-1]]

        lines, closed = [], []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type in ('Polygon', 'MultiPolygon'):
                closed.append(geom)
                continue
            for line in (geom.geoms if geom.geom_type == 'MultiLineString' else [geom]):
                coords = list(line.coords)
                if len(coords) < 3:
                    continue
                if coords[0] != coords[-1]:          # close open fronts so polygonize can use them
                    coords.append(coords[0])
                lines.append(LineString(coords))
        if lines:
            # geometry.py approach: polygonize the (noded) fronts; handles self-crossing loops
            poly = lines_to_polygon(MultiLineString(lines))
            if poly is None:                          # fallback: one polygon per closed ring
                poly = unary_union([make_valid(Polygon(l.coords)) for l in lines])
            closed.append(poly)
        if not closed:
            return None
        return validate_geom(unary_union([make_valid(g) for g in closed]))

    def cleanup(self):
        """PATCH 3: delete this run's temp files and logs (no-op when debug=True)."""
        if self.debug:
            return
        cleanup_farsite_outputs(self.id, self.tmpfolder)
        for suffix in ('.out', '.err'):
            Path(self.tmpfolder, 'farsite_logs', f'{self.id}{suffix}').unlink(missing_ok=True)


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
# HIGH-LEVEL FORWARD PASS — single call (unchanged except per-step cleanup)
# ============================================================================

def forward_pass_farsite(poly, params, start_time, lcppath,
                         dist_res=30, perim_res=60, debug=False):
    """
    Run FARSITE forward simulation for specified time period.

    Args:
        poly:        Initial fire perimeter (Shapely Polygon, EPSG:5070)
        params:      Dict with 'windspeed', 'winddirection', 'temperature',
                     'humidity', 'dt' (timedelta)
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

    # Run multiple FARSITE steps if needed
    number_of_farsites = dt.seconds // (MAX_SIM * 60)
    for i in range(number_of_farsites):
        new_params = {
            'windspeed':     params['windspeed'],
            'winddirection': params['winddirection'],
            'temperature':   params['temperature'],
            'humidity':      params['humidity'],
            'dt':            datetime.timedelta(minutes=MAX_SIM)
        }
        farsite = Farsite(poly, new_params, start_time=start_time,
                          lcppath=lcppath, dist_res=dist_res,
                          perim_res=perim_res, debug=debug)
        farsite.run()
        out = farsite.output_geom()
        if out is None:
            print("FARSITE output geometry is None")
            return None
        farsite.cleanup()
        poly = validate_geom(out)

    # Handle remaining time
    remaining_dt = dt - number_of_farsites * datetime.timedelta(minutes=MAX_SIM)
    if remaining_dt < datetime.timedelta(minutes=10):
        return poly

    new_params = {
        'windspeed':     params['windspeed'],
        'winddirection': params['winddirection'],
        'temperature':   params['temperature'],
        'humidity':      params['humidity'],
        'dt':            remaining_dt
    }
    farsite = Farsite(poly, new_params, start_time=start_time,
                      lcppath=lcppath, dist_res=dist_res,
                      perim_res=perim_res, debug=debug)
    farsite.run()
    out = farsite.output_geom()

    if out is None:
        print("No output perimeter produced; keeping outputs for inspection.")
        return None

    farsite.cleanup()
    return out


# ============================================================================
# MULTISTEP FORWARD PASS — 30-min steps, optional time-varying weather
# ============================================================================

def forward_pass_farsite_multistep(poly, params, start_time, lcppath,
                                   dist_res=30, perim_res=60, debug=False,
                                   dt=None, step_minutes=30,
                                   min_remainder_minutes=10,
                                   return_history=False, on_step=None,
                                   timeout=20, ncores=4, verbose=True, simplify_m=0):
    """
    Run FARSITE in fixed-length steps, feeding each step's output perimeter
    into the next step.

    Args:
        poly:        Initial perimeter (Shapely Polygon, EPSG:5070)
        params:      (A weather dict may also carry 'fm1', 'fm10', 'fm100' [and 'fm_herb',
                     'fm_woody'] in percent; that step then uses those fuel moistures
                     instead of FUEL_MOISTURES_DATA.)
                     EITHER a dict with 'windspeed', 'winddirection',
                     'temperature', 'humidity' and 'dt' (constant weather,
                     original behaviour)
                     OR a callable f(step_start, step_end) -> dict with the
                     four weather keys (time-varying weather). In that case
                     the total duration comes from `dt` (or params['dt'] is
                     not needed).
        start_time:  datetime or "YYYY-MM-DD HH:MM:SS"
        lcppath:     Path to .lcp landscape file
        dt:          Total duration (timedelta). Required if params is callable.
        step_minutes:          Length of each FARSITE call (default 30).
        min_remainder_minutes: A leftover shorter than step_minutes is folded into the
                               previous step. If the whole duration is shorter than
                               step_minutes, it runs only if >= this; otherwise skipped.
        return_history:        If True, return a list of dicts
                               {'time', 'poly', 'weather', 'step'} — one per
                               completed step (index 0 = initial state). If a
                               step fails, the history so far is returned and
                               the last entry has 'failed': True.
        on_step:     Optional callback f(record_dict) after each step.
        simplify_m:  If > 0, each step's input perimeter is simplified with this tolerance (m)
                     before FARSITE. Very jagged perimeters (thousands of tiny loops) can crash it.
        debug:       Keep each step's temp files if True.

    Returns:
        Final polygon (or None on failure), or the history list when
        return_history=True.
    """
    if callable(params):
        weather_fn = params
        if dt is None:
            raise ValueError('dt is required when params is a callable')
    else:
        const = {k: params[k] for k in ('windspeed', 'winddirection', 'temperature', 'humidity')}
        weather_fn = lambda t0, t1: const
        dt = dt if dt is not None else params['dt']

    if dist_res > 500:
        warnings.warn(f'dist_res ({dist_res}) must be 1-500. Setting to 500')
        dist_res = 500
    if perim_res > 500:
        warnings.warn(f'perim_res ({perim_res}) must be 1-500. Setting to 500')
        perim_res = 500

    # PATCH 7: FARSITE rejects very short runs ("Start/End dates ... too close"), so a leftover
    # piece is folded into the previous full step (steps of step_minutes .. 2*step_minutes-1).
    # Only a whole duration shorter than step_minutes runs as a short step, and only if it is
    # at least min_remainder_minutes long; otherwise nothing is simulated.
    total_minutes     = int(round(dt.total_seconds() / 60))
    number_of_steps   = total_minutes // step_minutes
    remaining_minutes = total_minutes % step_minutes
    step_lengths = [step_minutes] * number_of_steps
    if remaining_minutes:
        if step_lengths:
            step_lengths[-1] += remaining_minutes
        elif remaining_minutes >= min_remainder_minutes:
            step_lengths.append(remaining_minutes)

    current_time = (datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                    if isinstance(start_time, str) else start_time)

    poly = validate_geom(poly)
    history = [{'step': 0, 'time': current_time, 'poly': poly, 'weather': None}]

    for i, minutes in enumerate(step_lengths):
        step_end = current_time + datetime.timedelta(minutes=minutes)
        wx = dict(weather_fn(current_time, step_end))
        step_params = {
            'windspeed':     wx['windspeed'],
            'winddirection': wx['winddirection'],
            'temperature':   wx['temperature'],
            'humidity':      wx['humidity'],
            'dt':            datetime.timedelta(minutes=minutes),
        }
        if all(k in wx for k in ('fm1', 'fm10', 'fm100')):
            step_params['fuel_moistures'] = [fuel_moisture_row(wx)]
        poly_in = prepare_ignition(poly, simplify_m)
        farsite = Farsite(poly_in, step_params,
                          start_time=current_time,
                          lcppath=lcppath, dist_res=dist_res,
                          perim_res=perim_res, debug=debug)
        rc = farsite.run(timeout=timeout, ncores=ncores)
        out = farsite.output_geom()
        if out is None:
            reason = farsite.failure_reason() + f"; input perimeter {len(poly_in.exterior.coords)} vertices, {poly_in.area/1e6:.1f} km2"
            print(f"FARSITE output None at step {i+1}/{len(step_lengths)} "
                  f"({current_time} -> {step_end}): {reason}; "
                  f"files kept with id {farsite.id}")
            if return_history:
                history.append({'step': i + 1, 'time': step_end, 'poly': None,
                                'weather': wx, 'failed': True, 'run_id': farsite.id,
                                'reason': reason, 'start': current_time})
                return history
            return None
        farsite.cleanup()

        poly = out
        current_time = step_end
        rec = {'step': i + 1, 'time': current_time, 'poly': poly, 'weather': wx}
        history.append(rec)
        if on_step is not None:
            on_step(rec)
        if verbose:
            print(f"  Step {i+1}/{len(step_lengths)} @ {current_time}: "
                  f"{poly.area/1e6:.3f} km2  "
                  f"(wind {step_params['windspeed']:.0f} mph @ {step_params['winddirection']:.0f}°)")

    return history if return_history else poly