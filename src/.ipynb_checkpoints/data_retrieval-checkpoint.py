"""
Data retrieval and landscape generation for FARSITE simulations.

Provides functions to:
- Download LANDFIRE landscape data (elevation, fuel, canopy)
- Fetch WIFIRE Firemap weather data (wind speed, direction)
- Fetch WIFIRE Firemap fire perimeters
- Fetch NASA FIRMS hotspot data
- Generate landscape files (.lcp) from rasters
"""
import time
import requests
import zipfile
import io
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point
from osgeo import gdal, osr
import struct

# Enable GDAL exceptions
gdal.UseExceptions()
osr.UseExceptions()

# Paths
SCRIPT_DIR = Path(__file__).parent
LCPMAKE_EXECUTABLE = SCRIPT_DIR / "lcpmake"


# ============================================================================
# LANDFIRE DATA DOWNLOAD
# ============================================================================

def download_landfire_data(
    center_lat,
    center_lon,
    radius_miles,
    output_dir,
    email="h7ahmed@ucsd.edu",
    verbose=True
):
    """Download LANDFIRE landscape data for a fire location."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if verbose:
        print(f"Downloading LANDFIRE data around ({center_lat}, {center_lon})")
        print(f"Radius: {radius_miles} miles")
    
    # Create bounding box
    point_wgs84 = gpd.GeoSeries([Point(center_lon, center_lat)], crs="EPSG:4326")
    point_utm = point_wgs84.to_crs(point_wgs84.estimate_utm_crs())
    
    radius_m = radius_miles * 1609.34
    buffer_utm = point_utm.buffer(radius_m)
    
    minx, miny, maxx, maxy = buffer_utm.to_crs("EPSG:4326").total_bounds
    
    if verbose:
        print(f"Bounding box: [{minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f}]")
    
    # Submit LANDFIRE request
    LFPS_URL = "https://lfps.usgs.gov/api/job/submit"
    
    params = {
        "Email": email,
        "Layer_List": "250CBD;250CBH;250CC;250CH;250FBFM40;ASP2020;ELEV2020;SLPP2020",
        "Area_of_Interest": f"{minx} {miny} {maxx} {maxy}",
        "Output_Projection": "5070",
        "Resample_Resolution": "90",
        "Priority_Code": "K3LS9F"
    }
    
    if verbose:
        print("\nSubmitting LANDFIRE request...")
    
    response = requests.get(LFPS_URL, params=params, timeout=30)
    response.raise_for_status()
    
    job_id = response.json()["jobId"]
    
    if verbose:
        print(f"✓ Job ID: {job_id}")
    
    # Wait for processing
    status_url = f"https://lfps.usgs.gov/api/job/status?JobId={job_id}"
    
    if verbose:
        print("\nWaiting for LANDFIRE processing...")
    
    start_time = time.time()
    while True:
        response = requests.get(status_url, timeout=30)
        status_data = response.json()
        status = status_data.get("status", "").lower()
        
        elapsed = int(time.time() - start_time)
        
        if verbose:
            print(f"  [{elapsed}s] {status}")
        
        if status == "succeeded":
            if verbose:
                print(f"\n✓ Completed in {elapsed}s")
            download_url = status_data["outputFile"]
            break
        elif status in ("failed", "canceled"):
            raise RuntimeError(f"LANDFIRE job {status}")
        
        time.sleep(10)
    
    # Download and extract
    if verbose:
        print("Downloading...")
    
    zip_response = requests.get(download_url, stream=True, timeout=60)
    zip_response.raise_for_status()
    
    if verbose:
        print("Extracting...")
    
    with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zf:
        zf.extractall(output_dir)
    
    # Convert to ASCII
    multi_tif = next(output_dir.glob("*.tif"))
    layer_names = ["250CBD", "250CBH", "250CC", "250CH", "250FBFM40", "ASP2020", "ELEV2020", "SLPP2020"]
    
    if verbose:
        print(f"\nConverting to ASCII...")
    
    for band_idx, layer_name in enumerate(layer_names, start=1):
        asc_path = output_dir / f"{layer_name}.asc"
        gdal.Translate(str(asc_path), str(multi_tif), format="AAIGrid", bandList=[band_idx])
        if verbose:
            print(f"  ✓ {layer_name}.asc")
    
    result = {
        'elevation': output_dir / "ELEV2020.asc",
        'slope': output_dir / "SLPP2020.asc",
        'aspect': output_dir / "ASP2020.asc",
        'fuel': output_dir / "250FBFM40.asc",
        'canopy_cover': output_dir / "250CC.asc",
        'canopy_height': output_dir / "250CH.asc",
        'canopy_base': output_dir / "250CBH.asc",
        'canopy_density': output_dir / "250CBD.asc"
    }
    
    if verbose:
        print(f"\n✓ LANDFIRE data downloaded")
    
    return result


def fetch_fuel_moisture(lat, lon, date, verbose=True):
    """
    Fetch fuel moisture values for a location and date.
    
    Args:
        lat: Latitude
        lon: Longitude  
        date: Date (datetime or string)
        verbose: Print progress
        
    Returns:
        tuple: (month, fm1, fm10, fm100, fmherb, fmwood) in percent
        
    Note:
        Falls back to NFDRS-based estimates if query fails.
    """
    from datetime import datetime
    import math
    
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")
    
    # TODO: Query WIFIRE Firemap fuel moisture API when available
    # For now, use NFDRS-based estimates based on time of year
    
    month = date.month - 1  # FARSITE uses 0-based months
    
    # Estimate based on season and location
    # Winter fires (CA): drier live fuels, moderate dead fuels
    # Summer fires: driest conditions
    
    if date.month in [6, 7, 8, 9]:  # Peak fire season
        fm1 = 3
        fm10 = 4  
        fm100 = 5
        fmherb = 30
        fmwood = 60
    elif date.month in [10, 11, 12, 1, 2]:  # Winter (still fire season in CA)
        fm1 = 4
        fm10 = 5
        fm100 = 6
        fmherb = 60
        fmwood = 90
    else:  # Spring
        fm1 = 5
        fm10 = 6
        fm100 = 8
        fmherb = 80
        fmwood = 100
    
    if verbose:
        print(f"Using fuel moisture for {date.strftime('%B %Y')}: "
              f"1h={fm1}%, 10h={fm10}%, 100h={fm100}%, herb={fmherb}%, wood={fmwood}%")
    
    return (month, fm1, fm10, fm100, fmherb, fmwood)


# ============================================================================
# FIREMAP WEATHER DATA
# ============================================================================

def fetch_weather(lat, lon, start_dt, end_dt, verbose=True):
    """Fetch weather data from WIFIRE Firemap."""
    # Convert datetime to string if needed
    if isinstance(start_dt, datetime):
        start_dt = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(end_dt, datetime):
        end_dt = end_dt.strftime("%Y-%m-%dT%H:%M:%S")
    
    FIREMAP_WX_URL = "https://firemap.sdsc.edu/pylaski/stations/data"
    
    params = {
        'selection': 'closestTo',
        'lat': str(lat),
        'lon': str(lon),
        'observable': ['wind_speed', 'wind_direction'],
        'from': start_dt,
        'to': end_dt,
    }
    
    if verbose:
        print(f"\nQuerying weather near ({lat}, {lon})")
    
    try:
        response = requests.get(FIREMAP_WX_URL, params=params, timeout=30)
        response.raise_for_status()
        
        response_text = response.text.strip()
        if response_text.startswith('wxData('):
            json_text = response_text[len('wxData('):-1]
            weather_data = json.loads(json_text)
        else:
            weather_data = response.json()
        
        features = weather_data.get('features', [])
        if not features:
            raise ValueError("No weather stations found")
        
        station = features[0]['properties']
        wind_speeds = station.get('wind_speed', [])
        wind_directions = station.get('wind_direction', [])
        
        if not wind_speeds:
            raise ValueError("No wind data")
        
        mean_speed = sum(wind_speeds) / len(wind_speeds)
        mean_direction = sum(wind_directions) / len(wind_directions)
        
        if verbose:
            print(f"✓ Mean wind: {mean_speed:.1f} mph @ {mean_direction:.0f}°")
        
        return {
            'windspeed': mean_speed,
            'winddirection': mean_direction,
            'observations': pd.DataFrame({'wind_speed_mph': wind_speeds, 'wind_direction_deg': wind_directions})
        }
        
    except Exception as e:
        if verbose:
            print(f"⚠ Weather query failed: {e}")
            print(f"  Using defaults: 5 mph @ 270°")
        return {'windspeed': 5.0, 'winddirection': 270.0, 'observations': pd.DataFrame()}


# ============================================================================
# FIREMAP PERIMETER DATA
# ============================================================================

def fetch_fire_perimeters(fire_name, year, verbose=True):
    """Fetch fire perimeters from WIFIRE Firemap WFS service."""
    FIREMAP_WFS_URL = "https://firemap.sdsc.edu/geoserver/wfs"
    
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": "WIFIRE:view_historical_fires",
        "CQL_FILTER": f"fire_name = '{fire_name}' AND year = {year}",
        "OUTPUTFORMAT": "application/json",
        "SRSNAME": "EPSG:4326",
    }
    
    if verbose:
        print(f"\nQuerying Firemap WFS for '{fire_name}' ({year})")
    
    try:
        response = requests.get(FIREMAP_WFS_URL, params=params, timeout=30)
        response.raise_for_status()
        
        geojson_data = response.json()
        
        if not geojson_data.get('features'):
            if verbose:
                print(f"  ⚠ No perimeters found")
            return gpd.GeoDataFrame()
        
        perims_gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
        perims_gdf.crs = "EPSG:4326"
        perims_gdf = perims_gdf.to_crs("EPSG:5070")
        
        if verbose:
            print(f"  ✓ Retrieved {len(perims_gdf)} perimeters")
            print(f"  Columns: {list(perims_gdf.columns)}")
        
        # Parse dates - check multiple possible column names
        date_col = None
        for col in ['perimeter_timestamp', 'date', 'timestamp', 'acq_date', 'perimeter_date', 'datetime', 'DATE', 'alarm_date']:
            if col in perims_gdf.columns:
                date_col = col
                break
        
        if date_col:
            try:
                # Clean up the timestamp format (remove trailing 'Z' if it's just a date)
                dates = perims_gdf[date_col].astype(str).str.replace('Z$', '', regex=True)
                perims_gdf['datetime'] = pd.to_datetime(dates)
                perims_gdf = perims_gdf.sort_values('datetime').reset_index(drop=True)
                if verbose:
                    print(f"  Date range: {perims_gdf['datetime'].min()} to {perims_gdf['datetime'].max()}")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Could not parse dates from column '{date_col}': {e}")
                    print(f"  Sample value: {perims_gdf[date_col].iloc[0]}")
        else:
            if verbose:
                print(f"  ⚠ No date column found - perimeters not sorted by time")
        
        return perims_gdf
        
    except Exception as e:
        if verbose:
            print(f"  ⚠ Error: {e}")
        return gpd.GeoDataFrame()


# ============================================================================
# NASA FIRMS HOTSPOT DATA
# ============================================================================

def fetch_firms_hotspots(
    center_lat,
    center_lon,
    radius_km,
    start_date,
    end_date=None,
    day_range=1,
    map_key="b38da98e9b7e9389fd05a00c32f99783",
    source="LANDSAT_NRT",
    verbose=True
):
    """Fetch NASA FIRMS active fire detections."""
    from io import StringIO
    
    if isinstance(start_date, datetime):
        start_date = start_date.strftime("%Y-%m-%d")
    if isinstance(end_date, datetime):
        end_date = end_date.strftime("%Y-%m-%d")
    
    if verbose:
        print(f"\nQuerying FIRMS hotspots near ({center_lat}, {center_lon})")
    
    # Create bounding box
    point_gdf = gpd.GeoDataFrame(geometry=[Point(center_lon, center_lat)], crs="EPSG:4326")
    utm_crs = point_gdf.estimate_utm_crs()
    point_utm = point_gdf.to_crs(utm_crs)
    buffer_utm = point_utm.buffer(radius_km * 1000)
    buffer_wgs84 = buffer_utm.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = buffer_wgs84.total_bounds
    bbox_str = f"{minx},{miny},{maxx},{maxy}"
    
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source}/{bbox_str}/{day_range}/{start_date}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        hotspots_df = pd.read_csv(StringIO(response.text))
        
        if hotspots_df.empty:
            if verbose:
                print(f"  ⚠ No detections")
            return gpd.GeoDataFrame()
        
        # Parse datetime
        if 'acq_date' in hotspots_df.columns:
            hotspots_df['datetime'] = pd.to_datetime(
                hotspots_df['acq_date'] + ' ' + hotspots_df['acq_time'].astype(str).str.zfill(4),
                format='%Y-%m-%d %H%M'
            )
        elif 'ACQ_DATE' in hotspots_df.columns:
            hotspots_df['datetime'] = pd.to_datetime(
                hotspots_df['ACQ_DATE'] + ' ' + hotspots_df['ACQ_TIME'].astype(str).str.zfill(4),
                format='%Y-%m-%d %H%M'
            )
            hotspots_df.columns = hotspots_df.columns.str.lower()
        
        # Filter high confidence
        high_conf = hotspots_df[hotspots_df['confidence'].isin(['H', 'M'])].copy()
        
        if verbose:
            print(f"  ✓ Retrieved {len(high_conf)} high-confidence detections")
        
        if high_conf.empty:
            return gpd.GeoDataFrame()
        
        geometry = [Point(lon, lat) for lon, lat in zip(high_conf['longitude'], high_conf['latitude'])]
        hotspots_gdf = gpd.GeoDataFrame(high_conf, geometry=geometry, crs="EPSG:4326")
        hotspots_gdf = hotspots_gdf.to_crs("EPSG:5070")
        
        return hotspots_gdf
        
    except Exception as e:
        if verbose:
            print(f"  ⚠ Error: {e}")
        return gpd.GeoDataFrame()


# ============================================================================
# LCP GENERATION
# ============================================================================



def fix_canopy_cover(template_asc, output_asc, default_value=40, verbose=True):
    """
    Generate proper canopy cover file with realistic values.
    Uses default value since LANDFIRE data may be corrupted.
    
    Args:
        template_asc: Template ASC file (for dimensions/header)
        output_asc: Output canopy cover ASC file
        default_value: Default canopy cover percentage (0-100)
        verbose: Print progress
    """
    if verbose:
        print(f"Generating canopy cover with default value {default_value}%...")
    
    with open(template_asc, 'r') as fin, open(output_asc, 'w') as fout:
        for i, line in enumerate(fin):
            if i < 6:  # Copy header
                fout.write(line)
            else:  # Replace data with default value
                values = []
                for val in line.split():
                    if val.strip() == '-9999':
                        values.append('-9999')
                    else:
                        values.append(str(default_value))
                fout.write(' '.join(values) + '\n')
    
    if verbose:
        print(f"  ✓ Created {output_asc}")
    
    return Path(output_asc)


def generate_lcp_from_rasters(
    output_path,
    elevation_asc,
    slope_asc,
    aspect_asc,
    fuel_asc,
    canopy_cover_asc,
    canopy_height_asc,
    canopy_base_asc,
    canopy_density_asc,
    latitude=None,
    fuel_model="fb40",
    verbose=True
):
    """Generate FARSITE landscape file from rasters."""
    import os
    
    # Fix PROJ_LIB
    if 'PROJ_LIB' not in os.environ:
        os.environ['PROJ_LIB'] = '/opt/conda/share/proj'
    
    output_path = Path(output_path)
    lcpmake_exe = Path(LCPMAKE_EXECUTABLE)
    
    if not lcpmake_exe.exists():
        raise FileNotFoundError(f"lcpmake not found at {lcpmake_exe}")
    
    # Auto-detect latitude if needed
    if latitude is None:
        ds = gdal.Open(str(elevation_asc))
        if ds:
            gt = ds.GetGeoTransform()
            proj = ds.GetProjection()
            x_center = gt[0] + (ds.RasterXSize / 2) * gt[1]
            y_center = gt[3] + (ds.RasterYSize / 2) * gt[5]
            
            src_srs = osr.SpatialReference()
            src_srs.ImportFromWkt(proj)
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromEPSG(4326)
            transform = osr.CoordinateTransformation(src_srs, dst_srs)
            lon, lat, _ = transform.TransformPoint(x_center, y_center)
            latitude = lat
            ds = None
    
    cmd = [
        str(lcpmake_exe),
        "-latitude", str(latitude),
        "-landscape", str(output_path.with_suffix('')),
        "-elevation", str(elevation_asc),
        "-slope", str(slope_asc),
        "-aspect", str(aspect_asc),
        "-fuel", str(fuel_asc),
        "-cover", str(canopy_cover_asc),
        "-height", str(canopy_height_asc),
        "-base", str(canopy_base_asc),
        "-density", str(canopy_density_asc),
    ]
    
    if fuel_model.lower() in ["fb40", "fbfm40", "40"]:
        cmd.append("-fb40")
    elif fuel_model.lower() in ["fb13", "fbfm13", "13"]:
        cmd.append("-fb13")
    
    if verbose:
        print("\nRunning lcpmake...")
        print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if verbose and result.stdout:
        print(f"lcpmake stdout: {result.stdout}")
    if verbose and result.stderr:
        print(f"lcpmake stderr: {result.stderr}")
    
    if result.returncode != 0:
        raise RuntimeError(f"lcpmake failed: {result.stderr}")
    
    final_path = output_path.with_suffix('.lcp')
    
    if not final_path.exists():
        raise RuntimeError("LCP file not created")
    
    if verbose:
        print(f"✓ Landscape file created: {final_path}")
    
    return final_path