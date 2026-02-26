"""
Data retrieval and landscape generation for FARSITE simulations.

Provides functions to:
- Download LANDFIRE landscape data (elevation, fuel, canopy)
- Fetch WIFIRE Firemap weather data (wind speed, direction)
- Generate landscape files (.lcp) from rasters
- Validate landscape files
"""
import time
import requests
import zipfile
import io
import json
import subprocess
import numpy as np
from pathlib import Path
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
    email,
    verbose=True
):
    """
    Download LANDFIRE landscape data for a fire location.
    
    Args:
        center_lat: Center latitude (decimal degrees)
        center_lon: Center longitude (decimal degrees)
        radius_miles: Radius around center point (miles)
        output_dir: Directory to save downloaded rasters
        email: Valid email address (required by LANDFIRE)
        verbose: Print progress
        
    Returns:
        dict with paths to ASCII rasters:
        {
            'elevation': Path,
            'slope': Path,
            'aspect': Path,
            'fuel': Path,
            'canopy_cover': Path,
            'canopy_height': Path,
            'canopy_base': Path,
            'canopy_density': Path
        }
    """
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
        "Output_Projection": "5070",  # NAD83 Albers
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
        print(f"  Notification will be sent to {email}")
    
    # Wait for processing
    status_url = f"https://lfps.usgs.gov/api/job/status?JobId={job_id}"
    
    if verbose:
        print("\nWaiting for LANDFIRE processing (takes a couple minutes)...")
    
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
            raise RuntimeError(f"LANDFIRE job {status}: {status_data.get('message', '')}")
        
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
    
    # Convert multi-band TIFF to ASCII rasters
    multi_tif = next(output_dir.glob("*.tif"))
    layer_names = ["250CBD", "250CBH", "250CC", "250CH", "250FBFM40", "ASP2020", "ELEV2020", "SLPP2020"]
    
    if verbose:
        print(f"\nConverting {multi_tif.name} to ASCII rasters...")
    
    for band_idx, layer_name in enumerate(layer_names, start=1):
        asc_path = output_dir / f"{layer_name}.asc"
        gdal.Translate(str(asc_path), str(multi_tif), format="AAIGrid", bandList=[band_idx])
        if verbose:
            print(f"  ✓ {layer_name}.asc")
    
    # Return paths in friendly names
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
        print(f"\n✓ LANDFIRE data downloaded to {output_dir}/")
    
    return result


# ============================================================================
# FIREMAP WEATHER DATA
# ============================================================================

def fetch_weather(lat, lon, start_dt, end_dt, verbose=True):
    """
    Fetch weather data from WIFIRE Firemap for a location and time window.
    
    Args:
        lat: Latitude (decimal degrees)
        lon: Longitude (decimal degrees)
        start_dt: Start datetime (string "YYYY-MM-DDTHH:MM:SS" or datetime)
        end_dt: End datetime (string "YYYY-MM-DDTHH:MM:SS" or datetime)
        verbose: Print progress
        
    Returns:
        dict with:
        {
            'windspeed': float,        # Mean wind speed (mph)
            'winddirection': float,    # Mean wind direction (degrees)
            'observations': DataFrame  # Raw observations
        }
        
    Note:
        Falls back to default values (5.0 mph, 270°) if query fails.
    """
    import pandas as pd
    from datetime import datetime
    
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
        print(f"Time window: {start_dt} to {end_dt}")
    
    try:
        response = requests.get(FIREMAP_WX_URL, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse response (may have JSONP callback wrapper)
        response_text = response.text.strip()
        if response_text.startswith('wxData(') and response_text.endswith(')'):
            json_text = response_text[len('wxData('):-1]
            weather_data = json.loads(json_text)
        else:
            weather_data = response.json()
        
        # Extract station data
        features = weather_data.get('features', [])
        
        if not features:
            raise ValueError("No weather stations found")
        
        station = features[0]['properties']
        station_name = station.get('stationName', 'Unknown')
        
        # Get observations
        wind_speeds = station.get('wind_speed', [])
        wind_directions = station.get('wind_direction', [])
        
        if not wind_speeds:
            raise ValueError("No wind speed data")
        
        # Calculate means
        mean_speed = sum(wind_speeds) / len(wind_speeds)
        mean_direction = sum(wind_directions) / len(wind_directions)
        
        # Create DataFrame
        obs_df = pd.DataFrame({
            'wind_speed_mph': wind_speeds,
            'wind_direction_deg': wind_directions
        })
        
        if verbose:
            print(f"✓ Station: {station_name}")
            print(f"  Observations: {len(wind_speeds)}")
            print(f"  Mean wind: {mean_speed:.1f} mph @ {mean_direction:.0f}°")
        
        return {
            'windspeed': mean_speed,
            'winddirection': mean_direction,
            'observations': obs_df
        }
        
    except Exception as e:
        if verbose:
            print(f"⚠ Weather query failed: {e}")
            print(f"  Using default values: 5.0 mph @ 270°")
        
        # Return defaults
        return {
            'windspeed': 5.0,
            'winddirection': 270.0,
            'observations': pd.DataFrame()
        }


# ============================================================================
# LCP GENERATION FROM RASTERS
# ============================================================================

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
    """
    Generate FARSITE landscape (.lcp) file from ASCII rasters using lcpmake.
    
    Args:
        output_path: Output .lcp file path
        elevation_asc: Elevation raster (.asc) in meters
        slope_asc: Slope raster (.asc) in percent
        aspect_asc: Aspect raster (.asc) in degrees (0-360)
        fuel_asc: Fuel model raster (.asc) - integers matching fuel_model
        canopy_cover_asc: Canopy cover (.asc) in percent (0-100)
        canopy_height_asc: Canopy height (.asc) in meters * 10
        canopy_base_asc: Canopy base height (.asc) in meters * 10
        canopy_density_asc: Canopy bulk density (.asc) in kg/m³ * 100
        latitude: Center latitude in decimal degrees (auto-detected if None)
        fuel_model: Fuel model type - "fb40" (FBFM40) or "fb13" (FBFM13)
        verbose: Print lcpmake command
        
    Returns:
        Path to generated .lcp file
    """
    output_path = Path(output_path)
    lcpmake_exe = Path(LCPMAKE_EXECUTABLE)
    
    if not lcpmake_exe.exists():
        raise FileNotFoundError(
            f"lcpmake executable not found at {lcpmake_exe}\n"
            f"Place lcpmake in {SCRIPT_DIR}/"
        )
    
    # Auto-detect latitude from elevation raster if not provided
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
            if verbose:
                print(f"Auto-detected latitude: {latitude:.4f}")
    
    # Build lcpmake command
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
    else:
        raise ValueError(f"Unknown fuel model: {fuel_model}")
    
    if verbose:
        print("\nRunning lcpmake command:")
        print(" ".join(cmd))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(
            f"lcpmake failed with return code {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    
    final_path = output_path.with_suffix('.lcp')
    
    if not final_path.exists():
        raise RuntimeError(f"lcpmake succeeded but output file not found: {final_path}")
    
    if verbose:
        print(f"\n✓ Landscape file created: {final_path}")
        print(f"  Size: {final_path.stat().st_size / 1024:.1f} KB")
    
    return final_path