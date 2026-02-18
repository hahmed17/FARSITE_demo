"""
Landscape file (.lcp) generation utilities for FARSITE.

Provides functions to create landscape files from raster data or
generate simple synthetic landscapes for testing.
"""
import subprocess
import numpy as np
from pathlib import Path
from osgeo import gdal, osr
import struct

from config import LCPMAKE_EXECUTABLE, DATA_DIR


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
        
    Raises:
        FileNotFoundError: If lcpmake executable not found
        RuntimeError: If lcpmake fails
    """
    output_path = Path(output_path)
    lcpmake_exe = Path(LCPMAKE_EXECUTABLE)
    
    if not lcpmake_exe.exists():
        raise FileNotFoundError(
            f"lcpmake executable not found at {lcpmake_exe}\n"
            f"Place lcpmake in {DATA_DIR}/"
        )
    
    # Auto-detect latitude from elevation raster if not provided
    if latitude is None:
        ds = gdal.Open(str(elevation_asc))
        if ds:
            gt = ds.GetGeoTransform()
            proj = ds.GetProjection()
            # Get center pixel
            x_center = gt[0] + (ds.RasterXSize / 2) * gt[1]
            y_center = gt[3] + (ds.RasterYSize / 2) * gt[5]
            
            # Transform to WGS84
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
        "-landscape", str(output_path.with_suffix('')),  # lcpmake adds .lcp
        "-elevation", str(elevation_asc),
        "-slope", str(slope_asc),
        "-aspect", str(aspect_asc),
        "-fuel", str(fuel_asc),
        "-cover", str(canopy_cover_asc),
        "-height", str(canopy_height_asc),
        "-base", str(canopy_base_asc),
        "-density", str(canopy_density_asc),
    ]
    
    # Add fuel model flag
    if fuel_model.lower() in ["fb40", "fbfm40", "40"]:
        cmd.append("-fb40")
    elif fuel_model.lower() in ["fb13", "fbfm13", "13"]:
        cmd.append("-fb13")
    else:
        raise ValueError(f"Unknown fuel model: {fuel_model}. Use 'fb40' or 'fb13'")
    
    if verbose:
        print("\nRunning lcpmake command:")
        print(" ".join(cmd))
    
    # Run lcpmake
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(
            f"lcpmake failed with return code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    
    # lcpmake adds .lcp extension automatically
    final_path = output_path.with_suffix('.lcp')
    
    if not final_path.exists():
        raise RuntimeError(f"lcpmake succeeded but output file not found: {final_path}")
    
    if verbose:
        print(f"\n✓ Landscape file created: {final_path}")
        print(f"  Size: {final_path.stat().st_size / 1024:.1f} KB")
    
    return final_path


# ============================================================================
# SIMPLE SYNTHETIC LANDSCAPE GENERATION
# ============================================================================

def create_simple_landscape(
    output_path,
    size_km=10,
    resolution_m=30,
    fuel_model=1,
    elevation_m=1000,
    slope_pct=10,
    aspect_deg=180,
    canopy_cover_pct=40,
    canopy_height_m=15,
    canopy_base_m=2,
    canopy_density_kg_m3=0.15,
    latitude=34.0,
    verbose=True
):
    """
    Generate a simple uniform landscape for FARSITE testing.
    
    Creates a flat, uniform landscape with constant fuel, slope, and canopy
    properties. Perfect for quick demos without needing real terrain data.
    
    Args:
        output_path: Output .lcp file path
        size_km: Landscape size in kilometers (square)
        resolution_m: Cell size in meters
        fuel_model: FBFM40 fuel model code (1-40)
        elevation_m: Uniform elevation in meters
        slope_pct: Uniform slope in percent
        aspect_deg: Uniform aspect in degrees (0-360)
        canopy_cover_pct: Canopy cover percent (0-100)
        canopy_height_m: Canopy height in meters
        canopy_base_m: Canopy base height in meters
        canopy_density_kg_m3: Canopy bulk density in kg/m³
        latitude: Latitude for solar calculations
        verbose: Print progress
        
    Returns:
        Path to generated .lcp file
    """
    output_path = Path(output_path)
    temp_dir = output_path.parent / "temp_rasters"
    temp_dir.mkdir(exist_ok=True)
    
    if verbose:
        print(f"Creating simple {size_km}x{size_km} km landscape...")
    
    # Calculate raster dimensions
    ncols = int((size_km * 1000) / resolution_m)
    nrows = ncols
    
    # Create uniform arrays
    elevation = np.full((nrows, ncols), elevation_m, dtype=np.float32)
    slope = np.full((nrows, ncols), slope_pct, dtype=np.float32)
    aspect = np.full((nrows, ncols), aspect_deg, dtype=np.float32)
    fuel = np.full((nrows, ncols), fuel_model, dtype=np.int16)
    cover = np.full((nrows, ncols), canopy_cover_pct, dtype=np.int16)
    height = np.full((nrows, ncols), int(canopy_height_m * 10), dtype=np.int16)
    base = np.full((nrows, ncols), int(canopy_base_m * 10), dtype=np.int16)
    density = np.full((nrows, ncols), int(canopy_density_kg_m3 * 100), dtype=np.int16)
    
    # Define geotransform (arbitrary NAD83 Albers coordinates)
    # Origin roughly centered on continental US
    xmin, ymin = 500000, 4000000
    xmax = xmin + (ncols * resolution_m)
    ymax = ymin + (nrows * resolution_m)
    
    geotransform = (xmin, resolution_m, 0, ymax, 0, -resolution_m)
    
    # EPSG:5070 (NAD83 Conus Albers) - FARSITE standard
    projection = osr.SpatialReference()
    projection.ImportFromEPSG(5070)
    wkt = projection.ExportToWkt()
    
    # Save as ASCII rasters
    rasters = {
        'elevation': (elevation, np.float32),
        'slope': (slope, np.float32),
        'aspect': (aspect, np.float32),
        'fuel': (fuel, np.int16),
        'cover': (cover, np.int16),
        'height': (height, np.int16),
        'base': (base, np.int16),
        'density': (density, np.int16),
    }
    
    raster_paths = {}
    for name, (data, dtype) in rasters.items():
        asc_path = temp_dir / f"{name}.asc"
        _write_ascii_raster(asc_path, data, geotransform, wkt)
        raster_paths[name] = asc_path
    
    if verbose:
        print(f"  Grid: {nrows} x {ncols} cells ({resolution_m}m resolution)")
        print(f"  Fuel model: {fuel_model}")
        print(f"  Temporary rasters: {temp_dir}")
    
    # Generate LCP using lcpmake
    lcp_path = generate_lcp_from_rasters(
        output_path=output_path,
        elevation_asc=raster_paths['elevation'],
        slope_asc=raster_paths['slope'],
        aspect_asc=raster_paths['aspect'],
        fuel_asc=raster_paths['fuel'],
        canopy_cover_asc=raster_paths['cover'],
        canopy_height_asc=raster_paths['height'],
        canopy_base_asc=raster_paths['base'],
        canopy_density_asc=raster_paths['density'],
        latitude=latitude,
        fuel_model="fb40",
        verbose=verbose
    )
    
    # Cleanup temporary rasters
    import shutil
    shutil.rmtree(temp_dir)
    
    return lcp_path


def _write_ascii_raster(path, data, geotransform, projection_wkt):
    """Write numpy array as GDAL ASCII raster (.asc)."""
    driver = gdal.GetDriverByName('AAIGrid')
    nrows, ncols = data.shape
    
    ds = driver.Create(str(path), ncols, nrows, 1, 
                       gdal.GDT_Float32 if data.dtype == np.float32 else gdal.GDT_Int16)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection_wkt)
    ds.GetRasterBand(1).WriteArray(data)
    ds.GetRasterBand(1).SetNoDataValue(-9999)
    ds = None


# ============================================================================
# LCP FILE VALIDATION
# ============================================================================

def verify_lcp_file(lcp_path, verbose=True):
    """
    Verify that an LCP file is valid and readable.
    
    Args:
        lcp_path: Path to .lcp file
        verbose: Print file info
        
    Returns:
        dict with keys: valid (bool), ncols, nrows, cellsize, errors (list)
    """
    lcp_path = Path(lcp_path)
    
    if not lcp_path.exists():
        return {'valid': False, 'errors': [f"File not found: {lcp_path}"]}
    
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    try:
        with open(lcp_path, 'rb') as f:
            # Read LCP header (first 7316 bytes)
            header = f.read(7316)
            
            # Parse key fields (simplified - full spec has 148 fields)
            crown_fuels = struct.unpack('i', header[0:4])[0]
            ground_fuels = struct.unpack('i', header[4:8])[0]
            latitude = struct.unpack('i', header[8:12])[0]
            loeast = struct.unpack('d', header[4172:4180])[0]
            hieast = struct.unpack('d', header[4180:4188])[0]
            lonorth = struct.unpack('d', header[4188:4196])[0]
            hinorth = struct.unpack('d', header[4196:4204])[0]
            numelev = struct.unpack('i', header[4228:4232])[0]
            numslope = struct.unpack('i', header[4232:4236])[0]
            
            cellsize = struct.unpack('d', header[4248:4256])[0]
            
            # Calculate dimensions
            ncols = int((hieast - loeast) / cellsize)
            nrows = int((hinorth - lonorth) / cellsize)
            
            result['ncols'] = ncols
            result['nrows'] = nrows
            result['cellsize'] = cellsize
            result['latitude'] = latitude / 10000.0
            result['crown_fuels'] = crown_fuels
            result['ground_fuels'] = ground_fuels
            
            if verbose:
                print(f"\n✓ Valid LCP file: {lcp_path}")
                print(f"  Dimensions: {ncols} x {nrows}")
                print(f"  Cell size: {cellsize} m")
                print(f"  Latitude: {result['latitude']:.4f}°")
                print(f"  Layers: {ground_fuels} ground, {crown_fuels} crown")
                
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Error reading LCP: {str(e)}")
        
    return result


# ============================================================================
# DATA DOWNLOAD (PLACEHOLDER)
# ============================================================================

def download_landscape_data(bbox, output_dir, data_source='landfire', verbose=True):
    """
    Download landscape data (elevation, fuel, canopy) for a bounding box.
    
    Args:
        bbox: (minx, miny, maxx, maxy) in EPSG:5070
        output_dir: Directory to save downloaded rasters
        data_source: 'landfire' or 'usgs'
        verbose: Print progress
        
    Returns:
        dict with paths to downloaded rasters
        
    Note:
        This is a placeholder. Actual implementation would use:
        - LANDFIRE API for fuel/canopy data
        - USGS 3DEP for elevation
        - Or other public data sources
    """
    raise NotImplementedError(
        "Automatic landscape data download not yet implemented.\n"
        "Manual download options:\n"
        "  - LANDFIRE: https://landfire.gov/viewer/\n"
        "  - USGS 3DEP Elevation: https://apps.nationalmap.gov/downloader/\n"
        "  - OpenTopography: https://opentopography.org/\n"
    )
