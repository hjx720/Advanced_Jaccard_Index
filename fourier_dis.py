import geopandas as gpd
import numpy as np
from skimage.measure import find_contours
from numpy.fft import fft
import rasterio
from rasterio.features import rasterize
from rasterio.plot import show
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_polygon(gpkg_file, target_crs="EPSG:3857"):
    """Load the first polygon from a GeoPackage file and reproject it to the target CRS."""
    try:
        gdf = gpd.read_file(gpkg_file)
        if gdf.empty:
            raise ValueError(f"No geometry found in {gpkg_file}")
        polygon = gdf.geometry.iloc[0]
        if gdf.crs != target_crs:
            polygon = gdf.to_crs(target_crs).geometry.iloc[0]
        return polygon
    except Exception as e:
        logging.error(f"Error loading polygon from {gpkg_file}: {e}")
        raise

def extract_contour(polygon, transform=None, raster_shape=(1000, 1000)):
    """Extract the contour from a polygon using rasterization and find_contours."""
    try:
        if transform is None:
            transform = rasterio.transform.from_bounds(*polygon.bounds, *raster_shape)
        raster = rasterize([(polygon, 1)], out_shape=raster_shape, transform=transform, fill=0, all_touched=True)
        contours = find_contours(raster, 0.5)
        if contours:
            contour = max(contours, key=lambda x: len(x))  # Use the largest contour
            min_distance_index = np.argmin(np.linalg.norm(contour - contour[0], axis=1))
            contour = np.roll(contour, -min_distance_index, axis=0)
            return np.array(contour), raster, transform
        return None, None, None
    except Exception as e:
        logging.error(f"Error extracting contour: {e}")
        raise

def find_combined_bounds(polygon1, polygon2):
    """Find combined bounds of two polygons."""
    bounds1 = polygon1.bounds
    bounds2 = polygon2.bounds
    combined_bounds = (
        min(bounds1[0], bounds2[0])-10000,
        min(bounds1[1], bounds2[1])-10000,
        max(bounds1[2], bounds2[2])+10000,
        max(bounds1[3], bounds2[3])+10000
    )
    return combined_bounds

def resample_contour(contour, num_points):
    """Resample the contour to a fixed number of points."""
    try:
        length = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
        length = np.insert(length, 0, 0) / length[-1]
        interp_func = interp1d(length, contour, axis=0, kind='linear')
        return interp_func(np.linspace(0, 1, num_points))
    except Exception as e:
        logging.error(f"Error resampling contour: {e}")
        raise

def compute_similarity(gpkg_file_1, gpkg_file_2, num_points=1000, target_crs="EPSG:3857"):
    """Compute the similarity between two polygons using Fourier descriptors."""
    try:
        polygon1 = load_polygon(gpkg_file_1, target_crs)
        polygon2 = load_polygon(gpkg_file_2, target_crs)
        combined_bounds = find_combined_bounds(polygon1, polygon2)
        contour1, raster1, transform1 = extract_contour(polygon1, transform=rasterio.transform.from_bounds(*combined_bounds, *(1000, 1000)))
        contour2, raster2, transform2 = extract_contour(polygon2, transform=rasterio.transform.from_bounds(*combined_bounds, *(1000, 1000)))
        
        if contour1 is not None and contour2 is not None:
            contour1_resampled = resample_contour(contour1, num_points)
            contour2_resampled = resample_contour(contour2, num_points)
            
            dft1 = fft(contour1_resampled[:, 0] + 1j * contour1_resampled[:, 1])
            dft2 = fft(contour2_resampled[:, 0] + 1j * contour2_resampled[:, 1])

            similarity = np.abs(np.sum(np.conj(dft1) * dft2) / np.sqrt(np.sum(np.abs(dft1)**2) * np.sum(np.abs(dft2)**2)))

            return similarity, contour1, contour1_resampled, contour2, contour2_resampled, raster1, transform1, raster2, transform2
        else:
            raise ValueError("Contours could not be extracted.")
    except Exception as e:
        logging.error(f"Error computing similarity: {e}")
        raise

def plot_rasters(raster1, transform1, raster2, transform2):
    """Plot the rasters side by side."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    show(raster1, transform=transform1, ax=ax[0], title="Raster 1")
    show(raster2, transform=transform2, ax=ax[1], title="Raster 2")
    plt.show()

def plot_contours(contour1, contour1_resampled, contour2, contour2_resampled):
    """Plot the original and resampled contours for both polygons."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(contour1[:, 1], contour1[:, 0], label='Original Contour 1')
    ax[0].plot(contour1_resampled[:, 1], contour1_resampled[:, 0], label='Resampled Contour 1')
    ax[0].set_title('Contour 1')
    ax[0].legend()

    ax[1].plot(contour2[:, 1], contour2[:, 0], label='Original Contour 2')
    ax[1].plot(contour2_resampled[:, 1], contour2_resampled[:, 0], label='Resampled Contour 2')
    ax[1].set_title('Contour 2')
    ax[1].legend()

    plt.show()


# main.py

gpkg_file_1 = "example_data/Building_1.gpkg"
gpkg_file_2 = "example_data\Building_1_randomized_100_mm.gpkg"

try:
    similarity, contour1, contour1_resampled, contour2, contour2_resampled, raster1, transform1, raster2, transform2 = compute_similarity(gpkg_file_1, gpkg_file_2)
    print("Ähnlichkeit der Fourier-Deskriptoren:", similarity)
    
    # Plot the rasters
    plot_rasters(raster1, transform1, raster2, transform2)
    
    # Plot the contours
    plot_contours(contour1, contour1_resampled, contour2, contour2_resampled)
except Exception as e:
    print(f"Fehler: {e}")
