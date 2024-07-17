import geopandas as gpd
import numpy as np
import shapely.affinity as affinity
import random
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, MultiLineString, MultiPoint

def read_gpkg(file_path):
    """Reads a GeoPackage file and returns a GeoDataFrame."""
    return gpd.read_file(file_path)

def write_gpkg(gdf, file_path):
    """Writes a GeoDataFrame to a GeoPackage file."""
    gdf.to_file(file_path, driver="GPKG")

def scale(gdf, scale_factor_x, scale_factor_y):
    """
    Scale the GeoDataFrame by the given scale factors along the X and Y axes.

    Parameters:
        gdf (GeoDataFrame): Input GeoDataFrame to be scaled.
        scale_factor_x (float): Scale factor along the X axis.
        scale_factor_y (float): Scale factor along the Y axis.

    Returns:
        GeoDataFrame: Scaled GeoDataFrame.
    """
    # Copy the input GeoDataFrame
    scaled_gdf = gdf.copy()

    # Scale the geometries
    scaled_gdf.geometry = scaled_gdf.geometry.scale(scale_factor_x, scale_factor_y)

    return scaled_gdf

def shift(gdf, x_shift, y_shift):
    """Shifts the geometries in a GeoDataFrame by given offsets."""
    gdf_shifted = gdf.copy()
    gdf_shifted['geometry'] = gdf_shifted['geometry'].translate(xoff=x_shift, yoff=y_shift)
    return gdf_shifted

def rotate_centroid(gdf, angle_degrees):
    """Rotates the geometries in a GeoDataFrame around their centroids by a given angle (in degrees)."""
    gdf_rotated = gdf.copy()
    gdf_rotated['geometry'] = gdf_rotated['geometry'].rotate(angle_degrees, origin='centroid')
    return gdf_rotated

def rotate_point(gdf, angle_degrees, point):
    """Rotates the geometries in a GeoDataFrame around a given point by a given angle (in degrees)."""
    gdf_rotated = gdf.copy()
    gdf_rotated['geometry'] = gdf_rotated['geometry'].rotate(angle_degrees, origin=point)
    return gdf_rotated

def manipulate_coordinates(coords, x_range, y_range):
    # Example manipulation: Adding 1 to each coordinate
    manipulated_coords = [(x + random.uniform(-x_range, x_range), y + random.uniform(-y_range, y_range)) for x, y in coords]
    return manipulated_coords

def randomize(gdf, x_range, y_range):
    gdf_randomized = gdf.copy()
    for index, row in gdf_randomized.iterrows():
        geometry = row.geometry
        new_polygons = []
        if geometry.geom_type == 'MultiPolygon':
            for polygon in geometry.geoms:
                polygon_coords = manipulate_coordinates(list(polygon.exterior.coords), x_range, y_range)
                # Ensure the last coordinate is the same as the first
                polygon_coords[-1] = polygon_coords[0]
                new_polygons.append(Polygon(polygon_coords))
        elif geometry.geom_type == 'Polygon':
            polygon_coords = manipulate_coordinates(list(geometry.exterior.coords), x_range, y_range)
            # Ensure the last coordinate is the same as the first
            polygon_coords[-1] = polygon_coords[0]
            new_polygons.append(Polygon(polygon_coords))
        gdf_randomized.at[index, 'geometry'] = MultiPolygon(new_polygons)
    return gdf_randomized



# Example usage:
input_file = "Building_1.gpkg"
output_file = "output.gpkg"

# Read GeoPackage file
gdf = read_gpkg(input_file)

# Example manipulations
gdf_scaled = scale(gdf, 0.95, 0.95)
gdf_shifted = shift(gdf, 0.1, 0.1)
gdf_rotated_centroid = rotate_centroid(gdf, 1)
gdf_rotated_point = rotate_point(gdf, 1, (0, 0))
gdf_randomized = randomize(gdf, 0.1, 0.1)

# Write manipulated GeoDataFrames to GeoPackage files
write_gpkg(gdf_scaled, "scaled.gpkg")
write_gpkg(gdf_shifted, "shifted.gpkg")
write_gpkg(gdf_rotated_centroid, "rotated_centroid.gpkg")
write_gpkg(gdf_rotated_point, "rotated_point.gpkg")
write_gpkg(gdf_randomized, "randomized.gpkg")
