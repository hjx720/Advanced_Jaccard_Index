import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import random
import manipulate

def create_gpkg_file(polygons, name):
    # Create a list to hold Polygon objects
    poly_list = []
    #print(polygons)

    # Iterate over the polygons provided
    for points in polygons:
        # Create a Polygon object from the provided points
        polygon = Polygon(points)
        poly_list.append(polygon)

    # Create a single MultiPolygon object from the list of Polygons
    multi_polygon = MultiPolygon(poly_list)

    # Create a GeoDataFrame with the MultiPolygon as the geometry
    gdf = gpd.GeoDataFrame(geometry=[multi_polygon])
    coordinates_count = sum(
            sum(len(poly.exterior.coords) for poly in geom.geoms) if geom.geom_type == 'MultiPolygon' else len(geom.exterior.coords)
            for geom in gdf.geometry
        )
    print(coordinates_count)
    gdf_scaled_080 = manipulate.scale(gdf, 0.80, 0.80)
    gdf_scaled_095 = manipulate.scale(gdf, 0.95, 0.95)
    gdf_scaled_099 = manipulate.scale(gdf, 0.99, 0.99)
    gdf_scaled_101 = manipulate.scale(gdf, 1.01, 1.01)
    gdf_scaled_105 = manipulate.scale(gdf, 1.05, 1.05)
    gdf_scaled_120 = manipulate.scale(gdf, 1.20, 1.20)

    gdf_shifted_5cm = manipulate.shift(gdf, 0.05, 0.05)
    gdf_shifted_10cm = manipulate.shift(gdf, 0.1, 0.1)
    gdf_shifted_50cm = manipulate.shift(gdf, 0.5, 0.5)
    gdf_shifted_100cm = manipulate.shift(gdf, 1, 1)

    gdf_rotated_05gon = manipulate.rotate_centroid(gdf, 0.5*0.9)
    gdf_rotated_1gon = manipulate.rotate_centroid(gdf, 1*0.9)
    gdf_rotated_2gon = manipulate.rotate_centroid(gdf, 2*0.9)
    gdf_rotated_5gon = manipulate.rotate_centroid(gdf, 5*0.9)
    gdf_rotated_20gon = manipulate.rotate_centroid(gdf, 20*0.9)

    gdf_manipulated_low = manipulate.rotate_centroid(manipulate.shift(manipulate.scale(gdf, 1.01, 1.01), 0.05, 0.05), 0.5*0.9) 
    gdf_manipulated_medium = manipulate.rotate_centroid(manipulate.shift(manipulate.scale(gdf, 1.05, 1.05), 0.1, 0.1), 1*0.9) 
    gdf_manipulated_high = manipulate.rotate_centroid(manipulate.shift(manipulate.scale(gdf, 1.10, 1.10), 0.5, 0.5), 2*0.9) 
    gdf_manipulated_very_high = manipulate.rotate_centroid(manipulate.shift(manipulate.scale(gdf, 1.20, 1.20), 1, 1), 20*0.9) 
    
    # Dictionary to store modified variables

    # Get a list of variables to modify
    vars_to_randomize = [name for name in locals() if name.startswith('gdf')]


    for var_name in vars_to_randomize:
        locals()[var_name + '_randomized_5_mm'] = manipulate.randomize(locals()[var_name], 0.005, 0.005)
        locals()[var_name + '_randomized_10_mm'] = manipulate.randomize(locals()[var_name], 0.01, 0.01)
        locals()[var_name + '_randomized_50_mm'] = manipulate.randomize(locals()[var_name], 0.05, 0.05)
        locals()[var_name + '_randomized_100_mm'] = manipulate.randomize(locals()[var_name], 0.1, 0.1)
        #modified_var = apply_modification(var_value)
        #locals()[var_name] = modified_var
        #print(manipulate.randomize(locals()[var_name], 0.01, 0.01))
    
    vars_to_save = [name for name in locals() if name.startswith('gdf')]

    for var_name in vars_to_save:
        filename = 'example_data/'+name+var_name[3:]+'.gpkg'
        locals()[var_name].to_file(filename, driver="GPKG")
        #print(var_name, locals()[var_name])
    
    
    """
    variables = [xxx_scaled, xxx_shifted, xxx_rotated]

    for i, variable in enumerate(variables):
    """
    # Specify the GeoPackage file path
    name = name + '.gpkg'
    geopackage_file = name

    # Save the GeoDataFrame to a GeoPackage file
    gdf.to_file(geopackage_file, driver="GPKG")

    #print(f"GeoPackage file '{geopackage_file}' created successfully.")

# Example usage:
#points = [((0, 0), (2, 0), (2, 1), (0, 1))]  # Example square coordinates
#name = "basic_square_4"
#create_shape_file(points, name)

building_footprints = [
    # Original Buildings
    ("Building_1", [[(5, 5), (20, 10), (15, 25), (0, 20)]]),  # Building 1: Rotated Rectangle
    ("Building_2", [[(30, 10), (50, 10), (50, 20), (40, 20), (40, 30), (20, 30), (20, 20), (30, 20), (30, 10)]]),  # Building 2: L-shaped
    ("Building_3", [[(65, 15), (90, 0), (80, 25), (60, 35), (75, 20)]]),  # Building 3: Irregular shape with rotation
    ("Building_4", [[(85, 0), (100, 5), (110, 20), (95, 15)]]),  # Building 4: Trapezoidal shape
    ("Building_5", [[(10, 10), (30, 10), (30, 20), (10, 20)],[(15, 25), (20, 25), (20, 30), (15, 30)]])
]

for footprint in building_footprints:

    create_gpkg_file(footprint[1], footprint[0])


    