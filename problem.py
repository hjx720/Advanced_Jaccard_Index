import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, MultiPolygon, Polygon, MultiPoint
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import warnings
import pandas as pd
import turning_function
import math
import turning_function_new
import numpy as np
from shapely.affinity import scale

# Ignore specific warning
#warnings.filterwarnings("ignore", category=UserWarning)

# Load shapefile data
shp_file = "hu_EPSG25832_Shape/test_file_4.shp"
shp_gdf = gpd.read_file(shp_file)

def plot(shp_building, osm_building=None, geojson_building=None, AJI_building=None):
    fig, ax = plt.subplots()
    handles = []
    labels = []

    # Define a UTM projection or any other projection with metric units
    utm_projection = 'EPSG:25832'  # Example for UTM zone 33N, adjust as needed

    # Reproject shapefile building
    shp_building = shp_building.to_crs(utm_projection)
    shp_building.plot(ax=ax, edgecolor='black', facecolor='black', alpha=0.3, linewidth=1)
    handles.append(Patch(edgecolor='black', facecolor='black', alpha=0.3, linewidth=1))
    labels.append('Catastre Data')

    if osm_building is not None:
        osm_building = osm_building.to_crs(utm_projection)
        osm_building.plot(ax=ax, edgecolor='#377eb8', facecolor='none', alpha=0.5, linewidth=1)
        handles.append(Patch(edgecolor='#377eb8', facecolor='none', alpha=0.5, linewidth=1))
        labels.append('OSM Data')

    if geojson_building is not None:
        geojson_building = geojson_building.to_crs(utm_projection)
        geojson_building.plot(ax=ax, edgecolor='green', facecolor='none', alpha=0.5, linewidth=1)
        handles.append(Patch(edgecolor='green', facecolor='none', alpha=0.5, linewidth=1))
        labels.append('GeoJSON Building')

    if AJI_building is not None:
        AJI_building = AJI_building.to_crs(utm_projection)
        AJI_building.plot(ax=ax, edgecolor='#4daf4a', facecolor='none', alpha=0.5, linewidth=1)
        handles.append(Patch(edgecolor='#4daf4a', facecolor='none', alpha=0.5, linewidth=1))
        labels.append('OSM Data (Procrustes Transformed)')

    ax.set_aspect('equal', adjustable='box')
    #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.0f} m'.format(x)))
    #ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '{:.0f} m'.format(y)))

    ax.legend(handles, labels)
    plt.show()

def scale_gdf(gdf, scale_factor_x, scale_factor_y):
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

# Function to find the closest building containing the centroid
def get_closest_building(buildings, centroid):
    if buildings.empty:
        return None
    buildings = buildings.copy()
    buildings.loc[:, 'distance'] = buildings.geometry.distance(centroid)
    closest_building = buildings.loc[buildings['distance'].idxmin()]
    return closest_building

def plot_mbr(gdf_a, mbr_a, gdf_b= None, mbr_b=None):
    # Create a plot with a specific size
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams['hatch.linewidth'] = 1.0  # Increase the hatch line width

    # Plot each GeoDataFrame with hatching
    gdf_a.plot(ax=ax, color='cornflowerblue', edgecolor='cornflowerblue', label='Building A Footprint', alpha=0.5, zorder=1)
    #gdf_b.plot(ax=ax, color='lightcoral', edgecolor='lightcoral', label='Building B Footprint', alpha=0.5, zorder=1)

    # Plot the boundaries of the MBRs
    mbr_a.boundary.plot(ax=ax, edgecolor='blue', label='Minimum Bounding Box', zorder=2)
    #mbr_b.boundary.plot(ax=ax, edgecolor='red', label='Minimum Bounding Box', zorder=2)

    # Create custom legend handles
    handles = [
        Patch(facecolor='cornflowerblue', edgecolor='cornflowerblue', label='Building A Footprint', alpha=0.5),
        Patch(edgecolor='blue', facecolor='none', label='Minimum Bounding Box A'),
        Patch(facecolor='lightcoral', edgecolor='lightcoral', label='Building B Footprint', alpha=0.5),
        Patch(edgecolor='red', facecolor='none', label='Minimum Bounding Box B')
    ]



    # Add the arrow to the legend using the custom handler
    ax.legend(handles=handles)

    # Set plot title and labels
    plt.title('Minimum Bounding Box')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Display the plot
    plt.show()

# Function to find corresponding buildings in other datasets
def find_corresponding_buildings(building):
    centroid = building.geometry.centroid

    osm_building = None


    try:
        # Fetch OSM data
        osm_gdf = ox.features_from_point((centroid.y, centroid.x), tags={"building": True}, dist=5)

        # Convert OSM data to GeoDataFrame and filter for buildings
        osm_gdf = osm_gdf[osm_gdf.geometry.type == 'Polygon']

        # Check for corresponding buildings in OSM
        osm_candidates = osm_gdf[osm_gdf.contains(centroid)]
        if not osm_candidates.empty:
            osm_building = get_closest_building(osm_candidates, centroid)
            osm_data_found = True
    except ox._errors.InsufficientResponseError:
        print(f"No OSM data found for centroid at {centroid}")

    # Return the original building and any found matches
    if osm_building is not None:
        return building, osm_building
    return None

def find_rotated_mbr(gdf):
    # Extract all points from the geometries
    all_points = []
    for geom in gdf.geometry:
        print(geom)
        for p in geom.exterior.coords:
            print(p)
            all_points.append(p)

    multi_point = MultiPoint(all_points)
    
    # Compute the minimum oriented rectangle
    rectangle = multi_point.minimum_rotated_rectangle
    
    return rectangle

def geodataframe_to_coordinates_list(gdf):
    coordinates_list = []
    coordinates_list.extend(gdf.exterior.coords.xy)
        
    return coordinates_list[0]

def find_closest_point(ref_point, points):
    min_dist = float('inf')
    closest_point = None
    closest_index = 0
    for i, point in enumerate(points):
        dist = math.sqrt((ref_point[0] - point[0]) ** 2 + (ref_point[1] - point[1]) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest_point = point
            closest_index = i
    return closest_point, closest_index

def rotate_list_to_start_with_point(points, start_index):
    return points[start_index:] + points[:start_index]

def compute_area(points):
    n = len(points)
    area = 0.0
    for i in range(n - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        area += (x1 * y2 - x2 * y1)
    return area / 2.0

def align_coordinate_lists(list1, list2):
    if not list1 or not list2:
        return list1, list2
    area1 = compute_area(list1)
    area2 = compute_area(list2)
    
    if (area1 < 0 and area2 > 0) or (area1 > 0 and area2 < 0):
        list2 = list2[::-1]
    
    list2 = list2[0:-1]
    
    ref_point = list1[0]
    _, closest_index = find_closest_point(ref_point, list2)
    list2_aligned = rotate_list_to_start_with_point(list2, closest_index)
    list2_aligned.append(list2_aligned[0])
    return list2_aligned

def reverse_list(lst):
    return lst[::-1]

def plot_coordinate_list(list1, list2):
    # Extracting x and y coordinates
    x1_new, y1_new = zip(*list1)
    x2_new, y2_new = zip(*list2)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x1_new, y1_new, marker='o', linestyle='-', color='b', label='List 1')
    plt.plot(x2_new, y2_new, marker='x', linestyle='--', color='r', label='List 2')

    # Highlighting first points
    plt.scatter(x1_new[0], y1_new[0], color='blue', s=100, zorder=5)
    plt.scatter(x2_new[0], y2_new[0], color='red', s=100, zorder=5)

    for i, (x, y) in enumerate(zip(x1_new, y1_new)):
        plt.text(x, y, f'1_{i}', fontsize=12, color='blue', ha='right')

    for i, (x, y) in enumerate(zip(x2_new, y2_new)):
        plt.text(x, y, f'2_{i}', fontsize=12, color='red', ha='left')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Plot of New Coordinate Lists')
    plt.legend()
    plt.grid(True)

    # Set equal scaling
    plt.axis('equal')

    plt.show()
    
# Initialize a list to store results
results = []
utm_projection = 'EPSG:25832'
# Iterate through the first 100 buildings in the shapefile
total_buildings = min(len(shp_gdf), 1000)
for idx, building in shp_gdf.iloc[224:999].iterrows():
    result = find_corresponding_buildings(building)
    if result:
        shp_building, osm_building = result

        # Ensure the geometries are MultiPolygon
        
        # Create temporary GeoDataFrames for comparison
        shp_building_gdf = gpd.GeoDataFrame([shp_building], crs=shp_gdf.crs)
        shp_building_gdf = shp_building_gdf.to_crs(utm_projection)
        if osm_building is not None:
            osm_building_gdf = gpd.GeoDataFrame([osm_building], crs=shp_gdf.crs)
            osm_building_gdf = osm_building_gdf.to_crs(utm_projection)
        else:
            osm_building_gdf = None

        if osm_building_gdf is not None:
            #turn the gdf into coordinate lists
            print(type(shp_building_gdf.geometry.iloc))
            x, y = shp_building_gdf.geometry.iloc[0].exterior.coords.xy
            shp_coords = reverse_list(list(zip(x, y)))

            x, y = osm_building_gdf.geometry.iloc[0].exterior.coords.xy
            osm_coords = list(zip(x, y))
            
            #define mbr and turn into coordinate list
            shp_mbr = find_rotated_mbr(shp_building_gdf)
            x, y = shp_mbr.exterior.coords.xy
            shp_mbr_coords = reverse_list(list(zip(x, y)))
            #print(shp_coords, osm_coords, shp_mbr_coords)
            #align the coordinate lists osm_coords and shp_mbr_coords to shp_coords in order to prepare them for turning_function
            #they need to start at "same" coordinate
            #osm_coords_aligned = align_coordinate_lists(shp_coords, osm_coords)
            print(osm_coords)
            osm_coords_aligned = reverse_list(osm_coords[3:]+osm_coords[:4])#+[osm_coords_aligned[7]]
            shp_mbr_coords_aligned = align_coordinate_lists(shp_coords, shp_mbr_coords)
            
            #calculating turning distance for Distanz(A, B) und Distanz(A, MBR_A)
            distance_a_b, theta, ht_err, slope_err = turning_function.distance(shp_coords, osm_coords_aligned, brute_force_updates=True)
            distance_a_b_n, theta, ht_err, slope_err = turning_function.distance(shp_coords, osm_coords, brute_force_updates=True)
            distance_a_mbr, theta, ht_err, slope_err = turning_function.distance(shp_coords, shp_mbr_coords_aligned, brute_force_updates=True)
            print(distance_a_b, distance_a_b_n)
            #Calculating Rectangularity
            # Calculate the area of the building
            building_area = shp_building_gdf.geometry.area.iloc[0]
            #building_area = np.array([building_area])
            print(building_area)
            # Calculate the area of the minimum bounding rectangle (MBR)
            mbr_area = shp_mbr.area
            #mbr_area = np.array([mbr_area])
            print(mbr_area)
            # Calculate rectangularity
            #rectangularity_16 = building_area.astype(np.float16) / mbr_area.astype(np.float16)
            rectangularity_64 = building_area / mbr_area
            #print(rectangularity_16)
            print(rectangularity_64)


            #Normalize the Similarity
            normalized_similarity = 1 - distance_a_b *((1-rectangularity_64)/distance_a_mbr)

            #print((1-normalized_similarity)/distance_a_b)
            #print((1-rectangularity)/distance_a_mbr)
            
            print(f"Building {idx + 1}: distance A MBR_A: {distance_a_mbr}")
            print(f"Building {idx + 1}: rectangularity: {rectangularity_64}")
            print(f"Building {idx + 1}: Similarity A B: {distance_a_b}")
            print(f"Building {idx + 1}: Normalized Similarity A B: {normalized_similarity}")
            #plot_mbr(shp_building_gdf, shp_mbr, osm_building_gdf, shp_mbr)

            #using turning_function_new to do the same
            #calculating turning distance for Distanz(A, B) und Distanz(A, MBR_A)
            distance_a_b_v2 = turning_function_new.calculate_similarity(shp_coords[0:-1], osm_coords_aligned[0:-1])
            distance_a_mbr_v2 = turning_function_new.calculate_similarity(shp_coords[0:-1], shp_mbr_coords_aligned[0:-1])

            #Normalize the Similarity
            normalized_similarity_v2 = 1 - distance_a_b_v2 *((1-rectangularity_64)/distance_a_mbr_v2)

            print(f"Building {idx + 1}: distance_a_mbr: {distance_a_mbr_v2}")
            print(f"Building {idx + 1}: Similarity A B: {distance_a_b_v2}")
            print(f"Building {idx + 1}: Normalized Similarity: {normalized_similarity_v2}")
            print((1-rectangularity_64)/distance_a_mbr)
            #plot coordintes to check if alignement worked
            plot_coordinate_list(shp_coords, osm_coords_aligned)
            #Plot Turning Function A und B
            turning_function_new.plot_turning_functions(shp_coords, osm_coords_aligned)
            #plot coordintes to check if alignement worked
            plot_coordinate_list(shp_coords, shp_mbr_coords_aligned)
            #Plot Turning Function A und MBR_A
            turning_function_new.plot_turning_functions(shp_coords, shp_mbr_coords_aligned)

            

            #plot comparison polygons
            plot(shp_building_gdf, osm_building_gdf)
        
        # Append results

        results.append({
            'id': idx,
            'similarity': distance_a_b if distance_a_b is not None else None,
            'normalized similarity': normalized_similarity if normalized_similarity is not None else None,
            'similarity new': distance_a_b_v2 if distance_a_b_v2 is not None else None,
            'normalized similarity new': normalized_similarity_v2 if normalized_similarity_v2 is not None else None
        })



# Create a DataFrame from results
results_df = pd.DataFrame(results)

# Save results to Excel
results_df.to_excel("building_comparison_results.xlsx", index=False)

