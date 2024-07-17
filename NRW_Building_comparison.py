import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, MultiPolygon, Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from comparison import jaccard_index, advanced_jaccard_index, find_rotated_mbr
import warnings
import pandas as pd
import turning_function
import math
import turning_function_new

# Ignore specific warning
warnings.filterwarnings("ignore", category=UserWarning)

# Load shapefile data
shp_file = "hu_EPSG25832_Shape/test_file_4.shp"
shp_gdf = gpd.read_file(shp_file)

def plot(shp_building, osm_building=None, AJI_building=None):
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

    if AJI_building is not None:
        AJI_building = AJI_building.to_crs(utm_projection)
        AJI_building.plot(ax=ax, edgecolor='#4daf4a', facecolor='none', alpha=0.5, linewidth=1)
        handles.append(Patch(edgecolor='#4daf4a', facecolor='none', alpha=0.5, linewidth=1))
        labels.append('OSM Data (Transformed)')

    ax.set_aspect('equal', adjustable='box')
    #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.0f} m'.format(x)))
    #ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '{:.0f} m'.format(y)))

    ax.legend(handles, labels)
    plt.show()
# Ensure the geometries are MultiPolygon
def ensure_multipolygon(geom):
    if geom.geom_type == 'Polygon':
        return MultiPolygon([geom])
    return geom

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
    geojson_building = None
    osm_data_found = False
    geojson_data_found = False

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
    '''
    # Check for corresponding buildings in GeoJSON
    geojson_candidates = geojson_gdf[geojson_gdf.contains(centroid)]
    if not geojson_candidates.empty:
        geojson_building = get_closest_building(geojson_candidates, centroid)
        geojson_data_found = True
    '''
    # Print status messages
    print(f"OSM data found: {osm_data_found}, GeoJSON data found: {geojson_data_found}")
    '''
    # Plot the buildings
    fig, ax = plt.subplots()
    handles = []
    labels = []

    if osm_building is not None:
        gdf = gpd.GeoDataFrame([osm_building], crs=osm_gdf.crs)
        gdf.plot(ax=ax, color='blue', alpha=0.5)
        handles.append(Patch(color='blue', alpha=0.5))
        labels.append('OSM Building')
    if geojson_building is not None:
        gdf = gpd.GeoDataFrame([geojson_building], crs=geojson_gdf.crs)
        gdf.plot(ax=ax, color='green', alpha=0.5)
        handles.append(Patch(color='green', alpha=0.5))
        labels.append('GeoJSON Building')

    gdf = gpd.GeoDataFrame([building], crs=shp_gdf.crs)
    gdf.plot(ax=ax, color='red', alpha=0.5)
    handles.append(Patch(color='red', alpha=0.5))
    labels.append('Shapefile Building')

    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.5f}'.format(x)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '{:.5f}'.format(y)))

    ax.legend(handles, labels)
    plt.show()
    '''
    # Return the original building and any found matches
    if osm_building is not None or geojson_building is not None:
        return building, osm_building
    return None

def geodataframe_to_coordinates_list(gdf):
    coordinates_list = []
    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            coordinates = list(geom.exterior.coords)
            coordinates_list.append([(x, y) for x, y in coordinates])
        elif geom.geom_type == 'MultiPolygon':
            for polygon in geom.geoms:
                coordinates = list(polygon.exterior.coords)
                coordinates_list.append([(x, y) for x, y in coordinates])
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
    return list1, list2_aligned

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
for idx, building in shp_gdf.iloc[0:1000].iterrows():
    result = find_corresponding_buildings(building)
    if result:
        shp_building, osm_building = result

        # Ensure the geometries are MultiPolygon
        shp_building.geometry = ensure_multipolygon(shp_building.geometry)
        if osm_building is not None:
            osm_building.geometry = ensure_multipolygon(osm_building.geometry)
        
        # Create temporary GeoDataFrames for comparison
        shp_building_gdf = gpd.GeoDataFrame([shp_building], crs=shp_gdf.crs)
        shp_building_gdf = shp_building_gdf.to_crs(utm_projection)
        if osm_building is not None:
            osm_building_gdf = gpd.GeoDataFrame([osm_building], crs=shp_gdf.crs)
            osm_building_gdf = osm_building_gdf.to_crs(utm_projection)
        else:
            osm_building_gdf = None

        # Calculate Jaccard indices
        limits_1 = {
            "translation_limit_1": 2,
            "translation_limit_2": 1,
            "rotation_limit_1": 3,
            "rotation_limit_2": 2,
            "scale_limit_1": 1.1,
            "scale_limit_2": 1.05
        }

        limits_2 = {
            "translation_limit_1": 2,
            "translation_limit_2": 1,
            "rotation_limit_1": 3,
            "rotation_limit_2": 2,
            "scale_limit_1": 1.05,
            "scale_limit_2": 1.02
        }
        limits_3 = {
            "translation_limit_1": 2,
            "translation_limit_2": 1,
            "rotation_limit_1": 3,
            "rotation_limit_2": 2,
            "scale_limit_1": 1.02,
            "scale_limit_2": 1.01
        }
        limits_4 = {
            "translation_limit_1": 2,
            "translation_limit_2": 1,
            "rotation_limit_1": 3,
            "rotation_limit_2": 2,
            "scale_limit_1": 1.01,
            "scale_limit_2": 1.005
        }
        limits_5 = {
            "translation_limit_1": 2,
            "translation_limit_2": 1,
            "rotation_limit_1": 3,
            "rotation_limit_2": 2,
            "scale_limit_1": 1.005,
            "scale_limit_2": 1.002
        }
        limits_6 = {
            "translation_limit_1": 2,
            "translation_limit_2": 1,
            "rotation_limit_1": 3,
            "rotation_limit_2": 2,
            "scale_limit_1": 4,
            "scale_limit_2": 3
        }
        utm_projection = 'EPSG:25832'
        '''
        if geojson_building_gdf is not None:
            jaccard_geojson = jaccard_index(shp_building_gdf.to_crs(utm_projection), geojson_building_gdf.to_crs(utm_projection))[0]
            advanced_jaccard_geojson, AJI_building, info_geojson = advanced_jaccard_index(shp_building_gdf.to_crs(utm_projection), geojson_building_gdf.to_crs(utm_projection), limits)
            print(f"Building {idx + 1}: Jaccard Index (Shapefile vs GeoJSON): {jaccard_geojson}")
            print(f"Building {idx + 1}: Advanced Jaccard Index (Shapefile vs GeoJSON): {advanced_jaccard_geojson}")
        '''
        if osm_building_gdf is not None:
            jaccard_osm = jaccard_index(shp_building_gdf.to_crs(utm_projection), osm_building_gdf.to_crs(utm_projection))[0]
            
            advanced_jaccard_osm, AJI_building, info = advanced_jaccard_index(shp_building_gdf, osm_building_gdf, limits_1, rotation_off=True)
            #except Exception as e:
            #    advanced_jaccard_osm, AJI_building, info = None, None, None
            #    print("Error in advanced_jaccard_index with limits_1:", e)
            '''
            try:
                advanced_jaccard_osm_lim_2, AJI_building_2, info_2 = advanced_jaccard_index(shp_building_gdf, osm_building_gdf, limits_2, rotation_off=True)
            except Exception as e:
                advanced_jaccard_osm_lim_2, AJI_building_2, info_2 = None, None, None
                print("Error in advanced_jaccard_index with limits_2:", e)

            try:
                advanced_jaccard_osm_lim_3, AJI_building_3, info_3 = advanced_jaccard_index(shp_building_gdf, osm_building_gdf, limits_3, rotation_off=True)
            except Exception as e:
                advanced_jaccard_osm_lim_3, AJI_building_3, info_3 = None, None, None
                print("Error in advanced_jaccard_index with limits_3:", e)

            try:
                advanced_jaccard_osm_lim_4, AJI_building_4, info_4 = advanced_jaccard_index(shp_building_gdf, osm_building_gdf, limits_4, rotation_off=True)
            except Exception as e:
                advanced_jaccard_osm_lim_4, AJI_building_4, info_4 = None, None, None
                print("Error in advanced_jaccard_index with limits_4:", e)

            try:
                advanced_jaccard_osm_lim_5, AJI_building_5, info_5 = advanced_jaccard_index(shp_building_gdf, osm_building_gdf, limits_5, rotation_off=True)
            except Exception as e:
                advanced_jaccard_osm_lim_5, AJI_building_5, info_5 = None, None, None
                print("Error in advanced_jaccard_index with limits_5:", e)
            try:
                advanced_jaccard_osm_lim_6, AJI_building_6, info_6 = advanced_jaccard_index(shp_building_gdf, osm_building_gdf, limits_6, rotation_off=True)
            except Exception as e:
                advanced_jaccard_osm_lim_6, AJI_building_6, info_6 = None, None, None
            #try:
            #    advanced_jaccard_osm_lim_7, AJI_building_7, info_7 = advanced_jaccard_index(shp_building_gdf.to_crs(utm_projection), osm_building_gdf.to_crs(utm_projection), limits_6, rotation_off=True)
            #except Exception as e:
            #    advanced_jaccard_osm_lim_7, AJI_building_7, info_7 = None, None, None
            #    print("Error in advanced_jaccard_index with limits_7:", e)
            
            print(f"Building {idx + 1}: Jaccard Index (Shapefile vs OSM): {jaccard_osm}")
            print(f"Building {idx + 1}: Advanced Jaccard Index (Shapefile vs OSM): {advanced_jaccard_osm}")
            '''
            '''
            if jaccard_osm-advanced_jaccard_osm > 0.05 and jaccard_osm > 0.7:
                plot(shp_building_gdf, osm_building_gdf, AJI_building = AJI_building)
            elif jaccard_osm-advanced_jaccard_osm < -0.3:
                plot(shp_building_gdf, osm_building_gdf, AJI_building = AJI_building)
            '''




            shp_coords = reverse_list(geodataframe_to_coordinates_list(shp_building_gdf))
            osm_coords = reverse_list(geodataframe_to_coordinates_list(osm_building_gdf))
            shp_mbr = find_rotated_mbr(shp_building_gdf)
            #plot_mbr(shp_building_gdf, shp_mbr)
            shp_mbr_coords = reverse_list(geodataframe_to_coordinates_list(shp_mbr))

            aligned_list1, aligned_list2 = align_coordinate_lists(shp_coords, osm_coords)
   
            aligned_list3, aligned_list4 = align_coordinate_lists(shp_coords, shp_mbr_coords)
            print(aligned_list1, aligned_list2)   
            distance_a_b, theta, ht_err, slope_err = turning_function.distance(aligned_list1, aligned_list2, brute_force_updates=False)
            print(aligned_list3, aligned_list4)
            distance_a_mbr, theta, ht_err, slope_err = turning_function.distance(aligned_list3, aligned_list4, brute_force_updates=True)
            print(ht_err, slope_err)
            if ht_err > 0.001:
                plot_coordinate_list(aligned_list1, aligned_list2)

            if slope_err > 0.001:
                plot_coordinate_list(aligned_list1, aligned_list2)
            rectangularity = shp_building_gdf.geometry.area.sum()/shp_mbr.geometry.area.sum()
            normalized_similarity = 1 - distance_a_b *((1-rectangularity)/(distance_a_mbr))
            

            #print((1-normalized_similarity)/distance_a_b)
            print((1-rectangularity)/distance_a_mbr)
            print(distance_a_mbr, theta, ht_err, slope_err)
            print(f"Building {idx + 1}: distance_a_mbr: {distance_a_mbr}")
            print(f"Building {idx + 1}: rectangularity: {rectangularity}")
            print(f"Building {idx + 1}: Similarity: {distance_a_b}")
            print(f"Building {idx + 1}: Normalized Similarity: {normalized_similarity}")
            #plot_mbr(shp_building_gdf, shp_mbr, osm_building_gdf, shp_mbr)
            distance_a_b_new = turning_function_new.calculate_similarity(aligned_list1[0:-1], aligned_list2[0:-1])
            distance_a_mbr_new = turning_function_new.calculate_similarity(aligned_list1[0:-1], aligned_list4[0:-1])
            #if distance_a_mbr_new < 0.01:
            #    normalized_similarity_new = '-inf'
            #else:     
            normalized_similarity_new = 1 - distance_a_b_new *((1-rectangularity)/distance_a_mbr_new)

            print(f"Building {idx + 1}: distance_a_mbr: {distance_a_mbr_new}")
            print(f"Building {idx + 1}: Similarity: {distance_a_b_new}")
            print(f"Building {idx + 1}: Normalized Similarity: {normalized_similarity_new}")
            #turning_function_new.plot_turning_functions(aligned_list1[0:-1], aligned_list2[0:-1])
            #turning_function_new.plot_turning_functions(aligned_list1[0:-1], aligned_list4[0:-1])
            #
            plot_coordinate_list(aligned_list1, aligned_list2)
            plot_coordinate_list(aligned_list3, aligned_list4)

        plot(shp_building_gdf, osm_building_gdf, AJI_building)
        # Append results

        results.append({
            'id': idx,
            'stand. Jaccard': jaccard_osm if osm_building_gdf is not None else None,
            'Set 1': advanced_jaccard_osm if osm_building_gdf is not None else None,
            #'Set 2': advanced_jaccard_osm_lim_2 if osm_building_gdf is not None else None,
            #'Set 3': advanced_jaccard_osm_lim_3 if osm_building_gdf is not None else None,
            #'Set 4': advanced_jaccard_osm_lim_4 if osm_building_gdf is not None else None,
            #'Set 5': advanced_jaccard_osm_lim_5 if osm_building_gdf is not None else None,
            #'Set 6': advanced_jaccard_osm_lim_6 if osm_building_gdf is not None else None,
            'scaling det.': info[4] if osm_building_gdf is not None else None,
            'dangle_det.': info[6] if osm_building_gdf is not None else None,
            'dx det.': info[8] if osm_building_gdf is not None else None,
            'dy det.': info[9] if osm_building_gdf is not None else None,
            'similarity': distance_a_b if distance_a_b is not None else None,
            'normalized similarity': normalized_similarity if normalized_similarity is not None else None,
            'similarity new': distance_a_b_new if distance_a_b_new is not None else None,
            'normalized similarity new': normalized_similarity_new if normalized_similarity_new is not None else None,
            'rectangularity': rectangularity if rectangularity is not None else None,
            'distance_a_mbr': distance_a_mbr if distance_a_mbr is not None else None
        })


    # Print progress
    #print(f"{idx + 1} out of {total_buildings} objects were checked")

# Create a DataFrame from results
results_df = pd.DataFrame(results)

# Save results to Excel
results_df.to_excel("building_comparison_results.xlsx", index=False)

