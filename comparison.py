import geopandas as gpd
import numpy as np
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, LineString
import math
from shapely.affinity import rotate
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
def shift_gdf(gdf, shift_x, shift_y):
    """
    Shift the GeoDataFrame along the X and Y axes.

    Parameters:
        gdf (GeoDataFrame): Input GeoDataFrame to be shifted.
        shift_x (float): Shift distance along the X axis.
        shift_y (float): Shift distance along the Y axis.

    Returns:
        GeoDataFrame: Shifted GeoDataFrame.
    """
    # Copy the input GeoDataFrame
    shifted_gdf = gdf.copy()

    # Shift the geometries
    shifted_gdf.geometry = shifted_gdf.geometry.translate(xoff=shift_x, yoff=shift_y)

    return shifted_gdf

def rotate_gdf(gdf, angle, pivot_point=None):
    """
    Rotate the GeoDataFrame by the given angle around the pivot point.

    Parameters:
        gdf (GeoDataFrame): Input GeoDataFrame to be rotated.
        angle (float): Rotation angle in degrees (positive angle indicates counterclockwise rotation).
        pivot_point (Point or tuple): Pivot point for rotation. If None, the centroid of the geometry will be used.

    Returns:
        GeoDataFrame: Rotated GeoDataFrame.
    """

    # Copy the input GeoDataFrame
    rotated_gdf = gdf.copy()

    # If pivot point is not provided, use the centroid of the geometry
    if pivot_point is None:
        pivot_point = gdf.geometry.centroid.values[0]

    # Rotate the geometries
    rotated_geometries = [rotate(geom, angle, origin=pivot_point) for geom in rotated_gdf.geometry]
    rotated_gdf.geometry = rotated_geometries

    return rotated_gdf

# Example usage
# Assuming gdf is your GeoDataFrame and angle is the rotation angle in degrees
# rotated_gdf = rotate_gdf(gdf, angle)
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
def jaccard_index(gdf_a, gdf_b):
    # Calculate intersection
    intersection = gpd.overlay(gdf_a, gdf_b, how='intersection')

    # Calculate areas
    area_a = gdf_a.geometry.area.sum()
    area_b = gdf_b.geometry.area.sum()
    intersection_area = intersection.geometry.area.sum()

    # Calculate areas exclusive to A and B
    only_a = area_a - intersection_area
    only_b = area_b - intersection_area

    # Calculate Jaccard index
    jaccard_index = intersection_area / (area_a + area_b - intersection_area)
    return jaccard_index, intersection_area, only_a, only_b

def find_rotated_mbr(gdf):
    # Find the minimum area bounding rectangle for each polygon geometry
    mabr_geometries = []
    for geom in gdf.geometry:
        mabr_geometries.append(geom.minimum_rotated_rectangle)
    
    # Combine the resulting geometries into a MultiPolygon
    mabr_multipolygon = MultiPolygon(mabr_geometries)
    
    # Create a GeoDataFrame with the minimum bounding rectangle
    mabr_gdf = gpd.GeoDataFrame(geometry=[mabr_multipolygon], crs=gdf.crs)
    
    return mabr_gdf

# Function to calculate the length of a line segment
def length_of_line_segment(line):
    dx = line[1][0] - line[0][0]
    dy = line[1][1] - line[0][1]
    return np.sqrt(dx**2 + dy**2)

# Function to determine the coordinates of the longest side of the MBR
def longest_side_coordinates(mbr_series):
    # Extract the MultiPolygon from the GeoDataFrame
    mabr_multipolygon = mbr_series.iloc[0].geometry
    
    # Initialize variables to store longest side information
    max_length = 0
    longest_side_coords = None
    
    # Iterate over each polygon in the MultiPolygon
    for mbr in mabr_multipolygon.geoms:
        # Extract exterior coordinates of the MBR
        coords = list(mbr.exterior.coords)
        
        # Calculate lengths of line segments and find the longest side
        for i in range(len(coords) - 1):
            length = length_of_line_segment((coords[i], coords[i + 1]))
            if length > max_length:
                max_length = length
                longest_side_coords = (coords[i], coords[i + 1])
    
    return longest_side_coords

# Function to calculate the angle of a line segment
def angle_of_line_segment(line):
    dx = line[1][0] - line[0][0]
    dy = line[1][1] - line[0][1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle

def plot_gdfs(gdf1, gdf2, figsize=(10, 6), title=None, legend_labels=('GDF 1', 'GDF 2', 'GDF 3')):
    """
    Plot three GeoDataFrames on the same plot with transparent colors.

    Parameters:
        gdf1 (GeoDataFrame): First GeoDataFrame to plot.
        gdf2 (GeoDataFrame): Second GeoDataFrame to plot.
        gdf3 (GeoDataFrame): Third GeoDataFrame to plot.
        figsize (tuple): Size of the figure (width, height). Default is (10, 6).
        title (str): Title of the plot. Default is None.
        legend_labels (tuple): Labels for the legend. Default is ('GDF 1', 'GDF 2', 'GDF 3').

    Returns:
        None
    """
    # Create a new figure
    plt.figure(figsize=figsize)

    # Plot GeoDataFrame 1 with transparent color
    gdf1.plot(ax=plt.gca(), color='blue', alpha=0.5, label=legend_labels[0])

    # Plot GeoDataFrame 2 with transparent color
    gdf2.plot(ax=plt.gca(), color='red', alpha=0.5, label=legend_labels[1])

    # Plot GeoDataFrame 3 with transparent color
    #gdf3.plot(ax=plt.gca(), color='green', alpha=0.5, label=legend_labels[2])

    # Add title and legend
    if title:
        plt.title(title)
    plt.legend()

    # Show plot
    plt.show()

def compare_geopackages(file_path_a, file_path_b, limits, rotation_off=False):
    # Read GeoPackage files
    gdf_a = gpd.read_file(file_path_a)
    gdf_b = gpd.read_file(file_path_b)

    jaccard_index_original = jaccard_index(gdf_a,gdf_b)

    # Translation

    ## getting centroids

    centroid_a = gdf_a.centroid
    centroid_b = gdf_b.centroid

    ## calculate deltas

    dx = centroid_b.x[0] - centroid_a.x[0]  # Difference in X coordinates
    dy = centroid_b.y[0] - centroid_a.y[0]  # Difference in Y coordinates

    ## check wether delta are in limits
    dx_determined = dx
    if dx == 0:
        translation_x_note = "No need for X-Axis translation was determind"
    elif limits["translation_limit_2"] < dx <= limits["translation_limit_1"]:
        dx = limits["translation_limit_2"]
        translation_x_note = "limit 2 exceeded --> maximum translation of "+ str(limits["translation_limit_2"]) +" used"
    elif limits["translation_limit_1"] < dx:
        dx = 0
        translation_x_note = "limit 1 exceeded --> no translation"
    elif dx <= limits["translation_limit_2"]:
        translation_x_note = "full X-Axis translation of " + str(dx) + " used"

    dy_determined = dy
    if dy == 0:
        translation_y_note = "No need for Y-Axis translation was determind"
    elif limits["translation_limit_2"] < dy <= limits["translation_limit_1"]:
        dy = limits["translation_limit_2"]
        translation_y_note = "limit 2 exceeded --> maximum translation of "+ str(limits["translation_limit_2"]) +" used"
    elif limits["translation_limit_1"] < dy:
        dy = 0
        translation_y_note = "limit 1 exceeded --> no translation"
    elif dy <= limits["translation_limit_2"]:
        translation_y_note = "full Y-Axis translation of " + str(dy) + " used"
    
    ## translate gdf_b
    gdf_b_translation = shift_gdf(gdf_b, -dx, -dy)

    ## calculate new jaccard index
    jaccard_index_translation = jaccard_index(gdf_a,gdf_b_translation)
    
    # Ratation

    ## define minimum boundign rectangle
    mbr_a = find_rotated_mbr(gdf_a)
    mbr_b = find_rotated_mbr(gdf_b)

    ## define longest side of minimum boundign rectangle
    longest_side_a = longest_side_coordinates(mbr_a)
    longest_side_b = longest_side_coordinates(mbr_b)

    ## define angles of longes side
    angle_longest_side_a = angle_of_line_segment(longest_side_a)
    angle_longest_side_b = angle_of_line_segment(longest_side_b)

    ## define the delta between angles
    delta_angle = angle_longest_side_a-angle_longest_side_b
    if delta_angle < -90:
        delta_angle += 180
    elif delta_angle > 90:
        delta_angle -= 180

    
    delta_angle_determined = delta_angle
    ## check wether delta are in limits
    if delta_angle == 0:
        rotation_note = "No need for rotation was determined"
    elif limits["rotation_limit_2"] < abs(delta_angle) < limits["rotation_limit_1"]:
        delta_angle = limits["rotation_limit_2"] if delta_angle > 0 else -limits["rotation_limit_2"]
        rotation_note = "rotation limit 2 exceeded --> maximum rotation of " + str(delta_angle) + " used"
    elif abs(delta_angle) > limits["rotation_limit_1"]:
        delta_angle = 0
        rotation_note = "rotation limit 1 exceeded --> no rotation"
    elif abs(delta_angle) < limits["rotation_limit_2"]:
        rotation_note = "full rotation of " + str(delta_angle) + " used"

    ##
    gdf_b_rotate = rotate_gdf(gdf_b_translation, delta_angle)

    
    ## calculate new jaccard index
    jaccard_index_rotation = jaccard_index(gdf_a,gdf_b_rotate)

    if rotation_off == True:
        gdf_b_rotate = gdf_b_translation
    # Scaling
    area_a = gdf_a.geometry.area.sum()
    area_b = gdf_b_rotate.geometry.area.sum()
    
    scaling_factor = (area_a/area_b)**0.5
    scaling_factor_determined = scaling_factor
    if scaling_factor == 1:
        scaling_note = "No need for scaling was determined"
    elif (2-limits["scale_limit_2"]) < scaling_factor < limits["scale_limit_2"]:
        scaling_note = "full scaling of " + str(scaling_factor) + " used"
    elif (2-limits["scale_limit_1"]) < scaling_factor < (2-limits["scale_limit_2"]):
        scaling_factor = 2-limits["scale_limit_2"]
        scaling_note = "scale limit 2 exceeded --> maximum scaling of " + str(2-limits["scale_limit_2"]) + " used"
    elif (limits["scale_limit_2"]) < scaling_factor < (limits["scale_limit_1"]):
        scaling_factor = limits["scale_limit_2"]
        scaling_note = "scale limit 2 exceeded --> maximum scaling of " + str(limits["scale_limit_2"]) + " used"
    else:
        scaling_factor = 1
        scaling_note = "scaling limit 1 exceeded --> no scaling"

    gdf_b_scaling = scale_gdf(gdf_b_rotate, scaling_factor, scaling_factor)
    jaccard_index_scaling = jaccard_index(gdf_a,gdf_b_scaling)


    #plot_gdfs(gdf_a, gdf_b, gdf_b_rotate, title="Comparison of GeoDataFrames")
    # Print results
    results = {
    "Before Transformation": {
        "data A": gdf_a,
        "data B": gdf_b,        
        "Intersection area": jaccard_index_original[1],
        "Area only in A": jaccard_index_original[2],
        "Area only in B": jaccard_index_original[3],
        "Jaccard_Coefficient": jaccard_index_original[0]
    },
    "After Translation": {
        "data B": gdf_b_translation,
        "Translation Note X Axis": translation_x_note,
        "Determined X Translation": dx_determined,
        "X Translation used": dx,
        "Translation Note Y Axis": translation_y_note,
        "Determined Y Translation": dy_determined,
        "Y Translation used": dy,
        "Intersection area": jaccard_index_translation[1],
        "Area only in A": jaccard_index_translation[2],
        "Area only in B": jaccard_index_translation[3],
        "Jaccard_Coefficient": jaccard_index_translation[0]
    },
    "After Rotation": {
        "data B": gdf_b_rotate,
        "Rotation Note": rotation_note,
        "Determined Rotation Angle": delta_angle_determined,
        "Rotation Angle used": delta_angle,
        "Intersection area": jaccard_index_rotation[1],
        "Area only in A": jaccard_index_rotation[2],
        "Area only in B": jaccard_index_rotation[3],
        "Jaccard_Coefficient": jaccard_index_rotation[0]
    },
    "After Scaling": {
        "data B": gdf_b_scaling,
        "Scaling Note": scaling_note,
        "Determined Scaling Factor": scaling_factor_determined,
        "Scaling Factor used": scaling_factor,
        "Intersection area": jaccard_index_scaling[1],
        "Area only in A": jaccard_index_scaling[2],
        "Area only in B": jaccard_index_scaling[3],
        "Jaccard_Coefficient": jaccard_index_scaling[0]
    }
}

    return results

# Example usage
limits = {
    "translation_limit_1": 0.1,
    "translation_limit_2": 0.05,
    "rotation_limit_1": 4,
    "rotation_limit_2": 2,
    "scale_limit_1": 1.1,
    "scale_limit_2": 1.05
}
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch


def plot_geo_objects(gdf_a, mbr_a, longest_side, angle):
    # Create a plot with a specific size
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams['hatch.linewidth'] = 1.0  # Increase the hatch line width

    # Plot each GeoDataFrame with hatching
    gdf_a.plot(ax=ax, color='cornflowerblue', edgecolor='cornflowerblue', label='Building Footprint', alpha=0.5, zorder=1)

    # Plot the boundaries of the MBRs
    mbr_a.boundary.plot(ax=ax, edgecolor='blue', label='Minimum Bounding Box', zorder=2)

    # Plot the longest side as an arrow (ensure it's on top)
    start_point = longest_side[0]
    end_point = longest_side[1]
    arrow = FancyArrowPatch(
        (start_point[0], start_point[1]),
        (end_point[0], end_point[1]),
        color='red',
        arrowstyle='-|>',
        mutation_scale=15,
        linewidth=2,
        label='Longest Side',  # Label for the arrow in the legend
        zorder=3  # Ensure the arrow is on top
    )
    ax.add_patch(arrow)

    # Create custom legend handles
    handles = [
        Patch(facecolor='cornflowerblue', edgecolor='cornflowerblue', label='Building Footprint', alpha=0.5),
        Patch(edgecolor='blue', facecolor='none', label='Minimum Bounding Box')
    ]

    # Define a custom legend handler for the arrow
    class ArrowHandler(HandlerPatch):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            # Create a copy of the arrow with the same properties
            arrow_copy = FancyArrowPatch(
                (0, 4), (24, 4),
                arrowstyle=orig_handle.get_arrowstyle(),  # Use get_arrowstyle() method
                color=orig_handle.get_edgecolor(),
                mutation_scale=orig_handle.get_mutation_scale(),
                linewidth=orig_handle.get_linewidth(),
                transform=trans
            )
            return [arrow_copy]

    # Add the arrow to the legend using the custom handler
    ax.legend(handles=handles + [arrow], handler_map={FancyArrowPatch: ArrowHandler()}, loc='upper right')

    # Set plot title and labels
    plt.title('Minimum Bounding Box')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Display the plot
    plt.show()

def plot_mbr(gdf_a, mbr_a, gdf_b, mbr_b):
    # Create a plot with a specific size
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams['hatch.linewidth'] = 1.0  # Increase the hatch line width

    # Plot each GeoDataFrame with hatching
    gdf_a.plot(ax=ax, color='cornflowerblue', edgecolor='cornflowerblue', label='Building A Footprint', alpha=0.5, zorder=1)
    gdf_b.plot(ax=ax, color='lightcoral', edgecolor='lightcoral', label='Building B Footprint', alpha=0.5, zorder=1)

    # Plot the boundaries of the MBRs
    mbr_a.boundary.plot(ax=ax, edgecolor='blue', label='Minimum Bounding Box', zorder=2)
    mbr_b.boundary.plot(ax=ax, edgecolor='red', label='Minimum Bounding Box', zorder=2)

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

#compare_geopackages('Building_1.gpkg', 'Building_2.gpkg', limits)
def advanced_jaccard_index(gdf_a, gdf_b, limits, rotation_off=False):
    jaccard_index_original = jaccard_index(gdf_a,gdf_b)
    # Translation

    ## getting centroids
    dx = gdf_b.centroid.x.iloc[0] - gdf_a.centroid.x.iloc[0]  # Difference in X coordinates
    dy = gdf_b.centroid.y.iloc[0] - gdf_a.centroid.y.iloc[0]  # Difference in Y coordinates
    dx_determined = dx
    ## check wether delta are in limits
    if dx == 0:
        translation_x_note = "No need for X-Axis translation was determind"
    elif limits["translation_limit_2"] < dx <= limits["translation_limit_1"]:
        dx = limits["translation_limit_2"]
        translation_x_note = "limit 2 exceeded --> maximum translation of "+ str(limits["translation_limit_2"]) +" used"
    elif limits["translation_limit_1"] < dx:
        dx = 0
        translation_x_note = "limit 1 exceeded --> no translation"
    elif limits["translation_limit_2"] < -dx <= limits["translation_limit_1"]:
        dx = -limits["translation_limit_2"]
    elif limits["translation_limit_1"] < -dx:
        dx = 0
    elif dx <= limits["translation_limit_2"]:
        translation_x_note = "full X-Axis translation of " + str(dx) + " used"

    dy_determined = dy
    if dy == 0:
        translation_y_note = "No need for Y-Axis translation was determind"
    elif limits["translation_limit_2"] < dy <= limits["translation_limit_1"]:
        dy = limits["translation_limit_2"]
        translation_y_note = "limit 2 exceeded --> maximum translation of "+ str(limits["translation_limit_2"]) +" used"
    elif limits["translation_limit_1"] < dy:
        dy = 0
        translation_y_note = "limit 1 exceeded --> no translation"
    elif limits["translation_limit_2"] < -dy <= limits["translation_limit_1"]:
        dy = -limits["translation_limit_2"]
    elif limits["translation_limit_1"] < -dy:
        dy = 0
    elif dy <= limits["translation_limit_2"]:
        translation_y_note = "full Y-Axis translation of " + str(dy) + " used"
    ## translate gdf_b
    gdf_b_translation = shift_gdf(gdf_b, -dx, -dy)

    ## calculate new jaccard index
    jaccard_index_translation = jaccard_index(gdf_a,gdf_b_translation)

    # Ratation

    ## define minimum boundign rectangle
    mbr_a = find_rotated_mbr(gdf_a)
    mbr_b = find_rotated_mbr(gdf_b)
    #plot_mbr(gdf_a, mbr_a, gdf_b, mbr_b)
    ## define longest side of minimum boundign rectangle
    longest_side_a = longest_side_coordinates(mbr_a)
    longest_side_b = longest_side_coordinates(mbr_b)

    #plot_geo_objects(gdf_a, gdf_b, mbr_a, mbr_b)
    ## define angles of longes side
    angle_longest_side_a = angle_of_line_segment(longest_side_a)
    angle_longest_side_b = angle_of_line_segment(longest_side_b)
    #plot_geo_objects(gdf_a, mbr_a, longest_side_a, angle_longest_side_a)
    ## define the delta between angles
    delta_angle = angle_longest_side_a-angle_longest_side_b
    print(delta_angle)
    if -225 < delta_angle < -135:
        delta_angle += 180
    elif 225 > delta_angle > 135:
        delta_angle -= 180
    elif 45 < delta_angle and delta_angle < 135:
        delta_angle -= 90
    elif -45 > delta_angle and delta_angle > -135:
        delta_angle += 90
    elif -315 < delta_angle < -225:
        delta_angle += 270
    elif -360 < delta_angle < -315:
        delta_angle += 360
    elif 315 > delta_angle > 225:
        delta_angle -= 270
    elif 360 > delta_angle > 315:
        delta_angle -= 360

    
    delta_angle_determined = delta_angle
    ## check wether delta are in limits
    if delta_angle == 0:
        rotation_note = "No need for rotation was determined"
    elif limits["rotation_limit_2"] < abs(delta_angle) < limits["rotation_limit_1"]:
        delta_angle = limits["rotation_limit_2"] if delta_angle > 0 else -limits["rotation_limit_2"]
        rotation_note = "rotation limit 2 exceeded --> maximum rotation of " + str(delta_angle) + " used"
    elif abs(delta_angle) > limits["rotation_limit_1"]:
        delta_angle = 0
        rotation_note = "rotation limit 1 exceeded --> no rotation"
    elif abs(delta_angle) < limits["rotation_limit_2"]:
        rotation_note = "full rotation of " + str(delta_angle) + " used"

    ##
    gdf_b_rotate = rotate_gdf(gdf_b_translation, -delta_angle)

    if rotation_off == True:
        gdf_b_rotate = gdf_b_translation
    
    ## calculate new jaccard index
    jaccard_index_rotation = jaccard_index(gdf_a,gdf_b_rotate)

    # Scaling
    area_a = gdf_a.geometry.area.sum()
    area_b = gdf_b_rotate.geometry.area.sum()
    
    scaling_factor = (area_a/area_b)**0.5
    scaling_factor_determined = scaling_factor
    if scaling_factor == 1:
        scaling_note = "No need for scaling was determined"
    elif (2-limits["scale_limit_2"]) < scaling_factor < limits["scale_limit_2"]:
        scaling_note = "full scaling of " + str(scaling_factor) + " used"
    elif (2-limits["scale_limit_1"]) < scaling_factor < (2-limits["scale_limit_2"]):
        scaling_factor = 2-limits["scale_limit_2"]
        scaling_note = "scale limit 2 exceeded --> maximum scaling of " + str(2-limits["scale_limit_2"]) + " used"
    elif (limits["scale_limit_2"]) < scaling_factor < (limits["scale_limit_1"]):
        scaling_factor = limits["scale_limit_2"]
        scaling_note = "scale limit 2 exceeded --> maximum scaling of " + str(limits["scale_limit_2"]) + " used"
    else:
        scaling_factor = 1
        scaling_note = "scaling limit 1 exceeded --> no scaling"

    gdf_b_scaling = scale_gdf(gdf_b_rotate, scaling_factor, scaling_factor)
    jaccard_index_scaling = jaccard_index(gdf_a,gdf_b_scaling)
    print(f"Jaccard Index (Original): {jaccard_index_original}")
    print(f"Jaccard Index (Translation): {jaccard_index_translation}")
    print(f"Jaccard Index (Rotation): {jaccard_index_rotation}")
    print(f"Jaccard Index (Scaling): {jaccard_index_scaling}")
    print(f"Scaling Factor Determined: {scaling_factor_determined}")
    print(f"Scaling Factor: {scaling_factor}")
    print(f"Delta Angle Determined: {delta_angle_determined}")
    print(f"Delta Angle: {delta_angle}")
    print(f"Translation Determined (dx, dy): ({dx_determined}, {dy_determined})")
    print(f"Translation (dx, dy): ({dx}, {dy})")

    #plot_gdfs(gdf_a, gdf_b, gdf_b_rotate, title="Comparison of GeoDataFrames")
    # Print results
    results =  jaccard_index_scaling[0]
    variables_array = np.array([jaccard_index_original[0], 
                            jaccard_index_translation[0], 
                            jaccard_index_rotation[0], 
                            jaccard_index_scaling[0], 
                            scaling_factor_determined, 
                            scaling_factor, 
                            delta_angle_determined, 
                            delta_angle, 
                            dx_determined, 
                            dy_determined, 
                            dx, 
                            dy])
    return results, gdf_b_scaling, variables_array