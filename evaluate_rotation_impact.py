from comparison import jaccard_index
from comparison_processing import fetch_building_data_by_base_name
import geopandas as gpd
from manipulate import rotate_centroid
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import geopandas as gpd
from shapely.geometry import Polygon

# Define the coordinates of the polygon
coordinates = [(0, 0), (0, 10), (2, 10), (2, 3), (4,3), (4,10), (6,10), (6,0)]
building_1 = gpd.read_file('Building_1.gpkg')
# Create a Polygon object
polygon = Polygon(coordinates)

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(index=[0], geometry=[polygon])




#gdf_b = gpd.read_file(row[6])
original_jaccard = jaccard_index(building_1, building_1)
print(original_jaccard[0])
shift_list = np.arange(0, 91, 1)
jaccard_index_array = []
for i in shift_list:
    shift_jaccard = jaccard_index(building_1, rotate_centroid(building_1, i))
    jaccard_index_array.append((i, shift_jaccard[0], shift_jaccard[1]))
    print(i, shift_jaccard)
#print(jaccard_index_array)

import matplotlib.pyplot as plt

# Assuming jaccard_index_array is already computed and looks something like this:
# jaccard_index_array = [(0, 0.75), (1, 0.73), (2, 0.70), (3, 0.68), (4, 0.65)]

# Extracting shift values and Jaccard indices
shifts = [item[0] for item in jaccard_index_array]
jaccard_indices = [item[1] for item in jaccard_index_array]
overlap = [item[2] for item in jaccard_index_array]

# Plotting


plt.figure(figsize=(10, 5))
# Plot the first line with the first y-axis
line1, = plt.plot(shifts, jaccard_indices, marker=None, linestyle='-', color='blue', label='Jaccard Index')
plt.title('Rotation Effect on Jaccard Index')
plt.xlabel('Rotation Value [°]')
plt.ylabel('Jaccard Index')
plt.ylim(0.6, max(jaccard_indices) * 1.1)  # Adjust y-axis for Jaccard Index to start at 0

# Create a second y-axis
overlap_axis = plt.gca().twinx()
line2, = overlap_axis.plot(shifts, overlap, color='red', marker=None, linestyle='-', label='Overlapping area')
overlap_axis.set_ylabel('Overlapping area [m²]')
overlap_axis.set_ylim(150, max(overlap) * 1.1)  # Adjust y-axis for Overlapping area to start at 0

# Manually create a combined legend
lines = [line1, line2]
labels = [line.get_label() for line in lines]
plt.legend(lines, labels, loc='best')

plt.grid(True)
plt.show()



from matplotlib.patches import FancyArrowPatch, Arc

def plot_original_and_arrow(original_gdf, shift_vector, rotation_angle):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot original polygon
    original_gdf.plot(ax=ax, color='blue', alpha=0.5, edgecolor='black', label='Original')
    
    # Calculate the arrow's end point after applying the shift vector
    start_point = np.array([3, 4.5])
    end_point = start_point + np.array(shift_vector)
    
    # Apply rotation
    angle_rad = np.deg2rad(rotation_angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    rotated_shift_vector = rotation_matrix.dot(np.array(shift_vector))
    rotated_end_point = start_point + rotated_shift_vector
    
    # Plot rotated arrow
    rotated_arrow = FancyArrowPatch(
        start_point, rotated_end_point,
        color='red', 
        arrowstyle='-|>', 
        mutation_scale=15
    )
    ax.add_patch(rotated_arrow)
    
    # Plot the arc representing the rotation
    arc_center = start_point
    arc_radius = np.linalg.norm(shift_vector)
    arc = Arc(
        arc_center, 
        2 * arc_radius, 
        2 * arc_radius, 
        angle=0, 
        theta1=0, 
        theta2=rotation_angle, 
        color='green',
        linestyle='--'
    )
    ax.add_patch(arc)
    
    # Plot the arrowhead to indicate direction of rotation at the end of the arc
    arc_end_angle_rad = np.deg2rad(rotation_angle)
    arrowhead_position = arc_center + np.array([
        arc_radius * np.cos(arc_end_angle_rad),
        arc_radius * np.sin(arc_end_angle_rad)
    ])
    direction_arrow = FancyArrowPatch(
        arc_center, arrowhead_position,
        color='green',
        arrowstyle='->',
        mutation_scale=10
    )
    ax.add_patch(direction_arrow)
    
    # Create custom legend
    ax.set_xlim(original_gdf.total_bounds[0] - 1, original_gdf.total_bounds[2] + 1)
    ax.set_ylim(original_gdf.total_bounds[1] - 1, original_gdf.total_bounds[3] + 1)
    ax.set_title('Concave Polygon with Rotation Arrow')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(['Original Polygon', 'Rotated Arrow'], loc='upper right')
    plt.show()

# Example usage:
# Assuming gdf is the original GeoDataFrame
# Define shift vector, for example, (1, 0) for shifting 1 unit horizontally
shift_vector = (2, 0)
plot_original_and_arrow(gdf, shift_vector, 45)

shifted_coordinates = gdf.iloc[0].geometry.exterior.coords[:]
print("Coordinates of Shifted Polygon:")
for coord in shifted_coordinates:
    print(coord)


def calculate_jaccard_matrix_rotations(gdf, rotations):
    
    jaccard_indices = []
    
    for angle in rotations:
        rotated_gdf = rotate_centroid(gdf, angle)
        jaccard_index_value = jaccard_index(gdf, rotated_gdf)[0]
        jaccard_indices.append(jaccard_index_value)
    
    return np.array(jaccard_indices)


def plot_jaccard_indices_rotations(jaccard_indices, gdf_labels=None):
    rotations = np.arange(0, 5.1, 0.1)
    plt.figure(figsize=(12, 8))
    average_jaccard = np.mean(jaccard_indices, axis=0)
    print(average_jaccard)
    for i in range(len(average_jaccard)-1):
        print(rotations[i+1])
        print(average_jaccard[i]-average_jaccard[i+1])
    for idx, jaccard_index_values in enumerate(jaccard_indices):
        label = f'Jaccard Index: Building {idx+1}' if not gdf_labels else gdf_labels[idx]
        plt.plot(rotations, jaccard_index_values, label=label)
        
    plt.plot(rotations, average_jaccard, linestyle='--', color='black', label='Average Jaccard Index')
    plt.xlabel('Rotation Angle [°]')
    plt.ylabel('Jaccard Index')
    plt.title('Jaccard Index for Rotations')
    plt.legend()
    plt.grid(True)
    plt.show()



# Assuming you have 5 GeoDataFrames loaded
gdfs = [gpd.read_file('Building_1.gpkg'), 
        gpd.read_file('Building_2.gpkg'), 
        gpd.read_file('Building_3.gpkg'), 
        gpd.read_file('Building_4.gpkg'), 
        gpd.read_file('Building_5.gpkg')]

# Calculate Jaccard indices for rotations
rotations = np.arange(0, 5.1, 0.1)
jaccard_matrices = [calculate_jaccard_matrix_rotations(gdf, rotations) for gdf in gdfs]
rotations = np.arange(0, -5.1, -0.1)
print(rotations)
jaccard_matrices_negative = [calculate_jaccard_matrix_rotations(gdf, rotations) for gdf in gdfs]
# Plot Jaccard indices for each GeoDataFrame
plot_jaccard_indices_rotations(jaccard_matrices)
