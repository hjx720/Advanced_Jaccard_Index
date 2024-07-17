from comparison import jaccard_index
from comparison_processing import fetch_building_data_by_base_name
import geopandas as gpd
from manipulate import shift
import numpy as np
from matplotlib.lines import Line2D

import geopandas as gpd
from shapely.geometry import Polygon

# Define the coordinates of the polygon
coordinates = [(0, 0), (0, 10), (2, 10), (2, 3), (4,3), (4,10), (6,10), (6,0)]

# Create a Polygon object
polygon = Polygon(coordinates)

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(index=[0], geometry=[polygon])




#gdf_b = gpd.read_file(row[6])
original_jaccard = jaccard_index(gdf, gdf)
print(original_jaccard[0])
shift_list = np.arange(0, 1, 0.01)
jaccard_index_array = []
for i in shift_list:
    shift_jaccard = jaccard_index(gdf, shift(gdf, 0, i))
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
line1, = plt.plot(shifts, jaccard_indices, marker=None, linestyle='-', color='blue')
plt.title('Translation Effect on Jaccard Index')
plt.xlabel('Translation Value [m]')
plt.ylabel('Jaccard Index')

# Create a second y-axis
overlap_axis = plt.gca().twinx()
line2, = overlap_axis.plot(shifts, overlap, color='red', marker=None, linestyle='-', label='Overlapping area')
overlap_axis.set_ylabel('Overlapping area [m²]')

# Manually create a combined legend
plt.legend([line1, line2], ['Jaccard Index', 'Overlapping area'])

plt.grid(True)
plt.show()



def plot_original_and_arrow(original_gdf, shift_vector):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot original polygon
    original_gdf.plot(ax=ax, color='blue', alpha=0.5, edgecolor='black', label='Original')
    
    
    # Plot arrow
    ax.arrow(3, 4.5, shift_vector[0], shift_vector[1], 
             head_width=0.25, head_length=0.5, fc='red', ec='red')
    # Create custom legend
    ax.set_xlim(original_gdf.total_bounds[0] - 1, original_gdf.total_bounds[2] + 1)
    ax.set_ylim(original_gdf.total_bounds[1] - 1, original_gdf.total_bounds[3] + 1)
    ax.set_title('Concave Polygon')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# Example usage:
# Assuming gdf is the original GeoDataFrame
# Define shift vector, for example, (1, 0) for shifting 1 unit horizontally
shift_vector = (2, 0)
plot_original_and_arrow(gdf, shift_vector)

shifted_coordinates = gdf.iloc[0].geometry.exterior.coords[:]
print("Coordinates of Shifted Polygon:")
for coord in shifted_coordinates:
    print(coord)


def generate_shifts():
    directions = 40
    angle_increment = 360 / directions
    shifts = []
    
    for i in range(directions):
        angle = np.radians(i * angle_increment)
        for distance in np.arange(0, 1.01, 0.01):  # From 0 to 1 meter in 1 cm steps
            x_shift = distance * np.cos(angle)
            y_shift = distance * np.sin(angle)
            shifts.append((x_shift, y_shift))
    
    return shifts

def calculate_jaccard_matrix(gdf):
    shifts = generate_shifts()
    directions = 40
    steps = 101  # 0 to 1 meter in 1 cm steps
    jaccard_matrix = np.zeros((directions, steps))
    
    for i, (x_shift, y_shift) in enumerate(shifts):
        direction_index = i // steps
        step_index = i % steps
        shifted_gdf = shift(gdf, x_shift, y_shift)
        jaccard_index_value = jaccard_index(gdf, shifted_gdf)[0]
        
        jaccard_matrix[direction_index, step_index] = jaccard_index_value
    
    return jaccard_matrix

def plot_jaccard_indices(jaccard_matrices, shift_index):
    directions = 40
    angles = np.linspace(0, 360, directions, endpoint=False)
    
    plt.figure(figsize=(12, 8))
    for idx, jaccard_matrix in enumerate(jaccard_matrices):
        jaccard_indices = jaccard_matrix[:, shift_index]
        print(np.var(jaccard_indices))
        plt.plot(angles, jaccard_indices, marker='o', label=f'Jaccard Index: Building {idx+1}')
    
    plt.xlabel('Direction (degrees)')
    plt.ylabel('Jaccard Index')
    plt.title(f'Jaccard Index for {shift_index * 0.01:.2f} m Translation')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_average_jaccard_indices(jaccard_matrices):
    directions = 40
    steps = 101  # 0 to 1 meter in 1 cm steps
    distances = np.arange(0, 1.01, 0.01)
    
    # Calculate average Jaccard index per distance for each GDF
    average_jaccard_per_gdf = [jaccard_matrix.mean(axis=0) for jaccard_matrix in jaccard_matrices]
    
    # Calculate overall average Jaccard index per distance
    overall_average_jaccard = np.mean(average_jaccard_per_gdf, axis=0)
    print(overall_average_jaccard)
    print(overall_average_jaccard[::10])
    var_overall_average_jaccard = np.var(average_jaccard_per_gdf, axis=0)
    print(var_overall_average_jaccard)
    print(np.mean(var_overall_average_jaccard))

    plt.figure(figsize=(12, 8))
    
    for idx, avg_jaccard in enumerate(average_jaccard_per_gdf):
        plt.plot(distances, avg_jaccard, marker=None, label=f'Average Jaccard Index: Building {idx+1}')
    
    plt.plot(distances, overall_average_jaccard, marker=None, linestyle='--', color='black', label='Overall Average Jaccard')
    
    plt.xlabel('Shift Distance (meters)')
    plt.ylabel('Jaccard Index')
    plt.title('Average Jaccard Index per Translation Distance')
    plt.legend()
    plt.grid(True)
    plt.show()

gdfs = [gpd.read_file('Building_1.gpkg'), gpd.read_file('Building_2.gpkg'), gpd.read_file('Building_3.gpkg'), gpd.read_file('Building_4.gpkg'), gpd.read_file('Building_5.gpkg')]
jaccard_matrices = [calculate_jaccard_matrix(gdf) for gdf in gdfs]
# Plot Jaccard indices for a 1 cm shift (index 1)
plot_jaccard_indices(jaccard_matrices, shift_index=1)
plot_jaccard_indices(jaccard_matrices, shift_index=2)
plot_jaccard_indices(jaccard_matrices, shift_index=5)
plot_jaccard_indices(jaccard_matrices, shift_index=10)
plot_jaccard_indices(jaccard_matrices, shift_index=20)
plot_jaccard_indices(jaccard_matrices, shift_index=50)
plot_jaccard_indices(jaccard_matrices, shift_index=100)
plot_average_jaccard_indices(jaccard_matrices)