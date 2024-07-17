from comparison import jaccard_index
from comparison_processing import fetch_building_data_by_base_name
import geopandas as gpd
from manipulate import scale
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


# Example GeoDataFrame for testing
coordinates = [(0, 0), (0, 10), (2, 10), (2, 3), (4, 3), (4, 10), (6, 10), (6, 0)]
polygon = Polygon(coordinates)
gdf = gpd.GeoDataFrame(index=[0], geometry=[polygon])
building_1 = gpd.read_file('Building_1.gpkg')

building_3 = gpd.read_file('Building_3.gpkg')
#gdf_b = gpd.read_file(row[6])
original_jaccard = jaccard_index(building_1, building_1)
print(original_jaccard[0])
shift_list = np.arange(0.01, 2.01, 0.01)
jaccard_index_array = []
for i in shift_list:
    shift_jaccard = jaccard_index(building_1, scale(building_1, i, i))
    jaccard_index_array.append((i, shift_jaccard[0], shift_jaccard[1], shift_jaccard[2], shift_jaccard[3]))
    print(i, shift_jaccard)
#print(jaccard_index_array)

import matplotlib.pyplot as plt
# Scale the GeoDataFrame by a factor of 2
scaled_building_3 = scale(building_3, 2, 2)

# Plot both the original and scaled GeoDataFrames
def plot_original_and_scaled(original_gdf, scaled_gdf):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot original GeoDataFrame
    original_gdf.plot(ax=ax, color='blue', alpha=0.5, edgecolor='black', label='Original')
    
    # Plot scaled GeoDataFrame
    scaled_gdf.plot(ax=ax, color='red', alpha=0.5, edgecolor='black', label='Scaled')
    
    # Set plot limits
    ax.set_xlim(min(original_gdf.total_bounds[0], scaled_gdf.total_bounds[0]) - 1,
                max(original_gdf.total_bounds[2], scaled_gdf.total_bounds[2]) + 1)
    ax.set_ylim(min(original_gdf.total_bounds[1], scaled_gdf.total_bounds[1]) - 1,
                max(original_gdf.total_bounds[3], scaled_gdf.total_bounds[3]) + 1)
    
    # Add legend
    plt.legend()
    plt.title('Original and Scaled Building 3')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# Plot the original and scaled GeoDataFrames
plot_original_and_scaled(building_3, scaled_building_3)

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
plt.title('Scaling Effect on Jaccard Index')
plt.xlabel('Scaling Value')
plt.ylabel('Jaccard Index')
#plt.ylim(0.6, max(jaccard_indices) * 1.1)  # Adjust y-axis for Jaccard Index to start at 0

# Create a second y-axis
overlap_axis = plt.gca().twinx()
line2, = overlap_axis.plot(shifts, overlap, color='red', marker=None, linestyle='-', label='Overlapping area')
overlap_axis.set_ylabel('Overlapping area [m²]')
#overlap_axis.set_ylim(150, max(overlap) * 1.1)  # Adjust y-axis for Overlapping area to start at 0

# Manually create a combined legend
lines = [line1, line2]
labels = [line.get_label() for line in lines]
plt.legend(lines, labels, loc='best')

plt.grid(True)
plt.show()

def calculate_jaccard_matrix_scaling(gdf):
    scales = np.arange(0.95, 1.051, 0.001)  # Scaling from 0.5x to 1.5x
    jaccard_indices = []
    
    for scale_factor in scales:
        scaled_gdf = scale(gdf, scale_factor, scale_factor)
        jaccard_index_value = jaccard_index(gdf, scaled_gdf)
        jaccard_indices.append(jaccard_index_value[0])
    
    return np.array(jaccard_indices)


def plot_jaccard_indices_scaling(jaccard_indices, gdf_labels=None):
    scales = np.arange(0.95, 1.051, 0.001)
    plt.figure(figsize=(12, 8))
    
    for idx, jaccard_index_values in enumerate(jaccard_indices):
        label = f'Jaccard Index: Building {idx + 1}' if not gdf_labels else gdf_labels[idx]
        plt.plot(scales, jaccard_index_values, label=label)
    
    average_jaccard = np.mean(jaccard_indices, axis=0)
    print(average_jaccard)
    plt.plot(scales, average_jaccard, linestyle='--', color='black', label='Average Jaccard Index')
    for i in range(len(average_jaccard)-1):
        print(scales[i+1])
        print(average_jaccard[i]-average_jaccard[i+1])
    plt.xlabel('Scaling Factor')
    plt.ylabel('Jaccard Index')
    plt.title('Jaccard Index for Scaling')
    plt.legend()
    plt.grid(True)
    
    # Adding the note below the plot
    plt.figtext(0.5, 0.12, "Building 1, 2 and 4 have identical curves; only one is visible!", wrap=True, horizontalalignment='center', fontsize=12)
    
    plt.show()




# Assuming you have 5 GeoDataFrames loaded
gdfs = [gpd.read_file('Building_1.gpkg'), 
        gpd.read_file('Building_2.gpkg'), 
        gpd.read_file('Building_3.gpkg'), 
        gpd.read_file('Building_4.gpkg'), 
        gpd.read_file('Building_5.gpkg')]

# Calculate Jaccard indices for scaling
jaccard_matrices = [calculate_jaccard_matrix_scaling(gdf) for gdf in gdfs]

# Plot Jaccard indices for each GeoDataFrame
plot_jaccard_indices_scaling(jaccard_matrices)

