import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from manipulate import randomize
from comparison import jaccard_index, advanced_jaccard_index

def compute_statistics(jaccard_indices):
    avg_jaccard = np.mean(jaccard_indices)
    max_jaccard = np.max(jaccard_indices)
    min_jaccard = np.min(jaccard_indices)
    var_jaccard = np.var(jaccard_indices)
    return avg_jaccard, max_jaccard, min_jaccard, var_jaccard

def calculate_jaccard_for_randomizations(gdf, r_range=0.01, randomization_count=200):
    jaccard_indices = []
    Adv_jaccard_indices = []
    limits = {
        "translation_limit_1": 10000,
        "translation_limit_2": 10000,
        "rotation_limit_1": 360,
        "rotation_limit_2": 360,
        "scale_limit_1": 100,
        "scale_limit_2": 100
    }
    for _ in range(randomization_count):
        randomized_gdf = randomize(gdf, r_range, r_range)
        jaccard_index_value = jaccard_index(gdf, randomized_gdf)
        Adv_jaccard_index_value = advanced_jaccard_index(gdf, randomized_gdf, limits)
        if Adv_jaccard_index_value < jaccard_index_value[0]:
            if jaccard_index_value[0] - Adv_jaccard_index_value > 0.1:
                print(Adv_jaccard_index_value)
                print(jaccard_index_value[0])
                fig, ax = plt.subplots()
                gdf.plot(ax=ax, color='blue', alpha=0.5, label='Original')
                randomized_gdf.plot(ax=ax, color='red', alpha=0.5, label='Randomized')
                ax.set_title('title')
                ax.legend()
                plt.show()
        jaccard_indices.append(jaccard_index_value[0])
        Adv_jaccard_indices.append(Adv_jaccard_index_value)
    jaccard_indices = np.array(jaccard_indices)
    Adv_jaccard_indices = np.array(Adv_jaccard_indices)
    return jaccard_indices, Adv_jaccard_indices

def plot_jaccard_indices_randomizations(jaccard_indices_list, Adv_jaccard_indices_list, gdf_labels=None):
    if not gdf_labels:
        gdf_labels = [f'Building {i + 1}' for i in range(len(jaccard_indices_list))]

    merged_indices = []
    labels = []
    positions = []
    
    for i, (jaccard_indices, adv_jaccard_indices) in enumerate(zip(jaccard_indices_list, Adv_jaccard_indices_list)):
        base_position = i * 2 + 1
        positions.append(base_position)
        positions.append(base_position + 1)
        merged_indices.append(jaccard_indices)
        merged_indices.append(adv_jaccard_indices)
        labels.append('Jaccard')
        labels.append('Advanced Jaccard')
    
    plt.figure(figsize=(14, 10))
    
    # Create a box plot for each set of Jaccard indices
    box = plt.boxplot(merged_indices, positions=positions, patch_artist=True)
    
    colors = ['skyblue', 'lightgreen']
    for patch, color in zip(box['boxes'], colors * len(gdf_labels)):
        patch.set_facecolor(color)
    
    plt.ylabel('Jaccard Index')
    plt.title('Jaccard Index and Advanced Jaccard Index at Random Error (20 cm)')
    plt.xticks([base_position + 0.5 for base_position in range(1, len(gdf_labels) * 2 + 1, 2)], gdf_labels, rotation=45)
    plt.grid(True)
    plt.legend([box["boxes"][0], box["boxes"][1]], ['Jaccard', 'Advanced Jaccard'], loc='upper right')
    plt.show()

def print_statistics(statistics, gdf_labels=None):
    avg_jaccard, max_jaccard, min_jaccard, var_jaccard = statistics
    labels = [f'Building {i + 1}' for i in range(len(avg_jaccard))] if not gdf_labels else gdf_labels
    
    for i, label in enumerate(labels):
        print(f"{label}:")
        print(f"  Average Jaccard Index: {avg_jaccard[i]:.4f}")
        print(f"  Maximum Jaccard Index: {max_jaccard[i]:.4f}")
        print(f"  Minimum Jaccard Index: {min_jaccard[i]:.4f}")
        print(f"  Variance of Jaccard Index: {var_jaccard[i]:.4f}")
        print()

from shapely.geometry import Polygon, MultiPolygon
coordinates = [(0, 0), (0, 10), (7, 10), (7, 7), (18, 7), (18, 10), (25, 10), (25, 0)]
polygon = MultiPolygon([Polygon(coordinates)])
gdf = gpd.GeoDataFrame(index=[0], geometry=[polygon])

gdfs = [
    gpd.read_file('Building_1.gpkg'), 
    gpd.read_file('Building_2.gpkg'), 
    gpd.read_file('Building_3.gpkg'), 
    gpd.read_file('Building_4.gpkg'), 
    gpd.read_file('Building_5.gpkg'),
    gdf
]

jaccard_indices_list = []
Adv_jaccard_indices_list = []

for gdf in gdfs:
    jaccard_indices, adv_jaccard_indices = calculate_jaccard_for_randomizations(gdf)
    jaccard_indices_list.append(jaccard_indices)
    Adv_jaccard_indices_list.append(adv_jaccard_indices)

plot_jaccard_indices_randomizations(jaccard_indices_list, Adv_jaccard_indices_list)

statistics = [compute_statistics(jaccard_indices) for jaccard_indices in jaccard_indices_list]
avg_jaccard, max_jaccard, min_jaccard, var_jaccard = zip(*statistics)

print_statistics((avg_jaccard, max_jaccard, min_jaccard, var_jaccard))

statistics = [compute_statistics(jaccard_indices) for jaccard_indices in Adv_jaccard_indices_list]
avg_jaccard, max_jaccard, min_jaccard, var_jaccard = zip(*statistics)

print_statistics((avg_jaccard, max_jaccard, min_jaccard, var_jaccard))
print(np.mean(avg_jaccard))
def get_building_stats(gdfs):
    areas = []
    num_coords = []

    for gdf in gdfs:
        area = gdf.geometry.area.sum()
        coordinates_count = sum(
            sum(len(poly.exterior.coords) for poly in geom.geoms) if geom.geom_type == 'MultiPolygon' else len(geom.exterior.coords)
            for geom in gdf.geometry
        )
        areas.append(area)
        num_coords.append(coordinates_count)

    return areas, num_coords

def print_building_stats(areas, num_coords, gdf_labels=None):
    labels = [f'Building {i + 1}' for i in range(len(areas))] if not gdf_labels else gdf_labels
    
    for i, label in enumerate(labels):
        print(f"{label}:")
        print(f"  Surface Area: {areas[i]:.4f}")
        print(f"  Number of Coordinates: {num_coords[i]}")
        print()

# Get surface area and number of coordinates for each building
areas, num_coords = get_building_stats(gdfs)

# Print statistics for each building
print_building_stats(areas, num_coords)