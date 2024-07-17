import geopandas as gpd
import matplotlib.pyplot as plt

# List of file names
file_names = [
    "Building_1.gpkg",
    "Building_2.gpkg",
    "Building_3.gpkg",
    "Building_4.gpkg",
    "Building_5.gpkg"]

# Create a figure and axes for subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Adjust the spacing between rows
plt.subplots_adjust(hspace=1)

# Iterate over each file
for i, file_name in enumerate(file_names):
    # Read GeoPackage file
    gdf = gpd.read_file(file_name)

    # Plot GeoDataFrame in the corresponding subplot
    ax = axes.flatten()[i]
    gdf.plot(ax=ax)
    ax.set_title(f"{file_name[0:-5]}")
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

# Hide the extra subplot if there are fewer plots than subplots
if len(file_names) < len(axes.flatten()):
    axes.flatten()[-1].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()

plt.show()

