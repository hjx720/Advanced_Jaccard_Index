import geopandas as gpd
from frechetdist import frdist
from shapely.geometry import Polygon, MultiPolygon
from scipy.spatial.distance import directed_hausdorff



def hausdorff_distance(gdf_1, gdf_2):
    polygon1 = gdf_1.geometry.iloc[0]
    polygon2 = gdf_2.geometry.iloc[0]

    # Überprüfe, ob die Geometrien der Polygone geschlossen sind (optional)
    if isinstance(polygon1, MultiPolygon):
        polygon1 = list(polygon1.geoms)[0]  # Wähle das erste Polygon aus dem MultiPolygon aus
    if isinstance(polygon2, MultiPolygon):
        polygon2 = list(polygon2.geoms)[0]  # Wähle das erste Polygon aus dem MultiPolygon aus

    # Konvertiere die Polygone in einfache Polygone, falls sie es nicht bereits sind
    if not polygon1.is_empty and not polygon1.exterior.is_closed:
        polygon1 = Polygon(polygon1.exterior)
    if not polygon2.is_empty and not polygon2.exterior.is_closed:
        polygon2 = Polygon(polygon2.exterior)

    # Berechne die Fréchet-Distanz zwischen den beiden Polygonen
    print(list(polygon1.exterior.coords), list(polygon2.exterior.coords))
    #frechet_distance = frdist(polygon1.exterior.coords, polygon2.exterior.coords)
    hausdorff_distance = directed_hausdorff(polygon1.exterior.coords, polygon2.exterior.coords)[0]
    return hausdorff_distance

"""
# Lade die Polygone aus den GeoPackage-Dateien
gpkg_file_1 = "example_data\Building_1.gpkg"
gdf_1 = gpd.read_file(gpkg_file_1)

gpkg_file_2 = "example_data\Building_1_randomized_10_mm.gpkg"
gdf_2 = gpd.read_file(gpkg_file_2)
hausdorff_dist = hausdorff_distance(gdf_1, gdf_2)
#print("Fréchet-Distanz zwischen den Polygonen:", frechet_dist)
print("Gerichtete Hausdorff-Distanz zwischen den Polygonen:", hausdorff_dist)
"""