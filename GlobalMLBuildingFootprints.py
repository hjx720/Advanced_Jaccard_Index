"""
This snippet demonstrates how to access and convert the buildings
data from .csv.gz to geojson for use in common GIS tools. You will
need to install pandas, geopandas, and shapely.
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape

def main():
    # this is the name of the geography you want to retrieve. update to meet your needs
    location = 'Germany'

    dataset_links = pd.read_csv("https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv")
    greece_links = dataset_links[dataset_links.Location == location]
    for _, row in greece_links.iterrows():
        df = pd.read_json(row.Url, lines=True)
        df['geometry'] = df['geometry'].apply(shape)
        gdf = gpd.GeoDataFrame(df, crs=4326)
        try:
            gdf.to_file(f"GlobalMLBuildingFootprints/{row.QuadKey}.geojson", driver="GeoJSON")
            print(f"GlobalMLBuildingFootprints/{row.QuadKey}.geojson saved" )
        except:
            print(f"GlobalMLBuildingFootprints/{row.QuadKey}.geojson could not be saved" )
            None
    print('done')

if __name__ == "__main__":
    main()