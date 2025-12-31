# %%
import pandas as pd
import osmnx as ox
from MapMatcher import MapMatcher
import geopandas as gpd
import pickle

CITY = 'Singapore'
CITY = 'Chicago'
CITY = 'SingaporeArea'
CITY ='Jakarta'
# %%
with open(f'city_{CITY}_pro/raw_graph'+'.pkl', 'rb') as f:
    road_network = pickle.load(f)
gdf = gpd.read_parquet(f'city_{CITY}_pro/sorted_geom.parquet')
gdf.lon = gdf.geometry.x
gdf.lat = gdf.geometry.y
    
mapmatcher = MapMatcher(road_network)
gdf = mapmatcher.match(gdf, advanced_matching=False)
gdf = mapmatcher.clean_snap(gdf)

print('saving') 
gdf['road_geometry'] = gdf['road_geometry'].apply(lambda geom: geom.wkt)
gdf.to_parquet(f'city_{CITY}_pro/matched_geom.parquet', compression='snappy')
print(gdf.head())
