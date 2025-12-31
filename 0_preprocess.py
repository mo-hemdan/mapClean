# %%
import pandas as pd
import osmnx as ox
from MapMatcher import MapMatcher
import geopandas as gpd
import pickle

CITY = 'Singapore'
CITY = 'Chicago'
CITY ='Jakarta'

# %%
DatasetFolder = f'city={CITY}'
df = pd.read_parquet(DatasetFolder)
df = df.rename(columns={"trj_id": "vehicule_id", "rawlat": "lat", "rawlng": "lon", "pingtimestamp": "timestamp", "bearing": "angle"})
df = df.sort_values(by=['vehicule_id', 'timestamp'])
df = df.reset_index()
df.drop(columns=['index'], inplace=True)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
gdf.to_crs(crs=3857, inplace=True) 
gdf.lon = gdf.geometry.x
gdf.lat = gdf.geometry.y
gdf.to_parquet(f'city_{CITY}_pro/sorted_geom.parquet', index=True, compression="gzip", engine="pyarrow")

# %%
gdf = gpd.read_parquet(f'city_{CITY}_pro/sorted_geom.parquet')
gdf = gdf.to_crs("EPSG:3857")
minx, miny, maxx, maxy = gdf.total_bounds
buffer = 500  # meters

minx -= buffer
miny -= buffer
maxx += buffer
maxy += buffer

# %%
from pyproj import Transformer

def transform_bbox(minx, miny, maxx, maxy, crs_from="EPSG:3857", crs_to="EPSG:4326"):
    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)

    # transform the four corners
    x1, y1 = transformer.transform(minx, miny)
    x2, y2 = transformer.transform(minx, maxy)
    x3, y3 = transformer.transform(maxx, miny)
    x4, y4 = transformer.transform(maxx, maxy)

    # get new bounding box
    return min(x1,x2,x3,x4), min(y1,y2,y3,y4), max(x1,x2,x3,x4), max(y1,y2,y3,y4)


minx2, miny2, maxx2, maxy2 = transform_bbox(minx, miny, maxx, maxy)

# %%

bbox = (minx2, miny2, maxx2, maxy2) 

print(bbox)
G = ox.graph_from_bbox(
    bbox=bbox,
    network_type="drive",
    retain_all=True,
    truncate_by_edge=True,
)

with open(f"city_{CITY}_pro/raw_graph.pkl", "wb") as f:
    pickle.dump(G, f)

# %%
with open(f'city_{CITY}_pro/raw_graph'+'.pkl', 'rb') as f:
    road_network = pickle.load(f)
gdf = gpd.read_parquet(f'city_{CITY}_pro/sorted_geom.parquet')
gdf.lon = gdf.geometry.x
gdf.lat = gdf.geometry.y
    
mapmatcher = MapMatcher()
gdf = mapmatcher.match(gdf, road_network)
gdf = mapmatcher.clean_snap(gdf)

print('saving') 
gdf['road_geometry'] = gdf['road_geometry'].apply(lambda geom: geom.wkt)
gdf.to_parquet(f'city_{CITY}_pro/matched_geom.parquet', compression='snappy')
print(gdf.head())
