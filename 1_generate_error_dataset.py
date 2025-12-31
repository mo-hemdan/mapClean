# %%
import geopandas as gpd
import pickle
import random
import osmnx as ox
import numpy as np
from shapely.geometry import box
from pyproj import Transformer
from load_config import load_config
from tqdm import tqdm
from sklearn.neighbors import KDTree
from MapMatcher import MapMatcher
tqdm.pandas()

# config = load_config('configs/jakarta_m.json', same_gamma=True)
config = load_config(same_gamma=False)

# %%
gdf = gpd.read_parquet(f"city_{config['CITY']}_pro/matched_geom.parquet")

# Define transformer: WGS84 (lat, lon) -> Web Mercator (meters)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# Convert each (lon, lat) corner
min_x, min_y = transformer.transform(config['MIN_LON'], config['MIN_LAT'])
max_x, max_y = transformer.transform(config['MAX_LON'], config['MAX_LAT'])

print("Mercator coordinates:")
print("min_x:", min_x)
print("min_y:", min_y)
print("max_x:", max_x)
print("max_y:", max_y)

# %%
print('CRS: ', gdf.crs)
with open(f"city_{config['CITY']}_pro/raw_graph"+'.pkl', 'rb') as f:
    road_network = pickle.load(f)

# %%
gdf_edges = ox.graph_to_gdfs(road_network, nodes=False, edges=True)

# %%
gdf_edges.set_crs("EPSG:4326")
gdf = gdf.set_crs("EPSG:3857")
gdf_edges.to_crs("EPSG:3857", inplace=True)

# All here is EPSG:3857


# %% Before Error Injection
gdf.rename(columns={'matched_road_id': 'ture_matched_road_id', 'road_geometry': 'ture_road_geometry'}, inplace=True)
gdf['ture_distance_to_matched_road'] = 0
gdf['ture_toBeMatched'] = True
gdf['Perfect'] = True
gdf['ture_matched_road_id'] = gdf.ture_matched_road_id.apply(tuple)

# %%
if config['INJECT_ERROR_TO_AREAS']:
    minx, miny, maxx, maxy = min_x, min_y, max_x, max_y
    widthx, widthy = maxx - minx, maxy - miny 
    minx, miny, maxx, maxy = minx, miny, maxx - widthx/2, maxy - widthy/2
    total_area = (maxx - minx) * (maxy - miny)
    target_removal_area = total_area * config['REMOVAL_AREA_PERC']  # 0.1%
    area_per_square = (config['SQUARE_SIDE_LENGTH'] ** 2)
    num_squares = int(np.ceil(target_removal_area / area_per_square))
    print(f"Target number of small squares: {num_squares}")

    np.random.seed(config['RANDOM_SEED'])  # for reproducibility
    small_boxes = []
    for _ in tqdm(range(num_squares), total=num_squares, desc="Removing Squares"):
        rand_x = np.random.uniform(minx, maxx - config['SQUARE_SIDE_LENGTH'])
        rand_y = np.random.uniform(miny, maxy - config['SQUARE_SIDE_LENGTH'])
        small_box = box(rand_x, rand_y, rand_x + config['SQUARE_SIDE_LENGTH'], rand_y + config['SQUARE_SIDE_LENGTH'])
        small_boxes.append(small_box)

    # Combine all the boxes into one GeoDataFrame
    small_boxes_gdf = gpd.GeoDataFrame(geometry=small_boxes, crs=gdf_edges.crs)

    # 4. Find edges to remove
    gdf_edges_ri = gdf_edges.reset_index()
    edges_to_remove = gpd.sjoin(gdf_edges_ri, small_boxes_gdf, how="inner", predicate="intersects")

    print(f"Found {len(edges_to_remove)} edges to remove.")

    edges_to_remove_list = list(zip(edges_to_remove['u'], edges_to_remove['v'], edges_to_remove['key']))

    gdf.ture_matched_road_id = gdf.ture_matched_road_id.apply(tuple) 
    nPoints_toremoved = gdf[gdf.matched_road_id.isin(edges_to_remove_list)].nPoints.sum()
    print('Number of points exists in these cells: ', nPoints_toremoved)

    removed_roads = set(edges_to_remove_list)
    nGoodPoints = nPoints_toremoved

else:
    removed_roads = set()
    nGoodPoints = 0

# %%
prefix = 'ture_'

# %%
roads_set = None
if roads_set is None:
    roads_set = gdf[prefix+'matched_road_id'].unique().tolist()
    roads_set = list(filter(lambda item: item is not None, roads_set))
roads_set = set(roads_set)
# Remove previously removed edges
roads_set = roads_set - removed_roads
gdf_grouped = gdf.groupby(prefix+'matched_road_id')['vehicule_id'].count()


#%% Compute centroids for each road
centroids = np.vstack([gdf_edges.geometry.centroid.x, gdf_edges.geometry.centroid.y]).T
kdtree = KDTree(centroids)
# config['REMOVAL_ROADS_GROUPING'] = True # TODO to be fixed
# config['REMOVAL_ROAD_MAXLENGTH_OPTION'] = True # TODO To be fixed

# %%
nPerfectPoints = len(gdf[gdf['Perfect']])
print('Good and Perfect: ', nGoodPoints, nPerfectPoints)
random.seed(config['RANDOM_SEED'])
with tqdm(desc="Selecting perfect roads for removal", total=config['GAMMA_O'] * nPerfectPoints) as pbar:
    pbar.update(nGoodPoints)
    while nGoodPoints < config['GAMMA_O'] * nPerfectPoints:
        r = random.sample(list(roads_set), 1)[0]

        # Condition 1: Accept long roads only
        if config['REMOVAL_ROAD_MAXLENGTH_OPTION']:
            if gdf_edges.loc[r].length < config['MAX_ROAD_LENGTH_O']: 
                roads_set.remove(r)
                continue
        
        if r not in removed_roads:
            removed_roads.add(r)
            nAssociatedPoints = gdf_grouped.get(r, 0)
            nGoodPoints += nAssociatedPoints
            roads_set.discard(r)
            pbar.update(nAssociatedPoints)
            
        r2 = (r[1], r[0], r[2]) # Reversing direction
        if r2 not in removed_roads:
            removed_roads.add(r2)
            nAssociatedPoints = gdf_grouped.get(r2, 0)
            nGoodPoints += nAssociatedPoints
            roads_set.discard(r2)
            pbar.update(nAssociatedPoints)

        # New: 
        if config['REMOVAL_ROADS_GROUPING']:
            r_idx = gdf_edges.index.get_loc(r)  # get integer index
            r_coord = centroids[r_idx].reshape(1, -1)
            nearby_indices = kdtree.query_radius(r_coord, r=config['NEAREST_ROAD2ROAD_RANGE'])[0]  # 100 meters radius
            
            nearby_roads = gdf_edges.iloc[nearby_indices].index.tolist()

            # Now include all nearby roads
            for nearby_r in nearby_roads:
                if nearby_r not in roads_set:
                    continue
                
                if nearby_r not in removed_roads:
                    removed_roads.add(nearby_r)
                    nAssociatedPoints = gdf_grouped.get(nearby_r, 0)
                    nGoodPoints += nAssociatedPoints
                    roads_set.discard(nearby_r)
                    pbar.update(nAssociatedPoints)
                    
                r2 = (nearby_r[1], nearby_r[0], nearby_r[2])  # reverse direction
                if r2 not in removed_roads:
                    removed_roads.add(r2)
                    nAssociatedPoints = gdf_grouped.get(r2, 0)
                    nGoodPoints += nAssociatedPoints
                    roads_set.discard(r2)
                    pbar.update(nAssociatedPoints)        
        
print('Good and Perfect: ', nGoodPoints, nPerfectPoints)
# %%
removed_roads = list(removed_roads)

# %%
if config['VERBOSE']: print('Removing ', len(removed_roads), ' (', config['GAMMA_O']*100, '%) of provided roads from OSM graph...')
road_network.remove_edges_from(removed_roads)
gdf[prefix+'toBeMatched'] = True
gdf.loc[gdf[prefix+'matched_road_id'].isin(removed_roads), prefix+'toBeMatched'] = False 

# %%
number_of_p = len(gdf)
number_of_p_make = len(gdf[gdf[prefix+'toBeMatched'] == False])
p_make_percentage = number_of_p_make/number_of_p
if config['VERBOSE']:
    print(f"    {prefix}P_make Points(", 100*p_make_percentage, "%): ", number_of_p_make)

# %% 
# ADDING NOISE
prefix = 'ture_'
gdf[prefix+'isAddedNoise'] = False
        
random.seed(config['RANDOM_SEED'])
np.random.seed(config['RANDOM_SEED'])
gdf.lon, gdf.lat = gdf.geometry.x, gdf.geometry.y

# %%
# Adding to Matching Points
# minx, miny, maxx, maxy = gdf_edges.total_bounds
if config['INJECT_NOISE_TO_REGIONS']:
    minx, miny, maxx, maxy = min_x, min_y, max_x, max_y
    widthx, widthy = maxx - minx, maxy - miny 
    minx, miny, maxx, maxy = minx, miny, maxx - widthx/2, maxy
    bbox = box(minx, miny, maxx, maxy)
    matched_index = gdf[gdf.within(bbox) & gdf[prefix+'toBeMatched'] == True].index
else:
    matched_index = gdf[gdf[prefix+'toBeMatched'] == True].index

if config['P_NOISE_O'] == 1: noise_matched_index = matched_index
else: noise_matched_index = sorted(random.sample(list(matched_index), int(config['P_NOISE_O']*len(matched_index))))

if config['VERBOSE']: print('Adding noise to ', len(gdf.loc[noise_matched_index]), '(', config['P_NOISE_O']*100, '%) of Matching points...')
noise = np.random.normal(config['MU_O'], config['SIGMA_O'], [len(noise_matched_index),2])

# %%
gdf.loc[noise_matched_index, ['lat', 'lon']] += noise #.astype(int)
gdf.loc[noise_matched_index, prefix+'isAddedNoise'] = True

gdf.geometry = gpd.points_from_xy(gdf.lon, gdf.lat)

# %%
matcher = MapMatcher(road_network)
gdf = matcher.match(gdf, advanced_matching=True)


# %%
gdf_edges = ox.graph_to_gdfs(road_network, nodes=False, edges=True)

# %%

for i, row in gdf.iterrows():
    if row['matched_road_id'] not in gdf_edges.index:
        print('Not found')

# %%
gdf['road_geometry'] = gdf['road_geometry'].apply(lambda geom: geom.wkt)
gdf_filename = f"city_{config['CITY']}_pro/error_inj_g{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.parquet"
gdf.to_parquet(gdf_filename, compression='snappy')

# %%
road_network_file = f"city_{config['CITY']}_pro/road_network_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.pkl"
with open(road_network_file, 'wb') as file:
    pickle.dump(road_network, file)
# %%
