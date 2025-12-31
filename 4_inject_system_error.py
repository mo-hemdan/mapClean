# %%
import geopandas as gpd
import pickle
import random
from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
import osmnx as ox
from tqdm import tqdm
tqdm.pandas()
import json
from shapely import wkt
from shapely.geometry import base
from load_config import load_config
# config = load_config("configs/jakarta_m.json", same_gamma=False)
config = load_config(same_gamma=False)
# config['SUPER_POINT_SIZE'] = 1
def to_wkt_safe(g):
    if isinstance(g, base.BaseGeometry):
        return g.wkt
    return g   # already string or None
# %% Reading the files
gdf_name = f"city_{config['CITY']}_pro/perfect_b{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.parquet"
gdf = pd.read_parquet(gdf_name, columns=['lat', 'lon', 'speed', 'angle', 'road_angle', 'matched_road_id', 'distance_to_matched_road', 'r_p_sim', 'nPoints', 'type', 'road_geometry'])
gdf.matched_road_id = gdf.matched_road_id.apply(tuple)

gdf_original_points_name = f"city_{config['CITY']}_pro/original_points_C{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.pkl"
gdf_original_points = pd.read_pickle(gdf_original_points_name)

gdf = pd.concat([gdf, gdf_original_points], axis=1)

# %%
road_network_file = f"city_{config['CITY']}_pro/road_network_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.pkl"
with open(road_network_file, 'rb') as f:
    road_network = pickle.load(f)

perfect_roads_file = f"city_{config['CITY']}_pro/perfect_roads_b{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.pkl"
with open(perfect_roads_file, 'rb') as f:
    perfect_roads = pickle.load(f)


# %%
gdf_edges = ox.graph_to_gdfs(road_network, nodes=False, edges=True)
gdf_edges.to_crs("EPSG:3857", inplace=True)

# %% Preparing for the removal of roads
removed_roads = set()
nGoodPoints = 0
prefix = ''
roads_set = perfect_roads

if roads_set is None:
    roads_set = gdf[prefix+'matched_road_id'].unique().tolist()
    roads_set = list(filter(lambda item: item is not None, roads_set))
roads_set = set(roads_set)
roads_set = roads_set - removed_roads

gdf_grouped = gdf[gdf['type'] == 'perfect'].groupby(prefix+'matched_road_id')['nPoints'].sum()

# Compute centroids for each road
gdf_edges_notRemoved = gdf_edges[gdf_edges.index.isin(roads_set)].copy()
centroids = np.vstack([gdf_edges.geometry.centroid.x, gdf_edges.geometry.centroid.y]).T
kdtree = KDTree(centroids)

nPerfectPoints = gdf.loc[gdf['type'] == 'perfect', 'nPoints'].sum()
print('Good and Perfect: ', nGoodPoints, nPerfectPoints)

#%%
random.seed(config['RANDOM_SEED'])

with tqdm(desc="Selecting perfect roads for removal", total=config['GAMMA'] * nPerfectPoints) as pbar:
    pbar.update(nGoodPoints)
    while nGoodPoints < config['GAMMA'] * nPerfectPoints:
        # print('Length: ', len(roads_set))
        if len(roads_set) == 0: break
        r = random.sample(list(roads_set), 1)[0]
        
        # Condition 1: Accept long roads only
        if config['REMOVAL_ROAD_MAXLENGTH_OPTION']: # TODO: Fix this problem of try and except how come a road in roads set not be in gdf_edges 
            if gdf_edges.loc[r].length < config['MAX_ROAD_LENGTH']: 
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

removed_roads = list(removed_roads)

# %%
if config['VERBOSE']: print('Removing ', len(removed_roads), ' of provided roads from OSM graph...')

g_removed_edges = road_network.edge_subgraph(removed_roads).copy()
road_network.remove_edges_from(removed_roads)

#%%
gdf[prefix+'toBeMatched'] = None
gdf.loc[gdf['type'] != 'uncertain', prefix+'toBeMatched'] = True
gdf.loc[(gdf['type'] != 'uncertain') & gdf[prefix+'matched_road_id'].isin(removed_roads), prefix+'toBeMatched'] = False 
gdf.loc[(gdf['type'] != 'uncertain') & gdf[prefix+'matched_road_id'].isin(removed_roads), 'type'] = 'good' 

# %%
number_of_p = gdf.loc[gdf['type'] != 'uncertain', 'nPoints'].sum()
number_of_p_make = gdf.loc[gdf['type'] == 'good', 'nPoints'].sum()
p_make_percentage = number_of_p_make/number_of_p
if config['VERBOSE']:
    print(f"    {prefix}P_make Points(", 100*p_make_percentage, "%): ", number_of_p_make)

# %%  ADDING NOISE 
verbose = True
prefix = ''
gdf[prefix+'isAddedNoise'] = None
gdf.loc[gdf['type'] != 'uncertain', prefix+'isAddedNoise'] = False
# gdf[prefix+'isAddedNoise'] = False


random.seed(config['RANDOM_SEED'])
np.random.seed(config['RANDOM_SEED'])
import geopandas as gpd

# Step 1: Create geometry from lon/lat (in degrees)
gdf = gpd.GeoDataFrame(
    gdf,
    geometry=gpd.points_from_xy(gdf['lon'], gdf['lat']),
    crs="EPSG:3857"
)

gdf.lon, gdf.lat = gdf.geometry.x, gdf.geometry.y

# Adding to Matching Points
gdf.loc[gdf['type'] == 'perfect', 'type'] = 'old' # Changing them all to old points
matched_index = list(gdf[gdf['type'] == 'old'].index)
# noise_matched_index = sorted(random.sample(matched_index, int(1*len(matched_index))))
# %%

if config['P_NOISE'] == 1: noise_matched_index = matched_index
else: noise_matched_index = sorted(random.sample(list(matched_index), int(config['P_NOISE']*len(matched_index))))


if verbose: print('Adding noise to ', gdf.loc[noise_matched_index, 'nPoints'].sum(), '(', config['P_NOISE']*100, '%) of Matching points...')
noise = np.random.normal(config['MU'], config['SIGMA'], [len(noise_matched_index),2])

# %%
bad_points = gdf.loc[noise_matched_index].copy(deep=True)

# %%
bad_points.index = range(gdf.index.max()+1, gdf.index.max()+1 + len(bad_points))
bad_points['type'] = 'bad'
bad_points[['lat', 'lon']] += noise.astype(int) # TODO: Make sure the lat and long are the same as polygon or geometries
bad_points[prefix+'isAddedNoise'] = True
bad_points.geometry = gpd.points_from_xy(bad_points.lon, bad_points.lat)


# %% Map Matching

good_points = gdf[gdf['type'] == 'good'].copy(deep=True)
gdf = gdf[gdf['type'] != 'good'].copy(deep=True)

# %% MAP MATCHING

from MapMatcher import MapMatcher
matcher = MapMatcher(road_network)

bad_points_original_points = bad_points.original_dataframe.to_frame()
bad_points = bad_points.drop(columns=['original_dataframe'])

bad_points = matcher.match(bad_points, advanced_matching=True)
bad_points['road_geometry'] = bad_points['road_geometry'].apply(lambda geom: geom.wkt)

bad_points = pd.concat([bad_points, bad_points_original_points], axis=1)

if nGoodPoints > 0:
    good_points_original_points = good_points.original_dataframe.to_frame()
    good_points = good_points.drop(columns=['original_dataframe'])

    good_points = matcher.match(good_points, advanced_matching=True, prefix='new_')
    
    good_points = pd.concat([good_points, good_points_original_points], axis=1)
else:
    good_points['new_matched_road_id'] = None
    good_points['new_distance_to_matched_road'] = None
    good_points['new_road_geometry'] = None
    good_points['new_road_angle'] = None
    good_points['new_r_p_sim'] = None
    good_points['new_road_length'] = None

good_points['new_road_geometry'] = good_points['new_road_geometry'].apply(lambda geom: geom.wkt)

# %%
for c in ['matched_road_id', 'distance_to_matched_road', 'road_geometry', 'road_angle', 'r_p_sim']: #, 'road_length']:
    gdf['new_' + c] = gdf[c]
    bad_points['new_' + c] = bad_points[c]
    
# %%

gdf = pd.concat([gdf, good_points, bad_points], ignore_index=False)


# %% Reading the partioner to change representation
partitioner_filename = f"city_{config['CITY']}_pro/INDEXParitionerC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.pkl"
with open(partitioner_filename, 'rb') as file:
    partitioner = pickle.load(file)

cell_to_points_filename = f"city_{config['CITY']}_pro/INDEXC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.pkl"
with open(cell_to_points_filename, 'rb') as file:
    cell_to_points = pickle.load(file) 
    
# %%
bad_points = gdf[gdf['type'] == 'bad'].copy(deep=True)
# bad_points.geometry = bad_points.geometry.apply(wkt.loads)
bad_points['new_cell_id'] = bad_points.geometry.progress_apply(partitioner.calc_cell_id)
bad_cell_to_points = bad_points.groupby('new_cell_id').apply(lambda x: x.index.tolist()).to_dict()
for cell in bad_cell_to_points:
    if cell not in cell_to_points:
        cell_to_points[cell] = bad_cell_to_points[cell]
    else:
        cell_to_points[cell].extend(bad_cell_to_points[cell])

# %%
gdf_new = pd.concat([gdf[gdf['type']=='good'].copy(), gdf[gdf['type']=='bad'].copy(), gdf[gdf['type']=='uncertain'].copy(), gdf[gdf['type']=='old'].copy()], axis=0)
new_index_map = {idx: i for i, idx in enumerate(gdf_new.index)}
for cell in cell_to_points:
    cell_to_points[cell] = [new_index_map[p] for p in cell_to_points[cell]]
gdf_new.reset_index(drop=True, inplace=True)

# %%
output_gdf = gdf_new[gdf_new['type'] != 'old']

gdf_original_dfs = output_gdf.original_dataframe.copy()
gdf_original_dfs.to_pickle(f"city_{config['CITY']}_pro/OriginalDFs_g{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.pkl")

# , 'nMatchPoints', 'nMakePoints'
nPoints_df = output_gdf[['nPoints']].copy()
nPoints_df.to_csv(f"city_{config['CITY']}_pro/nPointsDf_g{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.csv")


# %%
keep_columns = ['lat', 'lon', 'speed', 'angle',  'nPoints', \
       'matched_road_id', 'distance_to_matched_road', \
       'type', 'toBeMatched','new_matched_road_id', \
        'new_distance_to_matched_road', \
        'new_road_angle', 'new_r_p_sim'
        ]

remove_columns = list(set(list(gdf_new.columns)) - set(keep_columns))
gdf_new.drop(columns=remove_columns, inplace=True)

gdf_new['lat']  = gdf_new['lat'].astype(int)
gdf_new['lon']  = gdf_new['lon'].astype(int)
gdf_new['speed']  = gdf_new['speed'].astype(float)
gdf_new['angle']  = gdf_new['angle'].astype(int)
gdf_new['nPoints'] = gdf_new['nPoints'].astype(int)
gdf_new['toBeMatched'] = gdf_new['toBeMatched'].map({True:'true', False:'false'})
def clean_tuple(t):
    # t = (np.int64, np.int64, np.int64)
    return f"({int(t[0])} {int(t[1])} {int(t[2])})"
gdf_new['matched_road_id'] = gdf_new['matched_road_id'].apply(clean_tuple)
gdf_new['new_matched_road_id'] = gdf_new['new_matched_road_id'].apply(clean_tuple)



# %%
ord_columns = ['lat', 'lon', 'speed', 'angle', 'nPoints', 'matched_road_id', 'distance_to_matched_road', 'type', 'toBeMatched', 'new_matched_road_id', 'new_distance_to_matched_road', 'new_road_angle', 'new_r_p_sim']
gdf_new = gdf_new[ord_columns]

#%% Saving the modified index
new_dict = {str(list(k)) :v for k, v in cell_to_points.items()}
with open(f"city_{config['CITY']}_pro/ErrorIndex_g{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.json", "w") as f:
    json.dump(new_dict, f)

with open(f"city_{config['CITY']}_pro/graph_g{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.pkl", 'wb') as file:
    pickle.dump(road_network, file)


# gdf_new.matched_road_id = gdf_new.matched_road_id.apply(
#     lambda t: [int(x) for x in t] if isinstance(t, tuple) else t
# )
# gdf_new.new_matched_road_id = gdf_new.new_matched_road_id.apply(
#     lambda t: [int(x) for x in t] if isinstance(t, tuple) else t
# )
gdf_new.to_csv(f"city_{config['CITY']}_pro/systemError_g{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.csv", index=False)

# Create the dictionary with your values
metadata = {
    'cell_width_x': partitioner.cell_width_x,
    'cell_width_y': partitioner.cell_width_y,
    'total_points': gdf_new.shape[0],
    'non_old_points': output_gdf.shape[0],
    'n_cells_x': partitioner.n_cells_x,
    'n_cells_y': partitioner.n_cells_y
}


with open(f"city_{config['CITY']}_pro/Format_g{int(100*config['GAMMA'])}s{config['SIGMA']}_PMEb{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.json", 'w') as f:
    json.dump(metadata, f, indent=4)

# %%
