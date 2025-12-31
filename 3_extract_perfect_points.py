# %%
import geopandas as gpd
import pickle
from load_config import load_config

# config = load_config('configs/jakarta_m.json', same_gamma=True)
config = load_config(same_gamma=False)

# %%
gdf_name = f"city_{config['CITY']}_pro/gdfC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.parquet"
gdf = gpd.read_parquet(gdf_name)
gdf.matched_road_id = gdf.matched_road_id.apply(tuple)

# %%
'''
Extracting Perfect Instances & Uncertain ones. Keep in mind this function duplicates the 
'''
prefix = ''
candidate_roads = gdf[prefix + 'matched_road_id'].unique()

road_to_perfect_count = dict.fromkeys(candidate_roads, 0)
road_to_perfect_points = {road: [] for road in candidate_roads}
road_to_match_count = dict.fromkeys(candidate_roads, 0)

from tqdm import tqdm
for index, row in tqdm(gdf.iterrows(), desc='Counting Perfect points', total=len(gdf)):
    n = row['nPoints']
    if row[prefix + 'distance_to_matched_road'] <= config['DELTA']:
        road_to_perfect_count[row['matched_road_id']] += n
        road_to_perfect_points[row['matched_road_id']].append(index)
    road_to_match_count[row['matched_road_id']] += n

# %%
perfect_roads = []
perfect_points = []
candidate_roads = road_to_perfect_count.keys()
for r in tqdm(candidate_roads, desc='Getting Perfect Roads', total=len(candidate_roads)):
    
    if (road_to_perfect_count[r] / road_to_match_count[r]) >= config['BETA']:
        perfect_roads.append(r)

# %%
for r in tqdm(perfect_roads):
    perfect_points.extend(road_to_perfect_points[r])

# %%
gdf['type'] = 'uncertain'
gdf.loc[perfect_points, 'type'] = 'perfect'

nPerfectPoints = gdf.loc[perfect_points, 'nPoints'].sum()
nPoints = gdf['nPoints'].sum()

if True:
    print("Total Number of Points: ", nPoints)
    print("   Number of Perfectly-Matched Points: ", nPerfectPoints)
    print("   Uncertain Points (inc. None) (", 100*(nPoints- nPerfectPoints)/nPoints, "%): ", nPoints- nPerfectPoints)
    print("   Perfect Roads: ", len(perfect_roads))


# %%
print('verifying the accuracy of the perfect extraction')

make_being_perfect = gdf[gdf['type'] == 'perfect']['nMakePoints'].sum() / gdf['nMakePoints'].sum()
match_being_perfect = gdf[gdf['type'] == 'perfect']['nMatchPoints'].sum() / gdf['nMatchPoints'].sum()

print('Sould be high: MatchBeingPerfect->', match_being_perfect)
print('should be low: MakeBeingPerfect', make_being_perfect)

# # vals = {'noNoiseBeingPerfect': no_noise_match_being_perfect, 'MatchBeingPerfect': match_being_perfect, 'MakeBeingPerfect': make_being_perfect}
# # with open(f"city_{config['CITY']}_pro/perfect_b{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.json", "w") as f:
# #     json.dump(vals, f)

# road_network_file = f"city_{config['CITY']}_pro/road_network_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.pkl"
# with open(road_network_file, 'rb') as f:
#     road_network = pickle.load(f)
# import osmnx as ox
# gdf_edges = ox.graph_to_gdfs(road_network, nodes=False, edges=True)

# for r in perfect_roads:
#     if r not in gdf_edges.index:
#         print('Not found')


# %%
# gdf['road_geometry'] = gdf['road_geometry'].apply(lambda geom: geom.wkt)
gdf_output_name = f"city_{config['CITY']}_pro/perfect_b{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.parquet"
gdf.to_parquet(gdf_output_name, compression='snappy')

perfect_roads_file = f"city_{config['CITY']}_pro/perfect_roads_b{int(100*config['BETA'])}g{config['DELTA']}_PreC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.pkl"
with open(perfect_roads_file, 'wb') as file:
    pickle.dump(perfect_roads, file)

# %%
