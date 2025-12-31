# %%
import geopandas as gpd
from SuperPointReducer import SuperPointReducer
import pickle
from MatchorMake.components.Partitioner import Partitioner
from tqdm import tqdm
tqdm.pandas()
from load_config import load_config
from shapely.geometry import base
def to_wkt_safe(g):
    if isinstance(g, base.BaseGeometry):
        return g.wkt
    return g   # already string or None
# config = load_config('configs/jakarta_m.json', same_gamma=True)
config = load_config(same_gamma=False)

# %%
print('Reading the GDF')
gdf_filename = f"city_{config['CITY']}_pro/error_inj_g{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.parquet"
gdf = gpd.read_parquet(gdf_filename)
gdf.ture_matched_road_id = gdf.ture_matched_road_id.apply(tuple)
gdf.matched_road_id = gdf.matched_road_id.apply(tuple)

# %% Merging points into superpoints
print('Reducing points into SuperPoints')
super_point_reducer = SuperPointReducer(config['SUPER_POINT_SIZE'])
gdf = super_point_reducer.apply(gdf)

# %%
print('Creating Index')
partitioner = Partitioner()
partitioner.partition(gdf, config['CELL_WIDTH'])
gdf['cell_id'] = gdf.geometry.progress_apply(partitioner.calc_cell_id)
# TODO: Replace it with gdf.groupby('cell_id').indices
cell_to_points = gdf.groupby('cell_id').apply(lambda x: x.index.tolist()).to_dict()

# %%
print('saving gdf')
gdf_original_points = gdf.original_dataframe.to_frame()
gdf = gdf.drop(columns=['original_dataframe'])

gdf_original_points_name = f"city_{config['CITY']}_pro/original_points_C{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.pkl"
gdf_original_points.to_pickle(gdf_original_points_name)

gdf.road_geometry = gdf.road_geometry.apply(to_wkt_safe)

gdf_name = f"city_{config['CITY']}_pro/gdfC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.parquet"
gdf.to_parquet(gdf_name, compression='snappy')

# %%
print('Saving Index and partitioner')
with open(f"city_{config['CITY']}_pro/INDEXC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.pkl", 'wb') as file:
    pickle.dump(cell_to_points, file)

with open(f"city_{config['CITY']}_pro/INDEXParitionerC{config['CELL_WIDTH']}sup{config['SUPER_POINT_SIZE']}_OERg{int(100*config['GAMMA_O'])}s{config['SIGMA_O']}p{int(100*config['P_NOISE_O'])}gr{int(config['REMOVAL_ROADS_GROUPING_O'])}mr{int(config['REMOVAL_ROAD_MAXLENGTH_OPTION_O'])}.pkl", 'wb') as file:
    pickle.dump(partitioner, file)

# %%
