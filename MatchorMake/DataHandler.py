import pandas as pd
import geopandas as gpd
import time
from mappymatch.utils.crs import XY_CRS
import os
import copy
import pickle
from shapely import wkt

def read_parqet_file(path, verbose=True):
    """
    Read the parqet data file from the dataset
    Args:
        path: [String] path to the parqet file
    """
    df =  pd.read_parquet(path)
    if verbose: print('Number of points: ', len(df))
    return df

def convert_data_to_gdf(df):
    """
    Convert the provided Pandas DataFrame of Points which include longitude and lattitude to a GeoPandas DataFrame
    Args:
        df: [DataFrame] the dataframe of the points
    """
    print('Converting the DataFrame into GeoPandas DataFrame')
    start = time.time()

    print('Reading timestamps...')
    print('Type of timestamp is ', df['pingtimestamp'].dtype)
    # df.loc[:, 'pingtimestamp'] = pd.to_datetime(df['pingtimestamp'], unit='s')
    # df.reset_index(drop=True, inplace=True)
    df.rename(columns={"trj_id": "vehicule_id", "rawlat": "lat", "rawlng": "lon", "pingtimestamp": "timestamp", "bearing": "angle"}, inplace=True)
    # df.timestamp = df.timestamp.dt.strftime('%Y-%m-%d %H:%M:%S+03')
    
    print('Creating GeoDataFrames...')
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    gdf.to_crs(crs=XY_CRS, inplace=True) 
    
    print('Sorting by vehicule_id and timestamp')
    gdf.sort_values(by=['vehicule_id', 'timestamp'], inplace=True)
    gdf.reset_index(drop=True, inplace=True)
    end = time.time()
    print('Took ', end-start, ' seconds')
    return gdf

def combine_uncertain_and_perfect(uncertain_points, perfect_points):
    # uncertain_points['toBeMatched'] = None
    perfect_points['Perfect'] = True
    uncertain_points['Perfect'] = False
    total_points = gpd.pd.concat([perfect_points, uncertain_points])
    # total_points = gpd.GeoDataFrame/(pd.concat([perfect_points, uncertain_points], ignore_index=True))
    total_points.sort_values(by=['vehicule_id', 'timestamp'], inplace=True)
    return total_points

def read_file(data_folder, filename):
    df = read_parqet_file(os.path.join(data_folder, filename))
    gdf = convert_data_to_gdf(df)
    return gdf

def read_dataset(data_folder):
    list_of_files = os.listdir(data_folder)
    dfs = []
    for filename in list_of_files:
        df = read_parqet_file(os.path.join(data_folder, filename))
        dfs.append(df)
    df_total = pd.concat(dfs)
    gdf = convert_data_to_gdf(df_total)
    return gdf

def save_gdf(gdf, file_path, col_tuple_list=[], col_geom_list=[]):
    gdf_c = copy.deepcopy(gdf)
    for col in col_tuple_list:
        gdf_c[col] = gdf_c[col].apply(lambda arr: tuple(arr))
    
    for col in col_geom_list:
        gdf_c[col] = gdf_c[col].apply(lambda geom: geom.wkt)

    gdf_c.to_parquet(file_path, index=True, compression="gzip", engine="pyarrow")
    del gdf_c
    
def load_gdf(file_path, col_tuple_list=[], col_geom_list=[]):
    gdf = gpd.read_parquet(file_path)

    for col in col_tuple_list:
        gdf[col] = gdf[col].apply(lambda arr: tuple(arr))
    
    for col in col_geom_list:
        gdf[col] = gdf[col].apply(wkt.loads)
    
    return gdf

def get_points_outside(gdf, partitioner):
    sindex = gdf.sindex
    return gdf.drop(list(sindex.intersection(partitioner.get_boundary().bounds)))

def get_points_within(gdf, partitioner):
    sindex = gdf.sindex
    return gdf.iloc[list(sindex.intersection(partitioner.get_boundary().bounds))]

def get_features_and_labels_for_model(gdf, feature_names, label_colname, is_copy=True):
    imp_features = feature_names
    imp_features.append(label_colname)
    # imp_features = ['total_number_of_nearby_points', 'n_high_match_points',
    #                 'n_low_match_points', 'high_match_ratio', 'low_match_ratio',
    #                 'n_distinct_high_matched_roads', 'n_distinct_low_matched_roads',
    #                 'n_distinct_matched_roads',  'n_distinct_match_directions', 
    #                 'n_distinct_low_match_directions', 'n_distinct_high_match_directions', 
    #                 label_colname]
    
    if is_copy: new_gdf = copy.deepcopy(gdf)
    else: new_gdf = gdf

    new_gdf = new_gdf[imp_features]

    y = new_gdf[label_colname]
    X = new_gdf.drop(columns=[label_colname])
    
    return X, y

def convert_types(X, y):
    y = y.astype(bool)#.to_numpy()
    
    float64_cols = X.select_dtypes(include=['float64']).columns
    X[float64_cols] = X[float64_cols].astype('float32')
    
    int64_cols = X.select_dtypes(include=['int64']).columns
    X[int64_cols] = X[int64_cols].astype('int32')
    # X.to_numpy()
    
    return X, y

def get_train_from_split(gdf, feature_names, label_name='toBeMatched', is_copy=True):
    perfect_points = gdf[gdf.Perfect]

    X_train, y_train = get_features_and_labels_for_model(perfect_points,   feature_names, label_colname=label_name, is_copy=is_copy)
    X_train, y_train = convert_types(X_train, y_train)
    
    return X_train, y_train

def get_test_from_split(gdf, feature_names, label_name='ture_toBeMatched', is_copy=True):
    uncertain_points = gdf[~gdf.Perfect]

    X_test,  y_test  = get_features_and_labels_for_model(uncertain_points, feature_names, label_colname=label_name, is_copy=is_copy)    
    X_test, y_test = convert_types(X_test, y_test)

    return X_test, y_test

def split_train_test(gdf, feature_names, is_copy=False): #It has to contain the 'Perfect' Column
    X_train, y_train = get_train_from_split(gdf, feature_names)
    X_test, y_test = get_test_from_split(gdf, feature_names)
    return X_train, X_test, y_train, y_test

def save_obj(obj, path):
        with open(path, 'wb') as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
            
def load_obj(path):
    with open(path, 'rb') as file:
        road_network = pickle.load(file)
    return road_network