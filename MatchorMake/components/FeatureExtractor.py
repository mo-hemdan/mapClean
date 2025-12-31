import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

class FeatureExtractor():
    def __init__(self, tau):
        self.tau = tau
        self.initial_features_values = {
            'n_nearby_points': 0.0,
            'n_high_match_weighted': 0.0,
            'n_low_match_weighted': 0.0,
            'high_match_ratio': 0.0,
            'low_match_ratio': 0.0,
            'n_distinct_high_weighted_directions': 0.0,
            'n_distinct_low_weighted_directions': 0.0,
            'n_distinct_high_weighted_roads': 0.0,
            'n_distinct_low_weighted_roads': 0.0,
        }
        # self.auxiliary_features = ['timestamp', 'driving_mode', 'osname', 'speed', 'angle', 'accuracy', 'lat', 'lon', 'distance_to_matched_road']
        self.auxiliary_features = ['speed', 'angle', 'accuracy', 'distance_to_matched_road'] #'lat', 'lon', 'timestamp', 
        self.prefix = ''
    
    def set_prefix(self, prefix):
        self.auxiliary_features.remove(self.prefix+'distance_to_matched_road')
        self.prefix = prefix
        self.auxiliary_features.append(self.prefix+'distance_to_matched_road')

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)  
    
    def get_feature_names(self):
        return list(self.initial_features_values.keys()) + self.auxiliary_features
    
    # TODO: For now do it for all values in the dataframe
    def extract_features(self, gdf, query_executor, list_of_ids=None):

        if list_of_ids is not None: list_of_ids_set = set(list_of_ids)
        gdf = gdf.assign(**self.initial_features_values)
        gdf['discrete_direction'] = (np.floor((gdf.angle + 22.5) % 360 / 45) + 1).astype(int)

        for cell_id in tqdm(query_executor.get_cell_ids(), desc="Processing Cells"):
            
            # Get the nearby points and current cell points
            nearby_cells_points = query_executor.get_nearbyPoints_for_cell(cell_id)
            points = set(query_executor.get_points_in_cell(cell_id))
            all_points_ids = set(nearby_cells_points).union(set(points))
            
            if list_of_ids is not None: points = points.intersection(list_of_ids_set)
            for point_id in points:
                nearby_points_ids = all_points_ids - {point_id}
                
                if len(nearby_points_ids) == 0:
                    print('WARNING: No Nearby Points are found')
                    continue

                nearby_points = gdf.loc[list(nearby_points_ids)]
                point_row = gdf.loc[point_id]
                distances = nearby_points.distance(point_row.geometry).to_dict() #TODO: There is a problem here

                self.add_weighted_features_vec(point_id, nearby_points, distances, gdf)
        return gdf
    
    def nearby_query(self, query_executor, list_of_ids=None):

        if list_of_ids is not None: list_of_ids_set = set(list_of_ids)
        nearby_point_dict = dict()

        for cell_id in tqdm(query_executor.get_cell_ids(), desc="Processing Cells"):
            
            # Get the nearby points and current cell points
            nearby_cells_points = query_executor.get_nearbyPoints_for_cell(cell_id)
            points = set(query_executor.get_points_in_cell(cell_id))
            all_points_ids = set(nearby_cells_points).union(set(points))
            
            if list_of_ids is not None: points = points.intersection(list_of_ids_set)
            for point_id in points:
                nearby_points_ids = all_points_ids - {point_id}
                nearby_point_dict[point_id] = nearby_points_ids

        return nearby_point_dict

    def add_weighted_features_vec(self, point_id, nearby_points_df, distances, gdf):
        # Ensure distances for all nearby points with default distance of 1
        distances_series = nearby_points_df.index.map(distances).fillna(1)
        distances_series = pd.Series(distances_series, index=nearby_points_df.index)

        # Calculate weights as inverse of distances (handling division by zero)
        weights = 1 / distances_series.replace(0, 1)

        # Total count weighted by distance
        n_nearby_points = weights.sum()

        # Boolean masks for high and low match points
        high_match_mask = nearby_points_df[self.prefix + 'distance_to_matched_road'] < self.tau
        low_match_mask = ~high_match_mask

        # Calculate weighted counts for high and low matches
        n_high_match_weighted = weights[high_match_mask].sum()
        n_low_match_weighted = weights[low_match_mask].sum()
        
        n_high_match_ratio = n_high_match_weighted / n_nearby_points if n_nearby_points > 0 else 0
        n_low_match_ratio = n_low_match_weighted / n_nearby_points if n_nearby_points > 0 else 0
        
        # Count distinct weighted roads and directions for high match
        n_distinct_high_weighted_roads = nearby_points_df.loc[high_match_mask, self.prefix + 'matched_road_id'].nunique() * n_high_match_ratio#weights[high_match_mask]
        n_distinct_high_weighted_directions = nearby_points_df.loc[high_match_mask, 'discrete_direction'].nunique() * n_low_match_ratio#weights[high_match_mask]

        # Count distinct weighted roads and directions for low match
        n_distinct_low_weighted_roads = nearby_points_df.loc[low_match_mask, self.prefix + 'matched_road_id'].nunique() * n_high_match_ratio#weights[low_match_mask]
        n_distinct_low_weighted_directions = nearby_points_df.loc[low_match_mask, 'discrete_direction'].nunique() * n_low_match_ratio#weights[low_match_mask]

        # Create a dictionary of updates
        updates = {
            'n_nearby_points': n_nearby_points,
            'n_high_match_weighted': n_high_match_weighted,
            'n_low_match_weighted': n_low_match_weighted,
            'high_match_ratio': n_high_match_ratio,
            'low_match_ratio': n_low_match_ratio,
            'n_distinct_high_weighted_roads': n_distinct_high_weighted_roads,
            'n_distinct_low_weighted_roads': n_distinct_low_weighted_roads,
            'n_distinct_high_weighted_directions': n_distinct_high_weighted_directions,
            'n_distinct_low_weighted_directions': n_distinct_low_weighted_directions
        }

        # Update the gdf using loc to set multiple values at once
        gdf.loc[point_id, list(updates.keys())] = list(updates.values())
    
    def add_weighted_features(self, point_id, nearby_points_df, distances, gdf):

        # # Calculate high and low match points based on distance to matched road
        # high_match_points = nearby_points_df[nearby_points_df['distance_to_matched_road'] < self.tau]
        # low_match_points = nearby_points_df[nearby_points_df['distance_to_matched_road'] >= self.tau]

        # Initialize weighted sums and counters
        n_nearby_points = 0
        n_high_match_weighted = 0
        n_low_match_weighted = 0
        n_distinct_high_weighted_roads = {}
        n_distinct_low_weighted_roads = {}
        n_distinct_high_weighted_directions = {}
        n_distinct_low_weighted_directions = {}

        # Iterate over nearby points and calculate weighted features
        for idx, row in nearby_points_df.iterrows():
            dist = distances[idx]  # Default to 1 if distance is missing
            # print(dist)
            weight = 1 / dist if dist > 0 else 1  # Define weight inversely proportional to distance

            # Total count weighted by distance
            n_nearby_points += weight
            
            # High and low match weighted points
            if row[self.prefix + 'distance_to_matched_road'] < self.tau:
                n_high_match_weighted += weight
                n_distinct_high_weighted_roads[row[self.prefix + 'matched_road_id']] = n_distinct_high_weighted_roads.get(row[self.prefix + 'matched_road_id'], 0) + weight
                n_distinct_high_weighted_directions[row['discrete_direction']] = n_distinct_high_weighted_directions.get(row['discrete_direction'], 0) + weight
            else:
                n_low_match_weighted += weight
                n_distinct_low_weighted_roads[row[self.prefix + 'matched_road_id']] = n_distinct_low_weighted_roads.get(row[self.prefix + 'matched_road_id'], 0) + weight
                n_distinct_low_weighted_directions[row['discrete_direction']] = n_distinct_low_weighted_directions.get(row['discrete_direction'], 0) + weight

        # Update gdf with computed features
        gdf.at[point_id, 'n_nearby_points'] = n_nearby_points
        gdf.at[point_id, 'n_high_match_weighted'] = n_high_match_weighted
        gdf.at[point_id, 'n_low_match_weighted'] = n_low_match_weighted
        gdf.at[point_id, 'high_match_ratio'] = n_high_match_weighted / n_nearby_points if n_nearby_points > 0 else 0
        gdf.at[point_id, 'low_match_ratio'] = n_low_match_weighted / n_nearby_points if n_nearby_points > 0 else 0
        gdf.at[point_id, 'n_distinct_high_weighted_roads'] = len(n_distinct_high_weighted_roads)
        gdf.at[point_id, 'n_distinct_low_weighted_roads'] = len(n_distinct_low_weighted_roads)
        gdf.at[point_id, 'n_distinct_high_weighted_directions'] = len(n_distinct_high_weighted_directions)
        gdf.at[point_id, 'n_distinct_low_weighted_directions'] = len(n_distinct_low_weighted_directions)
    
    def add_features(self, point_id, nearby_points_df, gdf):
                
        # Calculate high and low match points based on distance to matched road
        high_match_points = nearby_points_df[nearby_points_df[self.prefix + 'distance_to_matched_road'] < self.tau]
        low_match_points = nearby_points_df[nearby_points_df[self.prefix + 'distance_to_matched_road'] >= self.tau]
        n_high_match_points = len(high_match_points)
        n_low_match_points = len(low_match_points)
        n_nearby_points = len(nearby_points_df)
        
        # Find the index of the current point in gdf
        index = gdf.index[gdf['point_id'] == point_id].tolist()[0]
        
        # Update gdf with computed features
        gdf.at[index, 'n_nearby_points'] = n_nearby_points
        gdf.at[index, 'n_high_match_points'] = n_high_match_points
        gdf.at[index, 'n_low_match_points'] = n_low_match_points
        gdf.at[index, 'high_match_ratio'] = n_high_match_points / n_nearby_points
        gdf.at[index, 'low_match_ratio'] = n_low_match_points / n_nearby_points
        gdf.at[index, 'n_distinct_matched_roads'] = len(nearby_points_df[self.prefix + 'matched_road_id'].unique())
        gdf.at[index, 'n_distinct_high_matched_roads'] = len(high_match_points[self.prefix + 'matched_road_id'].unique())
        gdf.at[index, 'n_distinct_low_matched_roads'] = len(low_match_points[self.prefix + 'matched_road_id'].unique())
        gdf.at[index, 'n_distinct_match_directions'] = len(nearby_points_df['discrete_direction'].unique())
        gdf.at[index, 'n_distinct_low_match_directions'] = len(low_match_points['discrete_direction'].unique())
        gdf.at[index, 'n_distinct_high_match_directions'] = len(high_match_points['discrete_direction'].unique())
        
    # def extract_features_old(self, gdf, query_executor): # TODO: remove (Deprecated)
    #     # Initializing the feature set
    #     gdf['discrete_direction'] = (np.floor((gdf.angle + 22.5) % 360 / 45) + 1).astype(int)
    #     gdf['total_number_of_nearby_points'] = None
    #     gdf['n_high_match_points'] = None
    #     gdf['n_low_match_points'] = None
    #     gdf['high_match_ratio'] = None
    #     gdf['low_match_ratio'] = None
    #     gdf['n_distinct_high_match_directions'] = None
    #     gdf['n_distinct_low_match_directions'] = None
    #     gdf['n_distinct_matched_roads'] = None
    #     gdf['n_distinct_match_directions'] = None
    #     gdf['n_distinct_low_match_directions'] = None
    #     gdf['n_distinct_high_match_directions'] = None


        
    #     for index, row in tqdm(gdf.iterrows(), total=len(gdf), desc='Adding the Features'):
    #         cell_id = row.cell_id
            
    #         nearby_points_ids = query_executor.get_nearbyPoints_for_cell(cell_id)
                        
    #         nearby_points = gdf[gdf.point_id.isin(nearby_points_ids)]
    #         total_number_of_nearby_points = len(nearby_points)
    #         if total_number_of_nearby_points == 0: 
    #             gdf.at[index, 'total_number_of_nearby_points'] = 0
    #             gdf.at[index, 'n_high_match_points'] = 0
    #             gdf.at[index, 'n_low_match_points'] = 0
    #             gdf.at[index, 'high_match_ratio'] = 0
    #             gdf.at[index, 'low_match_ratio'] = 0
    #             gdf.at[index, 'n_distinct_matched_roads'] = 0
    #             gdf.at[index, 'n_distinct_high_matched_roads'] = 0
    #             gdf.at[index, 'n_distinct_low_matched_roads'] = 0
    #             gdf.at[index, 'n_distinct_match_directions'] = 0
    #             gdf.at[index, 'n_distinct_low_match_directions'] = 0
    #             gdf.at[index, 'n_distinct_high_match_directions'] = 0
    #         else:
    #             high_match_points = nearby_points[nearby_points['distance_to_matched_road'] < self.tau]
    #             low_match_points = nearby_points[nearby_points['distance_to_matched_road'] >= self.tau]
    #             n_high_match_points = len(high_match_points)
    #             n_low_match_points = len(low_match_points)
    #             gdf.at[index, 'total_number_of_nearby_points'] = total_number_of_nearby_points
    #             gdf.at[index, 'n_high_match_points'] = n_high_match_points
    #             gdf.at[index, 'n_low_match_points'] = n_low_match_points
    #             gdf.at[index, 'high_match_ratio'] = n_high_match_points / total_number_of_nearby_points
    #             gdf.at[index, 'low_match_ratio'] = n_low_match_points / total_number_of_nearby_points
    #             gdf.at[index, 'n_distinct_matched_roads'] = len(nearby_points.matched_road_id.unique())
    #             gdf.at[index, 'n_distinct_high_matched_roads'] = len(high_match_points.matched_road_id.unique())
    #             gdf.at[index, 'n_distinct_low_matched_roads'] = len(low_match_points.matched_road_id.unique())
    #             gdf.at[index, 'n_distinct_match_directions'] = len(nearby_points.discrete_direction.unique())
    #             gdf.at[index, 'n_distinct_low_match_directions'] = len(low_match_points.discrete_direction.unique())
    #             gdf.at[index, 'n_distinct_high_match_directions'] = len(high_match_points.discrete_direction.unique())
                
    #     return gdf
    