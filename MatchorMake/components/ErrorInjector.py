import geopandas as gpd
import numpy as np
import random
import copy
from mappymatch.maps.nx.nx_map import NxMap
from shapely.ops import nearest_points
import pickle

class ErrorInjector():
    def __init__(self, gamma, percentage_of_noise_points, mu, sigma, prefix, adding_noise_to_making=True, random_seed=80, verbose=True):
        self.gamma = gamma
        self.percentage_of_noise_points = percentage_of_noise_points
        self.mu = mu
        self.sigma = sigma
        self.random_seed = random_seed
        self.verbose = verbose
        self.prefix = prefix
        self.adding_noise_to_making = adding_noise_to_making 
    
    def set_prefix(self, prefix):
        self.prefix = prefix
    
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)  
    
    def remove_roads_from_network(self, gdf, road_network, is_copy=True, roads_set=None):
        # considered_roadtypes = ['motorway', 'trunk', 'primary', 'secondary']
        if roads_set is None:
            roads_set = gdf[self.prefix+'matched_road_id'].unique().tolist()
            roads_set = list(filter(lambda item: item is not None, roads_set))
        roads_set = set(roads_set)
        # filtered_roads_set = [r for r in roads_set if nx_map.g.edges[r]['highway'] in considered_roadtypes]
        # amp_factor = 2
        lambda x: set(x[:2])
        gdf_grouped = gdf.groupby(self.prefix+'matched_road_id').size()
        
        nGoodPoints = 0
        nPerfectPoints = sum(gdf['Perfect'])
        removed_roads = set()
        
        random.seed(self.random_seed)
        while nGoodPoints < self.gamma * nPerfectPoints:
            r = random.sample(list(roads_set), 1)[0]
            r2 = (r[1], r[0], r[2]) # Reversing direction
            removed_roads.add(r)
            removed_roads.add(r2)
            
            nAssociatedPoints = gdf_grouped.get(r, 0) + gdf_grouped.get(r2, 0)
            
            nGoodPoints += nAssociatedPoints
            roads_set = roads_set - set([r, r2])

        removed_roads = list(removed_roads)
        
        if self.verbose: print('Removing ', len(removed_roads), ' (', self.gamma*100, '%) of provided roads from OSM graph...')
        g_removed_edges = road_network.edge_subgraph(removed_roads).copy()
        road_network.remove_edges_from(removed_roads)
        gdf[self.prefix+'toBeMatched'] = True
        gdf.loc[gdf[self.prefix+'matched_road_id'].isin(removed_roads), self.prefix+'toBeMatched'] = False 
        number_of_p = len(gdf[self.prefix+'toBeMatched'])
        number_of_p_make = sum(gdf[self.prefix+'toBeMatched'] == False)
        p_make_percentage = number_of_p_make/number_of_p
        if self.verbose:
            print(f"    {self.prefix}P_make Points(", 100*p_make_percentage, "%): ", number_of_p_make)
        return road_network, removed_roads, g_removed_edges, p_make_percentage


    # def remove_roads_from_network(self, gdf, nx_map, is_copy=True, roads_set=None):
    #     considered_roadtypes = ['motorway', 'trunk', 'primary', 'secondary']
    #     if roads_set is None:
    #         roads_set = gdf[self.prefix+'matched_road_id'].unique().tolist()
    #         roads_set = list(filter(lambda item: item is not None, roads_set))
    #     # filtered_roads_set = [r for r in roads_set if nx_map.g.edges[r]['highway'] in considered_roadtypes]
    #     # amp_factor = 2
    #     filtered_roads_set = roads_set
    #     amp_factor = 1
    #     random.seed(self.random_seed)
    #     increase_factor = min(1, self.gamma*amp_factor)
    #     roads_to_removed = random.sample(filtered_roads_set, int(increase_factor*len(filtered_roads_set)))
    #     if self.gamma*amp_factor > 1:
    #         other_roads = list(set(roads_set) - set(filtered_roads_set))
    #         num_roads = self.gamma*len(roads_set) - len(filtered_roads_set)
    #         if num_roads > 0: 
    #             other_roads_to_removed = random.sample(other_roads, int(num_roads))
    #             roads_to_removed = roads_to_removed + other_roads_to_removed

    #     # Make sure the roads are removed from both directions
    #     set_of_roads_to_removed = set()
    #     for r in roads_to_removed:
    #         r2 = (r[1], r[0], r[2]) # Reversing direction
    #         set_of_roads_to_removed.add(r)
    #         set_of_roads_to_removed.add(r2)
    #     roads_to_removed = list(set_of_roads_to_removed)
        
    #     if self.verbose: print('Removing ', len(roads_to_removed), ' (', self.gamma*100, '%) of provided roads from OSM graph...')
    #     g_removed_edges = nx_map.g.edge_subgraph(roads_to_removed).copy()
    #     if is_copy: new_nx_map = NxMap(copy.deepcopy(nx_map.g)) #copy.deepcopy(nx_map)
    #     else: new_nx_map = nx_map
    #     new_nx_map.g.remove_edges_from(roads_to_removed)
    #     gdf[self.prefix+'toBeMatched'] = True
    #     gdf.loc[gdf[self.prefix+'matched_road_id'].isin(roads_to_removed), self.prefix+'toBeMatched'] = False 
    #     number_of_p = len(gdf[self.prefix+'toBeMatched'])
    #     number_of_p_make = sum(gdf[self.prefix+'toBeMatched'] == False)
    #     p_make_percentage = number_of_p_make/number_of_p
    #     if self.verbose:
    #         print(f"    {self.prefix}P_make Points(", 100*p_make_percentage, "%): ", number_of_p_make)
    #     return new_nx_map, roads_to_removed, g_removed_edges, p_make_percentage

    def add_noise_to_points(self, gdf, partitioner, tolerance=0.00001, adding_noise_to_making=False, snapping=False):
        gdf[self.prefix+'isAddedNoise'] = False
        
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        gdf.lon, gdf.lat = gdf.geometry.x, gdf.geometry.y
        
        # Adding to Matching Points
        matched_index = list(gdf[gdf[self.prefix+'toBeMatched'] == True].index)
        noise_matched_index = sorted(random.sample(matched_index, int(self.percentage_of_noise_points*len(matched_index))))
        if self.verbose: print('Adding noise to ', len(noise_matched_index), '(', self.percentage_of_noise_points*100, '%) of Matching points...')
        noise = np.random.normal(self.mu, self.sigma, [len(noise_matched_index),2])
        gdf.loc[noise_matched_index, ['lat', 'lon']] += noise # TODO: Make sure the lat and long are the same as polygon or geometries
        gdf.loc[noise_matched_index, self.prefix + 'isAddedNoise'] = True
        
        if adding_noise_to_making:
            maked_index = list(gdf[gdf[self.prefix+'toBeMatched'] == False].index)
            noise_maked_index = sorted(random.sample(maked_index, int(self.percentage_of_noise_points*len(maked_index))))
            if self.verbose: print('Adding noise to ', len(noise_maked_index), '(', self.percentage_of_noise_points*100, '%) of Making points...')
            noise = np.random.normal(self.mu, self.sigma, [len(noise_maked_index),2])
            gdf.loc[noise_maked_index, ['lat', 'lon']] += noise # TODO: Make sure the lat and long are the same as polygon or geometries
            gdf.loc[noise_maked_index, self.prefix + 'isAddedNoise'] = True 
        
        gdf.geometry = gpd.points_from_xy(gdf.lon, gdf.lat)

        if snapping:
            def snap_to_boundary(point, boundary):
                if not point.within(boundary):
                    # Find the nearest point on the boundary
                    snapped_point = nearest_points(point, boundary)[1]
                    return snapped_point
                return point

            # Apply snapping to all points in the GeoDataFrame
            boundary = partitioner.get_boundary().buffer(-tolerance)
            gdf['geometry'] = gdf['geometry'].apply(lambda point: snap_to_boundary(point, boundary))
        return gdf
    
    
    
    def run(self, gdf, road_network, is_road_network_copied, perfect_roads, partitioner):
        new_road_network, roads_to_removed, g_removed_edges, p_make_percentage = self.remove_roads_from_network(gdf, road_network, is_road_network_copied, perfect_roads)
        gdf = self.add_noise_to_points(gdf, partitioner, adding_noise_to_making=self.adding_noise_to_making)
        return gdf, new_road_network, roads_to_removed, g_removed_edges, p_make_percentage