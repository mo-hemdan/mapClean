from mappymatch.constructs.trace import Trace
from mappymatch.matchers.lcss.lcss import LCSSMatcher
from mappymatch.matchers.matcher_interface import MatchResult
from mappymatch.constructs.match import Match
from mappymatch.constructs.coordinate import Coordinate
from mappymatch.constructs.road import Road, RoadId
import numpy as np
from scipy.spatial.distance import cdist
import psutil
import os
from concurrent.futures import ProcessPoolExecutor
import os 
os.environ['MPLCONFIGDIR'] = "/project/cs-dmlab/hemdan/configs/"
import pandas as pd
import osmnx as ox
import time
from tqdm import tqdm
import pickle

class MapMatcher():
    def __init__(self, padding=200, delete_empty_matches=False, feature_prefix='', fix_empty_matches=True, nprocesses=None):
        self.padding=padding
        self.delete_empty_matches=delete_empty_matches
        self.feature_prefix=feature_prefix
        self.fix_empty_matches=fix_empty_matches
        self.nprocesses=nprocesses
    
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)  
    
    def set_prefix(self, new_prefix):
        self.feature_prefix = new_prefix

    def wkt_pack_geometry_col(self, gdf):
        gdf[self.feature_prefix+'matched_road_geom'] = gdf[self.feature_prefix+'matched_road_geom'].apply(lambda geom: geom.wkt)
        return gdf

    def wkt_unpack_geometry_col(self, gdf):
        gdf[self.feature_prefix+'matched_road_geom'] = gdf[self.feature_prefix+'matched_road_geom'].apply(lambda geom: geom.wkt)
        return gdf  
    
    def match(self, gdf, edges, tree):      
        # Split the GeoDataFrame into batches
        original_index = gdf.index
        def find_nearest_edge_and_distance_batch(points_batch):
            # Prepare results
            print(f"[Process ID]:{os.getpid()} Matching Batch..")
            matched_road_ids = []
            distances = []
            road_geometries = []
            indices = []

            for idx, point in points_batch.iterrows():        
                # Find the nearest edge using spatial index
                nearest_idx, nearest_d = tree.nearest(point.geometry, return_distance=True)
                nearest_idx = nearest_idx[1][0]
                nearest_edge = edges.iloc[nearest_idx]
                nearest_d = nearest_d[0]
                
                # Append results and original index
                matched_road_ids.append(nearest_edge.name)
                distances.append(nearest_d)
                road_geometries.append(nearest_edge.geometry)
                indices.append(original_index[idx])  # Preserve the original index
            
            # Return results as a DataFrame, keeping the original index
            return pd.DataFrame({
                'matched_road_id': matched_road_ids,
                'distance_to_matched_road': distances,
                'road_geometry': road_geometries
            }, index=indices)

        # Create a function to handle batching and parallelization
        def process_with_batches(gdf_points, edges, tree, batch_size=50000):
            # Split the GeoDataFrame into batches
            batches = [gdf_points.iloc[i:i+batch_size] for i in range(0, len(gdf_points), batch_size)]
            print(f"Head [Process ID]:{os.getpid()} Number of Lists = {len(batches)}")

            with ProcessPoolExecutor(max_workers=psutil.cpu_count(logical=False) - 1) as executor:
                results = list(executor.map(find_nearest_edge_and_distance_batch, batches))
            
            # Combine results, preserving the original index
            return pd.concat(results, ignore_index=False)


        results_df = process_with_batches(gdf, edges, tree)
        gdf[['matched_road_id', 'distance_to_matched_road', 'road_geometry']] = results_df
        return gdf

    # def match(self, gdf, road_network):
    #     list_of_traces, vechile_ids = self.process_traces(gdf)
    #     # Step 1: Do Map Matching
    #     print('Matching Traces... (in paralle): No output!')
    #     start_time = time.time()
    #     results_dict = self.LCSS_match(list_of_traces, road_network, vechile_ids)
    #     if len(gdf.vehicule_id.unique()) > len(results_dict.keys()):
    #         print("ERROR (not stopped): Found None Matches for trajectories after matching")
    #     end_time = time.time()
    #     print(f'Finished Matching Traces in {end_time-start_time} s')
    #     # Step 2: Assign the matched roads and distances to each point
    #     gdf[self.feature_prefix+'matched_road_id'] = len(gdf)*[None]
    #     gdf[self.feature_prefix+'matched_road_geom'] = len(gdf)*[None]
    #     gdf[self.feature_prefix+'matched_road_metadata'] = len(gdf)*[None]
    #     gdf[self.feature_prefix+'distance_to_matched_road'] = float('inf')
    #     for _, id in tqdm(enumerate(results_dict.keys()), desc="Adding Matching Information to GDF", total=len(results_dict)):
    #         values = results_dict[id]
    #         roads = [m.road for m in values.matches]
    #         if len(roads) != len(gdf[gdf.vehicule_id == id]):
    #             raise Exception("Error: Output Number of roads is different from the number of points for the same trajectory")
    #         if any(value is None for value in roads):
    #             print("WARNING: Found None Values in the roads list for trajectory ID (", id, "). Will be kept as None")
    #         list_of_tuples = [(r.road_id.start, r.road_id.end, r.road_id.key) if r != None else None for r in roads]
    #         df = pd.DataFrame(index=range(len(list_of_tuples)), columns=[self.feature_prefix+'matched_road_id'])
    #         df[self.feature_prefix+'matched_road_id'] = list_of_tuples
    #         gdf.loc[gdf.vehicule_id == id, self.feature_prefix+'matched_road_id'] = df[self.feature_prefix+'matched_road_id'].values
    #         # gdf.loc[gdf.vehicule_id == id, 'matched_road_end_id'] = [r.road_id.end if r != None else None for r in roads]
    #         gdf.loc[gdf.vehicule_id == id, self.feature_prefix+'matched_road_geom'] = [r.geom if r != None else None for r in roads]
    #         gdf.loc[gdf.vehicule_id == id, self.feature_prefix+'matched_road_metadata'] = [r.metadata if r != None else None for r in roads]
    #         distances = [m.distance for m in values.matches]
    #         gdf.loc[gdf.vehicule_id == id, self.feature_prefix+'distance_to_matched_road'] = distances
    #     end_time = time.time()
    #     # print("Took ", end_time - start_time, " seconds")
    #     return gdf

    def LCSS_match(self, list_of_traces, nx_map, vechile_ids):
        """
        Given a dataframe of different trajectory ids, give matching to each trajectory id. 
        Args:
            df: [DataFrame] the datapoints marked by their vehicule_id (can be GeoPandas DataFrame)
        Return:
            results: List[RoadMatch]
        """
        if self.nprocesses == None: self.nprocesses = len(list_of_traces)
        self.nprocesses = min(self.nprocesses, 2)
        # print(f"Matching in {nprocesses} processes")
        matcher = LCSSMatcher(nx_map)
        matches = matcher.match_trace_batch(
            trace_batch = list_of_traces, 
            processes=self.nprocesses
        )
        # print(f'WARNING: LCSS failed in {matcher.nearest_match_used} matches: Nearest Match is used')
        output_dict = dict(zip(vechile_ids, matches))
        # Check for empty matches
        is_empty_found = False
        list_to_be_deleted = []
        for id in output_dict.keys():
            result = output_dict[id].matches[0]
            if result.road == None:
                if not is_empty_found:
                    print("WARNING: some traces failed to find matching, will be deleted if delete_empty_matches is set to True")
                    print("Trajectory IDs:")
                    is_empty_found = True
                print("   ", id)
                list_to_be_deleted.append(id)
        if self.fix_empty_matches:
            for id in list_to_be_deleted:
                i = vechile_ids.index(id)
                trace = list_of_traces[i]
                print("Fixing ID: ", id)
                nearest_edges = ox.nearest_edges(nx_map.g, X=trace._frame.geometry.x, Y=trace._frame.geometry.y, return_dist=True)
                list_of_matches = []
                for i, (r, d) in enumerate(zip(nearest_edges[0], nearest_edges[1])):
                    point_coord = trace.coords[i]
                    point_geom = trace._frame.geometry.iloc[i]
                    point_crs = trace._frame.geometry.crs
                    corr = Coordinate(point_coord, point_geom, point_crs)
                    road_id = RoadId(r[0], r[1], r[2])
                    geom = nx_map.g.edges[r]['geometry']
                    road = Road(road_id, geom)
                    list_of_matches.append(Match(road, corr, d))
                output_dict[id] = MatchResult(list_of_matches, None)
        if self.delete_empty_matches:
            for item in list_to_be_deleted:
                del output_dict[item]
        return output_dict

    def process_traces(self, gdf):
        vechile_ids = sorted(gdf.vehicule_id.unique())

        print('Processing Traces for our input...!')
        start_time = time.time()
        list_of_traces = []
        for id in tqdm(vechile_ids):
            sub_df = gdf[gdf.vehicule_id == id]
            trace = Trace.from_geo_dataframe(frame=sub_df)
            list_of_traces.append(trace)
        end_time = time.time()
        print("Took ", end_time - start_time, " seconds")
        return list_of_traces, vechile_ids