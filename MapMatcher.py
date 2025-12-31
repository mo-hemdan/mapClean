from concurrent.futures import ProcessPoolExecutor
import psutil
import os
import pandas as pd
import osmnx as ox
from tqdm import tqdm
import numpy as np
from shapely.geometry import LineString, Point
tqdm.pandas()

class MapMatcher:
    def __init__(self, road_network):
        print('Getting roads and edges')
        _, edges = ox.graph_to_gdfs(road_network)
        if edges.crs == "EPSG:3857":
            print("Correct CRS")
        else:
            print("Wrong CRS:", edges.crs)
            edges.to_crs(epsg=3857, inplace=True)
            print('Edges Converted into the EPSG:3857')
        print('edges crs: ', edges.crs)
        self.edges = edges
        edge_geoms = edges.geometry
        self.tree = edge_geoms.sindex

    # original_index = gdf.index
    def vector_angle(self, dx, dy):
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        return angle_deg % 360

    def get_direction_at_projection(self, line: LineString, pt: Point, small_step=100) -> float:
        # Project point onto line
        proj_distance = line.project(pt)
        # proj_point = line.interpolate(proj_distance)

        # Find small segment around the projected point
        d1 = max(proj_distance - small_step, 0)
        d2 = min(proj_distance + small_step, line.length)

        p1 = line.interpolate(d1)
        p2 = line.interpolate(d2)

        dx = p2.x - p1.x
        dy = p2.y - p1.y

        return self.vector_angle(dx, dy)

    def find_near_edge_smp(self, points_batch):
        # Prepare results
        print(f"[Process ID]:{os.getpid()} Matching Batch..")
        matched_road_ids = []
        distances = []
        road_geometries = []

        for _, point in points_batch.iterrows():        
            # Find the nearest edge using spatial index
            nearest_idx, nearest_d = self.tree.nearest(point.geometry, return_distance=True)
            nearest_idx = nearest_idx[1][0]
            nearest_edge = self.edges.iloc[nearest_idx]
            nearest_d = nearest_d[0]
            
            # Append results and original index
            matched_road_ids.append(nearest_edge.name)
            distances.append(nearest_d)
            road_geometries.append(nearest_edge.geometry)
            # indices.append(original_index[idx])  # Preserve the original index
        
        # Return results as a DataFrame, keeping the original index
        return pd.DataFrame({
            'matched_road_id': matched_road_ids,
            'distance_to_matched_road': distances,
            'road_geometry': road_geometries
        }, index=points_batch.index)
    
    def find_near_edge_adv(self, points_batch):
        # Prepare results
        print(f"[Process ID]:{os.getpid()} Matching Batch..")
        matched_road_ids = []
        distances = []
        road_geometries = []
        road_angles = []
        r_p_similarities = []
        # oneway_list = []
        length_list = []

        for _, point in points_batch.iterrows():        
            # Find the nearest edge using spatial index
            nearest_idx, nearest_d = self.tree.nearest(point.geometry, return_distance=True)
            nearest_idx = nearest_idx[1][0]
            nearest_edge = self.edges.iloc[nearest_idx]
            nearest_d = nearest_d[0]        
            road_angle = self.get_direction_at_projection(nearest_edge.geometry, point.geometry)
            
            angle_diff = abs(road_angle - point.angle)
            angle_diff = angle_diff if angle_diff <= 180 else 360 - angle_diff
            # You can also add the length of the road as well

            if angle_diff > 90: # this case might have the point matching to the reverse direction
                edge_tuple = self.edges.index[nearest_idx]
                redge_tuple = (edge_tuple[1], edge_tuple[0], edge_tuple[2])
                if redge_tuple in self.edges.index:
                    nearest_edge = self.edges.loc[redge_tuple]
                    # nearest d is the same
                    road_angle = road_angle + 180
                    if road_angle > 360: road_angle = road_angle - 360
                    angle_diff = abs(road_angle - point.angle)
                    angle_diff = angle_diff if angle_diff <= 180 else 360 - angle_diff

            road_angles.append(road_angle)
            matched_road_ids.append(nearest_edge.name)
            distances.append(nearest_d)
            road_geometries.append(nearest_edge.geometry)
            r_p_similarities.append(angle_diff)
            # oneway_list.append(nearest_edge.oneway)
            length_list.append(nearest_edge.length)
        
        # Return results as a DataFrame, keeping the original index
        return pd.DataFrame({
            'matched_road_id': matched_road_ids,
            'distance_to_matched_road': distances,
            'road_geometry': road_geometries,
            'road_angles': road_angles,
            'r_p_sim': r_p_similarities,
            # 'road_oneway': oneway_list,
            'road_length': length_list
        }, index=points_batch.index)

    # Create a function to handle batching and parallelization
    def process_with_batches(self, points, advanced_matching, batch_size=50000):
        # Split the GeoDataFrame into batches
        batches = [points.iloc[i:i+batch_size] for i in range(0, len(points), batch_size)]
        print(f"Head [Process ID]:{os.getpid()} Number of Lists = {len(batches)}")

        if advanced_matching:
            with ProcessPoolExecutor(max_workers=psutil.cpu_count(logical=False) - 1) as executor:
                results = list(executor.map(self.find_near_edge_adv, batches))
        else:
            with ProcessPoolExecutor(max_workers=psutil.cpu_count(logical=False) - 1) as executor:
                results = list(executor.map(self.find_near_edge_smp, batches))
        
        # Combine results, preserving the original index
        return pd.concat(results, ignore_index=False)
    
    def match(self, points, advanced_matching, prefix=''): # IMPORTANT
        '''
        we need the gdf in EPSG:3857
        '''
        desired_cols = ['angle', 'geometry']
        points_df = points[desired_cols].copy(deep=True)
        matches = self.process_with_batches(points_df, advanced_matching=advanced_matching)

        if advanced_matching:
            col_names = ['matched_road_id', 'distance_to_matched_road', 'road_geometry', 'road_angle', 'r_p_sim', 'road_length']
        else: col_names = ['matched_road_id', 'distance_to_matched_road', 'road_geometry']

        if prefix != '': col_names = [prefix + col for col in col_names]

        points[col_names] = matches
        return points
    
    # Snapping the points to the road network
    def snap_point_to_nearest_edge(self, point):
        point_geom  = point.geometry
        edge_geom = point.road_geometry
        snapped_point = edge_geom.interpolate(edge_geom.project(point_geom))
        return snapped_point
    
    def clean_snap(self, gdf): # IMPORTANT
        print('Warning Geometry of points change at this point and distance to matched road change as well')
        gdf['snapped_geometry'] = gdf.progress_apply(self.snap_point_to_nearest_edge, axis=1)
        gdf['geometry'] = gdf['snapped_geometry']
        gdf.lon = gdf.geometry.x
        gdf.lat = gdf.geometry.y
        gdf.drop(columns=['snapped_geometry', 'distance_to_matched_road'], inplace=True)
        return gdf