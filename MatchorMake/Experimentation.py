import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from shapely.ops import nearest_points  


class RandomGPSDataGenerator():
    
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        pass
    
    def generate_points(self, n_points):
        # Step 2: Generate random points within these bounds
        x_coords = np.random.uniform(self.xmin, self.xmax, n_points)
        y_coords = np.random.uniform(self.ymin, self.ymax, n_points)
        # Step 3: Create a GeoDataFrame with the generated points
        points = [Point(x, y) for x, y in zip(x_coords, y_coords)]
        gdf = gpd.GeoDataFrame(pd.DataFrame({'geometry': points}), crs="EPSG:4326")
        return gdf
    
    def generate_roads(self, n_roads):
        roads = []
        for _ in range(n_roads):
            road_coords = [(np.random.uniform(self.xmin, self.xmax), 
                            np.random.uniform(self.ymin, self.ymax)) for _ in range(2)]
            roads.append(LineString(road_coords))
        roads_gdf = gpd.GeoDataFrame({'geometry': roads, 'road_id': range(n_roads)})
        return roads_gdf

    def plot_points(self, gdf):
        # Step 4: Plot the points
        gdf.plot(marker='o', color='blue', markersize=5)
        plt.title("Random Points in a Rectangular Space")
        plt.xlabel("Longitude (x)")
        plt.ylabel("Latitude (y)")
        plt.grid(True)
        plt.show()
    
    def map_matching(self, points_gdf, roads_gdf):
        '''
        Implements a simple Map Matching Technique in which we use the nearest points on lines.
        '''
        matched_road_ids = []
        distances = []
        
        for point in points_gdf['geometry']:
            # Find the nearest road
            nearest_road = None
            min_distance = float('inf')
            
            for idx, road in roads_gdf.iterrows():
                road_geom = road.geometry
                # Get the nearest point on the road to the point
                nearest_point_on_road = nearest_points(point, road_geom)[1]
                distance = point.distance(nearest_point_on_road)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_road = road_geom
                    matched_road_id = road.road_id  # Assuming 'road_id' is a column in roads_gdf
            
            matched_road_ids.append(matched_road_id)
            distances.append(min_distance)

        # Add matched road and distance to the points dataframe
        points_gdf['matched_road_id'] = matched_road_ids
        points_gdf['distance_to_matched_road'] = distances

        return points_gdf
    
    def plot_PME_results(self, points_gdf, roads_gdf, perfect_points, uncertain_points, perfect_roads, plot_geofence=False, delta=None):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot all roads
        if not roads_gdf.empty: roads_gdf.plot(ax=ax, color='gray', linewidth=2, label='Roads')

        # Highlight perfect roads
        perfect_roads_gdf = roads_gdf[roads_gdf['road_id'].isin(perfect_roads)]
        if not perfect_roads_gdf.empty: perfect_roads_gdf.plot(ax=ax, color='blue', linewidth=3, label='Perfect Roads')

        if plot_geofence and (not perfect_roads_gdf.empty): 
            perfect_roads_gdf['geofence'] = perfect_roads_gdf.geometry.buffer(delta)
            perfect_roads_gdf['geofence'].plot(ax=ax, color='lightblue', alpha=0.5, edgecolor='blue')
            # Custom legend handle for the geofence
            geofence_legend_handle = plt.Line2D([0], [0], color='lightblue', lw=4, alpha=0.5)
            plt.legend(handles=[geofence_legend_handle], labels=['Geofence'], loc='upper right')

        # Plot points
        if not points_gdf.empty: points_gdf.plot(ax=ax, color='black', markersize=20, label='All Points')
        if not perfect_points.empty: perfect_points.plot(ax=ax, color='green', markersize=50, label='Perfect Points')
        if not uncertain_points.empty: uncertain_points.plot(ax=ax, color='red', markersize=30, label='Uncertain Points')

        # Count and display the numbers of roads and points
        num_roads = len(roads_gdf)
        num_perfect_points = len(perfect_points)
        num_uncertain_points = len(uncertain_points)
        # Add text annotations to the plot
        ax.text(0.05, 0.98, f'Total Roads: {num_roads}', transform=ax.transAxes, fontsize=8, verticalalignment='top')
        ax.text(0.05, 0.96, f'Perfect Points: {num_perfect_points}', transform=ax.transAxes, fontsize=8, verticalalignment='top')
        ax.text(0.05, 0.94, f'Uncertain Points: {num_uncertain_points}', transform=ax.transAxes, fontsize=8, verticalalignment='top')

        # Annotate roads with their IDs
        for idx, row in roads_gdf.iterrows():
            ax.text(row.geometry.centroid.x, row.geometry.centroid.y, str(row.road_id), fontsize=10, color='red')

        # Annotate points with their IDs
        for idx, row in points_gdf.iterrows():
            ax.text(row.geometry.x, row.geometry.y, str(idx), fontsize=10, color='black')  # Assuming idx is the point ID

        
        # Update legend to include all items
        plt.legend()
        plt.title('Roads, Points, and Perfect/Uncertain Points Classification')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.show()