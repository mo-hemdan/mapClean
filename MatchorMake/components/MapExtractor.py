from mappymatch.maps.nx.nx_map import NxMap
import time
from mappymatch.constructs.geofence import Geofence
import dask_geopandas as gdd
from shapely.ops import unary_union
from shapely.ops import transform
from pyproj import Transformer
from mappymatch.utils.crs import LATLON_CRS, XY_CRS
import pickle
from shapely.geometry import box
from mappymatch.maps.nx.nx_map import NetworkType

class MapExtractor():
    def __init__(self, padding, is_parallel=False, npartitions=None, network_type=NetworkType.DRIVE):
        self.padding = padding
        self.is_parallel=is_parallel
        self.npartitions=npartitions
        self.geofence=None
        self.network_type = network_type

    def get_road_network(self, gdf, approximate=False):
        print('Creating Polygon...')
        if approximate:
            geofence = self.get_MBR_polygon(gdf)
        else: geofence = self.create_geofence_polygon(gdf)
        print('Converting Geofence CRS...')
        polygon = self.convert_polygon_crs(geofence, XY_CRS, LATLON_CRS)
        print('Creating Geofence...')
        self.geofence = Geofence(crs=LATLON_CRS, geometry=polygon)
    
        print('Getting Road Network From GeoFence... ')
        start_time = time.time()
        road_network = NxMap.from_geofence(self.geofence, network_type=self.network_type)
        end_time = time.time()
        print("Took ", end_time - start_time, " seconds")
        return road_network
    
    def create_geofence_polygon(self, gdf):
        """
        Create a GeoFence around the geometeries provided in the GeoPandas DataFrame
        Args:
            gdf: [GeoDataFrame] the GeoPandas DataFrame
            buffer_resolution: [Integer] the buffer around the geometries to create the fence
        """
        # Adding a buffer per each point in the GeoPandas Dataframe
        print('Calculating the Buffer Around Each Point (Parallel via GeoDask)...!')
        def square_buffer(point, width):
            x, y = point.x, point.y
            return box(x - width, y - width, x + width, y + width)

        start_time = time.time()
        if self.is_parallel:
            # raise Exception("Not Implemented")
            ddf = gdd.from_geopandas(gdf, npartitions=self.npartitions)
            # ddf['buffer'] = ddf.geometry.map_partitions(lambda part: part.apply(square_buffer, width=self.padding))
            ddf['buffer'] = ddf.geometry.map_partitions(lambda part: part.buffer(self.padding))
            gdf = ddf.compute()
        else:
            # gdf.loc[:, 'buffer'] = gdf.geometry.apply(square_buffer, width=self.padding)
            gdf.loc[:, 'buffer'] = gdf.geometry.buffer(self.padding)
        end_time = time.time()
        print("Took ", end_time - start_time, " seconds")

        print('Calculating the Union of Buffers...!')
        start_time = time.time()
        union = unary_union(gdf['buffer'])
        end_time = time.time()
        print("Took ", end_time - start_time, " seconds")
        
        return union
    
    def get_MBR_polygon(self, gdf):
        """
        Create a GeoFence around the geometeries provided in the GeoPandas DataFrame
        Args:
            gdf: [GeoDataFrame] the GeoPandas DataFrame
            buffer_resolution: [Integer] the buffer around the geometries to create the fence
        """
        # Adding a buffer per each point in the GeoPandas Dataframe
        print('Calculating the Buffer Around Each Point (Parallel via GeoDask)...!')
        min_x, min_y, max_x, max_y = gdf.total_bounds
        return box(min_x-self.padding, min_y-self.padding, max_x+self.padding, max_y+self.padding)

    @classmethod
    def convert_polygon_crs(cls, polygon, polygon_crs, target_crs):
        if polygon_crs == target_crs:
            return polygon
        project = Transformer.from_crs(
            polygon_crs, target_crs, always_xy=True
        ).transform
        return transform(project, polygon)
    
    @classmethod
    def save_road_network(cls, road_network, path):
        with open(path+'.pkl', 'wb') as file:
            pickle.dump(road_network, file, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_road_network(cls, path):
        with open(path+'.pkl', 'rb') as file:
            road_network = pickle.load(file)
        return road_network


    # def save_road_network(self, road_network):
    #     ox.settings.log_console = False

    

    # end_time = time.time()
    # print("Took ", end_time - start_time, " seconds")
    
    # print(f'Saving the NxMap ({i}/{len(files)})... !')
    # start_time = time.time()
    # ox.io.save_graphml(raw_graph, output_folder + filename+'.graphml')
    # with open(output_folder + filename+'.pkl', 'wb') as outp:
    #     pickle.dump(nx_map, outp, pickle.HIGHEST_PROTOCOL)
    # end_time = time.time()
    # print("Took ", end_time - start_time, " seconds")

    # def load_road_network(self, file_path):
    #     nx_graph = parse_osmnx_graph(raw_graph, nxmap_network_type)
    #     nx_map = NxMap(nx_graph)


        # import osmnx as ox
# 