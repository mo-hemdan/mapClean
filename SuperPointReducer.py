import pandas as pd
from shapely.geometry import Point, LineString
from shapely.geometry.base import BaseGeometry
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely import wkt
import numpy as np
import geopandas as gpd


class SuperPointReducer:
    def __init__(self, super_point_size):
        self.SUPER_POINT_SIZE = super_point_size
        tqdm.pandas()
    
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
    
    def angle_diff(self, a1, a2):
        return abs((a2 - a1 + 180) % 360 - 180)
    
    def get_road_angle(self, r, p):
        projected_distance = r.project(p)
        projected_point = r.interpolate(projected_distance)

        # Get coordinates of the road
        coords = list(r.coords)

        # Find the segment where the projected point lies
        for i in range(len(coords) - 1):
            segment = LineString([coords[i], coords[i+1]])
            if segment.project(projected_point) <= segment.length:
                break

        # Calculate the direction vector
        dx = coords[i+1][0] - coords[i][0]
        dy = coords[i+1][1] - coords[i][1]

        # Calculate the angle in degrees
        road_angle = np.degrees(np.arctan2(dy, dx)) % 360
        road_angle = int(road_angle)

        return road_angle

    def reducePointsIntoSuper(self, group):
        if len(group) == 0:
            return group

        avg_lon, avg_lat = group.geometry.x.mean(), group.geometry.y.mean()
        avg_point_geom = Point(avg_lon, avg_lat)      


        road_id = group['matched_road_id'].value_counts().idxmax()
        road_geom = group.loc[group['matched_road_id'] == road_id, 'road_geometry'].iat[0]
        road_length = road_geom.length

        road_angle = self.get_road_angle(road_geom, avg_point_geom)
        distance_to_road = avg_point_geom.distance(road_geom)

        angle_dif = (group['angle'] - road_angle).abs()
        angle_dif[angle_dif > 180] -= 180


        group_forward = group[angle_dif <= 90]
        group_reverse = group[angle_dif > 90]

        df = None
        if len(group_forward) > 0:
            df = group_forward[['lat', 'lon', 'speed', 'angle']].mean()
            df['road_angle'] = road_angle
            df['matched_road_id'] = road_id
            df['road_geometry'] = road_geom
            df['road_length'] = road_length
            df['distance_to_matched_road'] = distance_to_road
            df['r_p_sim'] = self.angle_diff(road_angle, df.angle)

            nPoints = len(group_forward)
            nMatchPoints = group_forward.ture_toBeMatched.sum()
            nMakePoints = nPoints - nMatchPoints
            df['nPoints'] = nPoints
            df['nMatchPoints'] = nMatchPoints
            df['nMakePoints'] = nMakePoints
            df['original_dataframe'] = group_forward
            
        if len(group_reverse) > 0:
            df_t = group_reverse[['lat', 'lon', 'speed', 'angle']].mean()
            # r_angle = angle + 180
            # if r_angle >= 360: r_angle -= 360
            # df_t['angle'] = r_angle
            df_t['road_angle'] = road_angle
            df_t['matched_road_id'] = road_id
            df_t['road_geometry'] = road_geom
            df_t['road_length'] = road_length
            df_t['distance_to_matched_road'] = distance_to_road
            df_t['r_p_sim'] = self.angle_diff(road_angle, df_t.angle)

            nPoints = len(group_reverse)
            nMatchPoints = group_reverse.ture_toBeMatched.sum()
            nMakePoints = nPoints - nMatchPoints
            df_t['nPoints'] = nPoints
            df_t['nMatchPoints'] = nMatchPoints
            df_t['nMakePoints'] = nMakePoints
            df_t['original_dataframe'] = group_reverse
            if df is not None: 
                # df = pd.concat([df, df_t], axis=1)
                df = pd.DataFrame([df, df_t])
            else: df = df_t
        
        return df
    
    def pixelize(self, gdf):
        """
        Input: gdf
        Output: gdf with new columns lat_pix and lon_pix which are the modified values of their point locations
        """
        if self.SUPER_POINT_SIZE != 0:
            gdf['lat_pix'] = gdf['lat'].progress_apply(lambda x: int(x)// self.SUPER_POINT_SIZE * self.SUPER_POINT_SIZE)
            gdf['lon_pix'] = gdf['lon'].progress_apply(lambda x: int(x)// self.SUPER_POINT_SIZE * self.SUPER_POINT_SIZE)
            print('Grouping by the pixel size')
            gdf_pixel = gdf.groupby(by=['lon_pix', 'lat_pix'])#.count()
        else:
            gdf['lat_pix'] = gdf['lat']
            gdf['lon_pix'] = gdf['lon']
            print('No Grouping by the lat/lon')
            gdf_pixel = gdf.copy()
        
        return gdf_pixel

    def group_pixels(self, gdf_pixel):
        if self.SUPER_POINT_SIZE != 0:
            gdf_result = gdf_pixel.progress_apply(self.reducePointsIntoSuper)
            first_two = gdf_result.index.droplevel(2)
            gdf_str = gdf_result[gdf_result.index.get_level_values(2).map(lambda x: isinstance(x, str))]
            gdf_int = gdf_result[gdf_result.index.get_level_values(2).map(lambda x: isinstance(x, int))]
            gdf_str_r = gdf_str.unstack(level=2)[0]
            gdf_result = pd.concat([gdf_int, gdf_str_r])
            gdf_result.drop(columns=[0], inplace=True)
        else:
            gdf_result = gdf_pixel.copy()
            gdf_result['nPoints'] = 1
        return gdf_result

    def apply(self, gdf):
        """
        Input Requirements:
        existing geometry column
        """
        self.check_input(gdf)

        gdf_pixel = self.pixelize(gdf)
        gdf = self.group_pixels(gdf_pixel)

        

        gdf['lat'] = gdf['lat'].apply(int)
        gdf['lon'] = gdf['lon'].apply(int)
        gdf = gdf.reset_index(drop=True)
        gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf["lon"], gdf["lat"]))
        
        return gdf
    

    def check_input(self, gdf):
        print('Checking the Road Geometry')
        val = gdf['road_geometry'].iloc[0]

        if isinstance(val, BaseGeometry):
            print("Already Shapely, no need to load WKT")
        else:
            print("Not Shapely, WKT load needed Will load it")
            gdf['road_geometry'] = gdf['road_geometry'].apply(wkt.loads)
            print('WARNING: Input has changed')

        xs = gdf["road_geometry"].apply(lambda g: g.bounds[0])  # min x
        ys = gdf["road_geometry"].apply(lambda g: g.bounds[1])  # min y

        if xs.between(-2e7, 2e7).all() and ys.between(-2e7, 2e7).all():
            print("Coordinates look like EPSG:3857")
        else:
            raise ValueError('Coordinates do NOT look like EPSG:3857')

        print('Checking the CRS of geometry column')
        if gdf.crs == "EPSG:3857":
            print("Correct CRS")
        else:
            print("Wrong CRS:", gdf.crs)
            gdf.to_crs(epsg=3857, inplace=True)
            print('Converted into the EPSG:3857')
        gdf['lat'] = gdf.geometry.y
        gdf['lon'] = gdf.geometry.x

        return True