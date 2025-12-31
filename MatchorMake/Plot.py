import numpy as np
import os 
os.environ['MPLCONFIGDIR'] = "/project/cs-dmlab/hemdan/configs/"
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import osmnx as ox
import pickle
import networkx as nx
import math
import copy
import random
import time
from mappymatch.maps.nx.nx_map import NxMap, NetworkType
from mappymatch.utils.crs import LATLON_CRS, XY_CRS
from mappymatch.constructs.trace import Trace
from mappymatch.constructs.geofence import Geofence
from mappymatch.utils.plot import plot_trace
from mappymatch.utils.plot import plot_map
from mappymatch.utils.plot import plot_geofence
from shapely.ops import unary_union
from shapely.ops import transform
from shapely.geometry import Point
from pyproj import CRS, Transformer
from sklearn import tree
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.metrics import log_loss
from helper_functions import plot_osmgraph, parse_osmnx_graph
from shapely.geometry import LineString, Polygon, mapping
from mappymatch.matchers.lcss.lcss import LCSSMatcher
#from mappymatch.utils.plot import plot_matches
from mappymatch.matchers.matcher_interface import MatchResult
from mappymatch.constructs.match import Match
from typing import List, Optional, Union
from geopy.distance import geodesic
import folium
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from datetime import timedelta
from mappymatch.constructs.coordinate import Coordinate
from mappymatch.constructs.road import Road, RoadId
import dask.dataframe as dd
import dask_geopandas as gdd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm




def visualize_matches_to_html(gdf, file_path='./viz/point.html', how_many=None):
    old_crs = None
    if not isinstance(gdf['ture_matched_road_geom'], gpd.geoseries.GeoSeries):
        gdf['ture_matched_road_geom'] = gpd.GeoSeries(gdf['ture_matched_road_geom'], crs=gdf.crs)
    if gdf.crs != LATLON_CRS:
        old_crs = gdf.crs
        gdf = gdf.to_crs(LATLON_CRS)
        gdf['ture_matched_road_geom'] = gdf['ture_matched_road_geom'].to_crs(LATLON_CRS)
    centriod = gdf.geometry.unary_union.centroid
    m = folium.Map(location=[centriod.y, centriod.x], zoom_start=11)
    d = 20
    point_color = 'red'
    line_color = 'black'
    vehicule_ids = gdf['vehicule_id'].unique()
    for indx, id in tqdm(enumerate(vehicule_ids), total= len(vehicule_ids)):
        if how_many is not None:
            if how_many <= indx+1:
                print('break at', how_many, ' ', indx)
                break
        sub_gdf = gdf[gdf['vehicule_id'] == id]
        for index, row in sub_gdf.iterrows():
            point = row['geometry']
            folium.Circle(
                location=(point.y, point.x), 
                radius=d, 
                color=point_color, 
                fill=True,
                fill_color='yellow',  # fill color of the circle
                fill_opacity=0.0,  # opacity of the fill color
                tooltip=index
            ).add_to(m)
            line = row['ture_matched_road_geom']
            folium.PolyLine(
                [(lat, lon) for lon, lat in line.coords],
                color=line_color,
                tooltip=index
            ).add_to(m)
    m.save(file_path)
    
    #returning back to old crs
    if old_crs is not None:
        gdf = gdf.to_crs(old_crs)
        gdf['ture_matched_road_geom'] = gdf['ture_matched_road_geom'].to_crs(old_crs)

def plot_points(gdf, m=None, point_color="yellow", prefix=''):
    """
    Plot some points.

    Args:
        trace: The trace.
        m: the folium map to plot on
        point_color: The color the points will be plotted in.

    Returns:
        An updated folium map with a plot of trace.
    """
    old_crs = None
    if not (gdf.crs == LATLON_CRS):
        old_crs = gdf.crs
        gdf = gdf.to_crs(LATLON_CRS)

    if not m:
        centriod = gdf.geometry.unary_union.centroid
        m = folium.Map(location=[centriod.y, centriod.x], zoom_start=11)

    for i, row in gdf.iterrows():
        folium.Circle(
            location=(row.geometry.y, row.geometry.x), 
            radius=3, 
            color=point_color, 
            fill=True,
            fill_color=point_color,  # fill color of the circle
            fill_opacity=1,  # opacity of the fill color
            tooltip=(i, row[prefix+'distance_to_matched_road']) # row[prefix+'distance_to_matched_road']+
        ).add_to(m)

    if old_crs is not None:
        gdf = gdf.to_crs(old_crs)

    return m

def plot_points_showing_features(gdf, m=None, point_color="yellow", prefix=''):
    """
    Plot some points.

    Args:
        trace: The trace.
        m: the folium map to plot on
        point_color: The color the points will be plotted in.

    Returns:
        An updated folium map with a plot of trace.
    """
    old_crs = None
    if not (gdf.crs == LATLON_CRS):
        old_crs = gdf.crs
        gdf = gdf.to_crs(LATLON_CRS)

    if not m:
        centriod = gdf.geometry.unary_union.centroid
        m = folium.Map(location=[centriod.y, centriod.x], zoom_start=11)

    for i, row in gdf.iterrows():
        folium.Circle(
            location=(row.geometry.y, row.geometry.x), 
            radius=5, 
            color=point_color, 
            fill=True,
            fill_color=point_color,  # fill color of the circle
            fill_opacity=1,  # opacity of the fill color
            tooltip=row[prefix+'matched_road_id'] # row[prefix+'distance_to_matched_road']+
        ).add_to(m)

    if old_crs is not None:
        gdf = gdf.to_crs(old_crs)

    return m