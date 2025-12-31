How to use this script

When you extract the Geofence
```python
from Plot import plot_geofence
m = plot_geofence(geofence)
m.save(folium_viz_folder+'1_geofence_around_points.html')
```

After Extracting the Road Network
```python 
from Plot import plot_map
m = plot_map(nx_map)
m.save(folium_viz_folder+'2_extracted_road_network.html')
```

After Matching
```python 
visualize_matches_to_html(gdf, file_path=folium_viz_folder+'3_matching_to_real_network.html', how_many=None)
```

```python 
from Plot import plot_points, plot_osmgraph
m = plot_map(beforeSystem_edgesRemoved_nx_map)
m = plot_osmgraph(beforeSystem_removedEdges_graph, crs=XY_CRS, color="black", m=m)
m = plot_points(gdf[gdf['ture_toBeMatched'] == True], m, point_color="yellow", prefix='ture_')
m = plot_points(gdf[gdf['ture_toBeMatched'] == False], m, point_color="blue", prefix='ture_')
m.save('./viz/true_road_network_edges_removed.html')
```

```python 

```

```python 

```

```python 

```

```python 

```