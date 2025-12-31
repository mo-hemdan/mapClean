This paper shows how to use the SpatialQueryExecutor

```python
from CustomSpatialQuery import SpatialQueryExecutor
query_executor = SpatialQueryExecutor()
query_executor.partition_space(gdf, cell_width)
nearby_data = query_executor.build_index(gdf, raduis)
```
nearby_data consists of two dictionaries: first is the cell to points mapping. Second is the nearby dictionary
they are in tuple format (cell_to_points, nearby_dicts)

How to plot nearby points in grid

```python
query_executor.plot_nearby_cells((5, 3))
query_executor.partitioner.plot_partitioning(gdf)
```


How to test it?
After plotting, choose a specific point and its id for testing, Change the point's location to a different cell as shown below, After u run the nearby points
```python
from shapely.geometry import Point
gdf.loc[30, 'geometry'] = [Point(35, 45)]
```

Make sure the cell changed its location, the id is the same and it's not in the nearby cells if it's nearby and it's in an existing cell
```python
query_executor.partitioner.plot_partitioning(gdf)
gdf.loc[97]
query_executor.plot_nearby_cells((6, 4))
```

You can also check the cell to points for each cell the new and the old one to see where the point is
```python
query_executor.cell_to_points[(6, 4)]
query_executor.cell_to_points[(3, 4)]
```

Now run the adjust nearby points. You specify the point that has changed instead of going through each point
```python
n_changes = query_executor.adjust_query_results(gdf, [30])
```

Now check the new cell id
```python
gdf.loc[30, 'cell_id']
```

See if the new cell appears for both new and old
```python
query_executor.plot_nearby_cells((6, 4))
query_executor.plot_nearby_cells((3, 4))
```

Look up the new the cell to points and see if the point changed its cell here
```python
query_executor.cell_to_points[(6, 4)]
query_executor.cell_to_points[(3, 4)]
```

You can also generate a sample dataset (random)
```python
from Experimentation import RandomGPSDataGenerator

xmin, ymin = 0, 0  # bottom-left corner
xmax, ymax = 100, 100  # top-right corner
num_points = 100  # Number of points you want to generate
points_be_plotted = 100
rg = RandomGPSDataGenerator(xmin, ymin, xmax, ymax)
gdf = rg.generate_points(num_points)
# gdf['point_id'] = range(0, len(gdf))
# gdf.set_index('point_id', inplace=True) -- you can later change this
```

