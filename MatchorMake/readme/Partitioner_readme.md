This is the Paritioning Module

You can use it as follows:

```python
from Partitioner import Partitioner
from Experimentation import RandomPointGenerator

xmin, ymin = 0, 0  # bottom-left corner
xmax, ymax = 100, 100  # top-right corner
num_points = 100  # Number of points you want to generate
points_be_plotted = 20
rg = RandomPointGenerator(xmin, ymin, xmax, ymax)
gdf = rg.generate(num_points)

partitioner = Partitioner()

cell_width = 10  # Adjust as needed
partitioner.partition(gdf, cell_width)
partitioner.calculate_ids(gdf)

partitioner.plot_partitioning(gdf.sample(points_be_plotted))
```


If you want to get the Ids of Nearby Cells and plot them
```python 
central_cell = (5, 5)
raduis = 30
nearby_cells = partitioner.get_Nearbycells_per_cell(central_cell, raduis)
partitioner.plot_nearby_cells_results(nearby_cells, central_cell, raduis)
```
