To use the Experimentation module, you can run the following


```python
from Experimentation import RandomGPSDataGenerator

xmin, ymin = 0, 0  # bottom-left corner
xmax, ymax = 100, 50  # top-right corner
rg = RandomGPSDataGenerator(xmin, ymin, xmax, ymax)
n_points = 100  # Number of points you want to generate
gdf = rg.generate_points(n_points)
rg.plot(gdf)
```