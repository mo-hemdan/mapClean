Please take note that in the process of adding noise to the dataset, we clip the geometry. That means the points are clipped into the geomtry

This is an example of how to this in polygons

```python
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

# Example GeoDataFrame with two polygons
data = {'geometry': [Polygon([(-10, 40), (-10, 50), (-5, 50), (-5, 40)]),
                     Polygon([(0, 40), (0, 50), (5, 50), (5, 40)])]}
gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

# Example boundary for clipping
boundary = Polygon([(-6, 42), (-6, 48), (-2, 48), (-2, 42)])  # Example boundary

# Clipping the GeoDataFrame with the boundary
clipped_gdf = gdf.clip(boundary)

# Plotting the original and clipped GeoDataFrames
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot the original GeoDataFrame
gdf.plot(ax=ax[0], color='lightblue', edgecolor='black')
ax[0].set_title('Original GeoDataFrame')

# Plot the clipped GeoDataFrame
clipped_gdf.plot(ax=ax[1], color='lightgreen', edgecolor='black')
ax[1].set_title('Clipped GeoDataFrame')

# Plot the boundary on both plots
gpd.GeoSeries([boundary], crs="EPSG:4326").plot(ax=ax[0], color='none', edgecolor='red', linestyle='--')
gpd.GeoSeries([boundary], crs="EPSG:4326").plot(ax=ax[1], color='none', edgecolor='red', linestyle='--')

plt.show()
```