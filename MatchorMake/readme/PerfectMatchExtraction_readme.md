# Perfect Match Extraction

You can experiment with this module with the RandomGPSDataGenerator. However, it's very hard to plot your results using this module as merely it generates roads as a table

```python
from Experimentation import RandomGPSDataGenerator
from PerfectMatchExtraction import PerfectMatchExtraction
rg = RandomGPSDataGenerator(0, 0, 100, 100)
n_points = 50
n_roads = 10
delta = 5
beta = 0.8
points = rg.generate_points(n_points)
roads = rg.generate_roads(n_roads)

# 3. Map matching: Assign each point to the nearest road
points_gdf = rg.map_matching(points, roads)

# 4. Instantiate PerfectMatchExtraction and apply it
extractor = PerfectMatchExtraction(delta, beta, prefix='')
perfect_points, uncertain_points, perfect_roads = extractor.extract_perfect_instances(
    points_gdf, 'distance_to_matched_road', verbose=True
)

# 5. Plot the results
rg.plot_PME_results(points_gdf, roads, perfect_points, uncertain_points, perfect_roads, plot_geofence=True, delta=delta)
```