


```python
gdf['ture_matched_road_geom'] = gdf['ture_matched_road_geom'].apply(lambda geom: geom.wkt)
gdf.to_parquet('file.parquet', compression="gzip", engine="pyarrow")
import geopandas as gpd
gdf = gpd.read_parquet('file.parquet')
gdf['ture_matched_road_id'] = gdf['ture_matched_road_id'].apply(lambda arr: tuple(arr))
```