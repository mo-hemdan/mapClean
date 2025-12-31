import numpy as np
import geopandas as gpd
import folium

def plot_labeled_points_on_map(points: np.ndarray,
                               labels: np.ndarray,
                               epsg: int,
                               map_file: str = "map.html"):
    """
    Plot labeled points on a Folium map.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2): [x, y] in the given EPSG.
    labels : np.ndarray
        Array of shape (N,) with values {0, 1, 2}.
    epsg : int
        EPSG code of the input coordinates.
    map_file : str
        Filename for the saved HTML map.
    """

    # --- Input validation ---
    assert points.shape[1] == 2, "Points must be (N, 2)"
    assert len(points) == len(labels), "Points and labels must have same length"

    # --- Color palette: label -> color ---
    palette = np.array([
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "cyan",
        "magenta",
        "yellow"
    ])

    point_colors = palette[labels]   # vectorized mapping

    # --- Construct GeoDataFrame ---
    gdf = gpd.GeoDataFrame(
        {"label": labels, "color": point_colors},
        geometry=gpd.points_from_xy(points[:, 0], points[:, 1]),
        crs=f"EPSG:{epsg}"
    )

    # --- Convert to EPSG:4326 (required by Folium) ---
    gdf = gdf.to_crs(4326)

    # --- Compute map center ---
    center_lat = gdf.geometry.y.mean()
    center_lon = gdf.geometry.x.mean()

    # --- Initialize Folium map ---
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # --- Add points to map ---
    for geom, color in zip(gdf.geometry, gdf["color"]):
        folium.CircleMarker(
            location=[geom.y, geom.x],
            radius=2,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8
        ).add_to(m)

    # --- Add Legend ---
    legend_html = """
    <div style="position: fixed; 
                bottom: 30px; left: 30px; width: 150px; 
                background-color: white; z-index:9999; 
                padding: 10px; border:2px solid grey;">
        <b>Labels</b><br>
        <i style="background:red; width:10px; height:10px; display:inline-block;"></i> 0 (red)<br>
        <i style="background:blue; width:10px; height:10px; display:inline-block;"></i> 1 (blue)<br>
        <i style="background:green; width:10px; height:10px; display:inline-block;"></i> 2 (green)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # --- Save map ---
    m.save(map_file)
    print(f"Map saved to {map_file}")
