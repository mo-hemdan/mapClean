# %%
import math
import json

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN

from sklearn.neighbors import NearestNeighbors
from shapely.ops import transform
from pyproj import Transformer


class MapMaker:
    """
    Class to infer new road centerlines from spatial point data.
    """

    def __init__(self, G):
        """
        Initializes the MapMaker with point data and base road network.

        Args:
            G (networkx.MultiDiGraph): OSMnx road network graph.
        """
        print("Extracting edge list...")
        self.G = G
        self.projector = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True
        ).transform
        self.edge_list = self.extract_edge_list(self.G)



    # ================================================================
    # 0. Helpers
    # ================================================================

    def angle_norm_180(self, angle):
        """Normalize angle to [0,180)."""
        angle = (angle + 360.0) % 360.0
        if angle >= 180:
            angle -= 180
        return angle


    def angle_diff(self, a, b):
        """Angular difference for undirected angles."""
        d = abs(a - b)
        return d if d <= 90.0 else 180.0 - d


    # ================================================================
    # 1. Extract base edges from OSMnx graph
    # ================================================================

    

    def extract_edge_list(self, G):
        """Extract Shapely LineStrings from OSMnx graph."""
        edges = []
        for u, v, data in G.edges(keys=False, data=True):
            if "geometry" in data:
                geom = data["geometry"]
            else:
                # Straight segment between nodes
                p0 = Point((G.nodes[u]["x"], G.nodes[u]["y"]))
                p1 = Point((G.nodes[v]["x"], G.nodes[v]["y"]))
                geom = LineString([p0, p1])
                geom = transform(self.projector, geom)
            edges.append({"geom": geom})
        return edges


    # ================================================================
    # 2. Filter out points hugging existing roads (<8m from base)
    # ================================================================

    def filter_points_by_distance(self, gdf_points, min_dist=8.0):
        # def pt_min_dist(pt):
        #     return float(min(pt.distance(e["geom"]) for e in self.edge_list))

        # gdf_points["d_to_base"] = gdf_points.geometry.apply(pt_min_dist)
        gdf_points["d_to_base"] = gdf_points['matching_score']
        out = gdf_points[gdf_points["d_to_base"] > min_dist].copy()
        out.drop(columns="d_to_base", inplace=True)
        return out


    # ================================================================
    # 3. Compute local orientation per point (only if "angle" not provided)
    # ================================================================

    def compute_point_orientations(self, gdf_points, k_neighbors=10):
        """Compute PCA angle for points."""
        coords = np.array([(p.x, p.y) for p in gdf_points.geometry])
        nn = NearestNeighbors(n_neighbors=k_neighbors).fit(coords)
        neighbors_idx = nn.kneighbors(return_distance=False)

        angles = np.zeros(len(coords))

        for i, idx in enumerate(neighbors_idx):
            local_pts = coords[idx]
            if len(local_pts) < 2:
                angles[i] = 0
                continue

            pca = PCA(n_components=2).fit(local_pts)
            vx, vy = pca.components_[0]
            angles[i] = self.angle_norm_180(math.degrees(math.atan2(vy, vx)))

        return angles


    # ================================================================
    # 4. Cluster points using distance + orientation
    # ================================================================

    def cluster_points(self, gdf_points, ANG_WEIGHT=2.0, min_cluster_size=50):

        coords = np.array([(p.x, p.y) for p in gdf_points.geometry])
        angles = np.deg2rad(gdf_points["angle"].values)

        # Encode angle as a vector
        ang_x = np.cos(angles) * ANG_WEIGHT
        ang_y = np.sin(angles) * ANG_WEIGHT

        # Final feature matrix: (x, y, angle_x, angle_y)
        X = np.column_stack([coords[:,0], coords[:,1], ang_x, ang_y])

        # Run HDBSCAN normally, with no custom metric
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            core_dist_n_jobs=32
        )

        labels = clusterer.fit_predict(X)
        gdf_points["cluster"] = labels
        return gdf_points

    # ================================================================
    # 5. Fit centerline via PCA per cluster
    # ================================================================

    def fit_centerlines_from_clusters(self, gdf_points):
        inferred = []
        coords = np.array([(p.x, p.y) for p in gdf_points.geometry])
        cluster_ids = sorted(gdf_points["cluster"].unique())

        for cid in cluster_ids:
            if cid == -1:
                continue

            idxs = np.where(gdf_points["cluster"].values == cid)[0]
            pts = coords[idxs]

            if len(pts) < 10:
                continue

            # Deduplicate slightly
            pts_unique = np.array(list({(round(x, 3), round(y, 3)) for x, y in pts}))
            if len(pts_unique) < 3:
                continue

            pca = PCA(n_components=1)
            proj = pca.fit_transform(pts_unique)
            order = np.argsort(proj[:, 0])
            ordered = pts_unique[order]

            # Subsample evenly
            k = min(30, len(ordered))
            keep = np.linspace(0, len(ordered)-1, k).astype(int)
            sampled = ordered[keep]

            ls = LineString(sampled.tolist()).simplify(0.6, preserve_topology=False)
            if ls.length >= 8.0:
                inferred.append((cid, ls))

        if not inferred:
            return gpd.GeoDataFrame(columns=["cluster", "support", "geometry"])

        out = gpd.GeoDataFrame(
            {
                "cluster": [cid for cid, _ in inferred],
                "support": [np.sum(gdf_points["cluster"] == cid) for cid, _ in inferred],
                "geometry": [ls for _, ls in inferred],
            },
            crs=gdf_points.crs,
        )
        return out


    # ================================================================
    # 6. Split base roads into horizontal/vertical groups
    # ================================================================

    def edge_groups_by_orientation(self, edge_list):
        horiz, vert = [], []
        for e in edge_list:
            coords = list(e["geom"].coords)
            x0, y0 = coords[0]
            x1, y1 = coords[-1]

            ang = abs(math.degrees(math.atan2(y1 - y0, x1 - x0)))
            ang = min(ang, 180 - ang)

            if ang <= 22.5:
                horiz.append(e["geom"])
            elif ang >= 67.5:
                vert.append(e["geom"])
            else:
                (horiz if ang < 45 else vert).append(e["geom"])

        return horiz, vert


    # ================================================================
    # 7. Snap inferred centerlines to base roads
    # ================================================================

    def nearest_on_group(self, p: Point, geoms):
        best = None
        for g in geoms:
            t = g.project(p)
            q = g.interpolate(t)
            d = p.distance(q)
            if (best is None) or (d < best[0]):
                best = (d, q)
        return best


    def snap_to_base(self, line: LineString, H_GROUP, V_GROUP):
        if line.is_empty or len(line.coords) < 2:
            return line

        start = Point(line.coords[0])
        end = Point(line.coords[-1])

        ds_h = self.nearest_on_group(start, H_GROUP)
        ds_v = self.nearest_on_group(start, V_GROUP)
        de_h = self.nearest_on_group(end, H_GROUP)
        de_v = self.nearest_on_group(end, V_GROUP)

        combo1 = (ds_v[0] if ds_v else 1e9) + (de_h[0] if de_h else 1e9)
        combo2 = (ds_h[0] if ds_h else 1e9) + (de_v[0] if de_v else 1e9)

        if combo1 <= combo2:
            ps = ds_v[1] if ds_v else start
            pe = de_h[1] if de_h else end
        else:
            ps = ds_h[1] if ds_h else start
            pe = de_v[1] if de_v else end

        new_coords = list(line.coords)
        new_coords[0] = (ps.x, ps.y)
        new_coords[-1] = (pe.x, pe.y)
        return LineString(new_coords)


    def snap_centerlines(self, gdf_lines, edge_list):
        H_GROUP, V_GROUP = self.edge_groups_by_orientation(edge_list)
        snapped = [self.snap_to_base(g, H_GROUP, V_GROUP) for g in gdf_lines.geometry]
        gdf_out = gdf_lines.copy()
        gdf_out["geometry"] = snapped
        return gdf_out


    # ================================================================
    # 8. MAIN FUNCTION
    # ================================================================

    def infer_roads(self, points, skip_snapping=False):
        """
        Input:
            points: GeoDataFrame with `geometry` in EPSG:3857 and optional `angle` column
        
        Output:
            snapped_centerlines_gdf: GeoDataFrame of inferred new roads
        """

        print("Filtering points hugging existing roads...")
        pts_far = self.filter_points_by_distance(points, min_dist=8.0)
        if len(pts_far) == 0:
            print("No unmatched points after distance filtering.")
            return None, 0

        print("Computing local orientations...")
        if "angle" not in pts_far.columns:
            pts_far["angle"] = self.compute_point_orientations(pts_far)
        else:
            pts_far["angle"] = pts_far["angle"].apply(self.angle_norm_180)

        print("Clustering points...")
        clustered = self.cluster_points(pts_far, ANG_WEIGHT=2.0, min_cluster_size=20)

        print("Fitting centerlines from clusters...")
        centerlines = self.fit_centerlines_from_clusters(clustered)
        if len(centerlines) == 0:
            print("No centerlines inferred.")
            return None, 0
        
        if skip_snapping:
            return centerlines, len(centerlines)

        print("Snapping centerlines to base map...")
        snapped = self.snap_centerlines(centerlines, self.edge_list)
        new_roads = len(snapped)

        return snapped, new_roads