from __future__ import annotations

import warnings

import numpy as np
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")


class ParkingEstimator:
    """Estimate a parking rectangle from accumulated boundary and empty-space points."""

    def __init__(self, safety_margin: float = 0.05, min_area: float = 0.1025):
        self.safety_margin = float(safety_margin)
        self.min_area = float(min_area)

    def estimate(self, boundary_pts: np.ndarray, empty_pts: np.ndarray) -> np.ndarray | None:
        # Convert input points to Nx2 float arrays
        boundary_pts = np.asarray(boundary_pts, dtype=np.float64).reshape(-1, 2)
        empty_pts = np.asarray(empty_pts, dtype=np.float64).reshape(-1, 2)

        # We need enough boundary points for a hull and enough empty points for a rectangle
        if len(boundary_pts) < 3 or len(empty_pts) < 4:
            return None

        # Build convex hull around the observed object boundary
        obj_hull = MultiPoint(boundary_pts).convex_hull

        # Keep only empty-space points that are inside the object hull
        empty_multipoint = MultiPoint(empty_pts)
        inside_empty = obj_hull.intersection(empty_multipoint)
        if inside_empty.is_empty:
            return None

        # Extract coordinates from Shapely intersection result
        inside_coords: list[list[float]] = []
        if hasattr(inside_empty, "geoms"):
            inside_coords = [
                [p.x, p.y]
                for p in inside_empty.geoms
                if p.geom_type == "Point" and not p.is_empty
            ]
        elif inside_empty.geom_type == "Point" and not inside_empty.is_empty:
            inside_coords = [[inside_empty.x, inside_empty.y]]

        # Not enough internal empty points to form a parking slot
        if len(inside_coords) < 4:
            return None

        inside = np.asarray(inside_coords, dtype=np.float64)

        # Remove empty points that are too close to the observed boundary
        dists = np.linalg.norm(inside[:, None, :] - boundary_pts[None, :, :], axis=2)
        safe_empty = inside[np.min(dists, axis=1) >= self.safety_margin]
        if len(safe_empty) < 4:
            return None

        # Cluster safe empty-space points and ignore DBSCAN noise points
        labels = DBSCAN(eps=0.10, min_samples=4).fit_predict(safe_empty)
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(unique) == 0:
            return None

        # Use the largest safe empty-space cluster as the parking-space hypothesis
        best_cluster = safe_empty[labels == unique[int(np.argmax(counts))]]
        if len(best_cluster) < 5:
            return None

        # Reject very thin clusters that cannot represent a useful slot
        dx = np.max(best_cluster[:, 0]) - np.min(best_cluster[:, 0])
        dy = np.max(best_cluster[:, 1]) - np.min(best_cluster[:, 1])
        if dx <= 0.15 or dy <= 0.15:
            return None

        # Build hull of the selected empty-space cluster
        parking_hull = MultiPoint(best_cluster).convex_hull
        if parking_hull.geom_type != "Polygon" or parking_hull.area <= 1e-4:
            return None

        # Approximate the empty-space hull by its minimum-area rotated rectangle
        rect = parking_hull.minimum_rotated_rectangle

        # Return rectangle corners if the estimated slot is large enough
        if rect.geom_type == "Polygon" and rect.area >= self.min_area:
            return np.asarray(rect.exterior.coords, dtype=np.float64)[:4]

        return None