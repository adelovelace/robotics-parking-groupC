from __future__ import annotations

import numpy as np


class GridMap:
    """Small point-cloud grid map for boundary and free-space observations."""

    def __init__(self, resolution: float = 0.02):
        self.resolution = float(resolution)
        self.boundary_points = np.empty((0, 2), dtype=np.float64)
        self.empty_points = np.empty((0, 2), dtype=np.float64)

    def reset(self) -> None:
        self.boundary_points = np.empty((0, 2), dtype=np.float64)
        self.empty_points = np.empty((0, 2), dtype=np.float64)

    def update_boundary(self, points: np.ndarray) -> None:
        self.boundary_points = self._append_snapped_unique(self.boundary_points, points)

    def update_empty_space(self, points: np.ndarray) -> None:
        self.empty_points = self._append_snapped_unique(self.empty_points, points)

    def get_points(self) -> tuple[np.ndarray, np.ndarray]:
        boundary = self.boundary_points[np.all(np.isfinite(self.boundary_points), axis=1)]
        empty = self.empty_points[np.all(np.isfinite(self.empty_points), axis=1)]
        return boundary, empty

    def _append_snapped_unique(self, current: np.ndarray, new_points: np.ndarray) -> np.ndarray:
        if new_points is None or len(new_points) == 0:
            return current
        pts = np.asarray(new_points, dtype=np.float64).reshape(-1, 2)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) == 0:
            return current
        snapped = np.round(pts / self.resolution) * self.resolution
        if len(current) == 0:
            return np.unique(snapped, axis=0)
        return np.unique(np.vstack([current, snapped]), axis=0)
