from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def normalize_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def min_distance(point: np.ndarray, points: np.ndarray) -> float:
    if points is None or len(points) == 0:
        return float("inf")
    p = np.asarray(point, dtype=np.float64).reshape(1, 2)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return float(np.min(np.linalg.norm(pts - p, axis=1)))


def rectangle_descriptor(corners: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    center = np.mean(pts, axis=0)
    # Normalize to cyclic polygon order so this works both for raw rectangle
    # corners and for parking-ordered [top_l, top_r, bottom_l, bottom_r].
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.linalg.norm(edges, axis=1)
    longest_idx = int(np.argmax(lengths))
    theta = float(np.arctan2(edges[longest_idx, 1], edges[longest_idx, 0]))
    area = float(Polygon(pts).area) if len(pts) == 4 else 0.0
    return center, theta, np.sort(lengths), area


def average_nearest_corner_distance(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    a = np.asarray(corners_a, dtype=np.float64).reshape(4, 2)
    b = np.asarray(corners_b, dtype=np.float64).reshape(4, 2)
    d_ab = np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2), axis=1)
    d_ba = np.min(np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2), axis=1)
    return float(max(np.mean(d_ab), np.mean(d_ba)))


def is_same_slot(
    new_corners: np.ndarray,
    ref_corners: np.ndarray,
    *,
    eps_position: float = 0.15,
    eps_size: float = 0.15,
    eps_corner: float = 0.16,
) -> tuple[bool, dict]:
    new_center, new_theta, new_sizes, _ = rectangle_descriptor(new_corners)
    ref_center, ref_theta, ref_sizes, _ = rectangle_descriptor(ref_corners)
    center_dist = float(np.linalg.norm(new_center - ref_center))
    theta_diff = abs(normalize_angle(new_theta - ref_theta))
    theta_diff = min(theta_diff, abs(np.pi - theta_diff))
    size_diff = float(np.max(np.abs(new_sizes - ref_sizes)))
    corner_dist = average_nearest_corner_distance(new_corners, ref_corners)
    same = center_dist <= eps_position and size_diff <= eps_size and corner_dist <= eps_corner
    return bool(same), {
        "center_dist": center_dist,
        "theta_diff": theta_diff,
        "size_diff": size_diff,
        "corner_dist": corner_dist,
        "eps_position": eps_position,
        "eps_size": eps_size,
        "eps_corner": eps_corner,
    }


def distance_points_to_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.empty((0,), dtype=np.float64)
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-12:
        return np.linalg.norm(pts - a.reshape(1, 2), axis=1)
    t = ((pts - a.reshape(1, 2)) @ ab) / ab2
    t = np.clip(t, 0.0, 1.0)
    closest = a.reshape(1, 2) + t.reshape(-1, 1) * ab.reshape(1, 2)
    return np.linalg.norm(pts - closest, axis=1)


def segment_clearance(a: np.ndarray, b: np.ndarray, boundary_pts: np.ndarray) -> float:
    if boundary_pts is None or len(boundary_pts) == 0:
        return float("inf")
    dists = distance_points_to_segment(boundary_pts, a, b)
    return float(np.min(dists)) if len(dists) > 0 else float("inf")


def side_wall_support(a: np.ndarray, b: np.ndarray, boundary_pts: np.ndarray, wall_dist: float = 0.06) -> float:
    if boundary_pts is None or len(boundary_pts) == 0:
        return 0.0
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    samples = np.array([a + t * (b - a) for t in np.linspace(0.1, 0.9, 9)], dtype=np.float64)
    covered = sum(1 for s in samples if min_distance(s, boundary_pts) <= wall_dist)
    return float(covered / len(samples))


def order_slot_by_entry_edge(corners: np.ndarray, edge_idx: int) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    bottom_l = pts[edge_idx]
    bottom_r = pts[(edge_idx + 1) % 4]
    top_r = pts[(edge_idx + 2) % 4]
    top_l = pts[(edge_idx + 3) % 4]
    return np.vstack([top_l, top_r, bottom_l, bottom_r])


def choose_safe_entry_edge(
    corners: np.ndarray,
    boundary_pts: np.ndarray,
    empty_pts: np.ndarray,
    robot_xy: np.ndarray,
    *,
    d_back: float = 0.50,
    side_wall_dist: float = 0.06,
    side_wall_block: float = 0.45,
    route_clearance: float = 0.20,
    point_clearance: float = 0.22,
) -> tuple[np.ndarray | None, int | None, dict]:
    """Try all rectangle sides as entry. Return ordered slot and debug for the best safe side."""
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    boundary = np.asarray(boundary_pts, dtype=np.float64).reshape(-1, 2)
    empty = np.asarray(empty_pts, dtype=np.float64).reshape(-1, 2)
    robot_xy = np.asarray(robot_xy, dtype=np.float64).reshape(2)
    center = np.mean(pts, axis=0)

    candidates = []
    for idx in range(4):
        a = pts[idx]
        b = pts[(idx + 1) % 4]
        edge = b - a
        edge_len = float(np.linalg.norm(edge))
        if edge_len < 1e-6:
            continue
        mid = 0.5 * (a + b)
        normal = np.array([-edge[1], edge[0]], dtype=np.float64) / edge_len
        if np.dot(normal, center - mid) > 0:
            normal = -normal
        prepark = mid + normal * d_back
        wall_support = side_wall_support(a, b, boundary, wall_dist=side_wall_dist)
        p_clearance = min_distance(prepark, boundary)
        approach_clearance = segment_clearance(robot_xy, prepark, boundary)
        entry_clearance = segment_clearance(prepark, center, boundary)
        empty_support = min_distance(prepark, empty)
        robot_dist = float(np.linalg.norm(prepark - robot_xy))

        reasons = []
        if wall_support >= side_wall_block:
            reasons.append("wall_support_high")
        if p_clearance < point_clearance:
            reasons.append("p_too_close")
        if approach_clearance < route_clearance:
            reasons.append("approach_blocked")
        if entry_clearance < route_clearance:
            reasons.append("entry_blocked")

        score = (
            2.0 * min(approach_clearance, 1.0)
            + 2.0 * min(entry_clearance, 1.0)
            + 1.0 * min(p_clearance, 1.0)
            - 2.0 * wall_support
            - 0.8 * min(empty_support, 1.0)
            - 0.2 * robot_dist
        )
        debug = {
            "edge_idx": idx,
            "mid": mid,
            "normal": normal,
            "prepark": prepark,
            "center": center,
            "wall_support": wall_support,
            "p_clearance": p_clearance,
            "approach_clearance": approach_clearance,
            "entry_clearance": entry_clearance,
            "empty_support": empty_support,
            "robot_dist": robot_dist,
            "score": float(score),
            "reasons": reasons,
        }
        candidates.append(debug)

    valid = [c for c in candidates if not c["reasons"]]
    if not valid:
        return None, None, {"candidates": candidates, "reason": "no_safe_entry"}
    best = max(valid, key=lambda c: c["score"])
    ordered = order_slot_by_entry_edge(pts, int(best["edge_idx"]))
    return ordered, int(best["edge_idx"]), {"selected": best, "candidates": candidates}
