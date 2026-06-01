from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def normalize_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def min_distance(point: np.ndarray, points: np.ndarray) -> float:
    """
    Compute the minimum Euclidean distance from one point to a set of points

    :param point: Query point [x, y]
    :param points: Point cloud of shape (N, 2)

    :return: Minimum distance from point to points. Returns infinity if points is empty
    """
    # No points means no obstacle, so clearance is infinite
    if points is None or len(points) == 0:
        return float("inf")

    # Convert query point and point cloud to fixed NumPy shapes
    p = np.asarray(point, dtype=np.float64).reshape(1, 2)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    # Compute all distances and return the smallest one
    return float(np.min(np.linalg.norm(pts - p, axis=1)))


def rectangle_descriptor(corners: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
    """
    Compute rectangle center, orientation, side lengths, and area

    :param corners: Rectangle corner coordinates of shape (4, 2). Corner order is arbitrary

    :return: (center, theta, lengths, area), where:
                - center: rectangle center [x, y]
                - theta: orientation of the longest edge [rad]
                - lengths: sorted edge lengths
                - area: rectangle area
    """

    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)

    # Compute the geometric center of the four corners.
    center = np.mean(pts, axis=0)

    # Sort corners by polar angle around the center
    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0],
    )
    pts = pts[np.argsort(angles)]

    # Compute edge vectors between consecutive polygon vertices.
    edges = np.roll(pts, -1, axis=0) - pts

    # Compute Euclidean length of each edge.
    lengths = np.linalg.norm(edges, axis=1)

    # Use the longest edge as the rectangle's main orientation axis
    longest_idx = int(np.argmax(lengths))

    # Convert the longest edge direction vector into an angle
    theta = float(np.arctan2(
        edges[longest_idx, 1],
        edges[longest_idx, 0],
    ))

    # Compute polygon area from cyclically ordered corners.
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
    """
    Check whether two detected rectangles describe the same parking slot

    :param new_corners: Corner coordinates of the newly detected rectangle, shape (4, 2)
    :param ref_corners: Corner coordinates of the reference rectangle, shape (4, 2)
    :param eps_position: Maximum allowed distance between rectangle centers
    :param eps_size: Maximum allowed difference between corresponding sorted edge lengths
    :param eps_corner: Maximum allowed average nearest-corner distance

    :return: (same, debug), where same is True if the rectangles match, and debug
              contains the measured differences and thresholds
    """
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
    """
    Compute distances from points to a line segment

    :param points: Point cloud of shape (N, 2)
    :param a: First segment endpoint [x, y]
    :param b: Second segment endpoint [x, y]

    :return: Array of distances from each point to segment [a, b]
    """
    # Convert point cloud to Nx2 array
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    # Empty input returns an empty distance array
    if len(pts) == 0:
        return np.empty((0,), dtype=np.float64)

    # Convert segment endpoints to 2D vectors
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)

    # Segment direction and squared length
    ab = b - a
    ab2 = float(np.dot(ab, ab))

    # Degenerate segment: distance to a single point
    if ab2 < 1e-12:
        return np.linalg.norm(pts - a.reshape(1, 2), axis=1)

    # Project every point onto the infinite line through a and b
    t = ((pts - a.reshape(1, 2)) @ ab) / ab2

    # Clamp projection to the actual segment
    t = np.clip(t, 0.0, 1.0)

    # Compute closest point on the segment for every input point
    closest = a.reshape(1, 2) + t.reshape(-1, 1) * ab.reshape(1, 2)

    # Return Euclidean distances to the closest segment points
    return np.linalg.norm(pts - closest, axis=1)


def segment_clearance(a: np.ndarray, b: np.ndarray, boundary_pts: np.ndarray) -> float:
    """
    Compute the minimum clearance between a segment and boundary points

    :param a: First segment endpoint [x, y]
    :param b: Second segment endpoint [x, y]
    :param boundary_pts: Boundary point cloud of shape (N, 2)

    :return: Minimum distance from boundary points to segment [a, b]
             Returns infinity if boundary_pts is empty.
    """
    # No boundary points means the segment is unobstructed
    if boundary_pts is None or len(boundary_pts) == 0:
        return float("inf")

    # Compute distance from every boundary point to the segment
    dists = distance_points_to_segment(boundary_pts, a, b)

    # Segment clearance is the nearest boundary distance
    return float(np.min(dists)) if len(dists) > 0 else float("inf")


def side_wall_support(a: np.ndarray, b: np.ndarray, boundary_pts: np.ndarray, wall_dist: float = 0.06) -> float:
    """
    Estimate how much of a rectangle side is supported by nearby boundary points

    :param a: First side endpoint [x, y]
    :param b: Second side endpoint [x, y]
    :param boundary_pts: Boundary point cloud of shape (N, 2)
    :param wall_dist: Maximum distance for a boundary point to count as supporting the side

    :return: Fraction of sampled side points that are close to the boundary
    """
    # No boundary points means there is no wall support
    if boundary_pts is None or len(boundary_pts) == 0:
        return 0.0

    # Convert side endpoints to 2D vectors
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)

    # Sample points along the middle part of the side
    samples = np.array([a + t * (b - a) for t in np.linspace(0.1, 0.9, 9)], dtype=np.float64)

    # Count how many sampled points are close to observed boundary points
    covered = sum(1 for s in samples if min_distance(s, boundary_pts) <= wall_dist)

    # Return normalized support score in [0, 1]
    return float(covered / len(samples))


def order_slot_by_entry_edge(corners: np.ndarray, edge_idx: int) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    bottom_l = pts[edge_idx]
    bottom_r = pts[(edge_idx + 1) % 4]
    top_r = pts[(edge_idx + 2) % 4]
    top_l = pts[(edge_idx + 3) % 4]
    return np.vstack([top_l, top_r, bottom_l, bottom_r])


def ordered_slot_fits(
    ordered_slot: np.ndarray,
    *,
    min_width: float,
    min_length: float,
) -> tuple[bool, dict]:
    pts = np.asarray(ordered_slot, dtype=np.float64).reshape(4, 2)
    top_l, top_r, bottom_l, bottom_r = pts
    bottom_width = float(np.linalg.norm(bottom_r - bottom_l))
    top_width = float(np.linalg.norm(top_r - top_l))
    left_length = float(np.linalg.norm(top_l - bottom_l))
    right_length = float(np.linalg.norm(top_r - bottom_r))
    dims = {
        "bottom_width": bottom_width,
        "top_width": top_width,
        "left_length": left_length,
        "right_length": right_length,
    }
    fits = min(bottom_width, top_width) >= min_width and min(left_length, right_length) >= min_length
    return bool(fits), dims


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
    """
    Select the safest rectangle side to use as the parking entry

    Each side is tested as a possible entry edge. For every side, the function
    computes a pre-parking point behind the edge, checks wall support, point
    clearance, approach clearance, and entry clearance, then selects the best
    valid side by score

    :param corners: Rectangle corners of the detected parking slot, shape (4, 2)
    :param boundary_pts: Observed boundary points used to detect walls and obstacles
    :param empty_pts: Observed empty-space points used as supporting free-space evidence
    :param robot_xy: Current robot position [x, y]
    :param d_back: Distance from the entry-edge midpoint to the pre-parking point
    :param side_wall_dist: Distance threshold for counting boundary points as wall support near an edge
    :param side_wall_block: Maximum allowed wall-support score for an entry side
    :param route_clearance: Minimum allowed clearance along approach and entry segments
    :param point_clearance: Minimum allowed clearance around the pre-parking point

    :return: (ordered, edge_idx, debug), where ordered is the slot reordered for the
              selected entry edge, edge_idx is the selected side index, and debug contains
              candidate scores and rejection reasons. If no side is safe, ordered and
              edge_idx are None
    """
    # Convert all geometry inputs to fixed NumPy shapes
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    boundary = np.asarray(boundary_pts, dtype=np.float64).reshape(-1, 2)
    empty = np.asarray(empty_pts, dtype=np.float64).reshape(-1, 2)
    robot_xy = np.asarray(robot_xy, dtype=np.float64).reshape(2)

    # Rectangle center is used to orient edge normals outward
    center = np.mean(pts, axis=0)

    candidates = []
    for idx in range(4):
        # Take one rectangle side as a possible entry edge
        a = pts[idx]
        b = pts[(idx + 1) % 4]
        edge = b - a
        edge_len = float(np.linalg.norm(edge))

        # Skip degenerate edges
        if edge_len < 1e-6:
            continue

        # Compute edge midpoint and outward normal
        mid = 0.5 * (a + b)
        normal = np.array([-edge[1], edge[0]], dtype=np.float64) / edge_len
        if np.dot(normal, center - mid) > 0:
            normal = -normal

        # Pre-parking point is placed behind the selected entry edge
        prepark = mid + normal * d_back

        # Estimate whether this edge is blocked by boundary points along the side
        wall_support = side_wall_support(a, b, boundary, wall_dist=side_wall_dist)

        # Check local clearance around the pre-parking point
        p_clearance = min_distance(prepark, boundary)

        # Check whether robot can safely reach the pre-parking point
        approach_clearance = segment_clearance(robot_xy, prepark, boundary)

        # Check whether robot can safely move from pre-parking point into the slot
        entry_clearance = segment_clearance(prepark, center, boundary)

        # Prefer pre-parking points that are not deep inside already observed empty points
        empty_support = min_distance(prepark, empty)

        # Prefer closer entry points when safety scores are similar
        robot_dist = float(np.linalg.norm(prepark - robot_xy))

        reasons = []

        # Reject sides that look like a wall rather than an opening
        if wall_support >= side_wall_block:
            reasons.append("wall_support_high")

        # Reject pre-parking point if it is too close to the boundary
        if p_clearance < point_clearance:
            reasons.append("p_too_close")

        # Reject route from robot to pre-parking point if it crosses too close to boundary
        if approach_clearance < route_clearance:
            reasons.append("approach_blocked")

        # Reject route from pre-parking point into slot if it crosses too close to boundary
        if entry_clearance < route_clearance:
            reasons.append("entry_blocked")

        # Score valid sides by route safety, point safety, wall support, empty support, and distance
        score = (
            2.0 * min(approach_clearance, 1.0)
            + 2.0 * min(entry_clearance, 1.0)
            + 1.0 * min(p_clearance, 1.0)
            - 2.0 * wall_support
            - 0.8 * min(empty_support, 1.0)
            - 0.2 * robot_dist
        )

        # Store full diagnostics for visualization and debugging
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

    # Keep only sides that passed all safety checks
    valid = [c for c in candidates if not c["reasons"]]

    # No side can currently be used as a safe entry
    if not valid:
        return None, None, {"candidates": candidates, "reason": "no_safe_entry"}

    # Select the safest valid side by score
    best = max(valid, key=lambda c: c["score"])

    # Reorder slot corners so downstream logic receives [top_l, top_r, bottom_l, bottom_r]
    ordered = order_slot_by_entry_edge(pts, int(best["edge_idx"]))

    return ordered, int(best["edge_idx"]), {"selected": best, "candidates": candidates}
