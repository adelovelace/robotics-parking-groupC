from __future__ import annotations

import numpy as np
from std_msgs.msg import Float32MultiArray


def _as_points(points: np.ndarray) -> np.ndarray:
    if points is None:
        return np.empty((0, 2), dtype=np.float64)
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return arr.reshape(-1, 2)


def pack_map_points(boundary_pts: np.ndarray, empty_pts: np.ndarray) -> Float32MultiArray:
    """
    Pack map boundary and empty points as [n_boundary, 2 boundary flatten points, n_empty, 2 empty flatten points].
    :param boundary_pts: a discrete list of an obstacle's boundary points
    :param empty_pts: a discrete list of empty space points
    :return: Float32MultiArray with encoded data
    """
    boundary = _as_points(boundary_pts)
    empty = _as_points(empty_pts)

    data = [float(len(boundary))]
    data.extend(boundary.reshape(-1).astype(float).tolist())

    data.append(float(len(empty)))
    data.extend(empty.reshape(-1).astype(float).tolist())

    msg = Float32MultiArray()
    msg.data = data

    return msg


def unpack_map_points(msg: Float32MultiArray) -> tuple[np.ndarray, np.ndarray]:
    """
    Unpack the message data from
    `[n_boundary, 2 boundary flatten points, n_empty, 2 empty flatten points]`
    format to
    tuple[np.ndarray, np.ndarray]
    :param msg: the message to unpack
    :return: tuple[np.ndarray, np.ndarray] - two arrays of boundary and empty points of shape
                                             (n_boundary, 2) and (n_empty, 2)
    """
    data = list(msg.data)
    if not data:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    idx = 0
    n_boundary = int(round(data[idx]))
    idx += 1

    boundary_flat = data[idx:idx + 2 * n_boundary]
    idx += 2 * n_boundary

    boundary = np.asarray(boundary_flat, dtype=np.float64).reshape(-1, 2)
    n_empty = int(round(data[idx])) if idx < len(data) else 0
    idx += 1

    empty_flat = data[idx:idx + 2 * n_empty]
    empty = np.asarray(empty_flat, dtype=np.float64).reshape(-1, 2)

    return boundary, empty


def pack_observation(robot_pose: tuple[float, float, float], boundary_pts: np.ndarray, empty_pts: np.ndarray) -> Float32MultiArray:
    """
    Pack observation data including robot pose, boundary points, and empty points into a `Float32MultiArray`.
    Packing scheme: [x, y, theta, n_boundary, boundary_points, n_empty, empty_points]

    :param robot_pose: Tuple containing the robot's x, y coordinates and orientation (theta).
    :param boundary_pts: A NumPy array representing the boundary points in the environment.
    :param empty_pts: A NumPy array representing the empty points in the environment.
    :return: A `Float32MultiArray` object containing the packed observation data.
    """
    boundary = _as_points(boundary_pts)
    empty = _as_points(empty_pts)

    x, y, theta = robot_pose
    data = [float(x), float(y), float(theta), float(len(boundary))]

    data.extend(boundary.reshape(-1).astype(float).tolist())

    data.append(float(len(empty)))
    data.extend(empty.reshape(-1).astype(float).tolist())

    msg = Float32MultiArray()
    msg.data = data

    return msg


def unpack_observation(msg: Float32MultiArray) -> tuple[tuple[float, float, float], np.ndarray, np.ndarray]:
    data = list(msg.data)
    if len(data) < 4:
        return (0.0, 0.0, 0.0), np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    x, y, theta = float(data[0]), float(data[1]), float(data[2])
    idx = 3
    n_boundary = int(round(data[idx])); idx += 1
    boundary_flat = data[idx:idx + 2 * n_boundary]; idx += 2 * n_boundary
    boundary = np.asarray(boundary_flat, dtype=np.float64).reshape(-1, 2)
    n_empty = int(round(data[idx])) if idx < len(data) else 0; idx += 1
    empty_flat = data[idx:idx + 2 * n_empty]
    empty = np.asarray(empty_flat, dtype=np.float64).reshape(-1, 2)
    return (x, y, theta), boundary, empty


def pack_slot(
    corners: np.ndarray | None,
    *,
    actionable: bool,
    entry_edge: int = -1,
    prepark: np.ndarray | None = None,
    center: np.ndarray | None = None,
) -> Float32MultiArray:
    """
    Pack slot as [valid, actionable, entry_edge, p_x, p_y, c_x, c_y, 8 corners].

    :param corners: array of shape (4, 2) defining the parking rectangle
    :param actionable: flag if we should proceed with the parking in this slop
    :param entry_edge: the edge of the rectangle that was used to enter the slot
    :param prepark: the point where the robot should be parked before entering the slot, `p` in the report
    :param center: the center point of the parking rectangle
    :return: encoded Float32MultiArray message
    """
    msg = Float32MultiArray()
    if corners is None:
        msg.data = [0.0, 0.0, -1.0, np.nan, np.nan, np.nan, np.nan]
        return msg
    pts = _as_points(corners).reshape(4, 2)
    if center is None:
        center = np.mean(pts, axis=0)
    if prepark is None:
        prepark = np.array([np.nan, np.nan], dtype=np.float64)
    data = [1.0, 1.0 if actionable else 0.0, float(entry_edge), float(prepark[0]), float(prepark[1]), float(center[0]), float(center[1])]
    data.extend(pts.reshape(-1).astype(float).tolist())
    msg.data = data
    return msg


def unpack_slot(msg: Float32MultiArray) -> dict:
    """
    Unpack the message data from Float32MultiArray.
    Follows the same format as `pack_slot`.

    :param msg: message to unpack
    :return: dictionary with unpacked data
                    - valid: bool
                    - actionable: bool
                    - entry_edge: int
                    - prepark: np.ndarray, (2,)
                    - center: np.ndarray,  (2,)
                    - corners: np.ndarray, (4, 2)
    """
    data = list(msg.data)
    if len(data) < 7 or data[0] < 0.5:
        return {"valid": False, "actionable": False, "entry_edge": -1, "corners": None}
    corners = None
    if len(data) >= 15:
        corners = np.asarray(data[7:15], dtype=np.float64).reshape(4, 2)
    return {
        "valid": True,
        "actionable": bool(data[1] >= 0.5),
        "entry_edge": int(round(data[2])),
        "prepark": np.asarray([data[3], data[4]], dtype=np.float64),
        "center": np.asarray([data[5], data[6]], dtype=np.float64),
        "corners": corners,
    }
