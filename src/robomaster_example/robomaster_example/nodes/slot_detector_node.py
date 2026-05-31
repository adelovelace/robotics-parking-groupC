from __future__ import annotations

import json
import sys

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from transforms3d._gohlketransforms import euler_from_quaternion

from robomaster_example.logic.parking_estimator import ParkingEstimator
from robomaster_example.logic.slot_geometry import choose_safe_entry_edge, rectangle_descriptor
from robomaster_example.logic.topic_codec import pack_slot, unpack_map_points


class SlotDetectorNode(Node):
    """Estimate parking hypotheses and decide whether they have a safe entry side."""

    VALIDATION_D_BACK = 0.50
    ENTRY_SIDE_WALL_DIST = 0.06
    ENTRY_SIDE_WALL_BLOCK = 0.45
    ENTRY_ROUTE_CLEARANCE = 0.20
    ENTRY_POINT_CLEARANCE = 0.22

    def __init__(self):
        super().__init__("slot_detector_node")
        self.estimator = ParkingEstimator(safety_margin=0.16, min_area=0.05)
        self.robot_pose = None
        self.boundary = np.empty((0, 2), dtype=np.float64)
        self.empty = np.empty((0, 2), dtype=np.float64)

        self.map_sub = self.create_subscription(Float32MultiArray, "map/global_points", self.map_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "odom", self.odom_callback, 10)
        self.candidate_pub = self.create_publisher(Float32MultiArray, "parking/candidate", 10)
        self.actionable_pub = self.create_publisher(Float32MultiArray, "parking/actionable", 10)
        self.debug_pub = self.create_publisher(String, "parking/debug", 10)

    def odom_callback(self, msg: Odometry) -> None:
        q = (msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        _, _, yaw = euler_from_quaternion(q)
        self.robot_pose = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y), float(yaw))

    def map_callback(self, msg: Float32MultiArray) -> None:
        self.boundary, self.empty = unpack_map_points(msg)
        self.estimate_and_publish()

    def estimate_and_publish(self) -> None:
        candidate = self.estimator.estimate(self.boundary, self.empty)
        if candidate is None:
            self.candidate_pub.publish(pack_slot(None, actionable=False))
            self.actionable_pub.publish(pack_slot(None, actionable=False))
            self.publish_debug({"valid": False, "reason": "no_candidate", "boundary": len(self.boundary), "empty": len(self.empty)})
            return

        self.candidate_pub.publish(pack_slot(candidate, actionable=False))
        center, theta, sizes, area = rectangle_descriptor(candidate)
        debug = {
            "valid": True,
            "actionable": False,
            "candidate_center": center.tolist(),
            "candidate_theta": theta,
            "candidate_sizes": sizes.tolist(),
            "candidate_area": area,
            "boundary": len(self.boundary),
            "empty": len(self.empty),
        }

        if self.robot_pose is None:
            self.actionable_pub.publish(pack_slot(None, actionable=False))
            debug["reason"] = "no_odom"
            self.publish_debug(debug)
            return

        robot_xy = np.array([self.robot_pose[0], self.robot_pose[1]], dtype=np.float64)
        ordered, edge_idx, entry_debug = choose_safe_entry_edge(
            candidate,
            self.boundary,
            self.empty,
            robot_xy,
            d_back=self.VALIDATION_D_BACK,
            side_wall_dist=self.ENTRY_SIDE_WALL_DIST,
            side_wall_block=self.ENTRY_SIDE_WALL_BLOCK,
            route_clearance=self.ENTRY_ROUTE_CLEARANCE,
            point_clearance=self.ENTRY_POINT_CLEARANCE,
        )
        debug["entry_debug"] = self._jsonify(entry_debug)
        if ordered is None:
            self.actionable_pub.publish(pack_slot(None, actionable=False))
            debug["reason"] = "no_safe_entry"
            self.publish_debug(debug)
            self.get_logger().info("[SLOT] Candidate detected, but no safe entry edge yet.")
            return

        selected = entry_debug.get("selected", {}) if isinstance(entry_debug, dict) else {}
        prepark = np.asarray(selected.get("prepark", [np.nan, np.nan]), dtype=np.float64)
        center_ordered = np.mean(ordered, axis=0)
        self.actionable_pub.publish(pack_slot(ordered, actionable=True, entry_edge=edge_idx, prepark=prepark, center=center_ordered))
        debug["actionable"] = True
        debug["entry_edge"] = int(edge_idx)
        debug["prepark"] = prepark.tolist()
        self.publish_debug(debug)
        self.get_logger().info(
            f"[SLOT] Actionable slot: entry_edge={edge_idx}, p=({prepark[0]:.3f},{prepark[1]:.3f}), "
            f"boundary={len(self.boundary)}, empty={len(self.empty)}"
        )

    def publish_debug(self, payload: dict) -> None:
        msg = String()
        msg.data = json.dumps(self._jsonify(payload))
        self.debug_pub.publish(msg)

    def _jsonify(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): self._jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._jsonify(v) for v in obj]
        return obj


def main(args=None):
    rclpy.init(args=args if args is not None else sys.argv)
    node = SlotDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
