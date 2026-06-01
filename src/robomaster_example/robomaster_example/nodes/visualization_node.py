from __future__ import annotations

import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from transforms3d._gohlketransforms import euler_from_quaternion

from robomaster_example.logic.topic_codec import unpack_map_points, unpack_slot
from robomaster_example.vision import FloorProjectiveTransform


class VisualizationNode(Node):
    """Debug-only OpenCV visualization for the global map and live camera overlay."""

    def __init__(self):
        super().__init__("visualization_node")
        self.map_size = 600
        self.scale = 100.0
        self.cx = self.map_size // 2
        self.cy = self.map_size // 2
        self.boundary = np.empty((0, 2), dtype=np.float64)
        self.empty = np.empty((0, 2), dtype=np.float64)
        self.robot_pose = (0.0, 0.0, 0.0)
        self.candidate = None
        self.actionable = None
        self.bridge = CvBridge()
        self.T = FloorProjectiveTransform.from_points()

        self.camera_sub = self.create_subscription(Image, "camera/image_color", self.camera_callback, 10)
        self.map_sub = self.create_subscription(Float32MultiArray, "map/global_points", self.map_callback, 10)
        self.candidate_sub = self.create_subscription(Float32MultiArray, "parking/candidate", self.candidate_callback, 10)
        self.actionable_sub = self.create_subscription(Float32MultiArray, "parking/actionable", self.actionable_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "odom", self.odom_callback, 10)
        self.timer = self.create_timer(1 / 10, self.draw)

    def camera_callback(self, msg: Image) -> None:
        """Show the old camera overlay: detected boundary/contact pixels and empty-space samples."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            result = self.T.estimate_boundaries(
                frame,
                min_component_area=50,
                bottom_band_px=1,
                max_gap_below_px=8,
                dbscan_eps=0.05,
                dbscan_min_samples=4,
                min_cluster_points=10,
                empty_space_stride=3,
            )
            overlay = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Old overlay logic: green contact/boundary pixels, blue-orange empty-space samples.
            for px in result.image_pixels:
                cv2.circle(overlay, (int(px[0]), int(px[1])), 2, (0, 255, 0), -1)
            for px in result.empty_image_pixels:
                cv2.circle(overlay, (int(px[0]), int(px[1])), 1, (255, 100, 0), -1)

            cv2.imshow("RoboMaster Camera - Boundary & Parking", overlay)
            cv2.waitKey(1)
        except Exception as exc:
            self.get_logger().warn(f"[VIS] Camera overlay failed: {exc}", throttle_duration_sec=1.0)

    def map_callback(self, msg: Float32MultiArray) -> None:
        self.boundary, self.empty = unpack_map_points(msg)

    def candidate_callback(self, msg: Float32MultiArray) -> None:
        parsed = unpack_slot(msg)
        self.candidate = parsed["corners"] if parsed["valid"] else None

    def actionable_callback(self, msg: Float32MultiArray) -> None:
        parsed = unpack_slot(msg)
        self.actionable = parsed["corners"] if parsed["valid"] and parsed["actionable"] else None

    def odom_callback(self, msg: Odometry) -> None:
        q = (msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        _, _, yaw = euler_from_quaternion(q)
        self.robot_pose = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y), float(yaw))

    def world_to_px(self, point: np.ndarray) -> tuple[int, int]:
        return int(self.cx - point[1] * self.scale), int(self.cy - point[0] * self.scale)

    def draw_points(self, img: np.ndarray, points: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
        for p in points:
            px, py = self.world_to_px(p)
            if 0 <= px < self.map_size and 0 <= py < self.map_size:
                cv2.circle(img, (px, py), radius, color, -1)

    def draw_rect(self, img: np.ndarray, corners: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
        if corners is None:
            return
        pts = np.array([self.world_to_px(p) for p in corners], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    def draw(self) -> None:
        img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.draw_points(img, self.empty, (255, 100, 100), 1)
        self.draw_points(img, self.boundary, (0, 0, 255), 2)
        self.draw_rect(img, self.candidate, (0, 180, 180), 1)
        self.draw_rect(img, self.actionable, (0, 255, 255), 2)

        x, y, theta = self.robot_pose
        rx, ry = self.world_to_px(np.array([x, y], dtype=np.float64))
        if 0 <= rx < self.map_size and 0 <= ry < self.map_size:
            cv2.circle(img, (rx, ry), 5, (255, 255, 0), -1)
            hx = int(rx - np.sin(theta) * 20)
            hy = int(ry - np.cos(theta) * 20)
            cv2.line(img, (rx, ry), (hx, hy), (255, 255, 0), 2)

        cv2.imshow("RoboMaster Global Map", img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args if args is not None else sys.argv)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
