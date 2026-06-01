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

        # Global-map canvas parameters
        self.map_size = 600
        self.scale = 100.0
        self.cx = self.map_size // 2
        self.cy = self.map_size // 2

        # Latest global-map state
        self.boundary = np.empty((0, 2), dtype=np.float64)
        self.empty = np.empty((0, 2), dtype=np.float64)
        self.robot_pose = (0.0, 0.0, 0.0)
        self.candidate = None
        self.actionable = None

        # Camera bridge and floor projection helper
        self.bridge = CvBridge()
        self.T = FloorProjectiveTransform.from_points()

        # Subscribe to live camera for old pixel-level overlay
        self.camera_sub = self.create_subscription(Image, "camera/image_color", self.camera_callback, 10)

        # Subscribe to global map, parking hypotheses, and robot odometry
        self.map_sub = self.create_subscription(Float32MultiArray, "map/global_points", self.map_callback, 10)
        self.candidate_sub = self.create_subscription(Float32MultiArray, "parking/candidate", self.candidate_callback, 10)
        self.actionable_sub = self.create_subscription(Float32MultiArray, "parking/actionable", self.actionable_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "odom", self.odom_callback, 10)

        # Redraw the global-map view at 10 FPS
        self.timer = self.create_timer(1 / 10, self.draw)

    def camera_callback(self, msg: Image) -> None:
        """Show the old camera overlay: detected boundary/contact pixels and empty-space samples."""
        try:
            # Convert ROS image message to OpenCV RGB frame
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            # Run the same image-space boundary detector used by observation logic
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

            # OpenCV display expects BGR
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
        # Update latest accumulated global boundary and empty-space points
        self.boundary, self.empty = unpack_map_points(msg)

    def candidate_callback(self, msg: Float32MultiArray) -> None:
        # Store raw parking candidate if the slot detector found one
        parsed = unpack_slot(msg)
        self.candidate = parsed["corners"] if parsed["valid"] else None

    def actionable_callback(self, msg: Float32MultiArray) -> None:
        # Store only safe actionable slots
        parsed = unpack_slot(msg)
        self.actionable = parsed["corners"] if parsed["valid"] and parsed["actionable"] else None

    def odom_callback(self, msg: Odometry) -> None:
        q = (msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        _, _, yaw = euler_from_quaternion(q)
        self.robot_pose = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y), float(yaw))

    def world_to_px(self, point: np.ndarray) -> tuple[int, int]:
        """
        Convert world coordinates to image-pixel coordinates on the debug map

        :param point: World point [x, y]

        :return: Pixel coordinates (px, py) on the OpenCV map canvas
        """
        return int(self.cx - point[1] * self.scale), int(self.cy - point[0] * self.scale)

    def draw_points(self, img: np.ndarray, points: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
        """
        Draw a point cloud on the debug map

        :param img: OpenCV image canvas
        :param points: Point cloud of shape (N, 2)
        :param color: BGR drawing color
        :param radius: Circle radius in pixels

        :return: None
        """
        # Draw only points that fall inside the map canvas
        for p in points:
            px, py = self.world_to_px(p)
            if 0 <= px < self.map_size and 0 <= py < self.map_size:
                cv2.circle(img, (px, py), radius, color, -1)

    def draw_rect(self, img: np.ndarray, corners: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
        """
        Draw a rectangle on the debug map

        :param img: OpenCV image canvas
        :param corners: Rectangle corners of shape (4, 2), or None
        :param color: BGR drawing color
        :param thickness: Line thickness in pixels

        :return: None
        """
        # Nothing to draw if the slot is not currently available
        if corners is None:
            return

        # Convert rectangle corners from world coordinates to map pixels
        pts = np.array([self.world_to_px(p) for p in corners], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    def draw(self) -> None:
        # Create a black debug-map canvas
        img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)

        # Draw map layers: empty space, boundary, candidate slot, actionable slot
        self.draw_points(img, self.empty, (255, 100, 100), 1)
        self.draw_points(img, self.boundary, (0, 0, 255), 2)
        self.draw_rect(img, self.candidate, (0, 180, 180), 1)
        self.draw_rect(img, self.actionable, (0, 255, 255), 2)

        # Draw robot position and heading
        x, y, theta = self.robot_pose
        rx, ry = self.world_to_px(np.array([x, y], dtype=np.float64))
        if 0 <= rx < self.map_size and 0 <= ry < self.map_size:
            cv2.circle(img, (rx, ry), 5, (255, 255, 0), -1)

            # Heading line uses the same world-to-pixel axis convention as the map
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
