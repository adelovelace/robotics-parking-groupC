from __future__ import annotations

import sys
import time

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, String
from transforms3d._gohlketransforms import euler_from_quaternion

from robomaster_example.logic.topic_codec import pack_observation
from robomaster_example.vision import FloorProjectiveTransform

SYNC_SLOP_S = 0.1


class VisionObserverNode(Node):
    """Produce one static world-frame boundary observation on request."""

    # Variable to define what we think the robot is stable at
    ODOM_LINEAR_STABLE = 0.02
    ODOM_ANGULAR_STABLE = 0.03
    CAPTURE_SETTLE_S = 0.75 # How much to wait until we start capturing

    def __init__(self):
        super().__init__("vision_observer_node")
        self.bridge = CvBridge()
        self.camera_offset_x = 0.02548
        self.camera_offset_y = -0.00047
        self.T = FloorProjectiveTransform.from_points()

        self.pending_request: str | None = None
        self.capture_ready_time: float | None = None
        self.latest_linear_speed = 0.0
        self.latest_angular_speed = 0.0

        # Publisher to publish the boundary observation (boundary and empty points)
        self.observation_pub = self.create_publisher(Float32MultiArray, "vision/boundary_observation", 10)
        # Publisher to flag the end of an observation
        self.done_pub = self.create_publisher(String, "vision/observation_done", 10)
        # Subscriber to get a request for a new observation
        self.request_sub = self.create_subscription(String, "mission/observation_request", self.request_callback, 10)

        image_sub = message_filters.Subscriber(self, Image, "camera/image_color")
        odom_sub = message_filters.Subscriber(self, Odometry, "odom")
        self.sync = message_filters.ApproximateTimeSynchronizer([image_sub, odom_sub], queue_size=20, slop=SYNC_SLOP_S)
        self.sync.registerCallback(self.synced_callback)

    def request_callback(self, msg: String) -> None:
        self.pending_request = msg.data or "capture"
        self.capture_ready_time = None
        self.get_logger().info(f"[VISION] Observation requested: {self.pending_request}")

    def robot_is_stable(self) -> bool:
        return abs(self.latest_linear_speed) <= self.ODOM_LINEAR_STABLE and abs(self.latest_angular_speed) <= self.ODOM_ANGULAR_STABLE

    def pose3d_to_2d(self, pose3) -> tuple[float, float, float]:
        q = (pose3.orientation.w, pose3.orientation.x, pose3.orientation.y, pose3.orientation.z)
        _, _, yaw = euler_from_quaternion(q)
        return float(pose3.position.x), float(pose3.position.y), float(yaw)

    def get_camera_pose_matrix(self, x: float, y: float, theta: float) -> np.ndarray:
        """
        Calculates the camera pose matrix, combining the transformations from the world to the base
        and the base to the camera. The result represents the transformation from the world frame
        to the camera frame.

        :param x: X-coordinate of the robot's base in the world frame.
        :param y: Y-coordinate of the robot's base in the world frame.
        :param theta: Orientation (in radians) of the robot's base relative to the world frame.
        :return: A 3x3 homogeneous transformation matrix representing the camera pose in the world frame.
        :rtype: numpy.ndarray
        """
        c, s = np.cos(theta), np.sin(theta)
        world_T_base = np.array([[c, -s, x], [s, c, y], [0, 0, 1]], dtype=np.float64)
        base_T_camera = np.array([[1, 0, self.camera_offset_x], [0, 1, self.camera_offset_y], [0, 0, 1]], dtype=np.float64)
        return world_T_base @ base_T_camera

    def synced_callback(self, image_msg: Image, odom_msg: Odometry) -> None:
        self.latest_linear_speed = float(odom_msg.twist.twist.linear.x)
        self.latest_angular_speed = float(odom_msg.twist.twist.angular.z)

        # Do nothing if there is no request
        if self.pending_request is None:
            return

        # Wait until robot stabilizes
        if not self.robot_is_stable():
            self.get_logger().info(
                f"[VISION] Waiting for stable odom before {self.pending_request}: "
                f"v={self.latest_linear_speed:.3f}, w={self.latest_angular_speed:.3f}",
                throttle_duration_sec=0.5,
            )
            return

        # Wait until we are ready to capture
        if self.capture_ready_time is None:
            self.capture_ready_time = time.monotonic() + self.CAPTURE_SETTLE_S
            self.get_logger().info(f"[VISION] Settling for {self.CAPTURE_SETTLE_S:.2f}s before {self.pending_request}")
            return
        if time.monotonic() < self.capture_ready_time:
            return

        request_name = self.pending_request
        self.pending_request = None
        self.capture_ready_time = None

        try:
            x, y, theta = self.pose3d_to_2d(odom_msg.pose.pose)
            camera_pose = self.get_camera_pose_matrix(x, y, theta)
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
            boundary, empty, _ = self.T.estimate_boundaries_world(
                frame,
                robot_pose=camera_pose,
                min_component_area=50,
                bottom_band_px=1,
                max_gap_below_px=8,
                dbscan_eps=0.05,
                dbscan_min_samples=4,
                min_cluster_points=10,
                empty_space_stride=3,
            )
            self.observation_pub.publish(pack_observation((x, y, theta), boundary, empty))
            done = String()
            done.data = request_name
            self.done_pub.publish(done)
            self.get_logger().info(
                f"[VISION] Captured {request_name}: boundary={len(boundary)}, empty={len(empty)}, "
                f"pose=({x:.2f},{y:.2f},{theta:.2f})"
            )
        except Exception as exc:
            self.get_logger().error(f"[VISION] Capture failed for {request_name}: {exc}")


def main(args=None):
    rclpy.init(args=args if args is not None else sys.argv)
    node = VisionObserverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
