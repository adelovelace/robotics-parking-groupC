import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from transforms3d._gohlketransforms import euler_from_quaternion

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry

import sys
import os
import termios
import numpy as np
import cv2
import warnings
from cv_bridge import CvBridge

import message_filters
from shapely.geometry import MultiPoint, Polygon
from sklearn.cluster import DBSCAN

# Suppress Shapely oriented_envelope / Runtime Warnings in the terminal
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")

# Absolute import fixed for ROS 2 package structure
from robomaster_example.vision import FloorProjectiveTransform

# Key bindings for movement
KEY_BINDINGS = {
    'w': (1.0, 0.0, 0.0),  # forward
    's': (-1.0, 0.0, 0.0),  # backward
    'a': (0.0, 1.0, 0.0),  # strafe left
    'd': (0.0, -1.0, 0.0),  # strafe right
    'q': (0.0, 0.0, 1.0),  # turn left
    'e': (0.0, 0.0, -1.0),  # turn right
}

# Movement speeds
LINEAR_SPEED = 0.5  # [m/s]
ANGULAR_SPEED = 1.0  # [rad/s]

# Synchronizer latency window
SYNC_SLOP_S = 0.1

USAGE_MSG = """
RoboMaster Modular Parking System
---------------------------------
   Q - turn left    W - forward     E - turn right
   A - strafe left  S - backward    D - strafe right

   [ F ] - Trigger Snapshot (Add currently visible area to global map)

Press any movement key to drive. Release to stop.
Press Ctrl+C to quit.
"""


class KeyboardReader:
    """Reads non-blocking keypresses from the terminal."""

    def __init__(self):
        self.fd = os.open('/dev/tty', os.O_RDONLY | os.O_NONBLOCK)
        self.old_settings = termios.tcgetattr(self.fd)
        new_settings = termios.tcgetattr(self.fd)
        new_settings[3] = new_settings[3] & ~(termios.ICANON | termios.ECHO)
        new_settings[6][termios.VMIN] = 0
        new_settings[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSADRAIN, new_settings)

    def read_key(self):
        try:
            ch = os.read(self.fd, 1)
            if ch:
                return ch.decode('utf-8', errors='ignore').lower()
        except (OSError, BlockingIOError):
            pass
        return None

    def close(self):
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
            os.close(self.fd)
        except Exception:
            pass


class GridMap:
    """Manages high-resolution global occupancy and empty space mapping."""

    def __init__(self, resolution: float = 0.02):
        self.resolution = resolution
        self.boundary_points = np.empty((0, 2), dtype=np.float64)
        self.empty_points = np.empty((0, 2), dtype=np.float64)

    def update_boundary(self, points: np.ndarray):
        """Sanitizes, snaps, and aggregates obstacle boundary points."""
        if len(points) == 0:
            return
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) > 0:
            snapped = np.round(points / self.resolution) * self.resolution
            self.boundary_points = np.vstack((self.boundary_points, snapped))
            self.boundary_points = np.unique(self.boundary_points, axis=0)

    def update_empty_space(self, points: np.ndarray):
        """Sanitizes, snaps, and aggregates traversable empty space points."""
        if len(points) == 0:
            return
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) > 0:
            snapped = np.round(points / self.resolution) * self.resolution
            self.empty_points = np.vstack((self.empty_points, snapped))
            self.empty_points = np.unique(self.empty_points, axis=0)

    def get_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns clean finite boundary and empty space maps."""
        b_pts = self.boundary_points[np.all(np.isfinite(self.boundary_points), axis=1)]
        e_pts = self.empty_points[np.all(np.isfinite(self.empty_points), axis=1)]
        return b_pts, e_pts


class ParkingEstimator:
    """Computes the optimal oriented rectangle for robot parking."""

    def __init__(self, safety_margin: float = 0.12, min_area: float = 0.05):
        self.safety_margin = safety_margin
        self.min_area = min_area

    def estimate(self, boundary_pts: np.ndarray, empty_pts: np.ndarray) -> np.ndarray | None:
        """Runs the spatial clustering & Shapely minimum bounding box estimation."""
        if len(boundary_pts) < 3 or len(empty_pts) < 4:
            return None

        # 1. Compute Convex Hull of the object's boundary
        obj_hull = MultiPoint(boundary_pts).convex_hull

        # 2. Extract empty space points residing inside the hull
        empty_multipoint = MultiPoint(empty_pts)
        inside_empty = obj_hull.intersection(empty_multipoint)

        inside_coords = []
        if hasattr(inside_empty, 'geoms'):
            inside_coords = [[p.x, p.y] for p in inside_empty.geoms if p.geom_type == 'Point']
        elif inside_empty.geom_type == 'Point':
            inside_coords = [[inside_empty.x, inside_empty.y]]

        if len(inside_coords) < 4:
            return None

        # 3. Apply safety distance margins (keep points away from walls)
        inside_coords_np = np.array(inside_coords)
        dists = np.linalg.norm(inside_coords_np[:, None, :] - boundary_pts[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        safe_empty = inside_coords_np[min_dists >= self.safety_margin]

        if len(safe_empty) < 4:
            return None

        # 4. Filter empty space into the largest contiguous cluster
        clustering = DBSCAN(eps=0.10, min_samples=4).fit(safe_empty)
        labels = clustering.labels_
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

        if len(unique_labels) == 0:
            return None

        biggest_label = unique_labels[np.argmax(counts)]
        best_cluster = safe_empty[labels == biggest_label]

        if len(best_cluster) < 5:
            return None

        # 5. Check dimensional span to guarantee 2D layout (avoids 1D collinearity warnings)
        dx = np.max(best_cluster[:, 0]) - np.min(best_cluster[:, 0])
        dy = np.max(best_cluster[:, 1]) - np.min(best_cluster[:, 1])
        if dx <= 0.15 or dy <= 0.15:
            return None

        # 6. Extract minimum rotated rectangle
        parking_hull = MultiPoint(best_cluster).convex_hull
        if parking_hull.geom_type == 'Polygon' and parking_hull.area > 1e-4:
            parking_rect = parking_hull.minimum_rotated_rectangle
            if parking_rect.geom_type == 'Polygon' and parking_rect.area >= self.min_area:
                return np.array(parking_rect.exterior.coords)[:4]

        return None


class VisualizationModule:
    """Handles CV-based visual overlays on camera frames and global 2D map canvas."""

    def __init__(self, map_size: int = 600, scale: float = 100.0):
        self.map_size = map_size
        self.scale = scale  # Pixels per meter
        self.cx = map_size // 2
        self.cy = map_size // 2

    def draw_camera_overlay(self, frame: np.ndarray, result) -> np.ndarray:
        """Draws projection indicators and contact points onto the live camera image."""
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw detected floor boundary contact line (Green)
        if len(result.image_pixels) > 0:
            for px in result.image_pixels:
                cv2.circle(bgr_frame, (int(px[0]), int(px[1])), 2, (0, 255, 0), -1)

        # Draw detected empty space samples (Blue-orange transition)
        if len(result.empty_image_pixels) > 0:
            for px in result.empty_image_pixels:
                cv2.circle(bgr_frame, (int(px[0]), int(px[1])), 1, (255, 100, 0), -1)

        return bgr_frame

    def draw_global_map(
            self, boundary_pts: np.ndarray, empty_pts: np.ndarray,
            parking_corners: np.ndarray | None, robot_pose2d: tuple[float, float, float]
            ) -> np.ndarray:
        """Renders the comprehensive 2D occupancy environment canvas."""
        map_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)

        # 1. Render empty floor workspace
        for point in empty_pts:
            px = int(self.cx - point[1] * self.scale)
            py = int(self.cy - point[0] * self.scale)
            if 0 <= px < self.map_size and 0 <= py < self.map_size:
                cv2.circle(map_img, (px, py), 1, (255, 100, 100), -1)

        # 2. Render obstacle wall boundaries
        for point in boundary_pts:
            px = int(self.cx - point[1] * self.scale)
            py = int(self.cy - point[0] * self.scale)
            if 0 <= px < self.map_size and 0 <= py < self.map_size:
                cv2.circle(map_img, (px, py), 2, (0, 0, 255), -1)

        # 3. Render parking box corners if calculated
        if parking_corners is not None:
            pts = []
            for point in parking_corners:
                px = int(self.cx - point[1] * self.scale)
                py = int(self.cy - point[0] * self.scale)
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(map_img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        # 4. Render current position and yaw of the robot
        rx = int(self.cx - robot_pose2d[1] * self.scale)
        ry = int(self.cy - robot_pose2d[0] * self.scale)
        theta = robot_pose2d[2]

        if 0 <= rx < self.map_size and 0 <= ry < self.map_size:
            cv2.circle(map_img, (rx, ry), 5, (255, 255, 0), -1)
            hx = int(rx - np.sin(theta) * 20)
            hy = int(ry - np.cos(theta) * 20)
            cv2.line(map_img, (rx, ry), (hx, hy), (255, 255, 0), 2)

        return map_img


class ControllerNode(Node):
    """ROS 2 wrapper managing robot topics, map visualizations, and callback runs."""

    def __init__(self):
        super().__init__('controller_node')

        self.odom_frame = 'odom'
        self.base_frame = 'base_link'
        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        parking_target_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.parking_target_publisher = self.create_publisher(
            Float32MultiArray,
            'parking_target',
            parking_target_qos,
        )

        self.br = CvBridge()
        self.camera_info = None

        # CoppeliaSim camera extrinsic frame translations
        self.camera_offset_x = 0.02548
        self.camera_offset_y = -0.00047

        # Keyboard Driving States
        self.target_linear_x = 0.0
        self.target_linear_y = 0.0
        self.target_angular_z = 0.0
        self.keyboard = KeyboardReader()

        # Instantiate Logic Modules
        self.grid_map = GridMap(resolution=0.02)
        self.parking_estimator = ParkingEstimator(safety_margin=0.12, min_area=0.05)
        self.visualization = VisualizationModule(map_size=600, scale=100.0)

        self.take_snapshot = False
        self.parking_corners = None
        self.prev_parking_corners = None  # Tracks changes for the change logger
        self.control_handed_off = False

        # Homography calibration
        self.T = FloorProjectiveTransform.from_points()

        # Synchronized Camera/Odom callback pipeline
        image_sub = message_filters.Subscriber(self, Image, 'camera/image_color')
        odom_sub = message_filters.Subscriber(self, Odometry, 'odom')

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, odom_sub],
            queue_size=20,
            slop=SYNC_SLOP_S,
        )
        self._sync.registerCallback(self.synced_callback)

        self.camera_info_sub = self.create_subscription(
            CameraInfo, 'camera/camera_info', self.camera_info_callback, 10
        )

    def start(self):
        self.timer = self.create_timer(1 / 60, self.update_callback)

    def stop(self):
        try:
            self.vel_publisher.publish(Twist())
        except Exception:
            pass
        self.keyboard.close()

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def pose3d_to_2d(self, pose3):
        quaternion = (
            pose3.orientation.w,
            pose3.orientation.x,
            pose3.orientation.y,
            pose3.orientation.z,
        )
        _, _, yaw = euler_from_quaternion(quaternion)
        return (pose3.position.x, pose3.position.y, yaw)

    def get_camera_pose_matrix(self, x, y, theta):
        c, s = np.cos(theta), np.sin(theta)
        world_T_base = np.array(
            [
                [c, -s, x],
                [s, c, y],
                [0, 0, 1],
            ]
        )
        base_T_camera = np.array(
            [
                [1, 0, self.camera_offset_x],
                [0, 1, self.camera_offset_y],
                [0, 0, 1],
            ]
        )
        return world_T_base @ base_T_camera

    def check_and_log_parking_updates(self):
        """Logs the parking spot coordinates when discovered, lost, or updated."""
        if self.parking_corners is not None and self.prev_parking_corners is None:
            # Parking spot discovered for the first time
            coords_str = ", ".join([f"[{pt[0]:.3f}, {pt[1]:.3f}]" for pt in self.parking_corners])
            self.get_logger().info(f"[PARKING STATUS] DISCOVERED! Corners: {coords_str}")
            self.publish_parking_target()

        elif self.parking_corners is None and self.prev_parking_corners is not None:
            # Parking spot lost
            self.get_logger().info("[PARKING STATUS] LOST! (Area requirements unmet or out of map bounds)")

        elif self.parking_corners is not None and self.prev_parking_corners is not None:
            # Check if coordinates updated significantly (threshold: 1.5 cm)
            if not np.allclose(self.parking_corners, self.prev_parking_corners, atol=0.015):
                coords_str = ", ".join([f"[{pt[0]:.3f}, {pt[1]:.3f}]" for pt in self.parking_corners])
                self.get_logger().info(f"[PARKING STATUS] UPDATED! New Corners: {coords_str}")

        # Cache current solution state for future evaluations
        if self.parking_corners is not None:
            self.prev_parking_corners = self.parking_corners.copy()
        else:
            self.prev_parking_corners = None

    def publish_parking_target(self):
        if self.control_handed_off or self.parking_corners is None or len(self.parking_corners) != 4:
            return

        msg = Float32MultiArray()
        msg.data = self.parking_corners.astype(np.float32).reshape(-1).tolist()
        self.parking_target_publisher.publish(msg)

        self.control_handed_off = True
        self.target_linear_x = 0.0
        self.target_linear_y = 0.0
        self.target_angular_z = 0.0
        self.vel_publisher.publish(Twist())
        self.get_logger().info('Published parking_target and released cmd_vel control to parking node.')

    def synced_callback(self, image_msg: Image, odom_msg: Odometry):
        try:
            x, y, theta = self.pose3d_to_2d(odom_msg.pose.pose)
            camera_pose_matrix = self.get_camera_pose_matrix(x, y, theta)

            # Vision boundary extraction
            current_frame = self.br.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
            world_points, empty_world_points, result = self.T.estimate_boundaries_world(
                current_frame,
                robot_pose=camera_pose_matrix,
                min_component_area=50,
                bottom_band_px=1,
                max_gap_below_px=8,
                dbscan_eps=0.05,
                dbscan_min_samples=4,
                min_cluster_points=10,
                empty_space_stride=3,
            )

            # Update Grid Mapping database on user keystroke snapshot
            if self.take_snapshot:
                self.grid_map.update_boundary(world_points)
                self.grid_map.update_empty_space(empty_world_points)
                self.take_snapshot = False

                b_pts, e_pts = self.grid_map.get_points()
                self.get_logger().info(f"Flashed snapshot! Boundary: {len(b_pts)}, Empty: {len(e_pts)}")

            # Continuously solve parking layout on current global map state
            boundary_map_pts, empty_map_pts = self.grid_map.get_points()
            self.parking_corners = self.parking_estimator.estimate(boundary_map_pts, empty_map_pts)

            # Check and log parking state differences
            self.check_and_log_parking_updates()

            # Delegate all display renderings to the VisualizationModule
            bgr_frame_with_overlays = self.visualization.draw_camera_overlay(current_frame, result)
            cv2.imshow("RoboMaster Camera - Boundary & Parking", bgr_frame_with_overlays)

            rendered_global_map = self.visualization.draw_global_map(
                boundary_map_pts, empty_map_pts, self.parking_corners, (x, y, theta)
            )
            cv2.imshow("RoboMaster Global Map", rendered_global_map)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Synced callback processing failed: {e}")

    def update_callback(self):
        if self.control_handed_off:
            return

        key = self.keyboard.read_key()

        if key is not None:
            if key == 'f':
                self.take_snapshot = True
                self.get_logger().info("Snapshot capture triggered!")
            elif key in KEY_BINDINGS:
                lin_x, lin_y, ang_z = KEY_BINDINGS[key]
                self.target_linear_x = lin_x * LINEAR_SPEED
                self.target_linear_y = lin_y * LINEAR_SPEED
                self.target_angular_z = ang_z * ANGULAR_SPEED
            else:
                self.target_linear_x = 0.0
                self.target_linear_y = 0.0
                self.target_angular_z = 0.0
        else:
            self.target_linear_x = 0.0
            self.target_linear_y = 0.0
            self.target_angular_z = 0.0

        cmd_vel = Twist()
        cmd_vel.linear.x = self.target_linear_x
        cmd_vel.linear.y = self.target_linear_y
        cmd_vel.angular.z = self.target_angular_z
        self.vel_publisher.publish(cmd_vel)


def main():
    rclpy.init(args=sys.argv)
    node = ControllerNode()

    try:
        node.start()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()


if __name__ == '__main__':
    print(USAGE_MSG)
    main()