import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from transforms3d._gohlketransforms import euler_from_quaternion

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry

import sys

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


# Movement speeds
LINEAR_SPEED = 0.5  # [m/s]
ANGULAR_SPEED = 1.0  # [rad/s]

# Synchronizer latency window
SYNC_SLOP_S = 0.1



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
            inside_coords = [[p.x, p.y] for p in inside_empty.geoms if p.geom_type == 'Point' and not p.is_empty]
        elif inside_empty.geom_type == 'Point' and not inside_empty.is_empty:
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

        # Publisher for parking target coordinates (x, y) in the robot's local frame
        self.parking_target_pub = self.create_publisher(Float32MultiArray, 'parking_target', parking_target_qos)

        self.br = CvBridge()
        self.camera_info = None

        # CoppeliaSim camera extrinsic frame translations
        self.camera_offset_x = 0.02548
        self.camera_offset_y = -0.00047

        
        self.target_linear_x = 0.0
        self.target_linear_y = 0.0
        self.target_angular_z = 0.0
       

        # Parking mode state
        self.parking_mode = False
        self.parking_target = None
        self.parking_delegated = False
        self.robot_pose_2d = (0.0, 0.0, 0.0)

        # Instantiate Logic Modules
        self.grid_map = GridMap(resolution=0.02)
        self.parking_estimator = ParkingEstimator(safety_margin=0.16, min_area=0.05)
        self.visualization = VisualizationModule(map_size=600, scale=100.0)


        self.parking_corners = None
        self.prev_parking_corners = None  # Tracks changes for the change logger

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

    def compute_parking_velocity(self):
        """Computes velocity commands to move robot into parking spot."""
        if self.parking_corners is None or self.parking_target is None:
            return 0.0, 0.0, 0.0

        x, y, theta = self.robot_pose_2d
        target_x, target_y = self.parking_target

        # Distance to target
        dx = target_x - x
        dy = target_y - y
        dist = np.sqrt(dx**2 + dy**2)

        # Target angle
        target_angle = np.arctan2(dy, dx)
        angle_diff = target_angle - theta
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        # Proportional control
        lin_speed = 0.3 * np.clip(dist, 0, 1)
        ang_speed = 1.0 * np.clip(angle_diff, -1, 1)

        if dist < 0.1:
            return 0.0, 0.0, 0.0

        return lin_speed, 0.0, ang_speed

    def synced_callback(self, image_msg: Image, odom_msg: Odometry):
        try:
            x, y, theta = self.pose3d_to_2d(odom_msg.pose.pose)
            self.robot_pose_2d = (x, y, theta)
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

            # Update global map with new snapshot data
            self.grid_map.update_boundary(world_points)
            self.grid_map.update_empty_space(empty_world_points)
            

            b_pts, e_pts = self.grid_map.get_points()
            self.get_logger().info(f"Flashed snapshot! Boundary: {len(b_pts)}, Empty: {len(e_pts)}")

            # Continuously solve parking layout on current global map state
            boundary_map_pts, empty_map_pts = self.grid_map.get_points()
            self.parking_corners = self.parking_estimator.estimate(boundary_map_pts, empty_map_pts)

            # Check and log parking state differences
            self.check_and_log_parking_updates()

            # Delegate to control park4 node if parking spot is found and not yet delegated
            if self.parking_corners is not None and not self.parking_delegated:
                
        
                msg = Float32MultiArray()
                msg.data = self.parking_corners.flatten().tolist()  #flatten points : [x1, y1, x2, y2, x3, y3, x4, y4]
                self.parking_target_pub.publish(msg)
                
                self.parking_delegated = True
                self.get_logger().info("==================================================")
                self.get_logger().info("PARKING SPOT FOUND: Delegating control to park4 node")
                self.get_logger().info("==================================================")

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
        cmd_vel = Twist()

        if self.parking_delegated:
            pass  # Control is delegated to park4 node, so we do not publish cmd_vel here
        else:
            b_pts, _ = self.grid_map.get_points()
            obstacle_near = False

            if len(b_pts) > 0:
                x, y, theta = self.robot_pose_2d
                # Check if any boundary points are within 0.4 meters of the robot's current position
                dists = np.sqrt((b_pts[:, 0] - x)**2 + (b_pts[:, 1] - y)**2)
                min_dist = np.min(dists)
                
                if min_dist < 0.25:
                    obstacle_near = True
            
            if obstacle_near:
                print()
                print(f"Obstacle detected nearby (distance: {min_dist:.2f} m). Rotating to scan surroundings.")
                self.get_logger().warn(f"Obstacle detected nearby (distance: {min_dist:.2f} m). Rotating to scan surroundings.")
                print()
                # Rotate in place to scan surroundings
                self.target_linear_x = 0.0
                self.target_angular_z = 1.0 * ANGULAR_SPEED
            else:
                # Move forward
                self.target_linear_x = 0.3 * LINEAR_SPEED
                self.target_angular_z = 0.07 

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

    main()