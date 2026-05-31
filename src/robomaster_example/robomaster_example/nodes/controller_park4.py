import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray
from transforms3d._gohlketransforms import euler_from_quaternion


class ControllerNode(Node):
    """Final parking controller.

    Input rectangle is expected in ordered entry format:
        [top_l, top_r, bottom_l, bottom_r]

    The bottom_l -> bottom_r side is the selected free entrance side. This node
    must not re-select the closest edge, because the closest edge can be a wall.
    """

    D_BACK = 0.50
    POSE_TOL = 0.07
    CENTER_TOL = 0.06
    ANGLE_TOL = np.deg2rad(6.0)
    MAX_LINEAR = 0.16
    MAX_LATERAL = 0.12
    MAX_ANGULAR = 0.35
    KP_XY = 0.9
    KP_THETA = 1.4

    def __init__(self):
        super().__init__('controller_park4')

        self.odom_pose = None
        self.odom_velocity = None
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.state = "IDLE"
        self.parking_slot_corners = None
        self.c = None
        self.m = None
        self.e = None
        self.n = None
        self.p = None
        self.parking_theta = None

        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        parking_target_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.odom_subscriber = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.parking_target_subscriber = self.create_subscription(
            Float32MultiArray,
            'parking_target',
            self.parking_target_callback,
            parking_target_qos,
        )

    def start(self):
        self.timer = self.create_timer(1 / 60, self.update_callback)

    def stop(self):
        self.vel_publisher.publish(Twist())

    def odom_callback(self, msg):
        self.odom_pose = msg.pose.pose
        self.odom_velocity = msg.twist.twist
        self.x, self.y, self.theta = self.pose3d_to_2d(self.odom_pose)

        self.get_logger().info(
            f"odometry: pose x={self.x:.2f}, y={self.y:.2f}, theta={self.theta:.2f}",
            throttle_duration_sec=0.5,
        )

    def pose3d_to_2d(self, pose3):
        quaternion = (
            pose3.orientation.w,
            pose3.orientation.x,
            pose3.orientation.y,
            pose3.orientation.z,
        )
        _, _, yaw = euler_from_quaternion(quaternion)
        return pose3.position.x, pose3.position.y, yaw

    @staticmethod
    def normalize_angle(angle):
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    @staticmethod
    def clamp(value, lo, hi):
        return float(np.clip(value, lo, hi))

    def parking_target_callback(self, msg):
        if len(msg.data) != 8:
            self.get_logger().warn(f'Ignoring parking_target with invalid size={len(msg.data)}.')
            return

        self.parking_slot_corners = np.array(msg.data, dtype=np.float64).reshape(4, 2)
        self.log_received_rectangle(self.parking_slot_corners)

        # Repeated latched/re-published targets are expected. Do not reset while
        # already executing the same maneuver, but do allow restart from IDLE.
        if self.state == "IDLE":
            self.state = "START"
            self.get_logger().info('[PARK4] Received parking_target. Switching IDLE -> START.')
        else:
            self.get_logger().info(f'[PARK4] Received parking_target while state={self.state}; keeping current state.')

    def log_received_rectangle(self, points):
        top_l, top_r, bottom_l, bottom_r = points
        center = np.mean(points, axis=0)
        self.get_logger().info(
            f"[PARK4] ordered target: top_l=({top_l[0]:.3f},{top_l[1]:.3f}), "
            f"top_r=({top_r[0]:.3f},{top_r[1]:.3f}), "
            f"bottom_l=({bottom_l[0]:.3f},{bottom_l[1]:.3f}), "
            f"bottom_r=({bottom_r[0]:.3f},{bottom_r[1]:.3f}), "
            f"center=({center[0]:.3f},{center[1]:.3f})"
        )

    def select_parking_slot_corners(self):
        if self.parking_slot_corners is None or self.odom_pose is None:
            return None

        points = self.parking_slot_corners
        top_l = points[0]
        top_r = points[1]
        bottom_l = points[2]
        bottom_r = points[3]

        self.get_logger().info(
            f"[PARK4] Using ordered entry edge bottom_l->bottom_r: "
            f"({bottom_l[0]:.2f},{bottom_l[1]:.2f}) -> ({bottom_r[0]:.2f},{bottom_r[1]:.2f})",
            throttle_duration_sec=1.0,
        )
        return top_l, top_r, bottom_l, bottom_r

    def compute_handy_points(self, top_l, top_r, bottom_l, bottom_r):
        c = (top_l + top_r + bottom_l + bottom_r) / 4.0
        m = (bottom_l + bottom_r) / 2.0
        e = bottom_r - bottom_l
        edge_norm = np.linalg.norm(e)
        if edge_norm < 1e-9:
            self.get_logger().warn('[PARK4] Cannot compute geometry: entry edge is degenerate.')
            return False

        n = np.array([-e[1], e[0]], dtype=np.float64) / edge_norm
        if np.dot(n, c - m) > 0:
            n = -n  # n must point outside the slot, toward the pre-parking point.

        p = m + n * self.D_BACK
        parking_theta = np.arctan2(c[1] - p[1], c[0] - p[0])

        self.c = c
        self.m = m
        self.e = e
        self.n = n
        self.p = p
        self.parking_theta = float(parking_theta)

        self.get_logger().info(
            f"[PARK4] geometry: c=({c[0]:.3f},{c[1]:.3f}), "
            f"m=({m[0]:.3f},{m[1]:.3f}), n=({n[0]:.3f},{n[1]:.3f}), "
            f"p=({p[0]:.3f},{p[1]:.3f}), parking_theta={parking_theta:.3f}"
        )
        return True

    def evaluate_space(self, top_l, top_r, bottom_l, bottom_r):
        robot_length = 0.4005842
        robot_width = 0.2424
        security_margin_length = -0.15
        security_margin_width = -0.10
        len_tolerance = 0.1
        angle_tolerance = 0.1

        width_bottom = np.linalg.norm(bottom_r - bottom_l)
        width_top = np.linalg.norm(top_r - top_l)
        len_left = np.linalg.norm(top_l - bottom_l)
        len_right = np.linalg.norm(top_r - bottom_r)

        self.get_logger().info(
            f"[PARK4] space dimensions: width_bottom={width_bottom:.3f}, width_top={width_top:.3f}, "
            f"len_left={len_left:.3f}, len_right={len_right:.3f}"
        )

        if min(width_bottom, width_top, len_left, len_right) < 1e-9:
            self.get_logger().warn('[PARK4] Rejected parking slot: degenerate side length.')
            return False

        if ((width_bottom - robot_width) < security_margin_width) or \
                ((width_top - robot_width) < security_margin_width) or \
                ((len_left - robot_length) < security_margin_length) or \
                ((len_right - robot_length) < security_margin_length):
            self.get_logger().warn('[PARK4] Rejected parking slot: dimensions too small.')
            return False

        if abs(len_left - len_right) > len_tolerance:
            self.get_logger().warn('[PARK4] Rejected parking slot: left/right lengths differ too much.')
            return False
        if abs(width_bottom - width_top) > len_tolerance:
            self.get_logger().warn('[PARK4] Rejected parking slot: top/bottom widths differ too much.')
            return False

        bottom_edge = bottom_r - bottom_l
        top_edge = top_r - top_l
        left_edge = top_l - bottom_l
        right_edge = top_r - bottom_r

        bottom_edge_norm = bottom_edge / width_bottom
        top_edge_norm = top_edge / width_top
        left_edge_norm = left_edge / len_left
        right_edge_norm = right_edge / len_right

        dots = [
            abs(np.dot(bottom_edge_norm, left_edge_norm)),
            abs(np.dot(left_edge_norm, top_edge_norm)),
            abs(np.dot(top_edge_norm, right_edge_norm)),
            abs(np.dot(right_edge_norm, bottom_edge_norm)),
        ]
        if any(v > angle_tolerance for v in dots):
            self.get_logger().warn(f'[PARK4] Rejected parking slot: non-rectangular angles, dot errors={np.round(dots, 3)}.')
            return False

        return True

    def drive_to_pose_holonomic(self, target_xy, target_theta, label, max_linear=None, max_lateral=None):
        if max_linear is None:
            max_linear = self.MAX_LINEAR
        if max_lateral is None:
            max_lateral = self.MAX_LATERAL

        robot_xy = np.array([self.x, self.y], dtype=np.float64)
        target_xy = np.asarray(target_xy, dtype=np.float64)
        delta = target_xy - robot_xy
        dist = float(np.linalg.norm(delta))
        angle_error = self.normalize_angle(target_theta - self.theta)

        c = np.cos(self.theta)
        s = np.sin(self.theta)
        forward_error = c * delta[0] + s * delta[1]
        left_error = -s * delta[0] + c * delta[1]

        cmd = Twist()
        if dist > self.POSE_TOL:
            cmd.linear.x = self.clamp(self.KP_XY * forward_error, -max_linear, max_linear)
            cmd.linear.y = self.clamp(self.KP_XY * left_error, -max_lateral, max_lateral)
        if abs(angle_error) > self.ANGLE_TOL:
            cmd.angular.z = self.clamp(self.KP_THETA * angle_error, -self.MAX_ANGULAR, self.MAX_ANGULAR)

        self.vel_publisher.publish(cmd)
        self.get_logger().info(
            f"[PARK4] {label}: target=({target_xy[0]:.3f},{target_xy[1]:.3f},{target_theta:.3f}), "
            f"pose=({self.x:.3f},{self.y:.3f},{self.theta:.3f}), dist={dist:.3f}, "
            f"angle_err={np.rad2deg(angle_error):.1f} deg, "
            f"body_err=(x={forward_error:.3f}, y={left_error:.3f}), "
            f"cmd=(x={cmd.linear.x:.3f}, y={cmd.linear.y:.3f}, w={cmd.angular.z:.3f})",
            throttle_duration_sec=0.4,
        )
        return dist, abs(angle_error)

    def update_callback(self):
        if self.state == "IDLE":
            return

        if self.state == "START":
            if self.parking_slot_corners is None:
                self.state = "IDLE"
                return
            if self.odom_pose is None:
                self.stop()
                return

            selected = self.select_parking_slot_corners()
            if selected is None:
                return
            top_l, top_r, bottom_l, bottom_r = selected

            if not self.evaluate_space(top_l, top_r, bottom_l, bottom_r):
                self.stop()
                return
            if not self.compute_handy_points(top_l, top_r, bottom_l, bottom_r):
                self.stop()
                return

            # Mission validation already moved the robot back to the pre-parking
            # point p. Do not restart the whole parking pipeline from MOVE_TO_P: at
            # very small p residuals the direction-to-p vector is unstable and can
            # make the robot spin. Park4 starts with final heading alignment.
            robot_xy = np.array([self.x, self.y], dtype=np.float64)
            dist_to_p = float(np.linalg.norm(self.p - robot_xy))
            self.get_logger().info(
                f'[PARK4] START: mission should already be near p; dist_to_p={dist_to_p:.3f}. '
                'Skipping MOVE_TO_P and starting ROTATE_TO_FACE_PARKING.'
            )
            self.state = "ROTATE_TO_FACE_PARKING"
            return

        if self.state == "MOVE_TO_P":
            # Kept only as a fallback/debug state. Normal refactored flow starts at
            # ROTATE_TO_FACE_PARKING because validation already returns to p.
            dist, angle = self.drive_to_pose_holonomic(self.p, self.parking_theta, "MOVE_TO_P")
            if dist <= self.POSE_TOL and angle <= self.ANGLE_TOL:
                self.stop()
                self.state = "ROTATE_TO_FACE_PARKING"
                self.get_logger().info('[PARK4] MOVE_TO_P complete -> ROTATE_TO_FACE_PARKING')
            return

        if self.state == "ROTATE_TO_FACE_PARKING":
            angle_error = self.normalize_angle(self.parking_theta - self.theta)
            cmd = Twist()
            if abs(angle_error) <= self.ANGLE_TOL:
                self.stop()
                self.state = "FORWARD"
                self.get_logger().info('[PARK4] ROTATE_TO_FACE_PARKING complete -> FORWARD')
                return
            cmd.angular.z = self.clamp(self.KP_THETA * angle_error, -self.MAX_ANGULAR, self.MAX_ANGULAR)
            self.vel_publisher.publish(cmd)
            self.get_logger().info(
                f"[PARK4] ROTATE_TO_FACE_PARKING: theta={self.theta:.3f}, "
                f"target={self.parking_theta:.3f}, angle_err={np.rad2deg(angle_error):.1f} deg, "
                f"cmd_w={cmd.angular.z:.3f}",
                throttle_duration_sec=0.4,
            )
            return

        if self.state == "FORWARD":
            # Drive to the center while keeping the entry orientation. This removes
            # the old premature "dist < 0.25 => PARKED" transition and corrects
            # small lateral errors instead of blindly driving straight.
            robot_xy = np.array([self.x, self.y], dtype=np.float64)
            dist_to_center = float(np.linalg.norm(self.c - robot_xy))
            old_pose_tol = self.POSE_TOL
            self.POSE_TOL = self.CENTER_TOL
            dist, angle = self.drive_to_pose_holonomic(
                self.c,
                self.parking_theta,
                "FORWARD_TO_CENTER",
                max_linear=0.12,
                max_lateral=0.08,
            )
            self.POSE_TOL = old_pose_tol
            if dist_to_center <= self.CENTER_TOL:
                self.stop()
                self.state = "PARKED"
                self.get_logger().info('[PARK4] FORWARD complete -> PARKED')
            return

        if self.state == "PARKED":
            self.stop()
            self.get_logger().info('[PARK4] PARKED: holding zero velocity.', throttle_duration_sec=1.0)
            return


def main():
    rclpy.init(args=sys.argv)
    node = ControllerNode()
    node.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()


if __name__ == '__main__':
    main()
