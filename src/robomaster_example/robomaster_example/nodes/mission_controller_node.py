from __future__ import annotations

import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, String
from transforms3d._gohlketransforms import euler_from_quaternion

from robomaster_example.logic.slot_geometry import is_same_slot, min_distance, normalize_angle
from robomaster_example.logic.topic_codec import unpack_map_points, unpack_slot


class MissionControllerNode(Node):
    """High-level SEE / ACT_ARC / VALIDATE_SLOT / DELEGATE state machine.

    This node decides *when* to observe, explore, validate, and delegate. It no
    longer performs vision, map fusion, or parking-rectangle estimation.
    """

    DESIRED_CLEARANCE = 0.45
    MIN_CLEARANCE = 0.28
    ARC_STEP = 0.35
    ARC_LINEAR_SPEED = 0.08
    ARC_MAX_ANGULAR_SPEED = 0.35
    ARC_CLEARANCE_GAIN = 1.2
    ARC_ANGLE_TOLERANCE = np.deg2rad(12.0)
    ARC_DIRECTION = 1
    SCAN_ANGLE = np.deg2rad(35.0)

    SEE_SCAN_ANGLE = np.deg2rad(18.0)
    SEE_ALIGN_TOL = np.deg2rad(7.0)
    SEE_MAX_ANGULAR_SPEED = 0.30
    SEE_FULL_SCAN_EVERY = 3
    SEE_SPARSE_BOUNDARY_POINTS = 40

    VALIDATION_D_BACK = 0.50
    VALIDATION_SIDE_OFFSET = 0.12
    VALIDATION_POSE_TOL = 0.05
    VALIDATION_ANGLE_TOL = np.deg2rad(7.0)
    VALIDATION_MAX_LINEAR = 0.16
    VALIDATION_MAX_LATERAL = 0.12
    VALIDATION_MAX_ANGULAR = 0.30

    DELEGATION_REPUBLISHES = 15
    VALIDATION_RECT_DELAY_S = 0.25  # let map_node and slot_detector process the new observation

    def __init__(self):
        super().__init__("mission_controller_node")
        self.state = "SEE"
        self.robot_pose = None
        self.boundary = np.empty((0, 2), dtype=np.float64)
        self.empty = np.empty((0, 2), dtype=np.float64)
        self.latest_candidate = None
        self.latest_actionable = None
        self.waiting_observation: str | None = None

        self.see_step = "ALIGN"
        self.see_center_theta = None
        self.see_use_side_views = True
        self.go_around_iterations = 0

        self.arc_mode = None
        self.arc_start_xy = None
        self.arc_start_theta = None
        self.scan_target_theta = None
        self.frozen_boundary = np.empty((0, 2), dtype=np.float64)

        self.validation_step = None
        self.validation_slot = None
        self.validation_rectangles: list[np.ndarray | None] = []
        self.pending_validation_next_step = None
        self.pending_validation_label = None
        self.pending_validation_store_time = None
        self.candidate_seq = 0
        self.actionable_seq = 0
        self.validation_capture_candidate_seq = -1
        self.validation_capture_actionable_seq = -1
        self.validation_c = None
        self.validation_p = None
        self.validation_theta = None
        self.validation_right_pose = None
        self.validation_left_pose = None
        self.final_parking_target = None
        self.delegation_publish_count = 0
        # Once the final parking target has been delegated, this node must stop
        # publishing cmd_vel. Otherwise it races with controller_park4 and the
        # robot appears to twitch or stall while park4 is commanding motion.
        self.cmd_vel_released_to_park4 = False

        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.obs_request_pub = self.create_publisher(String, "mission/observation_request", 10)
        self.state_pub = self.create_publisher(String, "mission/state", 10)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.parking_target_pub = self.create_publisher(Float32MultiArray, "parking_target", qos)

        self.odom_sub = self.create_subscription(Odometry, "odom", self.odom_callback, 10)
        self.map_sub = self.create_subscription(Float32MultiArray, "map/global_points", self.map_callback, 10)
        self.candidate_sub = self.create_subscription(Float32MultiArray, "parking/candidate", self.candidate_callback, 10)
        self.actionable_sub = self.create_subscription(Float32MultiArray, "parking/actionable", self.actionable_callback, 10)
        self.obs_done_sub = self.create_subscription(String, "vision/observation_done", self.observation_done_callback, 10)
        self.timer = self.create_timer(1 / 60, self.update_callback)
        self.reset_see_cycle()

    def odom_callback(self, msg: Odometry) -> None:
        q = (msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        _, _, yaw = euler_from_quaternion(q)
        self.robot_pose = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y), float(yaw))

    def map_callback(self, msg: Float32MultiArray) -> None:
        self.boundary, self.empty = unpack_map_points(msg)

    def candidate_callback(self, msg: Float32MultiArray) -> None:
        parsed = unpack_slot(msg)
        self.latest_candidate = parsed["corners"] if parsed["valid"] else None
        self.candidate_seq += 1

    def actionable_callback(self, msg: Float32MultiArray) -> None:
        parsed = unpack_slot(msg)
        self.latest_actionable = parsed if parsed["valid"] and parsed["actionable"] else None
        self.actionable_seq += 1

    def observation_done_callback(self, msg: String) -> None:
        if self.waiting_observation == msg.data:
            self.get_logger().info(f"[MISSION] Observation finished: {msg.data}")
            self.waiting_observation = None
            if self.state == "VALIDATE_SLOT" and self.validation_step == "WAIT_VALIDATION_RECT":
                self.pending_validation_store_time = time.monotonic() + self.VALIDATION_RECT_DELAY_S
                self.pending_validation_label = msg.data
                self.get_logger().info(
                    f"[VALIDATION] Observation {msg.data} arrived. Waiting {self.VALIDATION_RECT_DELAY_S:.2f}s "
                    "for map/slot topics before storing the global-map rectangle."
                )

    def set_state(self, state: str, reason: str = "") -> None:
        if self.state != state:
            self.get_logger().info(f"[STATE] {self.state} -> {state}" + (f" | {reason}" if reason else ""))
        self.state = state
        msg = String(); msg.data = state; self.state_pub.publish(msg)

    def publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    def request_observation(self, name: str) -> None:
        if self.waiting_observation is not None:
            return
        msg = String(); msg.data = name
        self.obs_request_pub.publish(msg)
        self.waiting_observation = name
        self.get_logger().info(f"[MISSION] Requested observation: {name}")

    def request_validation_observation(self, name: str, next_step: str) -> None:
        """Request a validation capture and later store the slot estimate after topic propagation."""
        if self.waiting_observation is not None:
            return
        self.validation_capture_candidate_seq = self.candidate_seq
        self.validation_capture_actionable_seq = self.actionable_seq
        self.pending_validation_next_step = next_step
        self.pending_validation_label = name
        self.pending_validation_store_time = None
        self.request_observation(name)
        self.validation_step = "WAIT_VALIDATION_RECT"

    def store_validation_rectangle_if_ready(self) -> bool:
        if self.waiting_observation is not None:
            self.publish_stop()
            return False
        if self.pending_validation_store_time is None:
            self.publish_stop()
            return False
        if time.monotonic() < self.pending_validation_store_time:
            self.publish_stop()
            return False

        rect = None if self.latest_candidate is None else self.latest_candidate.copy()
        self.validation_rectangles.append(rect)
        self.get_logger().info(
            f"[VALIDATION] Stored delayed global-map rectangle after {self.pending_validation_label}: "
            f"valid={rect is not None}, count={len(self.validation_rectangles)}/3, "
            f"candidate_seq={self.validation_capture_candidate_seq}->{self.candidate_seq}, "
            f"actionable_seq={self.validation_capture_actionable_seq}->{self.actionable_seq}, "
            f"actionable_now={self.latest_actionable is not None}"
        )

        self.validation_step = self.pending_validation_next_step
        self.pending_validation_next_step = None
        self.pending_validation_label = None
        self.pending_validation_store_time = None
        return True

    def reset_see_cycle(self) -> None:
        self.see_step = "ALIGN"
        self.see_center_theta = None
        self.see_use_side_views = len(self.boundary) < self.SEE_SPARSE_BOUNDARY_POINTS or self.go_around_iterations % self.SEE_FULL_SCAN_EVERY == 0
        mode = "3-view" if self.see_use_side_views else "center-only"
        self.get_logger().info(f"[SEE] New cycle: mode={mode}, boundary={len(self.boundary)}, iter={self.go_around_iterations}")

    def update_callback(self) -> None:
        if self.state == "DONE":
            # In the normal successful path controller_park4 owns cmd_vel now.
            # Do not keep publishing zero velocity here.
            if not self.cmd_vel_released_to_park4:
                self.publish_stop()
            return
        if self.robot_pose is None:
            self.publish_stop()
            return
        if self.state == "SEE":
            self.update_see()
        elif self.state == "ACT_ARC":
            if self.act_arc_step():
                self.reset_see_cycle()
                self.set_state("SEE", "arc step finished")
        elif self.state == "VALIDATE_SLOT":
            self.update_validation()
        elif self.state == "DELEGATE":
            self.update_delegate()
        else:
            self.publish_stop()

    def nearest_boundary_theta(self) -> float | None:
        if len(self.boundary) == 0:
            return None
        x, y, _ = self.robot_pose
        robot_xy = np.array([x, y], dtype=np.float64)
        idx = int(np.argmin(np.linalg.norm(self.boundary - robot_xy.reshape(1, 2), axis=1)))
        nearest = self.boundary[idx]
        return float(np.arctan2(nearest[1] - y, nearest[0] - x))

    def update_see(self) -> None:
        if self.waiting_observation is not None:
            self.publish_stop()
            return
        x, y, theta = self.robot_pose
        cmd = Twist()
        if self.see_step == "ALIGN":
            desired = self.nearest_boundary_theta()
            if desired is None:
                self.see_center_theta = theta
                self.see_step = "CAPTURE_CENTER"
                return
            err = normalize_angle(desired - theta)
            if abs(err) > self.SEE_ALIGN_TOL:
                cmd.angular.z = float(np.clip(1.2 * err, -self.SEE_MAX_ANGULAR_SPEED, self.SEE_MAX_ANGULAR_SPEED))
                self.cmd_pub.publish(cmd)
                return
            self.publish_stop()
            self.see_center_theta = theta
            self.see_step = "CAPTURE_CENTER"
            return
        if self.see_step == "CAPTURE_CENTER":
            self.request_observation("see_center")
            self.see_step = "AFTER_CENTER"
            return
        if self.see_step == "AFTER_CENTER":
            if self.latest_actionable is not None:
                if self.setup_validation(self.latest_actionable["corners"]):
                    self.set_state("VALIDATE_SLOT", "actionable slot after SEE")
                    return
            if self.see_use_side_views:
                self.see_step = "ROTATE_LEFT"
            else:
                self.start_arc()
            return
        if self.see_step == "ROTATE_LEFT":
            target = normalize_angle(self.see_center_theta + self.SEE_SCAN_ANGLE)
            if self.rotate_to_theta(target, "SEE left"):
                self.see_step = "CAPTURE_LEFT"
            return
        if self.see_step == "CAPTURE_LEFT":
            self.request_observation("see_left")
            self.see_step = "AFTER_LEFT"
            return
        if self.see_step == "AFTER_LEFT":
            if self.latest_actionable is not None and self.setup_validation(self.latest_actionable["corners"]):
                self.set_state("VALIDATE_SLOT", "actionable slot after left SEE")
                return
            self.see_step = "ROTATE_RIGHT"
            return
        if self.see_step == "ROTATE_RIGHT":
            target = normalize_angle(self.see_center_theta - self.SEE_SCAN_ANGLE)
            if self.rotate_to_theta(target, "SEE right"):
                self.see_step = "CAPTURE_RIGHT"
            return
        if self.see_step == "CAPTURE_RIGHT":
            self.request_observation("see_right")
            self.see_step = "AFTER_RIGHT"
            return
        if self.see_step == "AFTER_RIGHT":
            if self.latest_actionable is not None and self.setup_validation(self.latest_actionable["corners"]):
                self.set_state("VALIDATE_SLOT", "actionable slot after right SEE")
                return
            self.start_arc()

    def rotate_to_theta(self, target_theta: float, label: str) -> bool:
        _, _, theta = self.robot_pose
        err = normalize_angle(target_theta - theta)
        if abs(err) <= self.SEE_ALIGN_TOL:
            self.publish_stop()
            return True
        cmd = Twist(); cmd.angular.z = float(np.clip(1.2 * err, -self.SEE_MAX_ANGULAR_SPEED, self.SEE_MAX_ANGULAR_SPEED))
        self.cmd_pub.publish(cmd)
        self.get_logger().info(f"[{label}] rotate err={np.rad2deg(err):.1f}deg", throttle_duration_sec=0.5)
        return False

    def start_arc(self) -> None:
        x, y, theta = self.robot_pose
        self.arc_start_xy = np.array([x, y], dtype=np.float64)
        self.arc_start_theta = theta
        self.frozen_boundary = self.boundary.copy()
        if len(self.frozen_boundary) < 5:
            self.arc_mode = "SCAN"
            self.scan_target_theta = normalize_angle(theta + self.SCAN_ANGLE)
        else:
            self.arc_mode = "BOUNDARY"
        self.go_around_iterations += 1
        self.set_state("ACT_ARC", f"arc_mode={self.arc_mode}")

    def act_arc_step(self) -> bool:
        x, y, theta = self.robot_pose
        robot_xy = np.array([x, y], dtype=np.float64)
        cmd = Twist()
        if self.arc_mode == "SCAN":
            err = normalize_angle(self.scan_target_theta - theta)
            if abs(err) <= self.VALIDATION_ANGLE_TOL:
                self.publish_stop(); return True
            cmd.angular.z = float(np.clip(1.2 * err, -0.30, 0.30)); self.cmd_pub.publish(cmd); return False
        if self.arc_start_xy is None or len(self.frozen_boundary) == 0:
            self.publish_stop(); return True
        travelled = float(np.linalg.norm(robot_xy - self.arc_start_xy))
        if travelled >= self.ARC_STEP:
            self.publish_stop(); return True
        dists = np.linalg.norm(self.frozen_boundary - robot_xy.reshape(1, 2), axis=1)
        idx = int(np.argmin(dists)); nearest = self.frozen_boundary[idx]; clearance = float(dists[idx])
        radial = robot_xy - nearest
        rn = np.linalg.norm(radial)
        if rn < 1e-9:
            self.publish_stop(); return True
        radial = radial / rn
        if clearance < self.MIN_CLEARANCE:
            desired_dir = radial
        else:
            tangent = np.array([-radial[1], radial[0]], dtype=np.float64) if self.ARC_DIRECTION > 0 else np.array([radial[1], -radial[0]], dtype=np.float64)
            desired_dir = tangent + self.ARC_CLEARANCE_GAIN * (self.DESIRED_CLEARANCE - clearance) * radial
            desired_dir = desired_dir / max(np.linalg.norm(desired_dir), 1e-9)
        desired_theta = float(np.arctan2(desired_dir[1], desired_dir[0]))
        err = normalize_angle(desired_theta - theta)
        cmd.linear.x = 0.0 if abs(err) > self.ARC_ANGLE_TOLERANCE else self.ARC_LINEAR_SPEED
        cmd.angular.z = float(np.clip(1.5 * err, -self.ARC_MAX_ANGULAR_SPEED, self.ARC_MAX_ANGULAR_SPEED))
        self.cmd_pub.publish(cmd)
        self.get_logger().info(
            f"[ARC] travelled={travelled:.2f}/{self.ARC_STEP:.2f}, clearance={clearance:.2f}, cmd=({cmd.linear.x:.2f},{cmd.angular.z:.2f})",
            throttle_duration_sec=0.5,
        )
        return False

    def setup_validation(self, ordered_slot: np.ndarray) -> bool:
        pts = np.asarray(ordered_slot, dtype=np.float64).reshape(4, 2)
        top_l, top_r, bottom_l, bottom_r = pts
        c = (top_l + top_r + bottom_l + bottom_r) / 4.0
        m = (bottom_l + bottom_r) / 2.0
        e = bottom_r - bottom_l
        n = np.array([-e[1], e[0]], dtype=np.float64)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            return False
        n = n / n_norm
        if np.dot(n, c - m) > 0:
            n = -n
        p = m + n * self.VALIDATION_D_BACK
        theta = float(np.arctan2(c[1] - p[1], c[0] - p[0]))
        left = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
        right = -left
        self.validation_slot = pts.copy()
        self.validation_rectangles = []
        self.pending_validation_next_step = None
        self.pending_validation_label = None
        self.pending_validation_store_time = None
        self.delegation_publish_count = 0
        self.cmd_vel_released_to_park4 = False
        self.validation_c = c
        self.validation_p = p
        self.validation_theta = theta
        self.validation_right_pose = (float((p + self.VALIDATION_SIDE_OFFSET * right)[0]), float((p + self.VALIDATION_SIDE_OFFSET * right)[1]), theta)
        self.validation_left_pose = (float((p + self.VALIDATION_SIDE_OFFSET * left)[0]), float((p + self.VALIDATION_SIDE_OFFSET * left)[1]), theta)
        self.validation_step = "MOVE_TO_P"
        self.get_logger().info(f"[VALIDATION] Start: p=({p[0]:.2f},{p[1]:.2f}), c=({c[0]:.2f},{c[1]:.2f}), theta={theta:.2f}")
        return True

    def update_validation(self) -> None:
        if self.validation_step == "WAIT_VALIDATION_RECT":
            self.store_validation_rectangle_if_ready()
            return
        if self.waiting_observation is not None:
            self.publish_stop(); return
        if self.validation_step == "MOVE_TO_P":
            if self.drive_to_pose((self.validation_p[0], self.validation_p[1], self.validation_theta), "MOVE_TO_P"):
                self.validation_step = "CAPTURE_CENTER"
            return
        if self.validation_step == "CAPTURE_CENTER":
            self.request_validation_observation("validation_center", "MOVE_RIGHT"); return
        if self.validation_step == "MOVE_RIGHT":
            if self.drive_to_pose(self.validation_right_pose, "MOVE_RIGHT"):
                self.validation_step = "CAPTURE_RIGHT"
            return
        if self.validation_step == "CAPTURE_RIGHT":
            self.request_validation_observation("validation_right", "MOVE_LEFT"); return
        if self.validation_step == "MOVE_LEFT":
            if self.drive_to_pose(self.validation_left_pose, "MOVE_LEFT"):
                self.validation_step = "CAPTURE_LEFT"
            return
        if self.validation_step == "CAPTURE_LEFT":
            self.request_validation_observation("validation_left", "ESTIMATE"); return
        if self.validation_step == "ESTIMATE":
            self.finish_validation()
            return
        if self.validation_step == "RETURN_TO_P":
            if self.drive_to_pose((self.validation_p[0], self.validation_p[1], self.validation_theta), "RETURN_TO_P"):
                self.set_state("DELEGATE", "validation accepted; robot returned to pre-parking point p")
            return

    def drive_to_pose(self, target_pose: tuple[float, float, float], label: str) -> bool:
        tx, ty, target_theta = target_pose
        x, y, theta = self.robot_pose
        dx, dy = tx - x, ty - y
        dist = float(np.hypot(dx, dy))
        angle_err = normalize_angle(target_theta - theta)
        forward_err = np.cos(theta) * dx + np.sin(theta) * dy
        left_err = -np.sin(theta) * dx + np.cos(theta) * dy
        if dist <= self.VALIDATION_POSE_TOL and abs(angle_err) <= self.VALIDATION_ANGLE_TOL:
            self.publish_stop(); return True
        cmd = Twist()
        if dist > self.VALIDATION_POSE_TOL:
            cmd.linear.x = float(np.clip(0.8 * forward_err, -self.VALIDATION_MAX_LINEAR, self.VALIDATION_MAX_LINEAR))
            cmd.linear.y = float(np.clip(0.8 * left_err, -self.VALIDATION_MAX_LATERAL, self.VALIDATION_MAX_LATERAL))
        if abs(angle_err) > self.VALIDATION_ANGLE_TOL:
            cmd.angular.z = float(np.clip(1.2 * angle_err, -self.VALIDATION_MAX_ANGULAR, self.VALIDATION_MAX_ANGULAR))
        self.cmd_pub.publish(cmd)
        self.get_logger().info(f"[VALIDATION] {label}: dist={dist:.2f}, angle={np.rad2deg(angle_err):.1f}", throttle_duration_sec=0.5)
        return False

    def finish_validation(self) -> None:
        stable_rect, pair = self.select_stable_validation_rectangle()
        if stable_rect is None:
            self.get_logger().warn("[VALIDATION] Rejected: no 2-of-3 stable global rectangles.")
            self.reset_see_cycle(); self.start_arc(); return
        # Prefer the current safe-entry slot if available and consistent. If the actionable
        # topic is temporarily invalid because of asynchronous map/slot propagation, fall
        # back to the originally selected ordered validation slot. That slot already passed
        # the safe-entry gate before validation, and the 2-of-3 global-map stability check
        # confirms that the same rectangle still exists after the validation observations.
        final_target = None
        if self.latest_actionable is not None:
            same, dbg = is_same_slot(self.latest_actionable["corners"], stable_rect)
            self.get_logger().info(
                f"[VALIDATION] latest_actionable vs stable: same={same}, "
                f"center={dbg['center_dist']:.2f}, size={dbg['size_diff']:.2f}, corner={dbg['corner_dist']:.2f}"
            )
            if same:
                final_target = self.latest_actionable["corners"].copy()

        if final_target is None and self.validation_slot is not None:
            same, dbg = is_same_slot(self.validation_slot, stable_rect)
            self.get_logger().info(
                f"[VALIDATION] initial ordered slot vs stable: same={same}, "
                f"center={dbg['center_dist']:.2f}, size={dbg['size_diff']:.2f}, corner={dbg['corner_dist']:.2f}"
            )
            if same:
                final_target = self.validation_slot.copy()

        if final_target is None:
            self.get_logger().warn("[VALIDATION] Rejected: stable rectangle exists, but no consistent ordered entry target is available.")
            self.reset_see_cycle(); self.start_arc(); return

        self.final_parking_target = final_target
        self.validation_step = "RETURN_TO_P"
        self.get_logger().info(
            f"[VALIDATION] Accepted: stable_pair={pair}. Returning to pre-parking point p before delegation: "
            f"p=({self.validation_p[0]:.3f},{self.validation_p[1]:.3f}), theta={self.validation_theta:.3f}"
        )

    def select_stable_validation_rectangle(self) -> tuple[np.ndarray | None, tuple[int, int] | None]:
        valid = [(i, r) for i, r in enumerate(self.validation_rectangles) if r is not None]
        if len(valid) < 2:
            return None, None
        best_pair = None; best_rect = None
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                idx_a, a = valid[i]; idx_b, b = valid[j]
                same, dbg = is_same_slot(b, a)
                self.get_logger().info(
                    f"[VALIDATION] pair {idx_a}-{idx_b}: same={same}, center={dbg['center_dist']:.2f}, "
                    f"size={dbg['size_diff']:.2f}, corner={dbg['corner_dist']:.2f}, theta_diag={np.rad2deg(dbg['theta_diff']):.1f}"
                )
                if same and (best_pair is None or idx_b > best_pair[1]):
                    best_pair = (idx_a, idx_b); best_rect = b.copy()
        return best_rect, best_pair

    def update_delegate(self) -> None:
        # Do not keep publishing zero cmd_vel here. After delegation, controller_park4
        # owns the final parking motion. Repeated stop commands from this node can
        # race with park4 and make the parking maneuver look stuck.
        if self.final_parking_target is None:
            self.publish_stop()
            self.set_state("DONE", "no final target")
            return
        if self.delegation_publish_count == 0:
            self.publish_stop()
        if self.delegation_publish_count < self.DELEGATION_REPUBLISHES:
            msg = Float32MultiArray(); msg.data = self.final_parking_target.reshape(-1).astype(float).tolist()
            self.parking_target_pub.publish(msg)
            self.delegation_publish_count += 1
            subs = self.parking_target_pub.get_subscription_count()
            self.get_logger().info(f"[DELEGATE] parking_target publish {self.delegation_publish_count}/{self.DELEGATION_REPUBLISHES}, subscribers={subs}")
            return
        self.cmd_vel_released_to_park4 = True
        self.set_state("DONE", "parking target delegated; park4 owns cmd_vel")


def main(args=None):
    rclpy.init(args=args if args is not None else sys.argv)
    node = MissionControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
