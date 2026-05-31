import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray
from transforms3d._gohlketransforms import euler_from_quaternion

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry

import sys

import numpy as np

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_park4')
        
        # Create attributes to store odometry pose and velocity
        self.odom_pose = None
        self.odom_velocity = None
        self.state = "IDLE"
        self.c = None
        self.m = None
        self.e = None
        self.n = None
        self.p = None
        self.parking_slot_corners = None

        # Create a publisher for the topic 'cmd_vel'
        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        parking_target_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Create a subscriber to the topic 'odom', which will call 
        # self.odom_callback every time a message is received
        self.odom_subscriber = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.parking_target_subscriber = self.create_subscription(
            Float32MultiArray,
            'parking_target',
            self.parking_target_callback,
            parking_target_qos,
        )
        
        # NOTE: we're using relative names to specify the topics (i.e., without a 
        # leading /). ROS resolves relative names by concatenating them with the 
        # namespace in which this node has been started, thus allowing us to 
        # specify which RoboMaster should be controlled.
        
    def start(self):
        # Create and immediately start a timer that will regularly publish commands
        self.timer = self.create_timer(1/60, self.update_callback)
    
    def stop(self):
        # Set all velocities to zero
        cmd_vel = Twist()
        self.vel_publisher.publish(cmd_vel)
    
    def odom_callback(self, msg):
        self.odom_pose = msg.pose.pose
        self.odom_velocity = msg.twist.twist

        pose2d = self.pose3d_to_2d(self.odom_pose)
        self.x = pose2d[0]
        self.y = pose2d[1]
        self.theta = pose2d[2]

        self.get_logger().info(
            "odometry: received pose (x: {:.2f}, y: {:.2f}, theta: {:.2f})".format(*pose2d),
            throttle_duration_sec=0.5
        )
    
    def pose3d_to_2d(self, pose3):
        quaternion = (
            pose3.orientation.w,
            pose3.orientation.x,
            pose3.orientation.y,
            pose3.orientation.z
        )

        roll, pitch, yaw = euler_from_quaternion(quaternion)

        pose2 = (
            pose3.position.x,
            pose3.position.y,
            yaw
        )

        return pose2

    def parking_target_callback(self, msg):
        if len(msg.data) != 8:
            self.get_logger().warn('Ignoring parking_target with invalid size.')
            return

        self.parking_slot_corners = np.array(msg.data, dtype=np.float64).reshape(4, 2)
        if self.state == "IDLE":
            self.state = "START"
        self.get_logger().info('Received parking_target. Parking controller switching to START.')

    def select_parking_slot_corners(self):
        if self.parking_slot_corners is None or self.odom_pose is None:
            return None

        robot_position = np.array([self.x, self.y])
        points = self.parking_slot_corners

        closest_edge_index = 0
        closest_edge_distance = float('inf')
        for idx in range(4):
            edge_midpoint = (points[idx] + points[(idx + 1) % 4]) / 2.0
            edge_distance = np.linalg.norm(edge_midpoint - robot_position)
            if edge_distance < closest_edge_distance:
                closest_edge_distance = edge_distance
                closest_edge_index = idx

        bottom_l = points[closest_edge_index]
        bottom_r = points[(closest_edge_index + 1) % 4]
        top_r = points[(closest_edge_index + 2) % 4]
        top_l = points[(closest_edge_index + 3) % 4]
        return top_l, top_r, bottom_l, bottom_r
    
    def compute_handy_points(self, d_back, top_l,top_r,bottom_l,bottom_r):
        # 0. Compute handy points w.r.t world
        c = (top_l + top_r + bottom_l + bottom_r) / 4 # center of the parking slot
        m = (bottom_l + bottom_r) / 2 # center of bottom line
        e = bottom_r - bottom_l # edge of bottom
        n = np.array([-e[1],e[0]]) # line perpendicular to e
        if np.linalg.norm(n) > 1e-9: #not 0
            n = n / np.linalg.norm(n)
        if np.dot(n, c-m) > 0: # same direction
            n = -n # n should face towards outside the parking slot
        p = m + n * d_back

        self.c = c
        self.m = m
        self.e = e
        self.n = n
        self.p = p

        return
    
    def normalize_angle(self, angle): #[-pi, +pi1]
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return angle

    
    def evaluate_space(self, top_l,top_r,bottom_l,bottom_r):
        # robomaster dimensions
        length = 0.4005842
        width = 0.2424
        security_margin_length = -0.15
        security_margen_width = -0.10
        len_tolerance = 0.1
        angle_tolerance = 0.1

        # 1. check enough width and length
        width_bottom = np.linalg.norm(bottom_r - bottom_l)
        width_top = np.linalg.norm(top_r - top_l)
        len_left = np.linalg.norm(top_l - bottom_l)
        len_right = np.linalg.norm(top_r - bottom_r)
        if ((width_bottom - width) < security_margen_width) or \
            ((width_top - width) < security_margen_width) or \
            ((len_left - length) < security_margin_length) or \
            ((len_right - length) < security_margin_length):
            return False
        

        # 2. check that sides are approximatelly equal
        if abs(len_left - len_right) > len_tolerance:
            return False
        if abs(width_bottom - width_top) > len_tolerance:
            return False

        # 3. check angle are close to 90 degrees
        bottom_edge = bottom_r - bottom_l
        top_edge = top_r - top_l
        left_edge = top_l - bottom_l
        right_edge = top_r - bottom_r

        bottom_edge_norm = bottom_edge / width_bottom
        top_edge_norm = top_edge / width_top
        left_edge_norm = left_edge / len_left
        right_edge_norm = right_edge / len_right

        if abs(np.dot(bottom_edge_norm, left_edge_norm)) > angle_tolerance: # no perpendicular
            return False
        if abs(np.dot(left_edge_norm, top_edge_norm)) > angle_tolerance:
            return False
        if abs(np.dot(top_edge_norm, right_edge_norm)) > angle_tolerance:
            return False
        if abs(np.dot(right_edge_norm, bottom_edge_norm)) > angle_tolerance:
            return False
        
        return True

        
    def update_callback(self):

        if self.state == "IDLE":
            return

        cmd_vel = Twist()
        d_back = 0.5 # how far from the parking slot is p
        distance_tolerance = 0.05
        angle_tolerance = np.pi/32

        self.get_logger().info(
            f"state={self.state}, p={self.p}, center={self.c}",
            throttle_duration_sec=0.5
        )

        if self.state == "START":
            if self.parking_slot_corners is None:
                self.state = "IDLE"
                return
            if self.odom_pose is None:
                cmd_vel.linear.x  = 0.0 # [m/s]
                cmd_vel.angular.z = 0.0 # [rad/s]
                self.vel_publisher.publish(cmd_vel)
                return

            selected_corners = self.select_parking_slot_corners()

            if selected_corners is None:
                return
            
            top_l, top_r, bottom_l, bottom_r = selected_corners

            # evaluate dimentions of parking spot
            if self.evaluate_space(top_l,top_r,bottom_l,bottom_r) == False:
                cmd_vel.linear.x  = 0.0 # [m/s]
                cmd_vel.angular.z = 0.0 # [rad/s]
                self.vel_publisher.publish(cmd_vel)
                return
            # compute relevan points
            self.compute_handy_points(d_back, top_l,top_r,bottom_l,bottom_r)
            self.state = "MOVE_TO_P"
        elif self.state == "MOVE_TO_P":
            dx= self.p[0] - self.x
            dy = self.p[1] - self.y
            distance_error = np.sqrt(dx**2 + dy**2)
            target_angle = np.arctan2(dy, dx)
            angle_error = self.normalize_angle(target_angle - self.theta)
            # check distance first
            if abs(distance_error) < distance_tolerance:
                cmd_vel.angular.z = 0.0
                cmd_vel.linear.x  = 0.0
                self.vel_publisher.publish(cmd_vel)
                self.state = "ROTATE_TO_FACE_PARKING"
                return
            # face towards p
            if abs(angle_error) > angle_tolerance:
                cmd_vel.linear.x  = 0.0
                if angle_error > angle_tolerance:
                    cmd_vel.angular.z = 0.1
                else:
                    cmd_vel.angular.z = -0.1
            # move towards p
            elif abs(distance_error) > distance_tolerance:
                    cmd_vel.angular.z = 0.0
                    cmd_vel.linear.x  = 0.2
            else:
                cmd_vel.angular.z = 0.0
                cmd_vel.linear.x  = 0.0
                self.state = "ROTATE_TO_FACE_PARKING"
        elif self.state == "ROTATE_TO_FACE_PARKING":
            desired_dir = self.c - np.array([self.x, self.y])
            desired_theta = np.arctan2(desired_dir[1], desired_dir[0])
            angle_error = self.normalize_angle(desired_theta - self.theta)
            if abs(angle_error) > angle_tolerance:
                cmd_vel.linear.x  = 0.0
                if angle_error > angle_tolerance:
                    cmd_vel.angular.z = 0.1
                else:
                    cmd_vel.angular.z = -0.1
            else: #stop
                cmd_vel.angular.z = 0.0
                cmd_vel.linear.x  = 0.0
                self.state = "FORWARD"
        elif self.state == "FORWARD": 
                # move forward until c (center of parking spot)

                target_x = self.m[0] - (self.n[0] * 0.15)
                target_y = self.m[1] - (self.n[1] * 0.15)

                dx = target_x - self.x
                dy = target_y - self.y
                
                # dx= self.c[0] - self.x
                # dy = self.c[1] - self.y
                distance_error = np.sqrt(dx**2 + dy**2)
                print("Here")
                print(f"c: {self.c}, robot: ({self.x}, {self.y}), distance_error: {distance_error}")    

                # check distance first
                if abs(distance_error) < distance_tolerance:
                    cmd_vel.angular.z = 0.0
                    cmd_vel.linear.x  = 0.0
                    self.vel_publisher.publish(cmd_vel)
                    self.state = "PARKED"
                    return
                
                # keep checking angle
                desired_dir = self.c - np.array([self.x, self.y])
                desired_theta = np.arctan2(desired_dir[1], desired_dir[0])
                angle_error = self.normalize_angle(desired_theta - self.theta)
                
                
                if abs(distance_error) < 0.25:
                    cmd_vel.angular.z = 0.0
                    cmd_vel.linear.x  = 0.2
                    self.state = "PARKED"
                    self.vel_publisher.publish(cmd_vel)
                    return
                else:
                    # check angle before going forward
                    if abs(angle_error) > (angle_tolerance*2):
                        cmd_vel.linear.x  = 0.0
                        if angle_error > angle_tolerance:
                            cmd_vel.angular.z = 0.1
                        else:
                            cmd_vel.angular.z = -0.1
                    elif abs(distance_error) > distance_tolerance:
                        cmd_vel.linear.x  = 0.2
                        cmd_vel.angular.z = 0.0
                    else: #stop
                        cmd_vel.angular.z = 0.0
                        cmd_vel.linear.x  = 0.0
                        self.state = "PARKED"
        elif self.state == "PARKED":
             #stop
            cmd_vel.linear.x  = 0.0 # [m/s]
            cmd_vel.angular.z = 0.0 # [rad/s]
    
        # Publish the command
        self.vel_publisher.publish(cmd_vel)


def main():
    # Initialize the ROS client library
    rclpy.init(args=sys.argv)
    
    # Create an instance of your node class
    node = ControllerNode()
    node.start()
    
    # Keep processings events until someone manually shuts down the node
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    # Ensure the RoboMaster is stopped before exiting
    node.stop()


if __name__ == '__main__':
    main()
