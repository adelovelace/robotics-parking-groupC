from __future__ import annotations

import json
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from robomaster_example.logic.grid_map import GridMap
from robomaster_example.logic.topic_codec import pack_map_points, unpack_observation


class MapNode(Node):
    """Own the accumulated boundary/empty-space map."""

    def __init__(self):
        super().__init__("map_node")
        self.grid_map = GridMap(resolution=0.02)
        self.map_pub = self.create_publisher(Float32MultiArray, "map/global_points", 10)
        self.stats_pub = self.create_publisher(String, "map/stats", 10)
        self.obs_sub = self.create_subscription(Float32MultiArray, "vision/boundary_observation", self.observation_callback, 10)
        self.reset_sub = self.create_subscription(String, "map/reset", self.reset_callback, 10)

    def reset_callback(self, msg: String) -> None:
        self.grid_map.reset()
        self.publish_map("reset")
        self.get_logger().warn("[MAP] Reset requested.")

    def observation_callback(self, msg: Float32MultiArray) -> None:
        _, boundary, empty = unpack_observation(msg)
        self.grid_map.update_boundary(boundary)
        self.grid_map.update_empty_space(empty)
        self.publish_map("observation")

    def publish_map(self, reason: str) -> None:
        boundary, empty = self.grid_map.get_points()
        self.map_pub.publish(pack_map_points(boundary, empty))
        stats = String()
        stats.data = json.dumps({"reason": reason, "boundary": len(boundary), "empty": len(empty)})
        self.stats_pub.publish(stats)
        self.get_logger().info(f"[MAP] Updated from {reason}: boundary={len(boundary)}, empty={len(empty)}")


def main(args=None):
    rclpy.init(args=args if args is not None else sys.argv)
    node = MapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
