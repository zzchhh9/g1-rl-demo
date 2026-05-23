#!/usr/bin/env python3
"""Send ROS2 nav_msgs/Odometry as compact JSON over UDP.

This runs on the G1 Jetson, where ROS2/LIO is available but Unitree's Python
DDS stack may not be installed.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


def yaw_from_ros_quat(q) -> float:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    return float(math.atan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z)))


class OdomUdpSender(Node):
    def __init__(self, args):
        super().__init__("ros2_odom_udp_sender")
        self.args = args
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (args.udp_host, args.udp_port)
        self.count = 0
        self.last_print = time.time()

        qos = QoSProfile(depth=args.qos_depth)
        qos.history = HistoryPolicy.KEEP_LAST
        if args.best_effort:
            qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.sub = self.create_subscription(Odometry, args.ros_topic,
                                            self._callback, qos)
        print(f"[ros2-udp] {args.ros_topic} -> {args.udp_host}:{args.udp_port}")

    def _callback(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if stamp <= 0.0:
            stamp = time.time()

        payload = {
            "stamp": stamp,
            "recv_time": time.time(),
            "source": f"ros2:{self.args.ros_topic}",
            "frame_id": msg.header.frame_id,
            "child_frame_id": msg.child_frame_id,
            "x": float(p.x),
            "y": float(p.y),
            "z": float(p.z),
            "yaw": yaw_from_ros_quat(q),
            "qx": float(q.x),
            "qy": float(q.y),
            "qz": float(q.z),
            "qw": float(q.w),
            "vx": float(lin.x),
            "vy": float(lin.y),
            "vz": float(lin.z),
            "wz": float(ang.z),
        }
        self.sock.sendto(json.dumps(payload, separators=(",", ":")).encode(), self.addr)

        self.count += 1
        if self.args.print_every > 0 and self.count % self.args.print_every == 0:
            now = time.time()
            hz = self.args.print_every / max(now - self.last_print, 1e-6)
            self.last_print = now
            print(f"[ros2-udp] n={self.count} {hz:.1f}Hz "
                  f"x={payload['x']:+.3f} y={payload['y']:+.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ros-topic", default="/lio_sam_ros2/mapping/odometry")
    p.add_argument("--udp-host", required=True)
    p.add_argument("--udp-port", type=int, default=5070)
    p.add_argument("--best-effort", action="store_true")
    p.add_argument("--qos-depth", type=int, default=10)
    p.add_argument("--print-every", type=int, default=20)
    args = p.parse_args()

    rclpy.init()
    node = OdomUdpSender(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
