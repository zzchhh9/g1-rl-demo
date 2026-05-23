#!/usr/bin/env python3
"""Bridge ROS2 nav_msgs/Odometry to Unitree DDS JSON odometry.

This is the seam between a MID-360 LIO/SLAM stack and the SDK dodge controller:

    ROS2 LIO/SLAM /Odometry -> DDS std_msgs/String rt/dodge/odom

Run it from a shell where ROS2 is sourced. Keep Unitree DDS on domain 0; the
ROS2 domain can be set by ROS_DOMAIN_ID in the shell.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "third_party" / "unitree_sdk2_python"))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

try:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
except Exception as exc:  # pragma: no cover - depends on sourced ROS2 env
    raise SystemExit(
        "ROS2 Python packages are not importable. Source ROS2 first, e.g.\n"
        "  source /opt/ros/humble/setup.bash\n"
        "or the distro installed on this machine.\n"
        f"Original import error: {exc}"
    )


def yaw_from_ros_quat(q) -> float:
    # ROS geometry_msgs Quaternion order is x, y, z, w.
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    return float(math.atan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z)))


class Ros2OdomToDds(Node):
    def __init__(self, args):
        super().__init__("ros2_odom_to_unitree_dds")
        ChannelFactoryInitialize(args.dds_domain, args.net)
        self.pub = ChannelPublisher(args.dds_topic, String_)
        self.pub.Init()
        self.dds_msg = std_msgs_msg_dds__String_()
        self.args = args
        self.count = 0
        self.last_print = time.time()

        qos = QoSProfile(depth=args.qos_depth)
        qos.history = HistoryPolicy.KEEP_LAST
        if args.best_effort:
            qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.sub = self.create_subscription(Odometry, args.ros_topic,
                                            self._callback, qos)
        print(f"[bridge] ROS2 {args.ros_topic} -> DDS {args.dds_topic} "
              f"on {args.net} domain {args.dds_domain}")

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
        self.dds_msg.data = json.dumps(payload, separators=(",", ":"))
        self.pub.Write(self.dds_msg)

        self.count += 1
        now = time.time()
        if self.args.print_every > 0 and self.count % self.args.print_every == 0:
            dt = max(now - self.last_print, 1e-6)
            hz = self.args.print_every / dt
            self.last_print = now
            print(f"[odom] n={self.count} {hz:.1f}Hz "
                  f"x={payload['x']:+.3f} y={payload['y']:+.3f} "
                  f"yaw={math.degrees(payload['yaw']):+.1f}deg "
                  f"v=[{payload['vx']:+.2f},{payload['vy']:+.2f}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", default="eno1",
                   help="Unitree DDS network interface.")
    p.add_argument("--dds-domain", type=int, default=0,
                   help="Unitree DDS domain. Keep this 0 for G1 control.")
    p.add_argument("--dds-topic", default="rt/dodge/odom",
                   help="DDS std_msgs/String JSON odometry topic.")
    p.add_argument("--ros-topic", default="/Odometry",
                   help="ROS2 nav_msgs/Odometry topic from LIO/SLAM.")
    p.add_argument("--best-effort", action="store_true",
                   help="Use best-effort ROS2 QoS if the odom publisher does.")
    p.add_argument("--qos-depth", type=int, default=10)
    p.add_argument("--print-every", type=int, default=20)
    args = p.parse_args()

    rclpy.init()
    node = Ros2OdomToDds(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
