#!/usr/bin/env python3
"""Start the robot-side MID-360 livox_ros_driver2 node using paramiko."""

from __future__ import annotations

import argparse
import time

import paramiko

from start_lidar_raw_forward import put_text, run, stop_existing_livox


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="192.168.123.164")
    p.add_argument("--user", default="unitree")
    p.add_argument("--password", default="123")
    p.add_argument("--lidar-ip", default="192.168.123.120")
    p.add_argument("--ros-domain", type=int, default=42)
    p.add_argument("--legacy-mounted-extrinsic", action="store_true",
                   help="Use the old driver-level roll=0,pitch=90deg,yaw=180deg "
                        "point transform. Do not use this with LIO unless the "
                        "LIO IMU extrinsics are changed to match.")
    args = p.parse_args()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[lidar-ros2] connecting {args.user}@{args.host} ...")
    ssh.connect(args.host, username=args.user, password=args.password,
                timeout=8, banner_timeout=8, auth_timeout=8)
    stop_existing_livox(ssh)
    sftp = ssh.open_sftp()
    if args.legacy_mounted_extrinsic:
        extrinsic = '"roll":0.0,"pitch":1.57079632,"yaw":3.14159265,"x":0,"y":0,"z":0'
    else:
        # LIO consumes Livox point timestamps together with /livox/imu. Keep the
        # driver output in the raw MID-360 sensor frame; mounting transforms
        # belong in the downstream estimator, not only on the point cloud.
        extrinsic = '"roll":0.0,"pitch":0.0,"yaw":0.0,"x":0,"y":0,"z":0'

    config = f"""{{
  "lidar_summary_info" : {{"lidar_type": 8}},
  "MID360": {{
    "lidar_net_info" : {{"cmd_data_port":56100,"push_msg_port":56200,"point_data_port":56300,"imu_data_port":56400,"log_data_port":56500}},
    "host_net_info"  : {{"cmd_data_ip":"{args.host}","cmd_data_port":56101,"push_msg_ip":"{args.host}","push_msg_port":56201,"point_data_ip":"{args.host}","point_data_port":56301,"imu_data_ip":"{args.host}","imu_data_port":56401,"log_data_ip":"","log_data_port":56501}}
  }},
  "lidar_configs" : [
    {{"ip":"{args.lidar_ip}","pcl_data_type":1,"pattern_mode":0,
     "extrinsic_parameter":{{{extrinsic}}}}}
  ]
}}
"""
    start_driver = f"""#!/bin/bash
source /opt/ros/foxy/setup.bash
source /upgradePythonServer/temp/unitree/module/graph_pid_ws/install/setup.bash
export ROS_DOMAIN_ID={args.ros_domain}
exec ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \\
  -p user_config_path:=/tmp/livox-run/MID360_config_fixed.json \\
  -p xfer_format:=1 -p multi_topic:=0 -p data_src:=0 -p publish_freq:=10.0 \\
  -p output_data_type:=0 -p frame_id:=livox_frame \\
  -p lvx_file_path:=/tmp/livox-run/test.lvx \\
  -p cmdline_input_bd_code:=livox0000000001
"""
    put_text(sftp, "/tmp/livox-run/MID360_config_fixed.json", config)
    put_text(sftp, "/tmp/livox-run/start_driver_ros2.sh", start_driver, 0o755)
    sftp.close()

    live_check = f"""
source /opt/ros/foxy/setup.bash
source /upgradePythonServer/temp/unitree/module/graph_pid_ws/install/setup.bash
export ROS_DOMAIN_ID={args.ros_domain}
python3 - <<'PY'
import time
import rclpy
from sensor_msgs.msg import Imu
from livox_ros_driver2.msg import CustomMsg
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
rclpy.init()
node = rclpy.create_node("livox_ros2_live_check")
counts = {{"imu": 0, "custom": 0}}
qos = QoSProfile(depth=10)
qos.reliability = ReliabilityPolicy.RELIABLE
qos.history = HistoryPolicy.KEEP_LAST
node.create_subscription(Imu, "/livox/imu", lambda msg: counts.__setitem__("imu", counts["imu"] + 1), qos)
node.create_subscription(CustomMsg, "/livox/lidar", lambda msg: counts.__setitem__("custom", counts["custom"] + 1), qos)
t0 = time.time()
while time.time() - t0 < 6:
    rclpy.spin_once(node, timeout_sec=0.1)
print(f"[lidar-ros2] live_check imu={{counts['imu']}} custom={{counts['custom']}}")
node.destroy_node()
rclpy.shutdown()
raise SystemExit(0 if counts["imu"] > 0 and counts["custom"] > 0 else 3)
PY
"""

    for attempt in (1, 2):
        print(f"[lidar-ros2] starting driver on ROS_DOMAIN_ID={args.ros_domain} (attempt {attempt}) ...")
        run(ssh, "nohup /tmp/livox-run/start_driver_ros2.sh > /tmp/livox-run/driver.log 2>&1 "
                 "& echo $! > /tmp/livox-run/driver.pid")
        time.sleep(5)
        out = run(ssh, "grep -q 'successfully change work mode' /tmp/livox-run/driver.log "
                       "&& echo OK || (tail -40 /tmp/livox-run/driver.log; echo FAIL)")
        if "FAIL" in out:
            if attempt == 2:
                raise SystemExit("[lidar-ros2] driver handshake failed")
            stop_existing_livox(ssh)
            continue
        run(ssh, "pgrep -af livox_ros_driver2_node | head -1")
        try:
            run(ssh, live_check, timeout=20, check=True)
            break
        except RuntimeError:
            if attempt == 2:
                raise SystemExit("[lidar-ros2] driver started but no live /livox data")
            print("[lidar-ros2] no live data; restarting driver once ...")
            stop_existing_livox(ssh)
    ssh.close()
    print("[lidar-ros2] up.")


if __name__ == "__main__":
    main()
