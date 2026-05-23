#!/usr/bin/env python3
"""Start the robot-side MID-360 LIO-SAM odometry stack."""

from __future__ import annotations

import argparse
import shlex
import sys
import textwrap

import paramiko

from start_lidar_raw_forward import run, put_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="192.168.123.164")
    p.add_argument("--user", default="unitree")
    p.add_argument("--password", default="123")
    p.add_argument("--ros-domain", type=int, default=42)
    p.add_argument("--map-name-dir", default="/tmp/lio-run/pcd/default",
                   help="Robot-side LIO-SAM map output directory.")
    p.add_argument("--save-pcd", action="store_true",
                   help="Enable LIO-SAM PCD saving for mapping runs.")
    p.add_argument("--reset-map-dir", action="store_true",
                   help="Delete --map-name-dir before starting. Only allowed under /tmp/lio-run/pcd/.")
    p.add_argument("--scale-livox-imu-accel", dest="scale_livox_imu_accel",
                   action="store_true", default=True,
                   help="Republish /livox/imu to /livox/imu_scaled with linear "
                        "acceleration converted from g to m/s^2 for LIO-SAM.")
    p.add_argument("--no-scale-livox-imu-accel", dest="scale_livox_imu_accel",
                   action="store_false",
                   help="Disable Livox IMU acceleration scaling and feed "
                        "/livox/imu directly to LIO-SAM.")
    p.add_argument("--livox-imu-accel-scale", type=float, default=-9.80511,
                   help="Scale applied to Livox IMU linear_acceleration when "
                        "--scale-livox-imu-accel is enabled.")
    p.add_argument("--check", action="store_true",
                   help="Count odometry messages after startup.")
    args = p.parse_args()
    map_dir = args.map_name_dir.rstrip("/")
    map_dir_q = shlex.quote(map_dir)
    save_pcd = "true" if args.save_pcd else "false"
    imu_topic = "livox/imu_scaled" if args.scale_livox_imu_accel else "livox/imu"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[lio] connecting {args.user}@{args.host} ...")
    ssh.connect(args.host, username=args.user, password=args.password,
                timeout=8, banner_timeout=8, auth_timeout=8)

    start_lio = f"""#!/bin/bash
source /opt/ros/foxy/setup.bash
source /upgradePythonServer/temp/unitree/module/graph_pid_ws/install/setup.bash
export ROS_DOMAIN_ID={args.ros_domain}
cd /tmp/lio-run
exec ros2 launch lio_sam_ros2 lio_mapping_qt_mid360.launch.py \\
  use_sim_time:=False \\
  map_name_dir:={map_dir_q} \\
  mapping_param_dir:=/tmp/lio-run/params_mid360_runtime.yaml
"""

    imu_scale_script = f"""#!/usr/bin/env python3
import argparse
import os

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=float, default={float(args.livox_imu_accel_scale)})
    args = p.parse_args()
    scale = float(args.scale)
    rclpy.init()
    node = rclpy.create_node(f"livox_imu_accel_scaler_{{os.getpid()}}")
    pub = node.create_publisher(Imu, "/livox/imu_scaled", 50)
    qos = QoSProfile(depth=50)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    qos.history = HistoryPolicy.KEEP_LAST
    state = {{"count": 0}}

    def cb(msg):
        out = Imu()
        out.header = msg.header
        out.orientation = msg.orientation
        out.orientation_covariance = msg.orientation_covariance
        out.angular_velocity = msg.angular_velocity
        out.angular_velocity_covariance = msg.angular_velocity_covariance
        out.linear_acceleration.x = msg.linear_acceleration.x * scale
        out.linear_acceleration.y = msg.linear_acceleration.y * scale
        out.linear_acceleration.z = msg.linear_acceleration.z * scale
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance
        pub.publish(out)
        state["count"] += 1
        if state["count"] == 1:
            node.get_logger().info(
                f"scaling /livox/imu accel by {{scale:.5f}} to /livox/imu_scaled")

    sub = node.create_subscription(Imu, "/livox/imu", cb, qos)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_subscription(sub)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
"""

    sftp = ssh.open_sftp()
    if args.reset_map_dir:
        if not (map_dir.startswith("/tmp/lio-run/pcd/") and map_dir != "/tmp/lio-run/pcd"):
            raise SystemExit("--reset-map-dir is only allowed under /tmp/lio-run/pcd/")
        run(ssh, f"rm -rf {map_dir_q}")
    run(ssh, f"mkdir -p {map_dir_q}")
    put_text(sftp, "/tmp/lio-run/start_lio.sh", start_lio, 0o755)
    put_text(sftp, "/tmp/lio-run/livox_imu_scale.py", imu_scale_script, 0o755)
    sftp.close()

    remote = textwrap.dedent(f"""
    set -e
    source /opt/ros/foxy/setup.bash
    source /upgradePythonServer/temp/unitree/module/graph_pid_ws/install/setup.bash
    export ROS_DOMAIN_ID={args.ros_domain}

    mkdir -p {map_dir_q}
    if [ ! -f /tmp/lio-run/params_mid360_runtime.yaml ]; then
      pkg="$(ros2 pkg prefix lio_sam_ros2 2>/dev/null || true)"
      if [ -n "$pkg" ] && [ -f "$pkg/share/lio_sam_ros2/config/params_mid360.yaml" ]; then
        cp "$pkg/share/lio_sam_ros2/config/params_mid360.yaml" /tmp/lio-run/params_mid360_runtime.yaml
      else
        echo "[lio] missing params_mid360.yaml" >&2
        exit 2
      fi
    fi

    python3 - <<'PY'
import re
from pathlib import Path
p = Path("/tmp/lio-run/params_mid360_runtime.yaml")
s = p.read_text()
repls = {{
    'pointCloudTopic: "/livox/lidar"': 'pointCloudTopic: "livox/lidar"',
    'pointCloudTopic: "livox/lidar"': 'pointCloudTopic: "livox/lidar"',
    'imuTopic: "/livox/imu"': 'imuTopic: "{imu_topic}"',
    'imuTopic: "livox/imu"': 'imuTopic: "{imu_topic}"',
    'imuTopic: "/livox/imu_scaled"': 'imuTopic: "{imu_topic}"',
    'imuTopic: "livox/imu_scaled"': 'imuTopic: "{imu_topic}"',
    'savePCD: true': 'savePCD: {save_pcd}',
    'savePCD: True': 'savePCD: {save_pcd}',
    'savePCD: false': 'savePCD: {save_pcd}',
    'savePCD: False': 'savePCD: {save_pcd}',
}}
for a, b in repls.items():
    s = s.replace(a, b)
s = re.sub(r'(^\s*pointCloudTopic\s*:\s*)"?/?livox/lidar"?\s*$',
           r'\g<1>"livox/lidar"', s, flags=re.MULTILINE)
s = re.sub(r'(^\s*imuTopic\s*:\s*)"?/?(?:livox/imu|livox/imu_scaled|dog_imu_raw)"?\s*$',
           r'\g<1>"{imu_topic}"', s, flags=re.MULTILINE)
s = re.sub(r'(^\s*savePCD\s*:\s*).+$',
           r'\g<1>{save_pcd}', s, flags=re.MULTILINE)
# This deploy uses LIO as short-horizon local odometry.  The stock Unitree
# MID-360 config is for their mapping stack and enables loop closures plus IMU
# heading initialization.  With the MID-360 raw IMU and no GPS/magnetometer,
# those settings can inject yaw jumps/drift into the local odom stream.
s = re.sub(r'(^\s*useImuHeadingInitialization\s*:\s*).+$',
           r'\g<1>false', s, flags=re.MULTILINE)
s = re.sub(r'(^\s*imuRPYWeight\s*:\s*).+$',
           r'\g<1>0.0', s, flags=re.MULTILINE)
s = re.sub(r'(^\s*loopClosureEnableFlag\s*:\s*).+$',
           r'\g<1>false', s, flags=re.MULTILINE)
p.write_text(s)
PY

    python3 - <<'PY'
import os, signal, time
targets = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        comm = open(f"/proc/{{pid}}/comm").read().strip()
        cmd = open(f"/proc/{{pid}}/cmdline", "rb").read().replace(b"\\0", b" ").decode("utf-8", "ignore")
    except OSError:
        continue
    if comm.startswith("lio_sam_ros2_") or (comm == "ros2" and "lio_mapping_qt_mid360.launch.py" in cmd):
        targets.append(int(pid))
    if comm.startswith("python") and "/tmp/lio-run/livox_imu_scale.py" in cmd:
        targets.append(int(pid))
for sig in (signal.SIGTERM, signal.SIGKILL):
    for pid in targets:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
    time.sleep(1.0)
PY

    python3 - <<'PY'
import re
import time
from pathlib import Path

import rclpy
from livox_ros_driver2.msg import CustomMsg
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

p = Path("/tmp/lio-run/params_mid360_runtime.yaml")
s = p.read_text()
s = re.sub(r'imuTopic:\s*"/?dog_imu_raw"', 'imuTopic: "{imu_topic}"', s)
s = re.sub(r'imuTopic:\s*"/?livox/imu(?:_scaled)?"', 'imuTopic: "{imu_topic}"', s)
s = re.sub(r'pointCloudTopic:\s*"/?livox/lidar"', 'pointCloudTopic: "livox/lidar"', s)
s = re.sub(r'useImuHeadingInitialization:\s*true', 'useImuHeadingInitialization: false', s)
s = re.sub(r'imuRPYWeight:\s*[0-9.eE+-]+', 'imuRPYWeight: 0.0', s)
s = re.sub(r'loopClosureEnableFlag:\s*true', 'loopClosureEnableFlag: false', s)

lidar_id = None
try:
    rclpy.init()
    node = rclpy.create_node("peek_livox_lidar_id_for_lio")
    seen = []
    qos = QoSProfile(depth=10)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    qos.history = HistoryPolicy.KEEP_LAST
    def cb(msg):
        if not seen:
            seen.append(int(msg.lidar_id))
    sub = node.create_subscription(CustomMsg, "/livox/lidar", cb, qos)
    t0 = time.time()
    while time.time() - t0 < 3.0 and not seen:
        rclpy.spin_once(node, timeout_sec=0.1)
    if seen:
        lidar_id = seen[0]
    node.destroy_subscription(sub)
    node.destroy_node()
    rclpy.shutdown()
except Exception as exc:
    print(f"[lio] warning: failed to read Livox lidar_id: {{exc}}")

if lidar_id is not None:
    s = re.sub(r'lidarYsn:\s*"[^"]*"', f'lidarYsn: "{{lidar_id}}"', s)
    print(f"[lio] using livox imuTopic={imu_topic} lidarYsn={{lidar_id}}")
else:
    print("[lio] warning: no Livox CustomMsg seen; keeping existing lidarYsn")

p.write_text(s)
PY

    python3 - <<'PY'
import os, subprocess
found = False
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        comm = open(f"/proc/{{pid}}/comm").read().strip()
        cmd = open(f"/proc/{{pid}}/cmdline", "rb").read().replace(b"\\0", b" ").decode("utf-8", "ignore")
    except OSError:
        continue
    if comm == "ros2" and "topic pub" in cmd and "/check_out" in cmd:
        found = True
        break
if not found:
    subprocess.Popen(
        "source /opt/ros/foxy/setup.bash; "
        "source /upgradePythonServer/temp/unitree/module/graph_pid_ws/install/setup.bash; "
        "export ROS_DOMAIN_ID={args.ros_domain}; "
        "exec ros2 topic pub -r 1 /check_out std_msgs/msg/Bool '{{data: true}}'",
        shell=True,
        executable="/bin/bash",
        stdout=open("/tmp/lio_checkout_true.log", "ab"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
PY

    if [ "{'true' if args.scale_livox_imu_accel else 'false'}" = "true" ]; then
      nohup bash -lc 'source /opt/ros/foxy/setup.bash; source /upgradePythonServer/temp/unitree/module/graph_pid_ws/install/setup.bash; export ROS_DOMAIN_ID={args.ros_domain}; exec python3 /tmp/lio-run/livox_imu_scale.py --scale {float(args.livox_imu_accel_scale)}' > /tmp/lio-run/livox_imu_scale.log 2>&1 &
      echo $! > /tmp/lio-run/livox_imu_scale.pid
      sleep 1
      tail -5 /tmp/lio-run/livox_imu_scale.log || true
      python3 - <<'PY'
import time
import rclpy
from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
rclpy.init()
node = rclpy.create_node("check_livox_imu_scaled")
count = 0
last = None
qos = QoSProfile(depth=10)
qos.reliability = ReliabilityPolicy.RELIABLE
qos.history = HistoryPolicy.KEEP_LAST
def cb(msg):
    global count, last
    count += 1
    last = (msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z)
sub = node.create_subscription(Imu, "/livox/imu_scaled", cb, qos)
t0 = time.time()
while time.time() - t0 < 3.0:
    rclpy.spin_once(node, timeout_sec=0.1)
print(f"[lio] imu_scaled_count={{count}} last={{last}}")
node.destroy_subscription(sub)
node.destroy_node()
rclpy.shutdown()
if count <= 0:
    raise SystemExit("[lio] no scaled IMU from /livox/imu_scaled")
PY
    fi

    nohup /tmp/lio-run/start_lio.sh > /tmp/lio-run/lio.log 2>&1 &
    echo $! > /tmp/lio-run/lio.pid
    sleep 8
    tail -80 /tmp/lio-run/lio.log || true
    """)
    run(ssh, remote, timeout=75, check=True)

    if args.check:
        check = textwrap.dedent(f"""
        source /opt/ros/foxy/setup.bash
        source /upgradePythonServer/temp/unitree/module/graph_pid_ws/install/setup.bash
        export ROS_DOMAIN_ID={args.ros_domain}
        python3 - <<'PY'
import time
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
rclpy.init()
node = rclpy.create_node("count_lio_odom_check")
count = 0
last = None
qos = QoSProfile(depth=10)
qos.reliability = ReliabilityPolicy.RELIABLE
qos.history = HistoryPolicy.KEEP_LAST
def cb(msg):
    global count, last
    count += 1
    last = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
sub = node.create_subscription(Odometry, "/lio_sam_ros2/mapping/odometry", cb, qos)
t0 = time.time()
while time.time() - t0 < 6:
    rclpy.spin_once(node, timeout_sec=0.1)
print(f"[lio] odom_count={{count}} last={{last}}")
node.destroy_subscription(sub)
node.destroy_node()
rclpy.shutdown()
if count <= 0:
    raise SystemExit("[lio] no odometry from /lio_sam_ros2/mapping/odometry")
PY
        """)
        run(ssh, check, timeout=30, check=True)

    ssh.close()
    print("[lio] up.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[lio] failed: {exc}", file=sys.stderr)
        raise
