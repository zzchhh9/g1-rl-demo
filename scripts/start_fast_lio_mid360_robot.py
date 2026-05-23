#!/usr/bin/env python3
"""Start robot-side FAST-LIO2 MID-360 odometry.

This expects Ericsii/FAST_LIO_ROS2 to be built on the robot at
/home/unitree/fastlio2_ws.  It keeps the existing Livox driver contract:

    /livox/lidar      livox_ros_driver2/CustomMsg, xfer_format=1
    /livox/imu_scaled sensor_msgs/Imu, accel converted from g to m/s^2
    /Odometry         nav_msgs/Odometry from FAST-LIO
"""

from __future__ import annotations

import argparse
import shlex
import sys
import textwrap

import paramiko

from start_lidar_raw_forward import put_text, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="192.168.123.164")
    p.add_argument("--user", default="unitree")
    p.add_argument("--password", default="123")
    p.add_argument("--ros-domain", type=int, default=42)
    p.add_argument("--fastlio-ws", default="/home/unitree/fastlio2_ws")
    p.add_argument("--config-file", default="mid360.yaml")
    p.add_argument("--scale-livox-imu-accel", dest="scale_livox_imu_accel",
                   action="store_true", default=True)
    p.add_argument("--no-scale-livox-imu-accel", dest="scale_livox_imu_accel",
                   action="store_false")
    p.add_argument("--livox-imu-accel-scale", type=float, default=-9.80511)
    p.add_argument("--check", action="store_true",
                   help="Count /Odometry messages after startup.")
    args = p.parse_args()

    fastlio_ws = args.fastlio_ws.rstrip("/")
    fastlio_ws_q = shlex.quote(fastlio_ws)
    config_file_q = shlex.quote(args.config_file)
    imu_topic = "/livox/imu_scaled" if args.scale_livox_imu_accel else "/livox/imu"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[fast-lio] connecting {args.user}@{args.host} ...")
    ssh.connect(args.host, username=args.user, password=args.password,
                timeout=8, banner_timeout=8, auth_timeout=8)

    imu_scale_script = f"""#!/usr/bin/env python3
import argparse
import os

import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=float, default={float(args.livox_imu_accel_scale)})
    args = p.parse_args()
    rclpy.init()
    node = rclpy.create_node(f"livox_imu_accel_scaler_fastlio_{{os.getpid()}}")
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
        out.linear_acceleration.x = msg.linear_acceleration.x * args.scale
        out.linear_acceleration.y = msg.linear_acceleration.y * args.scale
        out.linear_acceleration.z = msg.linear_acceleration.z * args.scale
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance
        pub.publish(out)
        state["count"] += 1
        if state["count"] == 1:
            node.get_logger().info(
                f"scaling /livox/imu accel by {{args.scale:.5f}} to /livox/imu_scaled")

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
    run(ssh, "mkdir -p /tmp/fast-lio-run")
    put_text(sftp, "/tmp/fast-lio-run/livox_imu_scale.py", imu_scale_script, 0o755)
    sftp.close()

    remote = textwrap.dedent(f"""
    set -e
    source /opt/ros/foxy/setup.bash
    source /upgradePythonServer/temp/unitree/module/graph_pid_ws/install/setup.bash
    if [ ! -f {fastlio_ws_q}/install/setup.bash ]; then
      echo "[fast-lio] missing {fastlio_ws}/install/setup.bash; build FAST_LIO_ROS2 on the robot first" >&2
      exit 2
    fi
    source {fastlio_ws_q}/install/setup.bash
    export ROS_DOMAIN_ID={args.ros_domain}

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
    hit = False
    if comm.startswith("lio_sam_ros2_"):
        hit = True
    if comm == "ros2" and "lio_mapping_qt_mid360.launch.py" in cmd:
        hit = True
    if comm == "fastlio_mapping":
        hit = True
    if comm == "ros2" and "fast_lio" in cmd and "mapping.launch.py" in cmd:
        hit = True
    if comm.startswith("python") and "livox_imu_scale.py" in cmd:
        hit = True
    if hit:
        targets.append(int(pid))
for sig in (signal.SIGTERM, signal.SIGKILL):
    for pid in targets:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
    time.sleep(0.8)
if targets:
    print("[fast-lio] stopped previous odom pids:", sorted(set(targets)))
PY

    cfg="$(ros2 pkg prefix fast_lio)/share/fast_lio/config/{config_file_q}"
    if [ ! -f "$cfg" ]; then
      echo "[fast-lio] missing config: $cfg" >&2
      exit 3
    fi
    CFG="$cfg" python3 - <<'PY'
import os, re
from pathlib import Path
cfg = Path(os.environ["CFG"])
s = cfg.read_text()
s = re.sub(r'(imu_topic:\\s*)"?/livox/imu(?:_scaled)?"?', r'\\1"{imu_topic}"', s)
s = re.sub(r'(lid_topic:\\s*)"?/?livox/lidar"?', r'\\1"/livox/lidar"', s)
s = re.sub(r'(extrinsic_est_en:\\s*)true', r'\\1false', s)
s = re.sub(r'(pcd_save_en:\\s*)true', r'\\1false', s)
cfg.write_text(s)
print(f"[fast-lio] patched config {{cfg}} imu_topic={imu_topic}")
PY

    if [ "{'1' if args.scale_livox_imu_accel else '0'}" = "1" ]; then
      nohup bash -lc 'source /opt/ros/foxy/setup.bash; source /upgradePythonServer/temp/unitree/module/graph_pid_ws/install/setup.bash; export ROS_DOMAIN_ID={args.ros_domain}; exec python3 /tmp/fast-lio-run/livox_imu_scale.py --scale {float(args.livox_imu_accel_scale)}' > /tmp/fast-lio-run/livox_imu_scale.log 2>&1 &
      echo $! > /tmp/fast-lio-run/livox_imu_scale.pid
      sleep 1
      tail -5 /tmp/fast-lio-run/livox_imu_scale.log || true
      python3 - <<'PY'
import time
import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
rclpy.init()
node = rclpy.create_node("check_fastlio_imu_scaled")
count = 0
last = None
qos = QoSProfile(depth=10)
qos.reliability = ReliabilityPolicy.RELIABLE
qos.history = HistoryPolicy.KEEP_LAST
def cb(msg):
    global count, last
    count += 1
    last = (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z)
sub = node.create_subscription(Imu, "/livox/imu_scaled", cb, qos)
t0 = time.time()
while time.time() - t0 < 3.0:
    rclpy.spin_once(node, timeout_sec=0.1)
print(f"[fast-lio] imu_scaled_count={{count}} last={{last}}")
node.destroy_subscription(sub)
node.destroy_node()
rclpy.shutdown()
if count <= 0:
    raise SystemExit("[fast-lio] no scaled IMU from /livox/imu_scaled")
PY
    fi

    nohup bash -lc 'source /opt/ros/foxy/setup.bash; source /upgradePythonServer/temp/unitree/module/graph_pid_ws/install/setup.bash; source {fastlio_ws_q}/install/setup.bash; export ROS_DOMAIN_ID={args.ros_domain}; exec ros2 launch fast_lio mapping.launch.py config_file:={config_file_q} rviz:=false' > /tmp/fast-lio-run/fast_lio.log 2>&1 &
    echo $! > /tmp/fast-lio-run/fast_lio.pid
    sleep 8
    tail -80 /tmp/fast-lio-run/fast_lio.log || true
    """)
    run(ssh, remote, timeout=75, check=True)

    if args.check:
        check = textwrap.dedent(f"""
        source /opt/ros/foxy/setup.bash
        source /upgradePythonServer/temp/unitree/module/graph_pid_ws/install/setup.bash
        source {fastlio_ws_q}/install/setup.bash
        export ROS_DOMAIN_ID={args.ros_domain}
        python3 - <<'PY'
import time
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
rclpy.init()
node = rclpy.create_node("count_fastlio_odom_check")
count = 0
last = None
qos = QoSProfile(depth=10)
qos.reliability = ReliabilityPolicy.RELIABLE
qos.history = HistoryPolicy.KEEP_LAST
def cb(msg):
    global count, last
    count += 1
    last = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
sub = node.create_subscription(Odometry, "/Odometry", cb, qos)
t0 = time.time()
while time.time() - t0 < 6:
    rclpy.spin_once(node, timeout_sec=0.1)
print(f"[fast-lio] odom_count={{count}} last={{last}}")
node.destroy_subscription(sub)
node.destroy_node()
rclpy.shutdown()
if count <= 0:
    raise SystemExit("[fast-lio] no odometry from /Odometry")
PY
        """)
        run(ssh, check, timeout=30, check=True)

    ssh.close()
    print("[fast-lio] up.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[fast-lio] failed: {exc}", file=sys.stderr)
        raise
