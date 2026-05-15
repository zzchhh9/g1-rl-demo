#!/bin/bash
# Bring up the G1 Mid-360 LiDAR data pipeline:
#   1. SSH to robot (.164), briefly run livox_ros_driver2 to tell LiDAR to start streaming
#   2. Kill the driver (it's only used to do the handshake; UDP keeps flowing after)
#   3. Start a small Python UDP forwarder on robot that pumps .164:56301/56401/56201
#      → laptop:same port
# After this script returns, on the laptop run e.g.
#   uv run python test_lidar.py
#
# All robot-side state lives under /tmp/livox-run (cleared on reboot).
# To stop streaming: ./scripts/stop_lidar.sh
set -euo pipefail

ROBOT_USER="${ROBOT_USER:-unitree}"
ROBOT_HOST="${ROBOT_HOST:-192.168.123.164}"
LAPTOP_IP="${LAPTOP_IP:-192.168.123.222}"
LIDAR_IP="${LIDAR_IP:-192.168.123.120}"

ssh_run() {
    sshpass -p 123 ssh -o StrictHostKeyChecking=no "$ROBOT_USER@$ROBOT_HOST" "$@"
}
scp_to() {
    sshpass -p 123 scp -o StrictHostKeyChecking=no "$1" "$ROBOT_USER@$ROBOT_HOST:$2"
}

command -v sshpass >/dev/null || { echo "sshpass not installed (sudo apt install sshpass)"; exit 1; }

echo "[1/4] preparing /tmp/livox-run on robot ..."
ssh_run "mkdir -p /tmp/livox-run && pkill -9 -f livox_ros_driver2_node 2>/dev/null; pkill -f forward.py 2>/dev/null; sleep 1; true"

echo "[2/4] uploading MID360 config + forwarder + start scripts ..."
TMP=$(mktemp -d)
cat > "$TMP/MID360_config_fixed.json" <<EOF
{
  "lidar_summary_info" : {"lidar_type": 8},
  "MID360": {
    "lidar_net_info" : {"cmd_data_port":56100,"push_msg_port":56200,"point_data_port":56300,"imu_data_port":56400,"log_data_port":56500},
    "host_net_info"  : {"cmd_data_ip":"$ROBOT_HOST","cmd_data_port":56101,"push_msg_ip":"$ROBOT_HOST","push_msg_port":56201,"point_data_ip":"$ROBOT_HOST","point_data_port":56301,"imu_data_ip":"$ROBOT_HOST","imu_data_port":56401,"log_data_ip":"","log_data_port":56501}
  },
  "lidar_configs" : [
    {"ip":"$LIDAR_IP","pcl_data_type":1,"pattern_mode":0,
     "extrinsic_parameter":{"roll":0.0,"pitch":1.57079632,"yaw":3.14159265,"x":0,"y":0,"z":0}}
  ]
}
EOF
cat > "$TMP/forward.py" <<EOF
#!/usr/bin/env python3
"""Forward LiDAR UDP packets from robot:port → laptop:port for ports 56301/56401/56201."""
import socket, threading, signal, sys
LAPTOP_IP = "$LAPTOP_IP"
PORTS = [56301, 56401, 56201]
def fwd(p):
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind(("", p))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[fwd:{p}] → {LAPTOP_IP}:{p}", flush=True)
    n = 0
    while True:
        d, _ = rx.recvfrom(65535)
        tx.sendto(d, (LAPTOP_IP, p))
        n += 1
        if n % 5000 == 0:
            print(f"[fwd:{p}] {n} packets", flush=True)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
for p in PORTS:
    threading.Thread(target=fwd, args=(p,), daemon=True).start()
signal.pause()
EOF
cat > "$TMP/start_driver.sh" <<EOF
#!/bin/bash
source /opt/ros/foxy/setup.bash
source /upgradePythonServer/temp/unitree/module/graph_pid_ws/install/setup.bash
export ROS_DOMAIN_ID=42
exec ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \\
  -p user_config_path:=/tmp/livox-run/MID360_config_fixed.json \\
  -p xfer_format:=0 -p multi_topic:=0 -p data_src:=0 -p publish_freq:=10.0 \\
  -p output_data_type:=0 -p frame_id:=livox_frame \\
  -p lvx_file_path:=/tmp/livox-run/test.lvx \\
  -p cmdline_input_bd_code:=livox0000000001
EOF
chmod +x "$TMP/forward.py" "$TMP/start_driver.sh"
for f in MID360_config_fixed.json forward.py start_driver.sh; do
    scp_to "$TMP/$f" "/tmp/livox-run/$f" > /dev/null
done
rm -rf "$TMP"

echo "[3/4] running driver briefly to tell LiDAR to start streaming ..."
ssh_run "nohup /tmp/livox-run/start_driver.sh > /tmp/livox-run/driver.log 2>&1 & echo \$! > /tmp/livox-run/driver.pid"
sleep 5
# Verify the SDK config succeeded
if ssh_run "grep -q 'successfully change work mode' /tmp/livox-run/driver.log"; then
    echo "    ✓ LiDAR streaming started"
else
    echo "    ✗ driver init failed — log:"
    ssh_run "tail -20 /tmp/livox-run/driver.log"
    exit 2
fi
ssh_run "kill \$(cat /tmp/livox-run/driver.pid) 2>/dev/null; pkill -9 -f livox_ros_driver2_node 2>/dev/null; sleep 1; true"

echo "[4/4] starting UDP forwarder ..."
ssh_run "nohup python3 /tmp/livox-run/forward.py > /tmp/livox-run/forward.log 2>&1 & echo \$! > /tmp/livox-run/forward.pid; sleep 1; pgrep -af forward.py | head -1"

echo
echo "✓ LiDAR pipeline up. Test from laptop:"
echo "    uv run python test_lidar.py"
echo "    uv run python test_lidar.py --duration 30 --detect"
