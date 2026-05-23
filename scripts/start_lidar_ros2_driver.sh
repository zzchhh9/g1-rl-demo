#!/bin/bash
# Start the robot-side Livox ROS2 driver continuously for MID-360 LIO/SLAM.
#
# This differs from scripts/start_lidar.sh:
#   - start_lidar.sh does a one-time Livox handshake, kills the ROS driver, then
#     forwards raw UDP packets to the laptop for nearest-obstacle tools.
#   - this script keeps livox_ros_driver2 alive so FAST-LIO/Point-LIO/GLIM can
#     subscribe to /livox/lidar and /livox/imu on ROS_DOMAIN_ID.
set -euo pipefail

ROBOT_USER="${ROBOT_USER:-unitree}"
ROBOT_HOST="${ROBOT_HOST:-192.168.123.164}"
LIDAR_IP="${LIDAR_IP:-192.168.123.120}"
ROS_DOMAIN_ID_REMOTE="${ROS_DOMAIN_ID_REMOTE:-42}"

ssh_run() {
    sshpass -p 123 ssh -o StrictHostKeyChecking=no "$ROBOT_USER@$ROBOT_HOST" "$@"
}
scp_to() {
    sshpass -p 123 scp -o StrictHostKeyChecking=no "$1" "$ROBOT_USER@$ROBOT_HOST:$2"
}

command -v sshpass >/dev/null || { echo "sshpass not installed (sudo apt install sshpass)"; exit 1; }

echo "[1/3] preparing /tmp/livox-run on robot ..."
ssh_run "mkdir -p /tmp/livox-run && pkill -9 -f livox_ros_driver2_node 2>/dev/null; pkill -f forward.py 2>/dev/null; sleep 1; true"

echo "[2/3] uploading MID360 ROS2 driver config ..."
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
cat > "$TMP/start_driver_ros2.sh" <<EOF
#!/bin/bash
source /opt/ros/foxy/setup.bash
source /upgradePythonServer/temp/unitree/module/graph_pid_ws/install/setup.bash
export ROS_DOMAIN_ID=$ROS_DOMAIN_ID_REMOTE
exec ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \\
  -p user_config_path:=/tmp/livox-run/MID360_config_fixed.json \\
  -p xfer_format:=1 -p multi_topic:=0 -p data_src:=0 -p publish_freq:=10.0 \\
  -p output_data_type:=0 -p frame_id:=livox_frame \\
  -p lvx_file_path:=/tmp/livox-run/test.lvx \\
  -p cmdline_input_bd_code:=livox0000000001
EOF
chmod +x "$TMP/start_driver_ros2.sh"
scp_to "$TMP/MID360_config_fixed.json" "/tmp/livox-run/MID360_config_fixed.json" > /dev/null
scp_to "$TMP/start_driver_ros2.sh" "/tmp/livox-run/start_driver_ros2.sh" > /dev/null
rm -rf "$TMP"

echo "[3/3] starting livox_ros_driver2 on robot, ROS_DOMAIN_ID=$ROS_DOMAIN_ID_REMOTE ..."
ssh_run "nohup /tmp/livox-run/start_driver_ros2.sh > /tmp/livox-run/driver.log 2>&1 & echo \$! > /tmp/livox-run/driver.pid"
sleep 5
if ssh_run "grep -q 'successfully change work mode' /tmp/livox-run/driver.log"; then
    echo "    ✓ Livox ROS2 driver is running"
else
    echo "    ✗ driver init may have failed — log:"
    ssh_run "tail -40 /tmp/livox-run/driver.log"
    exit 2
fi
ssh_run "pgrep -af livox_ros_driver2_node | head -1"

echo
echo "Next checks on a ROS2 shell that can see ROS_DOMAIN_ID=$ROS_DOMAIN_ID_REMOTE:"
echo "    export ROS_DOMAIN_ID=$ROS_DOMAIN_ID_REMOTE"
echo "    ros2 topic list | grep livox"
echo "    ros2 topic hz /livox/lidar"
