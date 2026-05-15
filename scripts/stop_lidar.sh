#!/bin/bash
# Stop the LiDAR pipeline started by scripts/start_lidar.sh
# - Kills the UDP forwarder on the robot
# - Note: the Mid-360 will keep streaming to .164 (no traffic on the laptop after
#   the forwarder is gone). The LiDAR itself returns to idle on next power cycle.
set -euo pipefail
ROBOT_USER="${ROBOT_USER:-unitree}"
ROBOT_HOST="${ROBOT_HOST:-192.168.123.164}"

command -v sshpass >/dev/null || { echo "sshpass not installed"; exit 1; }

sshpass -p 123 ssh -o StrictHostKeyChecking=no "$ROBOT_USER@$ROBOT_HOST" "
pkill -f forward.py 2>/dev/null
pkill -9 -f livox_ros_driver2_node 2>/dev/null
sleep 1
echo 'forwarder stopped — robot-side processes:'
pgrep -af 'forward.py|livox_ros' || echo '(none)'
"
echo "✓ LiDAR pipeline stopped."
