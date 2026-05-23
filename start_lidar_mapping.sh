#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${G1_SESSION:-g1-lidar-map}"
NET="${NET:-eno1}"
ROBOT_IP="${ROBOT_IP:-192.168.123.164}"
LAPTOP_IP="${LAPTOP_IP:-192.168.123.222}"
ROBOT_SUBNET="${ROBOT_SUBNET:-${LAPTOP_IP%.*}.0/24}"
AUTO_FIX_NET="${AUTO_FIX_NET:-1}"
ROS_DOMAIN_ID_REMOTE="${ROS_DOMAIN_ID_REMOTE:-42}"
ROS_ODOM_TOPIC="${ROS_ODOM_TOPIC:-/lio_sam_ros2/mapping/odometry}"
ODOM_UDP_PORT="${ODOM_UDP_PORT:-5070}"
DDS_ODOM_TOPIC="${DDS_ODOM_TOPIC:-rt/dodge/odom}"
MAP_NAME="${MAP_NAME:-map_$(date +%Y%m%d_%H%M%S)}"
MAP_DIR="${MAP_DIR:-/tmp/lio-run/pcd/$MAP_NAME}"

command -v tmux >/dev/null || { echo "tmux is required: sudo apt install tmux" >&2; exit 1; }
if ! ip -br addr show "$NET" >/dev/null 2>&1; then
  echo "[map] interface not found: $NET" >&2
  exit 2
fi
if [[ "$AUTO_FIX_NET" != "0" ]]; then
  needs_fix=0
  if ! ip -br addr show "$NET" | grep -qw "$LAPTOP_IP/24"; then
    needs_fix=1
  elif ! ip route get "$ROBOT_IP" 2>/dev/null | grep -qw "dev $NET"; then
    needs_fix=1
  fi
  if [[ "$needs_fix" == "1" ]]; then
    echo "[map] auto-fixing $NET for robot subnet: $LAPTOP_IP/24 route $ROBOT_SUBNET"
    sudo_cmd=()
    if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
      sudo_cmd=(sudo)
    fi
    if ! "${sudo_cmd[@]}" ip link set "$NET" up \
      || ! "${sudo_cmd[@]}" ip addr flush dev "$NET" scope global \
      || ! "${sudo_cmd[@]}" ip addr add "$LAPTOP_IP/24" dev "$NET" \
      || ! "${sudo_cmd[@]}" ip route replace "$ROBOT_SUBNET" dev "$NET" src "$LAPTOP_IP"; then
      echo "[map] auto-fix failed. If sudo was just added, fully log out/in as zz4723 first." >&2
      echo "[map] manual fix:" >&2
      echo "  sudo ip link set $NET up" >&2
      echo "  sudo ip addr flush dev $NET scope global" >&2
      echo "  sudo ip addr add $LAPTOP_IP/24 dev $NET" >&2
      echo "  sudo ip route replace $ROBOT_SUBNET dev $NET src $LAPTOP_IP" >&2
      exit 2
    fi
    sleep 0.3
    echo "[map] after fix: $(ip -br addr show "$NET")"
  fi
fi

echo "[map] starting LiDAR-only mapping"
echo "[map] robot=$ROBOT_IP net=$NET laptop=$LAPTOP_IP"
echo "[map] robot map dir: $MAP_DIR"

uv run --with paramiko python "$ROOT/scripts/start_lidar_ros2_driver_robot.py" \
  --host "$ROBOT_IP" \
  --ros-domain "$ROS_DOMAIN_ID_REMOTE"

uv run --with paramiko python "$ROOT/scripts/start_lio_mid360_robot.py" \
  --host "$ROBOT_IP" \
  --ros-domain "$ROS_DOMAIN_ID_REMOTE" \
  --map-name-dir "$MAP_DIR" \
  --save-pcd \
  --reset-map-dir \
  --check

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" -n main "cd '$ROOT'; bash"
fi

for win in odom_udp odom_bridge odom_echo; do
  if tmux list-windows -t "$SESSION" -F '#W' | grep -qx "$win"; then
    tmux kill-window -t "$SESSION:$win"
  fi
done

tmux new-window -t "$SESSION" -n odom_udp \
  "cd '$ROOT'; PYTHONUNBUFFERED=1 uv run --with paramiko python scripts/ssh_ros2_odom_udp_sender.py --robot-ip '$ROBOT_IP' --local-ip '$LAPTOP_IP' --udp-port '$ODOM_UDP_PORT' --ros-topic '$ROS_ODOM_TOPIC' --print-every 10; bash"

tmux new-window -t "$SESSION" -n odom_bridge \
  "cd '$ROOT'; PYTHONUNBUFFERED=1 uv run python scripts/udp_odom_to_dds.py --net '$NET' --udp-port '$ODOM_UDP_PORT' --dds-topic '$DDS_ODOM_TOPIC' --print-every 10; bash"

tmux new-window -t "$SESSION" -n odom_echo \
  "cd '$ROOT'; uv run python scripts/dds_odom_echo.py '$NET' --topic '$DDS_ODOM_TOPIC'; bash"

cat <<EOF

[map] LiDAR-only mapping is running.
  Walk/push the robot slowly around the test area to build the map.
  Watch odom:   uv run python scripts/dds_odom_echo.py $NET --topic $DDS_ODOM_TOPIC --duration 10
  Attach logs:  tmux attach -t $SESSION
  Stop/save:    MAP_DIR='$MAP_DIR' ./stop_lidar_mapping.sh

EOF
