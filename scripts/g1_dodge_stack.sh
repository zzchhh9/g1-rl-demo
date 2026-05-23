#!/bin/bash
# Unified runner for the real G1 dodge stack.
#
# Usage:
#   scripts/g1_dodge_stack.sh sensors   # RGBD + YOLO + MID-360 driver + odom bridge
#   scripts/g1_dodge_stack.sh deploy    # run SDK dodge controller in foreground
#   scripts/g1_dodge_stack.sh status
#   scripts/g1_dodge_stack.sh stop
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${G1_SESSION:-g1-dodge}"
NET="${NET:-eno1}"
ROBOT_IP="${ROBOT_IP:-192.168.123.164}"
LAPTOP_IP="${LAPTOP_IP:-192.168.123.222}"
ROBOT_SUBNET="${ROBOT_SUBNET:-${LAPTOP_IP%.*}.0/24}"
AUTO_FIX_NET="${AUTO_FIX_NET:-1}"
DDS_ODOM_TOPIC="${DDS_ODOM_TOPIC:-rt/dodge/odom}"
DDS_YOLO_TOPIC="${DDS_YOLO_TOPIC:-rt/yolo/person}"
SLAM_BACKEND="${SLAM_BACKEND:-fast_lio}"  # fast_lio | lio_sam
if [ -z "${ROS_ODOM_TOPIC+x}" ]; then
    if [ "$SLAM_BACKEND" = "fast_lio" ]; then
        ROS_ODOM_TOPIC="/Odometry"
    else
        ROS_ODOM_TOPIC="/lio_sam_ros2/mapping/odometry"
    fi
fi
ODOM_UDP_PORT="${ODOM_UDP_PORT:-5070}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
ROS_DOMAIN_ID_REMOTE="${ROS_DOMAIN_ID_REMOTE:-42}"
YOLO_MODEL="${YOLO_MODEL:-yolov8m.pt}"
YOLO_SOURCE="${YOLO_SOURCE:-real}"  # real | fake
DEPTH_OFFSET="${DEPTH_OFFSET:-0.20}"
YOLO_PRINT_EVERY="${YOLO_PRINT_EVERY:-20}"
YOLO_LOCK_FIRST_TRACK="${YOLO_LOCK_FIRST_TRACK:-0}"
FAKE_YOLO_START_DELAY="${FAKE_YOLO_START_DELAY:-8.0}"
FAKE_YOLO_HOLD="${FAKE_YOLO_HOLD:-2.0}"
FAKE_YOLO_FADE="${FAKE_YOLO_FADE:-3.0}"
FAKE_YOLO_START_DIST="${FAKE_YOLO_START_DIST:-0.50}"
FAKE_YOLO_END_DIST="${FAKE_YOLO_END_DIST:-1.60}"
FAKE_YOLO_BEARING_DEG="${FAKE_YOLO_BEARING_DEG:-0.0}"
FAKE_YOLO_TRACK_ID="${FAKE_YOLO_TRACK_ID:-9001}"
MAX_VEL="${MAX_VEL:-0.40}"
# Keep the real evasive trigger close to the robot. By default, one meter is
# the boundary: dodge starts inside it, stops outside it after the return clear
# delay, then returns.
SAFETY_DIST="${SAFETY_DIST:-1.0}"
YOLO_LOCK_DIST="${YOLO_LOCK_DIST:-$SAFETY_DIST}"
CLEAR_MARGIN="${CLEAR_MARGIN:-0.00}"
ODOM_STALENESS="${ODOM_STALENESS:-0.35}"
RETURN_ODOM_SOURCE="${RETURN_ODOM_SOURCE:-external}"
RETURN_MODE="${RETURN_MODE:-geo}"
RETURN_FRAME_CALIB="${RETURN_FRAME_CALIB:-}"
RETURN_FRAME_ROTATE_WITH_SLAM_YAW="${RETURN_FRAME_ROTATE_WITH_SLAM_YAW:-0}"
RETURN_MAX_SLAM_YAW_DRIFT="${RETURN_MAX_SLAM_YAW_DRIFT:-0.45}"
if [ -z "${RETURN_PROBE+x}" ]; then
    if [ "$SLAM_BACKEND" = "fast_lio" ]; then
        RETURN_PROBE=0
    else
        RETURN_PROBE=1
    fi
fi
STARTUP_FRAME_CALIB="${STARTUP_FRAME_CALIB:-0}"
STARTUP_CALIB_DISTANCE="${STARTUP_CALIB_DISTANCE:-0.50}"
STARTUP_CALIB_SPEED="${STARTUP_CALIB_SPEED:-$MAX_VEL}"
STARTUP_CALIB_Y_SIGN="${STARTUP_CALIB_Y_SIGN:--1}"
STARTUP_CALIB_AXIS_TIMEOUT="${STARTUP_CALIB_AXIS_TIMEOUT:-10.0}"
STARTUP_CALIB_CMD_DURATION="${STARTUP_CALIB_CMD_DURATION:-1.0}"
STARTUP_CALIB_CMD_PERIOD="${STARTUP_CALIB_CMD_PERIOD:-0.25}"
STARTUP_CALIB_MIN_DELTA="${STARTUP_CALIB_MIN_DELTA:-0.35}"
if [ -z "${START_RGBD+x}" ]; then
    if [ "$YOLO_SOURCE" = "fake" ]; then
        START_RGBD=0
    else
        START_RGBD=1
    fi
fi
START_LIDAR_DRIVER="${START_LIDAR_DRIVER:-1}"
START_LIO="${START_LIO:-1}"
START_ODOM_BRIDGE="${START_ODOM_BRIDGE:-1}"
START_ODOM_ECHO="${START_ODOM_ECHO:-1}"
DEBUG_OBS="${DEBUG_OBS:-1}"
SAVE_LIO_MAP="${SAVE_LIO_MAP:-1}"
MAP_NAME="${MAP_NAME:-rhea_$(date +%Y%m%d_%H%M%S)}"
MAP_DIR="${MAP_DIR:-/tmp/lio-run/pcd/$MAP_NAME}"
START_WEB_VIZ="${START_WEB_VIZ:-1}"
WEB_VIZ_HOST="${WEB_VIZ_HOST:-127.0.0.1}"
WEB_VIZ_PORT="${WEB_VIZ_PORT:-8765}"
AUTO_START_SENSORS_BEFORE_DEPLOY="${AUTO_START_SENSORS_BEFORE_DEPLOY:-1}"
PREDEPLOY_TIMEOUT="${PREDEPLOY_TIMEOUT:-15}"
ODOM_PREDEPLOY_MIN_COUNT="${ODOM_PREDEPLOY_MIN_COUNT:-5}"
ODOM_PREDEPLOY_STABLE_S="${ODOM_PREDEPLOY_STABLE_S:-1.0}"
WAIT_YOLO_BEFORE_DEPLOY="${WAIT_YOLO_BEFORE_DEPLOY:-1}"
YOLO_STALENESS="${YOLO_STALENESS:-0.8}"
YOLO_PREDEPLOY_MIN_COUNT="${YOLO_PREDEPLOY_MIN_COUNT:-3}"
YOLO_PREDEPLOY_STABLE_S="${YOLO_PREDEPLOY_STABLE_S:-0.5}"
RECORD_DEPLOY="${RECORD_DEPLOY:-1}"
RENDER_DEPLOY_VIDEO="${RENDER_DEPLOY_VIDEO:-1}"
RUN_ID="${RUN_ID:-rhea_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/$RUN_ID}"

usage() {
    cat <<EOF
Usage:
  $0 sensors    Start camera, YOLO bridge, MID-360 ROS2 driver, odom bridge.
  $0 deploy     Run deploy_dodge_sdk_loco.py in the current terminal.
  $0 status     Show tmux windows and robot-side lidar processes.
  $0 stop       Stop tmux perception windows and robot-side lidar driver.

Common env overrides:
  NET=$NET
  ROBOT_IP=$ROBOT_IP
  LAPTOP_IP=$LAPTOP_IP
  ROS_SETUP=$ROS_SETUP
  ROS_DOMAIN_ID_REMOTE=$ROS_DOMAIN_ID_REMOTE
  SLAM_BACKEND=$SLAM_BACKEND       # fast_lio | lio_sam
  ROS_ODOM_TOPIC=$ROS_ODOM_TOPIC
  ODOM_UDP_PORT=$ODOM_UDP_PORT
  DDS_ODOM_TOPIC=$DDS_ODOM_TOPIC
  DDS_YOLO_TOPIC=$DDS_YOLO_TOPIC
  YOLO_SOURCE=$YOLO_SOURCE               # real | fake
  FAKE_YOLO_START_DELAY=$FAKE_YOLO_START_DELAY
  FAKE_YOLO_START_DIST=$FAKE_YOLO_START_DIST
  FAKE_YOLO_END_DIST=$FAKE_YOLO_END_DIST
  RETURN_ODOM_SOURCE=$RETURN_ODOM_SOURCE   # external | auto | cmd
  RETURN_MODE=$RETURN_MODE                 # geo | p; head requires ALLOW_RETURN_HEAD=1
  RETURN_PROBE=$RETURN_PROBE               # fast_lio default 0; lio_sam default 1
  RETURN_FRAME_CALIB=$RETURN_FRAME_CALIB   # optional manual body/SLAM frame JSON
  STARTUP_FRAME_CALIB=$STARTUP_FRAME_CALIB # 1 to actively calibrate before YOLO dodge
  SAVE_LIO_MAP=$SAVE_LIO_MAP
  MAP_DIR=$MAP_DIR
  AUTO_START_SENSORS_BEFORE_DEPLOY=$AUTO_START_SENSORS_BEFORE_DEPLOY
  RECORD_DEPLOY=$RECORD_DEPLOY
  RUN_DIR=$RUN_DIR
  YOLO_LOCK_FIRST_TRACK=$YOLO_LOCK_FIRST_TRACK # 0 by default; deploy locks first dodge track
  YOLO_LOCK_DIST=$YOLO_LOCK_DIST

Optional:
  LIO_CMD='ros2 launch ...'  # optional local ROS2 LIO override
EOF
}

require_tmux() {
    command -v tmux >/dev/null || {
        echo "tmux is required for sensors mode: sudo apt install tmux" >&2
        exit 1
    }
}

check_robot_network() {
    if ! ip -br addr show "$NET" >/dev/null 2>&1; then
        echo "[net] interface not found: $NET" >&2
        exit 1
    fi

    local link_line addr_line link_state
    link_line="$(ip -br link show "$NET")"
    addr_line="$(ip -br addr show "$NET")"
    link_state="$(awk '{print $2}' <<<"$link_line")"

    echo "[net] $addr_line"
    echo "[net] link $link_line"
    if [[ "$AUTO_FIX_NET" != "0" ]]; then
        local needs_fix=0
        if [[ "$link_state" != "UP" ]]; then
            needs_fix=1
        elif ! ip -br addr show "$NET" | grep -qw "$LAPTOP_IP/24"; then
            needs_fix=1
        elif ! ip route get "$ROBOT_IP" 2>/dev/null | grep -qw "dev $NET"; then
            needs_fix=1
        fi
        if [[ "$needs_fix" == "1" ]]; then
            echo "[net] auto-fixing $NET for robot subnet: $LAPTOP_IP/24 route $ROBOT_SUBNET"
            local sudo_cmd=()
            if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
                sudo_cmd=(sudo)
            fi
            if ! "${sudo_cmd[@]}" ip link set "$NET" up \
                || ! "${sudo_cmd[@]}" ip addr flush dev "$NET" scope global \
                || ! "${sudo_cmd[@]}" ip addr add "$LAPTOP_IP/24" dev "$NET" \
                || ! "${sudo_cmd[@]}" ip route replace "$ROBOT_SUBNET" dev "$NET" src "$LAPTOP_IP"; then
                echo "[net] auto-fix failed. If sudo was just added, fully log out/in as zz4723 first." >&2
                echo "[net] manual fix:" >&2
                echo "  sudo ip link set $NET up" >&2
                echo "  sudo ip addr flush dev $NET scope global" >&2
                echo "  sudo ip addr add $LAPTOP_IP/24 dev $NET" >&2
                echo "  sudo ip route replace $ROBOT_SUBNET dev $NET src $LAPTOP_IP" >&2
                exit 1
            fi
            sleep 0.3
            link_line="$(ip -br link show "$NET")"
            addr_line="$(ip -br addr show "$NET")"
            link_state="$(awk '{print $2}' <<<"$link_line")"
            echo "[net] after fix: $addr_line"
            echo "[net] link after fix: $link_line"
            echo "[net] route: $(ip route get "$ROBOT_IP" 2>/dev/null | head -1)"
        fi
    fi
    link_line="$(ip -br link show "$NET")"
    link_state="$(awk '{print $2}' <<<"$link_line")"
    if [[ "$link_state" != "UP" ]] || grep -qw "NO-CARRIER" <<<"$link_line"; then
        echo "[net] $NET has no Ethernet carrier: $link_line" >&2
        echo "[net] check the robot is powered, Ethernet cable/adapter is seated, and the robot LAN port/link LED is active." >&2
        exit 1
    fi
    if ! timeout 3 bash -lc ":</dev/tcp/$ROBOT_IP/22" >/dev/null 2>&1; then
        echo "[net] cannot reach $ROBOT_IP:22 from $NET" >&2
        echo "[net] current route:" >&2
        ip route get "$ROBOT_IP" >&2 || true
        echo "[net] neighbor entry:" >&2
        ip neigh show "$ROBOT_IP" >&2 || true
        echo "[net] fix Ethernet/robot power first, then rerun ./start_yolo_lidar.sh" >&2
        exit 1
    fi
}

ensure_session() {
    require_tmux
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux new-session -d -s "$SESSION" -n main "cd '$ROOT'; bash"
    fi
}

run_window() {
    local name="$1"
    local cmd="$2"
    ensure_session
    if tmux list-windows -t "$SESSION" -F '#W' | grep -qx "$name"; then
        tmux kill-window -t "$SESSION:$name"
    fi
    tmux new-window -t "$SESSION" -n "$name" "cd '$ROOT'; $cmd"
}

ros_prefix() {
    cat <<EOF
if [ -f '$ROS_SETUP' ]; then
  source '$ROS_SETUP'
else
  echo '[WARN] ROS_SETUP not found: $ROS_SETUP'
fi
export ROS_DOMAIN_ID='$ROS_DOMAIN_ID_REMOTE'
EOF
}

start_sensors() {
    cd "$ROOT"
    echo "[stack] starting sensors/perception for G1 dodge"
    echo "[stack] tmux session: $SESSION"
    check_robot_network

    if [ "$START_RGBD" = "1" ]; then
        echo "[stack] restarting robot RGBD publisher ..."
        uv run --with paramiko python scripts/restart_rgbd_robot.py --host "$ROBOT_IP"
    else
        echo "[stack] skipping RGBD restart (START_RGBD=$START_RGBD)"
    fi

    if [ "$START_LIDAR_DRIVER" = "1" ]; then
        echo "[stack] starting robot-side MID-360 ROS2 driver ..."
        uv run --with paramiko python scripts/start_lidar_ros2_driver_robot.py \
          --host "$ROBOT_IP" \
          --ros-domain "$ROS_DOMAIN_ID_REMOTE"
    else
        echo "[stack] skipping MID-360 driver (START_LIDAR_DRIVER=$START_LIDAR_DRIVER)"
    fi

    if [ -n "${LIO_CMD:-}" ]; then
        run_window "lio" "$(ros_prefix)
echo '[stack] running local LIO_CMD'
$LIO_CMD
bash"
    elif [ "$START_LIO" = "1" ]; then
        case "$SLAM_BACKEND" in
            fast_lio)
                echo "[stack] starting robot-side MID-360 FAST-LIO ..."
                uv run --with paramiko python scripts/start_fast_lio_mid360_robot.py \
                  --host "$ROBOT_IP" \
                  --ros-domain "$ROS_DOMAIN_ID_REMOTE" \
                  --check
                ;;
            lio_sam)
                echo "[stack] starting robot-side MID-360 LIO-SAM ..."
                lio_args=(
                  --host "$ROBOT_IP"
                  --ros-domain "$ROS_DOMAIN_ID_REMOTE"
                  --check
                )
                if [ "$SAVE_LIO_MAP" = "1" ]; then
                    echo "[stack] LIO-SAM map saving enabled: $MAP_DIR"
                    lio_args+=(--map-name-dir "$MAP_DIR" --save-pcd --reset-map-dir)
                fi
                uv run --with paramiko python scripts/start_lio_mid360_robot.py "${lio_args[@]}"
                ;;
            *)
                echo "[stack] unknown SLAM_BACKEND=$SLAM_BACKEND; expected lio_sam or fast_lio" >&2
                exit 2
                ;;
        esac
    else
        echo "[stack] skipping LIO startup (START_LIO=$START_LIO)"
    fi

    case "$YOLO_SOURCE" in
        real)
            yolo_lock_arg=""
            if [ "$YOLO_LOCK_FIRST_TRACK" = "1" ]; then
                yolo_lock_arg="--lock-first-track --lock-first-track-dist '$YOLO_LOCK_DIST'"
            fi
            run_window "yolo" \
                "uv run --with 'pillow==9.5.0' --with 'ultralytics==8.4.51' --with 'lap' \
python scripts/yolo_to_dds_laptop.py \
  --robot-ip '$ROBOT_IP' \
  --net '$NET' \
  --model '$YOLO_MODEL' \
  --depth-offset '$DEPTH_OFFSET' \
  --print-every '$YOLO_PRINT_EVERY' \
  $yolo_lock_arg; bash"
            ;;
        fake)
            echo "[stack] using fake YOLO obstacle publisher"
            run_window "yolo" \
                "PYTHONUNBUFFERED=1 uv run python scripts/fake_yolo_obstacle_dds.py '$NET' \
  --topic '$DDS_YOLO_TOPIC' \
  --start-delay '$FAKE_YOLO_START_DELAY' \
  --hold '$FAKE_YOLO_HOLD' \
  --fade '$FAKE_YOLO_FADE' \
  --start-dist '$FAKE_YOLO_START_DIST' \
  --end-dist '$FAKE_YOLO_END_DIST' \
  --bearing-deg '$FAKE_YOLO_BEARING_DEG' \
  --track-id '$FAKE_YOLO_TRACK_ID' \
  --print-every '$YOLO_PRINT_EVERY'; bash"
            ;;
        *)
            echo "[stack] unknown YOLO_SOURCE=$YOLO_SOURCE; expected real or fake" >&2
            exit 2
            ;;
    esac

    if [ "$START_ODOM_BRIDGE" = "1" ]; then
        run_window "odom_udp" \
            "PYTHONUNBUFFERED=1 uv run --with paramiko python scripts/ssh_ros2_odom_udp_sender.py \
  --robot-ip '$ROBOT_IP' \
  --local-ip '$LAPTOP_IP' \
  --udp-port '$ODOM_UDP_PORT' \
  --ros-topic '$ROS_ODOM_TOPIC' \
  --print-every 10; bash"

        run_window "odom_bridge" \
            "PYTHONUNBUFFERED=1 uv run python scripts/udp_odom_to_dds.py \
  --net '$NET' \
  --udp-port '$ODOM_UDP_PORT' \
  --dds-topic '$DDS_ODOM_TOPIC' \
  --print-every 10; bash"
    else
        echo "[stack] skipping odom bridge (START_ODOM_BRIDGE=$START_ODOM_BRIDGE)"
    fi

    if [ "$START_ODOM_ECHO" = "1" ]; then
        run_window "odom_echo" \
            "uv run python scripts/dds_odom_echo.py '$NET' --topic '$DDS_ODOM_TOPIC'; bash"
    fi

    if [ "$START_WEB_VIZ" = "1" ]; then
        run_window "web_viz" \
            "uv run python scripts/dds_odom_web_viz.py '$NET' --topic '$DDS_ODOM_TOPIC' --host '$WEB_VIZ_HOST' --port '$WEB_VIZ_PORT'; bash"
    fi

    echo
    echo "[stack] sensors/perception launched."
    echo "  Attach logs: tmux attach -t $SESSION"
    echo "  Windows:     tmux list-windows -t $SESSION"
    echo "  Deploy:      ./deploy.sh"
    echo "  Web viz:     http://$WEB_VIZ_HOST:$WEB_VIZ_PORT"
    echo "  SLAM:        $SLAM_BACKEND ROS_ODOM_TOPIC=$ROS_ODOM_TOPIC"
    echo "  Map dir:     $MAP_DIR"
    echo "  Odom check:  uv run python scripts/dds_odom_echo.py $NET --topic $DDS_ODOM_TOPIC --duration 10"
}

run_deploy() {
    cd "$ROOT"
    if [ "$AUTO_START_SENSORS_BEFORE_DEPLOY" = "1" ]; then
        echo "[deploy] pre-starting SLAM/map + YOLO stack before controller"
        start_sensors
        echo "[deploy] waiting for fresh SLAM odometry on $DDS_ODOM_TOPIC"
        uv run python scripts/dds_wait_json.py "$NET" \
          --topic "$DDS_ODOM_TOPIC" \
          --stale "$ODOM_STALENESS" \
          --timeout "$PREDEPLOY_TIMEOUT" \
          --min-count "$ODOM_PREDEPLOY_MIN_COUNT" \
          --stable-seconds "$ODOM_PREDEPLOY_STABLE_S"
        if [ "$WAIT_YOLO_BEFORE_DEPLOY" = "1" ]; then
            echo "[deploy] waiting for YOLO heartbeat on $DDS_YOLO_TOPIC"
            uv run python scripts/dds_wait_json.py "$NET" \
              --topic "$DDS_YOLO_TOPIC" \
              --stale "$YOLO_STALENESS" \
              --timeout "$PREDEPLOY_TIMEOUT" \
              --min-count "$YOLO_PREDEPLOY_MIN_COUNT" \
              --stable-seconds "$YOLO_PREDEPLOY_STABLE_S"
        fi
    fi

    if [ "$RETURN_MODE" = "head" ] && [ "${ALLOW_RETURN_HEAD:-0}" != "1" ]; then
        echo "[stack] RETURN_MODE=head is replay/debug only; overriding to RETURN_MODE=geo for real deploy." >&2
        echo "[stack] Set ALLOW_RETURN_HEAD=1 only when deliberately comparing the checkpoint head." >&2
        RETURN_MODE="geo"
    fi
    args=(
        "$NET"
        --source yolo
        --max_vel "$MAX_VEL"
        --safety_dist "$SAFETY_DIST"
        --clear_margin "$CLEAR_MARGIN"
        --return_mode "$RETURN_MODE"
        --return_odom_source "$RETURN_ODOM_SOURCE"
        --odom_topic "$DDS_ODOM_TOPIC"
        --odom_staleness "$ODOM_STALENESS"
    )
    if [ "$DEBUG_OBS" = "1" ]; then
        args+=(--debug_obs)
    fi
    if [ -n "$RETURN_FRAME_CALIB" ]; then
        args+=(--return_frame_calib "$RETURN_FRAME_CALIB")
    fi
    if [ "$RETURN_FRAME_ROTATE_WITH_SLAM_YAW" = "1" ]; then
        args+=(--return_frame_rotate_with_slam_yaw)
    fi
    args+=(--return_max_slam_yaw_drift "$RETURN_MAX_SLAM_YAW_DRIFT")
    if [ "$RETURN_PROBE" = "1" ]; then
        args+=(--return_probe)
    else
        args+=(--no_return_probe)
    fi
    if [ "$STARTUP_FRAME_CALIB" = "1" ]; then
        args+=(
            --startup_frame_calib
            --startup_calib_distance "$STARTUP_CALIB_DISTANCE"
            --startup_calib_speed "$STARTUP_CALIB_SPEED"
            --startup_calib_y_sign "$STARTUP_CALIB_Y_SIGN"
            --startup_calib_axis_timeout "$STARTUP_CALIB_AXIS_TIMEOUT"
            --startup_calib_cmd_duration "$STARTUP_CALIB_CMD_DURATION"
            --startup_calib_cmd_period "$STARTUP_CALIB_CMD_PERIOD"
            --startup_calib_min_delta "$STARTUP_CALIB_MIN_DELTA"
        )
    fi

    mkdir -p "$RUN_DIR"
    record_file="$RUN_DIR/odom.jsonl"
    video_file="$RUN_DIR/odom_trace.mp4"
    deploy_log="$RUN_DIR/deploy.log"
    deploy_artifacts_finalized=0

    finalize_deploy_artifacts() {
        local deploy_status="${1:-0}"
        if [ "$deploy_artifacts_finalized" = "1" ]; then
            return 0
        fi
        deploy_artifacts_finalized=1

        trap - EXIT INT TERM
        set +e
        echo "[deploy] finalizing artifacts after controller exit (status=$deploy_status)"
        if [ "$RECORD_DEPLOY" = "1" ]; then
            if tmux list-windows -t "$SESSION" -F '#W' 2>/dev/null | grep -qx "odom_record"; then
                tmux send-keys -t "$SESSION:odom_record" C-c >/dev/null 2>&1 || true
                sleep 1.0
                tmux kill-window -t "$SESSION:odom_record" >/dev/null 2>&1 || true
            fi
            if [ "$RENDER_DEPLOY_VIDEO" = "1" ] && [ -s "$record_file" ]; then
                echo "[deploy] rendering odom video at script end ..."
                if uv run python scripts/render_odom_video.py --input "$record_file" --out "$video_file"; then
                    echo "[deploy] video ready: $video_file"
                else
                    echo "[deploy] video render failed; odom recording kept at $record_file" >&2
                fi
            else
                echo "[deploy] odom recording: $record_file"
            fi
        fi
        echo "[deploy] deploy log: $deploy_log"
        set -e
        return 0
    }

    if [ "$RECORD_DEPLOY" = "1" ]; then
        echo "[deploy] recording odom to $record_file"
        run_window "odom_record" \
            "PYTHONUNBUFFERED=1 uv run python scripts/dds_odom_record.py '$NET' --topic '$DDS_ODOM_TOPIC' --out '$record_file'; bash"
        sleep 0.5
    fi

    echo "[deploy] run dir: $RUN_DIR"
    echo "[deploy] map dir: $MAP_DIR"
    echo "[deploy] video will be: $video_file"

    trap 'st=$?; finalize_deploy_artifacts "$st"; exit "$st"' EXIT
    trap 'finalize_deploy_artifacts 130; exit 130' INT
    trap 'finalize_deploy_artifacts 143; exit 143' TERM

    set +e
    uv run python "$ROOT/deploy_dodge_sdk_loco.py" "${args[@]}" "$@" 2>&1 | tee "$deploy_log"
    status=${PIPESTATUS[0]}
    set -e

    finalize_deploy_artifacts "$status"
    trap - EXIT INT TERM
    exit "$status"
}

status_stack() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux list-windows -t "$SESSION"
    else
        echo "[status] no tmux session: $SESSION"
    fi
    echo
    echo "[status] robot-side lidar processes:"
    ROBOT_IP="$ROBOT_IP" uv run --with paramiko python - <<'PY'
import os
import paramiko
import shlex

host = os.environ["ROBOT_IP"]
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, username="unitree", password="123", timeout=8)
cmd = (
    "ps -eo pid,ppid,etime,comm,args | "
        "egrep 'livox_ros_driver2|lio_sam_ros2|fastlio_mapping|ros2_odom_udp_sender|check_out' | "
    "egrep -v 'egrep|bash -lc' || true"
)
_, out, err = ssh.exec_command("bash -lc " + shlex.quote(cmd), timeout=12)
print(out.read().decode(errors="replace"))
e = err.read().decode(errors="replace")
if e:
    print(e)
ssh.close()
PY
}

stop_stack() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux kill-session -t "$SESSION"
        echo "[stop] killed tmux session $SESSION"
    else
        echo "[stop] no tmux session $SESSION"
    fi
    echo "[stop] stopping robot-side lidar/LIO/odom processes ..."
    ROBOT_IP="$ROBOT_IP" uv run --with paramiko python - <<'PY'
import os
import paramiko
import shlex

host = os.environ["ROBOT_IP"]
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, username="unitree", password="123", timeout=8)
cmd = r'''
python3 - <<'PY2'
import os, signal, time
targets = []
lio_targets = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        comm = open(f"/proc/{pid}/comm").read().strip()
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "ignore")
    except OSError:
        continue
    hit = False
    if comm.startswith("lio_sam_ros2_"):
        hit = True
        lio_targets.append(int(pid))
    if comm == "ros2" and ("lio_mapping_qt_mid360.launch.py" in cmd or "/check_out" in cmd):
        hit = True
        if "lio_mapping_qt_mid360.launch.py" in cmd:
            lio_targets.append(int(pid))
    if comm == "fastlio_mapping":
        hit = True
    if comm == "ros2" and "fast_lio" in cmd and "mapping.launch.py" in cmd:
        hit = True
    if "livox_ros_driver2_node" in cmd and comm in ("ros2", "livox_ros_drive"):
        hit = True
    if comm == "python3" and ("ros2_odom_udp_sender.py" in cmd or "/tmp/livox-run/forward.py" in cmd or "livox_imu_scale.py" in cmd):
        hit = True
    if hit:
        targets.append(int(pid))
for pid in sorted(set(lio_targets)):
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        pass
if lio_targets:
    print("sigint lio pids:", sorted(set(lio_targets)))
    time.sleep(15.0)
for sig in (signal.SIGTERM, signal.SIGKILL):
    for pid in targets:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
    time.sleep(1.0)
print("stopped pids:", targets)
PY2
'''
_, out, err = ssh.exec_command("bash -lc " + shlex.quote(cmd), timeout=20)
print(out.read().decode(errors="replace"))
e = err.read().decode(errors="replace")
if e:
    print(e)
ssh.close()
PY
}

cmd="${1:-}"
shift || true
case "$cmd" in
    sensors|perception|start)
        start_sensors "$@"
        ;;
    deploy|run)
        run_deploy "$@"
        ;;
    status)
        status_stack
        ;;
    stop)
        stop_stack
        ;;
    -h|--help|help|"")
        usage
        ;;
    *)
        echo "unknown command: $cmd" >&2
        usage >&2
        exit 2
        ;;
esac
