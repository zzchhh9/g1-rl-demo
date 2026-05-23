#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${G1_SESSION:-g1-lidar-map}"
ROBOT_IP="${ROBOT_IP:-192.168.123.164}"
ROBOT_USER="${ROBOT_USER:-unitree}"
ROBOT_PASSWORD="${ROBOT_PASSWORD:-123}"
MAP_DIR="${MAP_DIR:-/tmp/lio-run/pcd/default}"
STOP_LIDAR="${STOP_LIDAR:-1}"

echo "[map-stop] stopping LiDAR-only mapping"
echo "[map-stop] robot=$ROBOT_IP map_dir=$MAP_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
  echo "[map-stop] killed tmux session $SESSION"
else
  echo "[map-stop] no tmux session $SESSION"
fi

cd "$ROOT"
uv run --with paramiko python - "$ROBOT_IP" "$ROBOT_USER" "$ROBOT_PASSWORD" "$MAP_DIR" "$STOP_LIDAR" <<'PY'
from __future__ import annotations

import shlex
import sys

import paramiko

host, user, password, map_dir, stop_lidar = sys.argv[1:6]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print(f"[map-stop] connecting {user}@{host} ...")
ssh.connect(host, username=user, password=password, timeout=8, banner_timeout=8, auth_timeout=8)

remote = (
    "set +e\n"
    f"export MAP_DIR={shlex.quote(map_dir)}\n"
    f"export STOP_LIDAR={shlex.quote(stop_lidar)}\n"
    + r'''
echo "[map-stop] sending SIGINT to LIO so it can flush PCD files"
python3 - <<'PY2'
import os, signal

def matches(pid):
    try:
        comm = open(f"/proc/{pid}/comm").read().strip()
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "ignore")
    except OSError:
        return False
    if comm.startswith("lio_sam_ros2_"):
        return True
    if "lio_mapping_qt_mid360.launch.py" in cmd:
        return True
    return False

pids = [int(pid) for pid in os.listdir("/proc") if pid.isdigit() and matches(pid)]
for pid in pids:
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        pass
print("sigint_lio_pids:", pids)
PY2

sleep 15

echo "[map-stop] cleaning remaining helper processes"
python3 - <<'PY2'
import os, signal, time

stop_lidar = os.environ.get("STOP_LIDAR", "1") not in ("0", "false", "False", "no")

def kind(pid):
    try:
        comm = open(f"/proc/{pid}/comm").read().strip()
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "ignore")
    except OSError:
        return None
    if comm.startswith("lio_sam_ros2_") or "lio_mapping_qt_mid360.launch.py" in cmd:
        return "lio"
    if comm == "ros2" and "/check_out" in cmd:
        return "checkout"
    if "ros2_odom_udp_sender.py" in cmd:
        return "odom_sender"
    if stop_lidar and ("livox_ros_driver2_node" in cmd or "start_livox.sh" in cmd):
        return "livox"
    return None

targets = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    k = kind(pid)
    if k:
        targets.append((int(pid), k))

for sig in (signal.SIGTERM, signal.SIGKILL):
    for pid, _ in targets:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
    time.sleep(1.0)
print("stopped_pids:", targets)
PY2

echo "[map-stop] map directory: $MAP_DIR"
if [ -d "$MAP_DIR" ]; then
  du -sh "$MAP_DIR" || true
  find "$MAP_DIR" -maxdepth 2 -type f -printf "%p %s bytes\n" | sort | tail -80 || true
else
  echo "[map-stop] WARNING: map directory does not exist"
fi
'''
)

stdin, stdout, stderr = ssh.exec_command("bash -lc " + shlex.quote(remote), get_pty=True, timeout=60)
out = stdout.read().decode("utf-8", "replace")
err = stderr.read().decode("utf-8", "replace")
if out:
    print(out, end="")
if err:
    print(err, end="", file=sys.stderr)
ssh.close()
PY

echo "[map-stop] done"
