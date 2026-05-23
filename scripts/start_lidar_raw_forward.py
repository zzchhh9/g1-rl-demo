#!/usr/bin/env python3
"""Start MID-360 raw UDP forwarding through the robot using paramiko.

This is a sshpass-free equivalent of scripts/start_lidar.sh. It briefly starts
livox_ros_driver2 on the robot to configure the LiDAR, kills the driver, then
keeps a small UDP forwarder running:

    robot :56301/:56401/:56201 -> laptop same ports
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import textwrap
import time

import paramiko


def run(ssh, cmd: str, timeout: float = 15.0, check: bool = False) -> str:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    if out:
        print(out.rstrip())
    if err:
        print(err.rstrip(), file=sys.stderr)
    rc = stdout.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError(f"remote command failed rc={rc}: {cmd}")
    return out


def put_text(sftp, path: str, text: str, mode: int = 0o644):
    with sftp.open(path, "w") as f:
        f.write(text)
    sftp.chmod(path, mode)


def stop_existing_livox(ssh):
    run(ssh, r"""
mkdir -p /tmp/livox-run
if [ -f /tmp/livox-run/forward.pid ]; then
  kill -9 "$(cat /tmp/livox-run/forward.pid)" 2>/dev/null || true
fi
if [ -f /tmp/livox-run/driver.pid ]; then
  kill -9 "$(cat /tmp/livox-run/driver.pid)" 2>/dev/null || true
fi
python3 - <<'PY'
import os, signal
needles = ("livox_ros_driver2_node", "/tmp/livox-run/forward.py")
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "ignore")
    except OSError:
        continue
    if any(n in cmd for n in needles):
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
PY
sleep 1
""")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="192.168.123.164")
    p.add_argument("--user", default="unitree")
    p.add_argument("--password", default="123")
    p.add_argument("--laptop-ip", default="192.168.123.222")
    p.add_argument("--lidar-ip", default="192.168.123.120")
    args = p.parse_args()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[lidar-raw] connecting {args.user}@{args.host} ...")
    ssh.connect(args.host, username=args.user, password=args.password,
                timeout=8, banner_timeout=8, auth_timeout=8)

    stop_existing_livox(ssh)
    sftp = ssh.open_sftp()
    config = f"""{{
  "lidar_summary_info" : {{"lidar_type": 8}},
  "MID360": {{
    "lidar_net_info" : {{"cmd_data_port":56100,"push_msg_port":56200,"point_data_port":56300,"imu_data_port":56400,"log_data_port":56500}},
    "host_net_info"  : {{"cmd_data_ip":"{args.host}","cmd_data_port":56101,"push_msg_ip":"{args.host}","push_msg_port":56201,"point_data_ip":"{args.host}","point_data_port":56301,"imu_data_ip":"{args.host}","imu_data_port":56401,"log_data_ip":"","log_data_port":56501}}
  }},
  "lidar_configs" : [
    {{"ip":"{args.lidar_ip}","pcl_data_type":1,"pattern_mode":0,
     "extrinsic_parameter":{{"roll":0.0,"pitch":1.57079632,"yaw":3.14159265,"x":0,"y":0,"z":0}}}}
  ]
}}
"""
    forward = f'''#!/usr/bin/env python3
import socket, threading, signal, sys
LAPTOP_IP = "{args.laptop_ip}"
PORTS = [56301, 56401, 56201]
def fwd(p):
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind(("", p))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[fwd:{{p}}] -> {{LAPTOP_IP}}:{{p}}", flush=True)
    n = 0
    while True:
        d, _ = rx.recvfrom(65535)
        tx.sendto(d, (LAPTOP_IP, p))
        n += 1
        if n % 5000 == 0:
            print(f"[fwd:{{p}}] {{n}} packets", flush=True)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
for p in PORTS:
    threading.Thread(target=fwd, args=(p,), daemon=True).start()
signal.pause()
'''
    start_driver = """#!/bin/bash
source /opt/ros/foxy/setup.bash
source /upgradePythonServer/temp/unitree/module/graph_pid_ws/install/setup.bash
export ROS_DOMAIN_ID=42
exec ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \\
  -p user_config_path:=/tmp/livox-run/MID360_config_fixed.json \\
  -p xfer_format:=0 -p multi_topic:=0 -p data_src:=0 -p publish_freq:=10.0 \\
  -p output_data_type:=0 -p frame_id:=livox_frame \\
  -p lvx_file_path:=/tmp/livox-run/test.lvx \\
  -p cmdline_input_bd_code:=livox0000000001
"""
    put_text(sftp, "/tmp/livox-run/MID360_config_fixed.json", config)
    put_text(sftp, "/tmp/livox-run/forward.py", forward, 0o755)
    put_text(sftp, "/tmp/livox-run/start_driver.sh", start_driver, 0o755)
    sftp.close()

    print("[lidar-raw] running driver handshake ...")
    run(ssh, "nohup /tmp/livox-run/start_driver.sh > /tmp/livox-run/driver.log 2>&1 "
             "& echo $! > /tmp/livox-run/driver.pid")
    time.sleep(5)
    out = run(ssh, "grep -q 'successfully change work mode' /tmp/livox-run/driver.log "
                   "&& echo OK || (tail -40 /tmp/livox-run/driver.log; echo FAIL)")
    if "FAIL" in out:
        raise SystemExit("[lidar-raw] driver handshake failed")
    run(ssh, "kill $(cat /tmp/livox-run/driver.pid) 2>/dev/null; "
             "pkill -9 -f livox_ros_driver2_node 2>/dev/null; sleep 1; true")
    print("[lidar-raw] starting UDP forwarder ...")
    run(ssh, "nohup python3 /tmp/livox-run/forward.py > /tmp/livox-run/forward.log 2>&1 "
             "& echo $! > /tmp/livox-run/forward.pid; sleep 1; "
             "pgrep -af forward.py | head -1")
    ssh.close()
    print("[lidar-raw] up. Test with: uv run python test_lidar.py --duration 5")


if __name__ == "__main__":
    main()
