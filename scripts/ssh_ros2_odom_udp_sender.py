#!/usr/bin/env python3
"""Run the robot-side ROS2 odom UDP sender over a persistent SSH channel.

On this G1 image, a detached/nohup rclpy subscriber can appear in the graph but
fail to receive callbacks. Keeping the sender attached to an SSH exec channel
matches the foreground behavior that reliably receives LIO odometry.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

import paramiko


def q(s: str) -> str:
    return shlex.quote(s)


def run_short(ssh: paramiko.SSHClient, cmd: str, timeout: float = 10.0) -> str:
    stdin, stdout, stderr = ssh.exec_command("bash -lc " + q(cmd), timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    if err.strip():
        out += "\nSTDERR:\n" + err
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot-ip", default="192.168.123.164")
    p.add_argument("--user", default="unitree")
    p.add_argument("--password", default="123")
    p.add_argument("--local-ip", default="192.168.123.222")
    p.add_argument("--udp-port", type=int, default=5070)
    p.add_argument("--ros-topic", default="/lio_sam_ros2/mapping/odometry")
    p.add_argument("--remote-script", default="/tmp/ros2_odom_udp_sender.py")
    p.add_argument("--print-every", type=int, default=10)
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    local_sender = repo / "scripts" / "ros2_odom_udp_sender.py"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(args.robot_ip, username=args.user, password=args.password, timeout=8)

    sftp = ssh.open_sftp()
    sftp.put(str(local_sender), args.remote_script)
    sftp.close()

    # Stop detached senders from previous attempts. Keep LIO and Livox untouched.
    ps = run_short(ssh, "ps -eo pid,args", timeout=10)
    pids = []
    for line in ps.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and args.remote_script in parts[1]:
            pids.append(parts[0])
    if pids:
        run_short(ssh, "kill -TERM " + " ".join(pids) + " 2>/dev/null || true", timeout=5)
        time.sleep(0.3)

    cmd = (
        "source /opt/ros/foxy/setup.bash; "
        "source /upgradePythonServer/temp/unitree/module/graph_pid_ws/install/setup.bash; "
        "export ROS_DOMAIN_ID=42; "
        f"exec env PYTHONUNBUFFERED=1 python3 -u {q(args.remote_script)} "
        f"--ros-topic {q(args.ros_topic)} "
        f"--udp-host {q(args.local_ip)} "
        f"--udp-port {args.udp_port} "
        f"--print-every {args.print_every}"
    )
    print(f"[ssh-ros2-udp] starting on {args.robot_ip}: {args.ros_topic} -> "
          f"{args.local_ip}:{args.udp_port}", flush=True)
    stdin, stdout, stderr = ssh.exec_command("bash -lc " + q(cmd))
    chan = stdout.channel
    try:
        while not chan.exit_status_ready():
            if chan.recv_ready():
                sys.stdout.write(chan.recv(65535).decode("utf-8", "replace"))
                sys.stdout.flush()
            if chan.recv_stderr_ready():
                sys.stderr.write(chan.recv_stderr(65535).decode("utf-8", "replace"))
                sys.stderr.flush()
            time.sleep(0.1)
        while chan.recv_ready():
            sys.stdout.write(chan.recv(65535).decode("utf-8", "replace"))
        while chan.recv_stderr_ready():
            sys.stderr.write(chan.recv_stderr(65535).decode("utf-8", "replace"))
        raise SystemExit(chan.recv_exit_status())
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
