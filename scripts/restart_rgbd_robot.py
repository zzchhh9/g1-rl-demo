#!/usr/bin/env python3
"""Restart the robot-side RealSense RGBD TCP publisher.

Uploads scripts/rgbd_publisher_robot.py to the G1 Jetson /tmp and starts it as
`python3 /tmp/rgbd_publisher.py`, listening on TCP :5005.

Run from the laptop:
    uv run --with paramiko python scripts/restart_rgbd_robot.py
"""

from __future__ import annotations

import argparse
import pathlib
import shlex
import sys

import paramiko


def run(ssh: paramiko.SSHClient, cmd: str, timeout: float = 10.0):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    if out:
        print(out.rstrip())
    if err:
        print(err.rstrip(), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.123.164")
    parser.add_argument("--user", default="unitree")
    parser.add_argument("--password", default="123")
    parser.add_argument("--local-script",
                        default=str(pathlib.Path(__file__).with_name(
                            "rgbd_publisher_robot.py")))
    parser.add_argument("--remote-script", default="/tmp/rgbd_publisher.py")
    parser.add_argument("--remote-launch", default="/tmp/launch_rgbd.sh")
    parser.add_argument("--keep-videohub", action="store_true",
                        help="Do not kill Unitree video_hub_pc4 before starting RGBD.")
    parser.add_argument("--rgbd-w", type=int, default=640)
    parser.add_argument("--rgbd-h", type=int, default=480)
    parser.add_argument("--rgbd-fps", type=int, default=30,
                        help="Cap RealSense capture fps (lower = less robot CPU).")
    parser.add_argument("--rgbd-rate", type=float, default=10.0,
                        help="Encode+send rate Hz (lower = less robot CPU).")
    parser.add_argument("--nice", type=int, default=0,
                        help="nice level for the publisher (higher = lower priority, "
                             "keeps CPU for the locomotion controller).")
    args = parser.parse_args()

    local_script = pathlib.Path(args.local_script).resolve()
    if not local_script.exists():
        raise SystemExit(f"missing local script: {local_script}")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[rgbd] connecting {args.user}@{args.host} ...")
    ssh.connect(args.host, username=args.user, password=args.password,
                timeout=8, banner_timeout=8, auth_timeout=8)
    print("[rgbd] connected")

    run(ssh, "set +e; echo '[before]'; "
             "ps -eo pid,comm,args | "
             "egrep 'rgbd_publisher|launch_rgbd|video_hub_pc4|"
             "master_service__video_hub_pc4' | grep -v egrep || true")
    run(ssh, "set +e; "
             "pkill -f '/tmp/rgbd_publisher.py' || true; "
             "pkill -f 'rgbd_publisher_robot.py' || true; "
             "pkill -f '/tmp/launch_rgbd.sh' || true; "
             "sleep 1")
    if not args.keep_videohub:
        # video_hub_pc4 holds the RealSense; rgbd_publisher can't open the depth stream
        # until it is stopped. It is a root service supervised by master_service, so a
        # plain `pkill` fails (EPERM) and it respawns. Stop it the supported way via mscli
        # (sets enable=0, no respawn); fall back to a sudo pkill just in case.
        pw = shlex.quote(args.password)
        run(ssh, "set +e; "
                 f"echo {pw} | sudo -S /unitree/sbin/mscli stopservice video_hub_pc4 2>&1 || true; "
                 f"echo {pw} | sudo -S pkill -9 -f videohub_pc4 2>/dev/null || true; "
                 "sleep 2")

    print(f"[rgbd] upload {local_script} -> {args.remote_script}")
    sftp = ssh.open_sftp()
    sftp.put(str(local_script), args.remote_script)
    sftp.chmod(args.remote_script, 0o755)
    launch = f"""#!/bin/bash
set -e
export LD_LIBRARY_PATH=/opt/ros/noetic/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export RGBD_W={args.rgbd_w}
export RGBD_H={args.rgbd_h}
export RGBD_FPS={args.rgbd_fps}
export RGBD_RATE={args.rgbd_rate}
nohup nice -n {args.nice} python3 {args.remote_script} > /tmp/rgbd_publisher.log 2>&1 &
echo $! > /tmp/rgbd_publisher.pid
sleep 3
cat /tmp/rgbd_publisher.pid
"""
    with sftp.open(args.remote_launch, "w") as f:
        f.write(launch)
    sftp.chmod(args.remote_launch, 0o755)
    sftp.close()

    print("[rgbd] launching ...")
    run(ssh, f"bash {args.remote_launch}", timeout=10)
    run(ssh, "tail -n 80 /tmp/rgbd_publisher.log || true; "
             "echo '[port]'; "
             "(ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) "
             "| grep 5005 || true; "
             "echo '[ps]'; "
             "ps -eo pid,comm,args | grep /tmp/rgbd_publisher.py "
             "| grep -v grep || true")
    ssh.close()


if __name__ == "__main__":
    main()
