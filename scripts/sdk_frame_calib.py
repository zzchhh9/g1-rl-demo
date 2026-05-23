#!/usr/bin/env python3
"""Low-speed SDK SetVelocity frame calibration using external SLAM odom."""

from __future__ import annotations

import argparse
import sys
import subprocess
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy_dodge_sdk_loco import (
    ExternalOdomMonitor,
    _fit_probe_return_frame,
    _fmt_mat2,
    _fmt_vec,
    _loco_status,
)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient


def _wait_odom_delta(odom: ExternalOdomMonitor,
                     start_xy: np.ndarray,
                     duration: float,
                     staleness: float):
    deadline = time.time() + duration
    last = None
    while time.time() < deadline:
        snap = odom.snapshot(staleness)
        if snap.get("fresh"):
            last = snap
        time.sleep(0.02)
    if last is None:
        return None
    xy = np.array([last["x"], last["y"]], dtype=np.float32)
    return xy - start_xy


def _zero(loco: LocoClient, duration: float = 0.05, n: int = 6):
    for _ in range(n):
        loco.SetVelocity(0.0, 0.0, 0.0, duration=duration)
        time.sleep(0.03)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("--topic", default="rt/dodge/odom")
    ap.add_argument("--staleness", type=float, default=0.35)
    ap.add_argument("--speed-x", type=float, default=0.12)
    ap.add_argument("--speed-y", type=float, default=0.10)
    ap.add_argument("--duration", type=float, default=0.60)
    ap.add_argument("--settle", type=float, default=0.40)
    ap.add_argument("--min-delta", type=float, default=0.040)
    ap.add_argument("--max-axis-cos", type=float, default=0.70)
    ap.add_argument("--execute", action="store_true",
                    help="Actually send nonzero SetVelocity commands.")
    args = ap.parse_args()

    try:
        ip_info = subprocess.run(
            ["ip", "-br", "addr", "show", args.net],
            text=True, capture_output=True, check=False)
    except Exception as exc:
        raise SystemExit(f"[calib] failed to inspect interface {args.net}: {exc}")
    if ip_info.returncode != 0:
        raise SystemExit(f"[calib] interface {args.net} not found")
    print(f"[calib] net {ip_info.stdout.strip()}")
    if "192.168.123." not in ip_info.stdout:
        raise SystemExit(
            f"[calib] {args.net} has no 192.168.123.x address. "
            "Fix Ethernet route first, e.g. sudo ip addr add "
            f"192.168.123.222/24 dev {args.net}"
        )

    ChannelFactoryInitialize(0, args.net)
    odom = ExternalOdomMonitor(args.topic)
    print("[calib] waiting for fresh odom")
    snap = odom.wait_stable(5.0, args.staleness, 5, 0.5)
    if snap is None:
        raise SystemExit("[calib] no stable odom; abort")
    print(f"[calib] odom fresh x={snap['x']:+.3f} y={snap['y']:+.3f} "
          f"yaw={snap.get('yaw', None)}")

    loco = LocoClient()
    loco.SetTimeout(2.0)
    loco.Init()
    print(f"[calib] loco {_loco_status(loco)}")

    if not args.execute:
        print("[calib] dry-run only. Add --execute to send low-speed commands.")
        return

    try:
        print(f"[calib] settle zero {args.settle:.2f}s")
        _zero(loco)
        time.sleep(args.settle)
        snap = odom.wait_fresh(2.0, args.staleness)
        if snap is None:
            raise RuntimeError("no odom before +x")
        start_x = np.array([snap["x"], snap["y"]], dtype=np.float32)
        print(f"[calib] probe +x speed={args.speed_x:.2f} duration={args.duration:.2f}")
        t_end = time.time() + args.duration
        while time.time() < t_end:
            loco.SetVelocity(float(args.speed_x), 0.0, 0.0, duration=0.12)
            time.sleep(0.05)
        _zero(loco)
        dx = _wait_odom_delta(odom, start_x, args.settle, args.staleness)
        if dx is None:
            raise RuntimeError("no odom after +x")
        print(f"[calib] +x delta_w={_fmt_vec(dx, 3)} norm={np.linalg.norm(dx):.3f}m")

        print(f"[calib] settle zero {args.settle:.2f}s")
        _zero(loco)
        time.sleep(args.settle)
        snap = odom.wait_fresh(2.0, args.staleness)
        if snap is None:
            raise RuntimeError("no odom before +y")
        start_y = np.array([snap["x"], snap["y"]], dtype=np.float32)
        print(f"[calib] probe +y speed={args.speed_y:.2f} duration={args.duration:.2f}")
        t_end = time.time() + args.duration
        while time.time() < t_end:
            loco.SetVelocity(0.0, float(args.speed_y), 0.0, duration=0.12)
            time.sleep(0.05)
        _zero(loco)
        dy = _wait_odom_delta(odom, start_y, args.settle, args.staleness)
        if dy is None:
            raise RuntimeError("no odom after +y")
        print(f"[calib] +y delta_w={_fmt_vec(dy, 3)} norm={np.linalg.norm(dy):.3f}m")

        rot, reason = _fit_probe_return_frame(
            dx, dy, args.min_delta, args.max_axis_cos)
        if rot is None:
            raise SystemExit(f"[calib] FAIL {reason}")
        print(f"[calib] PASS {reason} rot={_fmt_mat2(rot, 4)}")
    finally:
        print("[calib] final zero")
        try:
            _zero(loco)
        except Exception as exc:
            print(f"[calib] final zero failed: {exc}")


if __name__ == "__main__":
    main()
