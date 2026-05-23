#!/usr/bin/env python3
"""Measure loop-closure drift from DDS odometry.

Start this after LiDAR/LIO odometry is publishing, move the robot through the
test path, return it to the same floor mark and heading, then press Ctrl-C.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "third_party" / "unitree_sdk2_python"))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("net", nargs="?", default="eno1")
    p.add_argument("--topic", default="rt/dodge/odom")
    p.add_argument("--stale", type=float, default=0.35)
    p.add_argument("--wait-timeout", type=float, default=20.0)
    p.add_argument("--duration", type=float, default=0.0,
                   help="Optional auto-stop duration in seconds. Default: run until Ctrl-C.")
    p.add_argument("--print-every", type=float, default=1.0)
    args = p.parse_args()

    ChannelFactoryInitialize(0, args.net)
    state = {"data": None, "recv": 0.0, "count": 0}

    def cb(msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        state["data"] = data
        state["recv"] = time.time()
        state["count"] += 1

    sub = ChannelSubscriber(args.topic, String_)
    sub.Init(cb, 10)
    print(f"[loop] waiting for fresh odom on {args.topic} via {args.net}")

    t0 = time.time()
    while True:
        time.sleep(0.05)
        data = state["data"]
        age = time.time() - state["recv"] if data is not None else float("inf")
        if data is not None and age <= args.stale:
            break
        if time.time() - t0 > args.wait_timeout:
            raise SystemExit("[loop] no fresh odom; check LiDAR/LIO/UDP/DDS bridge")

    start = dict(state["data"])
    start_t = time.time()
    sx = float(start.get("x", 0.0))
    sy = float(start.get("y", 0.0))
    syaw = float(start.get("yaw", 0.0))
    max_dist = 0.0
    last_print = 0.0
    final = start

    print(f"[loop] start x={sx:+.3f} y={sy:+.3f} yaw={syaw:+.3f} src={start.get('source', '?')}")
    print("[loop] move the robot, return to the same floor mark and heading, then press Ctrl-C")

    try:
        while args.duration <= 0.0 or time.time() - start_t < args.duration:
            time.sleep(0.05)
            data = state["data"]
            if data is None:
                continue
            final = dict(data)
            x = float(data.get("x", 0.0))
            y = float(data.get("y", 0.0))
            yaw = float(data.get("yaw", 0.0))
            dx = x - sx
            dy = y - sy
            dist = math.hypot(dx, dy)
            max_dist = max(max_dist, dist)
            now = time.time()
            if now - last_print >= args.print_every:
                last_print = now
                age = now - state["recv"]
                dyaw = wrap_pi(yaw - syaw)
                status = "fresh" if age <= args.stale else "STALE"
                print(f"[loop] {status} dx={dx:+.3f} dy={dy:+.3f} "
                      f"dist={dist:.3f} dyaw={dyaw:+.3f} max={max_dist:.3f}")
    except KeyboardInterrupt:
        pass

    fx = float(final.get("x", 0.0))
    fy = float(final.get("y", 0.0))
    fyaw = float(final.get("yaw", 0.0))
    dx = fx - sx
    dy = fy - sy
    dist = math.hypot(dx, dy)
    dyaw = wrap_pi(fyaw - syaw)
    print(f"[loop] final x={fx:+.3f} y={fy:+.3f} yaw={fyaw:+.3f}")
    print(f"[loop] loop_error dx={dx:+.3f} dy={dy:+.3f} "
          f"dist={dist:.3f} dyaw={dyaw:+.3f} max_dist={max_dist:.3f}")


if __name__ == "__main__":
    main()
