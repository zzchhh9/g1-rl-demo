#!/usr/bin/env python3
"""Print DDS odometry JSON consumed by deploy_dodge_sdk_loco.py."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "third_party" / "unitree_sdk2_python"))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


def main():
    p = argparse.ArgumentParser()
    p.add_argument("net", nargs="?", default="eno1")
    p.add_argument("--topic", default="rt/dodge/odom")
    p.add_argument("--stale", type=float, default=0.35)
    p.add_argument("--duration", type=float, default=0.0)
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
    print(f"[echo] listening to {args.topic} on {args.net}")

    start = time.time()
    last_count = 0
    last_t = start
    while args.duration <= 0 or time.time() - start < args.duration:
        time.sleep(1.0)
        data = state["data"]
        age = time.time() - state["recv"] if data is not None else float("inf")
        count = int(state["count"])
        hz = (count - last_count) / max(time.time() - last_t, 1e-6)
        last_count = count
        last_t = time.time()
        if data is None:
            print("[echo] no odom")
            continue
        status = "fresh" if age <= args.stale else "STALE"
        print(f"[echo] {status} age={age:.2f}s hz={hz:.1f} n={count} "
              f"x={float(data.get('x', 0.0)):+.3f} "
              f"y={float(data.get('y', 0.0)):+.3f} "
              f"yaw={float(data.get('yaw', 0.0)):+.3f} "
              f"src={data.get('source', '?')}")


if __name__ == "__main__":
    main()
