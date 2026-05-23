#!/usr/bin/env python3
"""Wait until a DDS std_msgs/String JSON topic is fresh."""

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
    p.add_argument("--topic", required=True)
    p.add_argument("--stale", type=float, default=0.5)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--stable-seconds", type=float, default=0.5,
                   help="Require this much continuous fresh data before success.")
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

    start = time.time()
    last_print = 0.0
    stable_start = None
    stable_first_count = None
    print(f"[wait-json] waiting for {args.topic} on {args.net}")
    while time.time() - start < args.timeout:
        time.sleep(0.05)
        data = state["data"]
        age = time.time() - state["recv"] if data is not None else float("inf")
        count = int(state["count"])
        if data is not None and age <= args.stale:
            if stable_start is None:
                stable_start = time.time()
                stable_first_count = count
            seen = count - int(stable_first_count) + 1
            stable_elapsed = time.time() - stable_start
            if seen >= args.min_count and stable_elapsed >= args.stable_seconds:
                print(f"[wait-json] OK topic={args.topic} age={age:.2f}s "
                      f"count={count} stable={stable_elapsed:.2f}s "
                      f"seen={seen} data_keys={sorted(data.keys())[:8]}")
                return
        else:
            stable_start = None
            stable_first_count = None
        now = time.time()
        if now - last_print >= args.print_every:
            last_print = now
            if data is None:
                print(f"[wait-json] no data yet topic={args.topic}")
            else:
                stable_elapsed = 0.0 if stable_start is None else time.time() - stable_start
                seen = 0 if stable_first_count is None else count - int(stable_first_count) + 1
                print(f"[wait-json] waiting topic={args.topic} age={age:.2f}s "
                      f"count={count} stable={stable_elapsed:.2f}s seen={seen}")

    raise SystemExit(f"[wait-json] TIMEOUT topic={args.topic} "
                     f"count={state['count']} timeout={args.timeout:.1f}s")


if __name__ == "__main__":
    main()
