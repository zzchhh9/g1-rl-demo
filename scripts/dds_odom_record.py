#!/usr/bin/env python3
"""Record DDS odometry JSON to JSONL."""

from __future__ import annotations

import argparse
import json
import signal
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
    p.add_argument("--out", required=True)
    p.add_argument("--print-every", type=int, default=25)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stop = {"value": False}

    def on_signal(_sig, _frame):
        stop["value"] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    ChannelFactoryInitialize(0, args.net)
    state = {"count": 0, "last": None}
    f = out_path.open("a", buffering=1)

    def cb(msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        rec = {
            "t": time.time(),
            "x": float(data.get("x", 0.0)),
            "y": float(data.get("y", 0.0)),
            "yaw": float(data.get("yaw", 0.0)),
            "source": data.get("source", "?"),
            "raw": data,
        }
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        state["count"] += 1
        state["last"] = rec
        if args.print_every > 0 and state["count"] % args.print_every == 0:
            print(f"[record] n={state['count']} "
                  f"x={rec['x']:+.3f} y={rec['y']:+.3f} yaw={rec['yaw']:+.3f}",
                  flush=True)

    sub = ChannelSubscriber(args.topic, String_)
    sub.Init(cb, 10)
    print(f"[record] writing {args.topic} on {args.net} -> {out_path}", flush=True)

    try:
        while not stop["value"]:
            time.sleep(0.1)
    finally:
        f.flush()
        f.close()
        print(f"[record] stopped n={state['count']} out={out_path}", flush=True)


if __name__ == "__main__":
    main()
