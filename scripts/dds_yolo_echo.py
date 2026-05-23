#!/usr/bin/env python3
"""Print YOLO person DDS messages consumed by deploy_dodge_sdk_loco.py."""

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


def _fmt(v, default="N/A"):
    if v is None:
        return default
    try:
        return f"{float(v):+.2f}"
    except Exception:
        return str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("net", nargs="?", default="eno1")
    p.add_argument("--topic", default="rt/yolo/person")
    p.add_argument("--stale", type=float, default=0.5)
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
            print("[echo] no yolo")
            continue
        status = "fresh" if age <= args.stale else "STALE"
        print(
            f"[echo] {status} age={age:.2f}s hz={hz:.1f} n={data.get('n', 0)} "
            f"dist={_fmt(data.get('dist'))} "
            f"x={_fmt(data.get('x_fwd'))} y={_fmt(data.get('y_left'))} "
            f"bear={_fmt(data.get('bearing'))} "
            f"track={data.get('track_id', -1)} "
            f"held={data.get('held', False)} "
            f"boxes={data.get('raw_box_n', '?')} "
            f"cand={data.get('candidate_n', '?')} "
            f"depth_rej={data.get('depth_reject_n', '?')} "
            f"locked={data.get('track_locked', False)} "
            f"locked_id={data.get('locked_track_id', None)}"
        )


if __name__ == "__main__":
    main()
