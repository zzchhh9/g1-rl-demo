#!/usr/bin/env python3
"""Publish a deterministic fake YOLO obstacle on rt/yolo/person.

This is for controller integration tests without a real person/camera:
idle heartbeat -> close obstacle -> slowly disappearing obstacle -> idle.
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

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


def _empty_payload(frame_id: int, track_id: int) -> dict:
    return {
        "frame_id": frame_id,
        "ts_ms": int(time.time() * 1000),
        "n": 0,
        "track_locked": True,
        "locked_track_id": track_id,
        "candidate_n": 0,
        "ignored_n": 0,
        "raw_box_n": 0,
        "depth_reject_n": 0,
        "held": False,
        "held_age": None,
        "source": "fake_yolo_obstacle",
    }


def _detect_payload(frame_id: int,
                    track_id: int,
                    dist: float,
                    bearing_deg: float,
                    conf: float) -> dict:
    bearing_rad = math.radians(bearing_deg)
    return {
        "frame_id": frame_id,
        "ts_ms": int(time.time() * 1000),
        "n": 1,
        "x_fwd": float(dist * math.cos(bearing_rad)),
        "y_left": float(-dist * math.sin(bearing_rad)),
        "z": 0.85,
        "dist": float(dist),
        "bearing": float(bearing_deg),
        "track_id": int(track_id),
        "conf": float(conf),
        "track_locked": True,
        "locked_track_id": int(track_id),
        "candidate_n": 1,
        "ignored_n": 0,
        "raw_box_n": 1,
        "depth_reject_n": 0,
        "held": False,
        "held_age": 0.0,
        "source": "fake_yolo_obstacle",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("net", nargs="?", default="eno1")
    p.add_argument("--topic", default="rt/yolo/person")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--start-delay", type=float, default=15.0,
                   help="Seconds of n=0 heartbeat before the fake obstacle appears.")
    p.add_argument("--hold", type=float, default=2.0,
                   help="Seconds to hold the fake obstacle at --start-dist.")
    p.add_argument("--fade", type=float, default=3.0,
                   help="Seconds to ramp distance from --start-dist to --end-dist.")
    p.add_argument("--start-dist", type=float, default=0.50)
    p.add_argument("--end-dist", type=float, default=1.60)
    p.add_argument("--bearing-deg", type=float, default=0.0,
                   help="YOLO bearing in degrees; 0 means straight ahead.")
    p.add_argument("--track-id", type=int, default=9001)
    p.add_argument("--conf", type=float, default=0.99)
    p.add_argument("--loop", action="store_true",
                   help="Repeat the scenario instead of staying idle after one pass.")
    p.add_argument("--print-every", type=int, default=10)
    args = p.parse_args()

    if args.hz <= 0:
        raise SystemExit("--hz must be > 0")
    if args.end_dist < args.start_dist:
        raise SystemExit("--end-dist must be >= --start-dist")

    ChannelFactoryInitialize(0, args.net)
    pub = ChannelPublisher(args.topic, String_)
    pub.Init()
    msg = std_msgs_msg_dds__String_()

    period = 1.0 / args.hz
    frame_id = 0
    cycle_start = time.time()
    print(
        f"[fake-yolo] publishing {args.topic} on {args.net}: "
        f"idle {args.start_delay:.1f}s -> {args.start_dist:.2f}m "
        f"for {args.hold:.1f}s -> fade to {args.end_dist:.2f}m "
        f"over {args.fade:.1f}s -> idle",
        flush=True,
    )

    while True:
        now = time.time()
        t = now - cycle_start
        frame_id += 1

        detect = False
        dist = args.start_dist
        phase = "idle"
        if t < args.start_delay:
            phase = "pre_idle"
        elif t < args.start_delay + args.hold:
            phase = "hold"
            detect = True
            dist = args.start_dist
        elif t < args.start_delay + args.hold + args.fade:
            phase = "fade"
            detect = True
            a = (t - args.start_delay - args.hold) / max(args.fade, 1e-6)
            a = max(0.0, min(1.0, a))
            dist = args.start_dist + a * (args.end_dist - args.start_dist)
        elif args.loop:
            cycle_start = now
            phase = "pre_idle"
        else:
            phase = "post_idle"

        payload = (
            _detect_payload(frame_id, args.track_id, dist, args.bearing_deg, args.conf)
            if detect else _empty_payload(frame_id, args.track_id)
        )
        payload["phase"] = phase
        msg.data = json.dumps(payload, separators=(",", ":"))
        pub.Write(msg)

        if args.print_every > 0 and frame_id % args.print_every == 0:
            if detect:
                print(f"[fake-yolo] frame={frame_id} phase={phase} dist={dist:.2f}m "
                      f"bearing={args.bearing_deg:+.1f}deg", flush=True)
            else:
                print(f"[fake-yolo] frame={frame_id} phase={phase} n=0", flush=True)

        sleep_s = period - (time.time() - now)
        if sleep_s > 0:
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()
