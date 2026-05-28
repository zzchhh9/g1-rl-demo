#!/usr/bin/env python3
"""Publish a deterministic fake YOLO obstacle on rt/yolo/person.

This is for controller integration tests without a real person/camera:
idle heartbeat -> close obstacle -> slowly disappearing obstacle -> idle.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
        "ready": True,
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
        "ready": True,
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
    p.add_argument("--trajectory", action="store_true",
                   help="Randomized straight-line crossing instead of the fixed radial "
                        "approach: the person enters from a random bearing "
                        "(front/diagonal/back), walks slowly past the robot at a random "
                        "impact offset, then exits. Distance+bearing evolve smoothly in "
                        "the robot's view.")
    p.add_argument("--cross-speed", type=float, default=0.3,
                   help="Person walking speed (m/s) in --trajectory mode.")
    p.add_argument("--cross-len", type=float, default=2.0,
                   help="Half-length of the crossing path (m): enter/exit at ~this range.")
    p.add_argument("--impact-min", type=float, default=-0.30,
                   help="Min signed closest-approach offset (m) in --trajectory mode.")
    p.add_argument("--impact-max", type=float, default=0.30,
                   help="Max signed closest-approach offset (m).")
    p.add_argument("--heading-min", type=float, default=0.0,
                   help="Min walk heading (deg), sampled uniformly per pass.")
    p.add_argument("--heading-max", type=float, default=360.0,
                   help="Max walk heading (deg). Full range gives front/diagonal/back.")
    p.add_argument("--detect-range", type=float, default=2.2,
                   help="Publish a detection only within this range (m); heartbeat beyond.")
    p.add_argument("--gap", type=float, default=6.0,
                   help="Idle seconds between crossings in --loop --trajectory mode.")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for reproducible random crossings.")
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
    rng = random.Random(args.seed)
    # --trajectory per-pass state (robot at origin, +x fwd, +y left)
    traj = {"active": False, "done": False, "t0": 0.0, "next_t": 0.0,
            "b": 0.0, "wx": 1.0, "wy": 0.0, "px": 0.0, "py": 1.0}

    def _start_pass(now_):
        h = math.radians(rng.uniform(args.heading_min, args.heading_max))
        b = rng.uniform(args.impact_min, args.impact_max)
        traj.update(active=True, t0=now_, b=b,
                    wx=math.cos(h), wy=math.sin(h),
                    px=-math.sin(h), py=math.cos(h))
        ex = b * traj["px"] - args.cross_len * traj["wx"]
        ey = b * traj["py"] - args.cross_len * traj["wy"]
        print(f"[fake-yolo] new pass: enter_bearing="
              f"{math.degrees(math.atan2(-ey, ex)):+.0f}deg impact={b:+.2f}m "
              f"heading={math.degrees(h):.0f}deg", flush=True)

    if args.trajectory:
        print(f"[fake-yolo] publishing {args.topic} on {args.net}: trajectory mode "
              f"(random crossings @ {args.cross_speed:.2f}m/s, "
              f"impact[{args.impact_min:+.2f},{args.impact_max:+.2f}]m, "
              f"detect<{args.detect_range:.1f}m), idle {args.start_delay:.1f}s first",
              flush=True)
    else:
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
        bearing_deg = args.bearing_deg
        phase = "idle"

        if args.trajectory:
            if t < args.start_delay:
                phase = "pre_idle"
            elif traj["done"]:
                phase = "post_idle"
            else:
                if not traj["active"] and now >= traj["next_t"]:
                    _start_pass(now)
                if traj["active"]:
                    s = -args.cross_len + args.cross_speed * (now - traj["t0"])
                    if s > args.cross_len:
                        traj["active"] = False
                        if args.loop:
                            traj["next_t"] = now + args.gap
                            phase = "between"
                        else:
                            traj["done"] = True
                            phase = "post_idle"
                    else:
                        px = traj["b"] * traj["px"] + s * traj["wx"]
                        py = traj["b"] * traj["py"] + s * traj["wy"]
                        dist = math.hypot(px, py)
                        bearing_deg = math.degrees(math.atan2(-py, px))
                        detect = dist <= args.detect_range
                        phase = "cross" if detect else "approach"
                else:
                    phase = "between"
        elif t < args.start_delay:
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
            _detect_payload(frame_id, args.track_id, dist, bearing_deg, args.conf)
            if detect else _empty_payload(frame_id, args.track_id)
        )
        payload["phase"] = phase
        msg.data = json.dumps(payload, separators=(",", ":"))
        pub.Write(msg)

        if args.print_every > 0 and frame_id % args.print_every == 0:
            if detect:
                print(f"[fake-yolo] frame={frame_id} phase={phase} dist={dist:.2f}m "
                      f"bearing={bearing_deg:+.1f}deg", flush=True)
            else:
                print(f"[fake-yolo] frame={frame_id} phase={phase} n=0", flush=True)

        sleep_s = period - (time.time() - now)
        if sleep_s > 0:
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()
