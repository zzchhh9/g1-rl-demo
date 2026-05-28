#!/usr/bin/env python3
"""Run a sequence of N random fake-YOLO crossings, one trigger at a time.

Each crossing: a "person" walks a random straight line past the robot (enters
from a random bearing front/diagonal/back, passes close, exits the far side).
After the crossing the script watches rt/dodge/odom and waits until the robot
has *moved then settled* (i.e. dodged and finished recovering), then waits
--post-recover-gap seconds and fires the next crossing. Stops after
--max-passes triggers.

Publishes rt/yolo/person and subscribes to rt/dodge/odom. Run it as the sole
YOLO publisher (do not also run fake_yolo_obstacle_dds.py).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the import below

from unitree_sdk2py.core.channel import (  # noqa: E402
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_  # noqa: E402
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_  # noqa: E402
from fake_yolo_obstacle_dds import _detect_payload, _empty_payload  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("net", nargs="?", default="eno1")
    p.add_argument("--topic", default="rt/yolo/person")
    p.add_argument("--odom-topic", default="rt/dodge/odom")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--max-passes", type=int, default=5,
                   help="Number of crossings (triggers) to run, then idle.")
    p.add_argument("--start-delay", type=float, default=15.0,
                   help="Idle heartbeat seconds before the first crossing.")
    p.add_argument("--post-recover-gap", type=float, default=5.0,
                   help="Seconds to wait after the robot settles before the next crossing.")
    # crossing geometry (same model as fake_yolo_obstacle_dds.py --trajectory)
    p.add_argument("--cross-speed", type=float, default=0.3)
    p.add_argument("--cross-len", type=float, default=2.0)
    p.add_argument("--impact-min", type=float, default=-0.30)
    p.add_argument("--impact-max", type=float, default=0.30)
    p.add_argument("--heading-min", type=float, default=0.0)
    p.add_argument("--heading-max", type=float, default=360.0)
    p.add_argument("--detect-range", type=float, default=2.2)
    p.add_argument("--track-id", type=int, default=9001)
    p.add_argument("--conf", type=float, default=0.99)
    # return/settle detection (from odom)
    p.add_argument("--settle-speed", type=float, default=0.06,
                   help="Odom speed (m/s) below which the robot counts as still.")
    p.add_argument("--settle-time", type=float, default=1.5,
                   help="Seconds of low speed (after moving) = recovered.")
    p.add_argument("--moved-speed", type=float, default=0.15,
                   help="Odom speed (m/s) that confirms the dodge actually started.")
    p.add_argument("--return-timeout", type=float, default=25.0,
                   help="Max seconds to wait for settle before forcing the next pass.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--print-every", type=int, default=10)
    args = p.parse_args()

    if args.hz <= 0:
        raise SystemExit("--hz must be > 0")

    ChannelFactoryInitialize(0, args.net)
    pub = ChannelPublisher(args.topic, String_)
    pub.Init()
    msg = std_msgs_msg_dds__String_()

    odom = {"x": 0.0, "y": 0.0, "px": None, "py": None, "pt": 0.0, "speed": 0.0,
            "recv": 0.0}

    def odom_cb(m):
        try:
            d = json.loads(m.data)
        except Exception:
            return
        x = float(d.get("x", 0.0))
        y = float(d.get("y", 0.0))
        now = time.time()
        if odom["px"] is None:
            odom["px"], odom["py"], odom["pt"] = x, y, now
        else:
            dt = now - odom["pt"]
            if dt >= 0.1:  # ~10 Hz speed estimate over a 0.1 s window, EMA-smoothed
                sp = math.hypot(x - odom["px"], y - odom["py"]) / dt
                odom["speed"] = 0.5 * odom["speed"] + 0.5 * sp
                odom["px"], odom["py"], odom["pt"] = x, y, now
        odom["x"], odom["y"], odom["recv"] = x, y, now

    sub = ChannelSubscriber(args.odom_topic, String_)
    sub.Init(odom_cb, 10)

    rng = random.Random(args.seed)
    period = 1.0 / args.hz
    frame_id = 0

    def publish(detect, dist=0.5, bearing=0.0, phase="idle"):
        nonlocal frame_id
        frame_id += 1
        payload = (_detect_payload(frame_id, args.track_id, dist, bearing, args.conf)
                   if detect else _empty_payload(frame_id, args.track_id))
        payload["phase"] = phase
        msg.data = json.dumps(payload, separators=(",", ":"))
        pub.Write(msg)

    def idle(duration, phase):
        t0 = time.time()
        while time.time() - t0 < duration:
            publish(False, phase=phase)
            time.sleep(period)

    def run_pass(i):
        h = math.radians(rng.uniform(args.heading_min, args.heading_max))
        b = rng.uniform(args.impact_min, args.impact_max)
        wx, wy = math.cos(h), math.sin(h)
        px, py = -math.sin(h), math.cos(h)
        ex, ey = b * px - args.cross_len * wx, b * py - args.cross_len * wy
        print(f"[seq] pass {i}/{args.max_passes}: enter_bearing="
              f"{math.degrees(math.atan2(-ey, ex)):+.0f}deg impact={b:+.2f}m "
              f"heading={math.degrees(h):.0f}deg", flush=True)
        t0 = time.time()
        n = 0
        while True:
            s = -args.cross_len + args.cross_speed * (time.time() - t0)
            if s > args.cross_len:
                break
            X, Y = b * px + s * wx, b * py + s * wy
            dist = math.hypot(X, Y)
            bearing = math.degrees(math.atan2(-Y, X))
            detect = dist <= args.detect_range
            publish(detect, dist, bearing, phase="cross" if detect else "approach")
            n += 1
            if args.print_every > 0 and n % args.print_every == 0 and detect:
                print(f"[seq]   dist={dist:.2f}m bearing={bearing:+.0f}deg", flush=True)
            time.sleep(period)

    def wait_recover():
        t0 = time.time()
        moved = False
        settled_since = None
        while time.time() - t0 < args.return_timeout:
            publish(False, phase="recovering")
            sp = odom["speed"]
            if sp >= args.moved_speed:
                moved = True
            if moved and sp < args.settle_speed:
                if settled_since is None:
                    settled_since = time.time()
                elif time.time() - settled_since >= args.settle_time:
                    return True
            elif sp >= args.settle_speed:
                settled_since = None
            time.sleep(period)
        print("[seq] recover-wait timed out; continuing", flush=True)
        return False

    print(f"[seq] {args.max_passes} random crossings on {args.topic} "
          f"(odom={args.odom_topic}); idle {args.start_delay:.0f}s, then "
          f"pass -> wait recover -> +{args.post_recover_gap:.0f}s -> next", flush=True)
    idle(args.start_delay, "pre_idle")
    for i in range(1, args.max_passes + 1):
        run_pass(i)
        publish(False, phase="cleared")
        wait_recover()
        print(f"[seq] pass {i} recovered; gap {args.post_recover_gap:.0f}s", flush=True)
        idle(args.post_recover_gap, "gap")
    print(f"[seq] done: {args.max_passes} passes complete; idling", flush=True)
    while True:
        publish(False, phase="post_idle")
        time.sleep(period)


if __name__ == "__main__":
    main()
