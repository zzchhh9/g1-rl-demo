#!/usr/bin/env python3
"""Send repeated high-level zero-velocity commands to a Unitree G1.

This is a software stop for the built-in high-level locomotion controller
(`LocoClient.SetVelocity`). It is not a replacement for physical E-stop/power
cutoff. Use it from a separate terminal when an SDK high-level velocity command
keeps executing after the main script exits.
"""

from __future__ import annotations

import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("net", nargs="?", default="eno1",
                        help="DDS network interface, e.g. eno1")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="How long to keep sending zero commands.")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Zero-command refresh rate. Keep modest; RPC can overload.")
    parser.add_argument("--duration", type=float, default=0.5,
                        help="Duration field for each zero SetVelocity command.")
    parser.add_argument("--timeout", type=float, default=0.5,
                        help="RPC timeout seconds.")
    args = parser.parse_args()

    hz = max(1.0, float(args.hz))
    period = 1.0 / hz
    seconds = max(0.1, float(args.seconds))

    ChannelFactoryInitialize(0, args.net)
    loco = LocoClient()
    loco.SetTimeout(float(args.timeout))
    loco.Init()

    print(f"[G1-STOP] SetVelocity(0,0,0,duration={args.duration}) "
          f"for {seconds:.1f}s @ {hz:.1f}Hz on {args.net}")
    deadline = time.time() + seconds
    n_ok = 0
    n_fail = 0
    n = 0
    while time.time() < deadline:
        t0 = time.time()
        try:
            code = loco.SetVelocity(0.0, 0.0, 0.0, duration=float(args.duration))
            if code == 0:
                n_ok += 1
            else:
                n_fail += 1
            print(f"[G1-STOP] n={n} code={code}", flush=True)
        except Exception as e:
            n_fail += 1
            print(f"[G1-STOP] n={n} failed: {e}", flush=True)
        n += 1
        sleep_s = period - (time.time() - t0)
        if sleep_s > 0:
            time.sleep(sleep_s)

    print(f"[G1-STOP] done ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()
