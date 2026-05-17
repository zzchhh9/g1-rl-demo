#!/usr/bin/env python3
"""Software L2+B damping watchdog for Unitree G1.

This listens to rt/lowstate, detects remote L2+B held for a configurable time,
then sends repeated zero velocity commands followed by LocoClient.Damp().

It is a convenience layer for development. It does not replace the robot's
firmware-level remote shortcut or physical power cutoff.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "third_party" / "unitree_rl_gym" / "deploy" / "deploy_real"))

from common.remote_controller import KeyMap, RemoteController
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG


class RemoteMonitor:
    def __init__(self):
        self.remote = RemoteController()
        self.seen = False
        self.sub = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.sub.Init(self._callback, 10)

    def _callback(self, msg):
        self.remote.set(msg.wireless_remote)
        self.seen = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("net", nargs="?", default="eno1")
    parser.add_argument("--hold", type=float, default=2.0,
                        help="Seconds L2+B must be held before damping.")
    parser.add_argument("--zero_count", type=int, default=10,
                        help="Zero velocity commands to send before Damp().")
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.net)
    mon = RemoteMonitor()
    loco = LocoClient()
    loco.SetTimeout(1.0)
    loco.Init()

    print(f"[L2B-DAMP] running on {args.net}; hold L2+B for {args.hold:.1f}s")
    held_since = None
    last_wait_print = 0.0
    while True:
        now = time.time()
        if not mon.seen:
            if now - last_wait_print > 2.0:
                print("[L2B-DAMP] waiting for rt/lowstate ...")
                last_wait_print = now
            time.sleep(0.02)
            continue

        pressed = (mon.remote.button[KeyMap.L2] == 1
                   and mon.remote.button[KeyMap.B] == 1)
        if pressed:
            if held_since is None:
                held_since = now
                print("[L2B-DAMP] L2+B detected")
            elif now - held_since >= args.hold:
                print("[L2B-DAMP] hold reached; sending zero velocity + Damp()")
                for i in range(max(0, int(args.zero_count))):
                    code = loco.SetVelocity(0.0, 0.0, 0.0, duration=0.05)
                    print(f"[L2B-DAMP] zero {i} code={code}")
                    time.sleep(0.02)
                code = loco.Damp()
                print(f"[L2B-DAMP] Damp code={code}")
                return
        else:
            held_since = None
        time.sleep(0.02)


if __name__ == "__main__":
    main()
