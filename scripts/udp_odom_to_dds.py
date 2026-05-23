#!/usr/bin/env python3
"""Forward UDP odometry JSON to Unitree DDS std_msgs/String."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "third_party" / "unitree_sdk2_python"))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--udp-port", type=int, default=5070)
    p.add_argument("--net", default="eno1")
    p.add_argument("--dds-domain", type=int, default=0)
    p.add_argument("--dds-topic", default="rt/dodge/odom")
    p.add_argument("--print-every", type=int, default=20)
    args = p.parse_args()

    ChannelFactoryInitialize(args.dds_domain, args.net)
    pub = ChannelPublisher(args.dds_topic, String_)
    pub.Init()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.udp_port))
    print(f"[udp-dds] UDP {args.bind}:{args.udp_port} -> DDS {args.dds_topic} on {args.net}")

    count = 0
    last_print = time.time()
    while True:
        data, addr = sock.recvfrom(65535)
        try:
            payload = json.loads(data.decode())
        except Exception:
            continue
        payload["bridge_recv_time"] = time.time()
        msg = std_msgs_msg_dds__String_()
        msg.data = json.dumps(payload, separators=(",", ":"))
        pub.Write(msg)
        count += 1
        if args.print_every > 0 and count % args.print_every == 0:
            now = time.time()
            hz = args.print_every / max(now - last_print, 1e-6)
            last_print = now
            print(f"[udp-dds] n={count} {hz:.1f}Hz from={addr[0]} "
                  f"x={float(payload.get('x', 0.0)):+.3f} "
                  f"y={float(payload.get('y', 0.0)):+.3f}")


if __name__ == "__main__":
    main()
