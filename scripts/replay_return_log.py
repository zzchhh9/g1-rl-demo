#!/usr/bin/env python3
"""Replay/analyze deploy return logs without touching the robot."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


_VEC_RE = re.compile(r"\[([+-]?\d+(?:\.\d+)?(?:,[+-]?\d+(?:\.\d+)?)+)\]")


def _parse_vec(label: str, line: str):
    m = re.search(rf"{re.escape(label)}=(\[[^\]]+\]|None)", line)
    if not m or m.group(1) == "None":
        return None
    return np.array([float(x) for x in m.group(1).strip("[]").split(",")],
                    dtype=np.float32)


def _parse_float(label: str, line: str):
    m = re.search(rf"{re.escape(label)}=([+-]?\d+(?:\.\d+)?)", line)
    return None if not m else float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--bad-cos", type=float, default=-0.30)
    ap.add_argument("--good-cos", type=float, default=0.30)
    args = ap.parse_args()

    lines = args.log.read_text(errors="replace").splitlines()
    returns = []
    starts = []
    probes = []
    aborts = []
    gates = []
    current_cmd = None
    current_state_line = None

    for lineno, line in enumerate(lines, 1):
        if "[DODGE GATE]" in line:
            gates.append((lineno, line.strip()))
        if "[DODGE START]" in line:
            starts.append((lineno, line.strip()))
        if "[RETURN PROBE]" in line or "[RETURN FRAME]" in line:
            probes.append((lineno, line.strip()))
        if "[RETURN ABORT]" in line or "[RETURN NO PROGRESS]" in line:
            aborts.append((lineno, line.strip()))
        if line.startswith("[RETURN]"):
            current_state_line = lineno
            current_cmd = _parse_vec("cmd", line)
        if "[DBG-RETURN]" in line:
            cmd_odom = _parse_vec("cmd_odom", line)
            target_odom = _parse_vec("target_odom", line)
            back_w = _parse_vec("back_w", line)
            disp_w = _parse_vec("disp_w", line)
            dodom_w = _parse_vec("dodom_w", line)
            returns.append({
                "line": lineno,
                "state_line": current_state_line,
                "cmd": current_cmd,
                "disp_w": disp_w,
                "back_w": back_w,
                "cmd_odom": cmd_odom,
                "target_odom": target_odom,
                "dodom_w": dodom_w,
                "prog_cos": _parse_float("prog_cos", line),
                "actual_dot": _parse_float("actual_dot", line),
                "actual_cos": _parse_float("actual_cos", line),
                "raw": line.strip(),
            })

    predicted_wrong = []
    actual_away = []
    for r in returns:
        cmd = r["cmd"]
        if cmd is None or np.linalg.norm(cmd[:2]) < 0.03:
            continue
        pc = r["prog_cos"]
        if pc is not None and pc < args.good_cos:
            predicted_wrong.append(r)
        ac = r["actual_cos"]
        ad = r["actual_dot"]
        if ac is not None and ad is not None and ac < args.bad_cos and ad < -0.005:
            actual_away.append(r)

    print(f"[replay] log={args.log}")
    print(f"[replay] dodge_starts={len(starts)} gates={len(gates)} "
          f"return_debug={len(returns)} probes={len(probes)} aborts={len(aborts)}")
    if starts:
        print("[replay] first dodge starts:")
        for lineno, text in starts[:5]:
            print(f"  L{lineno}: {text}")
    if gates:
        print("[replay] gates:")
        for lineno, text in gates[:5]:
            print(f"  L{lineno}: {text}")
    if probes:
        print("[replay] probe/frame events:")
        for lineno, text in probes[:10]:
            print(f"  L{lineno}: {text}")
    print(f"[replay] predicted_wrong={len(predicted_wrong)} "
          f"actual_away={len(actual_away)}")
    if predicted_wrong:
        print("[replay] predicted wrong samples:")
        for r in predicted_wrong[:5]:
            print(f"  L{r['line']}: cmd={r['cmd']} prog_cos={r['prog_cos']} "
                  f"back_w={r['back_w']} cmd_odom={r['cmd_odom']}")
    if actual_away:
        print("[replay] actual moved away samples:")
        for r in actual_away[:8]:
            print(f"  L{r['line']}: cmd={r['cmd']} actual_dot={r['actual_dot']:+.3f} "
                  f"actual_cos={r['actual_cos']:+.2f} dodom_w={r['dodom_w']}")
    if aborts:
        print("[replay] abort/no-progress events:")
        for lineno, text in aborts[:8]:
            print(f"  L{lineno}: {text}")

    if actual_away and not predicted_wrong:
        print("[replay] verdict=controller math pointed toward origin, "
              "but measured motion moved away.")
    elif predicted_wrong:
        print("[replay] verdict=controller command direction can be wrong in this log.")
    else:
        print("[replay] verdict=no bad return samples found.")


if __name__ == "__main__":
    main()
