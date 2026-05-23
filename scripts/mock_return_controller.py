#!/usr/bin/env python3
"""Mock checks for return frame math without robot hardware."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy_dodge_sdk_loco import (
    _disp_world_to_return_frame,
    _fit_probe_return_frame,
    _fmt_mat2,
    _fmt_vec,
    _shape_return_velocity,
)


def _world_cmd_from_disp(disp_w, rot, args):
    disp_b, src = _disp_world_to_return_frame(
        np.asarray(disp_w, dtype=np.float32),
        yaw=0.0,
        odom_yaw=0.0,
        frame=None,
        frozen_rot=rot,
        frozen_yaw_ref=None,
    )
    target = _shape_return_velocity(
        -disp_b,
        gain=args.gain,
        lat_gain=args.lat_gain,
        max_vel=args.max_vel,
        max_lat_vel=args.max_lat_vel,
        min_vel=args.min_vel,
        done_dist=args.done_dist,
    )
    cmd_w = rot @ target[:2]
    back_w = -np.asarray(disp_w, dtype=np.float32)
    denom = float(np.linalg.norm(cmd_w) * np.linalg.norm(back_w))
    cos = 0.0 if denom < 1e-6 else float(np.dot(cmd_w, back_w) / denom)
    return disp_b, target, cmd_w, cos, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_vel", type=float, default=0.25)
    ap.add_argument("--max_lat_vel", type=float, default=0.08)
    ap.add_argument("--gain", type=float, default=0.8)
    ap.add_argument("--lat_gain", type=float, default=0.35)
    ap.add_argument("--min_vel", type=float, default=0.12)
    ap.add_argument("--done_dist", type=float, default=0.10)
    ap.add_argument("--min_cos", type=float, default=0.60)
    ap.add_argument("--probe_min_delta", type=float, default=0.080)
    ap.add_argument("--probe_max_axis_cos", type=float, default=0.70)
    args = ap.parse_args()

    cases = [
        ("identity", np.array([0.12, 0.0]), np.array([0.0, 0.10]), True),
        ("rotated_90", np.array([0.0, -0.12]), np.array([0.10, 0.0]), True),
        ("rotated_45", np.array([0.085, 0.085]), np.array([-0.071, 0.071]), True),
        ("mirrored_y", np.array([0.12, 0.0]), np.array([0.0, -0.10]), True),
        ("weak_but_orthogonal_y", np.array([0.12, 0.0]),
         np.array([0.0, 0.06]), False),
        ("contaminated_collinear", np.array([-0.057, -0.106]),
         np.array([0.002, -0.050]), False),
        ("too_small_x", np.array([0.04, 0.0]), np.array([0.0, 0.10]), False),
        ("too_small_y", np.array([0.12, 0.0]), np.array([0.0, 0.04]), False),
    ]
    all_ok = True
    for name, dx, dy, should_fit in cases:
        rot, reason = _fit_probe_return_frame(
            dx, dy,
            min_delta=args.probe_min_delta,
            max_axis_cos=args.probe_max_axis_cos,
        )
        print(f"[mock] case={name} fit={reason} rot={_fmt_mat2(rot, 3)}")
        if rot is None:
            ok = not should_fit
            all_ok &= ok
            print(f"  {'OK rejected' if ok else 'BAD rejected valid probe'}")
            continue
        if not should_fit:
            all_ok = False
            print("  BAD accepted contaminated probe")
            continue
        for disp_w in (np.array([-0.4, -0.8]),
                       np.array([0.3, -0.5]),
                       np.array([-0.8, 0.05])):
            disp_b, target, cmd_w, cos, src = _world_cmd_from_disp(disp_w, rot, args)
            ok = cos > args.min_cos
            all_ok &= ok
            print(f"  disp_w={_fmt_vec(disp_w, 2)} disp_b={_fmt_vec(disp_b, 2)} "
                  f"target={_fmt_vec(target, 2)} cmd_w={_fmt_vec(cmd_w, 2)} "
                  f"cos={cos:+.2f} {'OK' if ok else 'BAD'} src={src}")
    print(f"[mock] verdict={'PASS' if all_ok else 'FAIL'}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
