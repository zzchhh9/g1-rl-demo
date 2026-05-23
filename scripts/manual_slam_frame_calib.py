#!/usr/bin/env python3
"""Manual body-frame to SLAM-frame calibration from pushed odom motion.

This script does not command the robot. It only watches rt/dodge/odom while you
physically move the robot forward, then sideways, and writes a return-frame JSON
that deploy_dodge_sdk_loco.py can use instead of an active SDK probe.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from deploy_dodge_sdk_loco import ExternalOdomMonitor, _fmt_mat2, _fmt_vec
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


def _wrap_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _mean_yaw(values: list[float]) -> float | None:
    if not values:
        return None
    s = float(np.mean(np.sin(values)))
    c = float(np.mean(np.cos(values)))
    return float(math.atan2(s, c))


def _collect_pose(odom: ExternalOdomMonitor,
                  label: str,
                  window_s: float,
                  staleness: float,
                  min_count: int) -> dict:
    samples: list[tuple[float, float, float, float | None]] = []
    deadline = time.time() + window_s
    while time.time() < deadline:
        snap = odom.snapshot(staleness)
        if snap.get("fresh"):
            yaw = snap.get("yaw")
            if yaw is not None:
                try:
                    yaw = float(yaw)
                    if not np.isfinite(yaw):
                        yaw = None
                except Exception:
                    yaw = None
            samples.append((
                time.time(),
                float(snap["x"]),
                float(snap["y"]),
                yaw,
            ))
        time.sleep(0.02)

    if len(samples) < min_count:
        raise RuntimeError(
            f"{label}: only {len(samples)} fresh odom samples in "
            f"{window_s:.1f}s; need {min_count}")

    t = np.array([s[0] for s in samples], dtype=np.float64)
    xy = np.array([[s[1], s[2]] for s in samples], dtype=np.float64)
    yaws = [s[3] for s in samples if s[3] is not None]
    first_xy = xy[0]
    last_xy = xy[-1]
    mean_xy = np.mean(xy, axis=0)
    drift = float(np.linalg.norm(last_xy - first_xy))
    pose = {
        "label": label,
        "count": len(samples),
        "span_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "xy": mean_xy.astype(float).tolist(),
        "first_xy": first_xy.astype(float).tolist(),
        "last_xy": last_xy.astype(float).tolist(),
        "window_drift_m": drift,
        "xy_std_m": np.std(xy, axis=0).astype(float).tolist(),
    }
    yaw_mean = _mean_yaw([float(y) for y in yaws])
    if yaw_mean is not None:
        pose["yaw"] = yaw_mean
        pose["yaw_first"] = float(yaws[0])
        pose["yaw_last"] = float(yaws[-1])
        pose["yaw_window_change"] = _wrap_pi(float(yaws[-1]) - float(yaws[0]))
    return pose


def _wait_stationary_pose(odom: ExternalOdomMonitor,
                          label: str,
                          window_s: float,
                          staleness: float,
                          min_count: int,
                          max_drift: float,
                          timeout_s: float) -> dict:
    deadline = time.time() + timeout_s
    last_warn = 0.0
    last_pose = None
    while time.time() < deadline:
        pose = _collect_pose(odom, label, window_s, staleness, min_count)
        last_pose = pose
        drift = float(pose["window_drift_m"])
        if drift <= max_drift:
            return pose
        now = time.time()
        if now - last_warn > 0.5:
            print(f"[manual-calib] waiting for {label} to settle: "
                  f"window_drift={drift:.3f}m > {max_drift:.3f}m")
            last_warn = now
    if last_pose is not None:
        raise RuntimeError(
            f"{label} did not settle: last window_drift="
            f"{last_pose['window_drift_m']:.3f}m > {max_drift:.3f}m")
    raise RuntimeError(f"{label} did not settle before timeout")


def _wait_push_distance(odom: ExternalOdomMonitor,
                        label: str,
                        start_xy: np.ndarray,
                        target_dist: float,
                        staleness: float,
                        timeout_s: float,
                        progress_every_s: float) -> np.ndarray:
    deadline = time.time() + timeout_s
    last_print = 0.0
    best_dist = 0.0
    best_delta = np.zeros(2, dtype=np.float32)
    while time.time() < deadline:
        snap = odom.snapshot(staleness)
        if snap.get("fresh"):
            xy = np.array([snap["x"], snap["y"]], dtype=np.float32)
            delta = xy - start_xy
            dist = float(np.linalg.norm(delta))
            if dist > best_dist:
                best_dist = dist
                best_delta = delta.copy()
            if dist >= target_dist:
                print(f"[manual-calib] {label} reached "
                      f"{dist:.3f}m delta={_fmt_vec(delta, 3)}")
                return xy
            now = time.time()
            if now - last_print >= progress_every_s:
                print(f"[manual-calib] {label} progress "
                      f"{dist:.3f}/{target_dist:.3f}m "
                      f"delta={_fmt_vec(delta, 3)}")
                last_print = now
        time.sleep(0.03)
    raise RuntimeError(
        f"{label} push did not reach {target_dist:.3f}m before timeout; "
        f"best={best_dist:.3f}m delta={_fmt_vec(best_delta, 3)}")


def _pose_xy(pose: dict) -> np.ndarray:
    return np.asarray(pose["xy"], dtype=np.float32).reshape(2)


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _calibrate_frame(start: dict,
                     forward: dict,
                     side: dict,
                     side_name: str,
                     min_axis_delta: float,
                     max_axis_cos: float) -> tuple[np.ndarray, dict]:
    forward_delta = _pose_xy(forward) - _pose_xy(start)
    side_delta = _pose_xy(side) - _pose_xy(forward)
    forward_norm = _norm(forward_delta)
    side_norm = _norm(side_delta)
    if forward_norm < min_axis_delta:
        raise RuntimeError(
            f"forward push too small: {forward_norm:.3f}m "
            f"< {min_axis_delta:.3f}m")
    if side_norm < min_axis_delta:
        raise RuntimeError(
            f"{side_name} push too small: {side_norm:.3f}m "
            f"< {min_axis_delta:.3f}m")

    # Body convention used by the controller: +x forward, +y left. If the
    # operator pushed the robot to its right, that measured axis is body -y.
    x_axis = forward_delta / max(forward_norm, 1e-6)
    body_y_delta = side_delta if side_name == "left" else -side_delta
    raw_y_norm = _norm(body_y_delta)
    axis_cos = float(np.dot(x_axis, body_y_delta / max(raw_y_norm, 1e-6)))
    if abs(axis_cos) > max_axis_cos:
        raise RuntimeError(
            f"forward and {side_name} pushes are too collinear: "
            f"cos={axis_cos:+.2f}, max={max_axis_cos:.2f}")

    y_orth = body_y_delta - float(np.dot(body_y_delta, x_axis)) * x_axis
    y_orth_norm = _norm(y_orth)
    if y_orth_norm < min_axis_delta * 0.7:
        raise RuntimeError(
            f"side push has too little orthogonal motion: "
            f"{y_orth_norm:.3f}m after removing forward component")
    y_axis = y_orth / max(y_orth_norm, 1e-6)

    rot = np.column_stack([x_axis, y_axis]).astype(np.float32)
    det = float(np.linalg.det(rot))
    if det < 0.5:
        raise RuntimeError(
            f"calibrated frame is mirrored det={det:+.2f}; check --side. "
            "Use --side right if you pushed robot-right, --side left if you "
            "pushed robot-left.")

    yaw_values = []
    for pose in (start, forward, side):
        if "yaw" in pose:
            yaw_values.append(float(pose["yaw"]))
    yaw_ref = _mean_yaw(yaw_values)
    yaw_change = None
    if len(yaw_values) >= 2:
        yaw_change = max(abs(_wrap_pi(y - yaw_values[0])) for y in yaw_values[1:])

    meta = {
        "forward_delta_w": forward_delta.astype(float).tolist(),
        "side_delta_w": side_delta.astype(float).tolist(),
        "body_left_delta_w": body_y_delta.astype(float).tolist(),
        "forward_norm_m": forward_norm,
        "side_norm_m": side_norm,
        "body_left_orth_norm_m": y_orth_norm,
        "axis_cos": axis_cos,
        "det": det,
        "yaw_ref": yaw_ref,
        "yaw_change_rad": yaw_change,
    }
    return rot, meta


def _fail(message: str, force: bool) -> None:
    if force:
        print(f"[manual-calib] WARN {message}")
    else:
        raise RuntimeError(message)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Manually calibrate G1 body +x/+y axes in SLAM/world frame.")
    p.add_argument("net", nargs="?", default="eno1")
    p.add_argument("--topic", default="rt/dodge/odom")
    p.add_argument("--staleness", type=float, default=0.35)
    p.add_argument("--sample-window", type=float, default=0.80,
                   help="Seconds to average each marked stationary pose.")
    p.add_argument("--min-samples", type=int, default=5)
    p.add_argument("--side", choices=["right", "left"], default="right",
                   help="Which side you will push after the forward push.")
    p.add_argument("--push-distance", type=float, default=0.60,
                   help="Distance the script waits for on each manual push.")
    p.add_argument("--push-timeout", type=float, default=25.0,
                   help="Seconds allowed for each manual push.")
    p.add_argument("--settle-timeout", type=float, default=12.0,
                   help="Seconds allowed for SLAM to settle after each push.")
    p.add_argument("--progress-every", type=float, default=0.5,
                   help="Progress print interval while waiting for a push.")
    p.add_argument("--min-axis-delta", type=float, default=0.30,
                   help="Minimum accepted forward/side push distance.")
    p.add_argument("--max-axis-cos", type=float, default=0.35,
                   help="Reject if forward and side pushes are not close to perpendicular.")
    p.add_argument("--max-capture-drift", type=float, default=0.05,
                   help="Reject a marked pose if SLAM moves this much while averaging it.")
    p.add_argument("--max-yaw-change-deg", type=float, default=12.0,
                   help="Reject if yaw changes this much across manual pushes.")
    p.add_argument("--drift-window", type=float, default=2.0,
                   help="Final stationary SLAM drift check duration.")
    p.add_argument("--max-stationary-drift", type=float, default=0.03,
                   help="Reject if final still pose drifts this much over --drift-window.")
    p.add_argument("--out", default=str(_REPO / "configs" / "manual_return_frame_calib.json"))
    p.add_argument("--force", action="store_true",
                   help="Write JSON despite validation warnings.")
    args = p.parse_args()

    print("[manual-calib] This script only reads SLAM odom; it sends no SDK commands.")
    print("[manual-calib] Keep deploy/controller stopped. Push the robot slowly, "
          "without yawing it.")
    print("[manual-calib] The output JSON can be reused by deploy. It stores the "
          "body/SLAM frame plus yaw_ref; deploy rotates it with live SLAM yaw.")
    print("[manual-calib] Body convention: +x=forward, +y=left. "
          f"You selected side push: robot-{args.side}.")

    ChannelFactoryInitialize(0, args.net)
    odom = ExternalOdomMonitor(args.topic)
    print("[manual-calib] waiting for stable odom ...")
    snap = odom.wait_stable(8.0, args.staleness, args.min_samples, 0.8)
    if snap is None:
        raise SystemExit("[manual-calib] no stable odom; start SLAM/odom bridge first")
    print(f"[manual-calib] odom ready x={float(snap['x']):+.3f} "
          f"y={float(snap['y']):+.3f} yaw={snap.get('yaw')}")

    print("\n[manual-calib] Keep the robot still. Recording start pose now ...")
    start = _wait_stationary_pose(
        odom, "start", args.sample_window, args.staleness,
        args.min_samples, args.max_capture_drift, args.settle_timeout)
    print(f"[manual-calib] start xy={_fmt_vec(_pose_xy(start), 3)} "
          f"window_drift={start['window_drift_m']:.3f}m")

    input(f"\n1/2 Press Enter, then push the robot straight FORWARD "
          f"{args.push_distance:.2f}m. I will detect it automatically.")
    _wait_push_distance(
        odom, "forward push", _pose_xy(start), args.push_distance,
        args.staleness, args.push_timeout, args.progress_every)
    print("[manual-calib] forward push detected; keep robot still, "
          "recording forward pose ...")
    forward = _wait_stationary_pose(
        odom, "forward", args.sample_window, args.staleness,
        args.min_samples, args.max_capture_drift, args.settle_timeout)
    print(f"[manual-calib] forward xy={_fmt_vec(_pose_xy(forward), 3)} "
          f"window_drift={forward['window_drift_m']:.3f}m")

    input(f"\n2/2 Forward axis captured. Press Enter, then push the robot "
          f"straight to its {args.side.upper()} {args.push_distance:.2f}m. "
          "I will detect it automatically.")
    _wait_push_distance(
        odom, f"{args.side} push", _pose_xy(forward), args.push_distance,
        args.staleness, args.push_timeout, args.progress_every)
    print(f"[manual-calib] {args.side} push detected; keep robot still, "
          f"recording {args.side} pose ...")
    side = _wait_stationary_pose(
        odom, args.side, args.sample_window, args.staleness,
        args.min_samples, args.max_capture_drift, args.settle_timeout)
    print(f"[manual-calib] {args.side} xy={_fmt_vec(_pose_xy(side), 3)} "
          f"window_drift={side['window_drift_m']:.3f}m")

    for pose in (start, forward, side):
        if float(pose["window_drift_m"]) > args.max_capture_drift:
            _fail(f"{pose['label']} pose drifted {pose['window_drift_m']:.3f}m "
                  f"during capture; SLAM/robot was not stationary", args.force)

    rot, meta = _calibrate_frame(start, forward, side, args.side,
                                 args.min_axis_delta, args.max_axis_cos)

    max_yaw_change = math.radians(args.max_yaw_change_deg)
    yaw_change = meta.get("yaw_change_rad")
    if yaw_change is not None and yaw_change > max_yaw_change:
        _fail(f"yaw changed {math.degrees(yaw_change):.1f}deg during pushes; "
              "push again with less rotation", args.force)

    print(f"[manual-calib] frame rot={_fmt_mat2(rot, 4)} "
          f"det={meta['det']:+.2f} axis_cos={meta['axis_cos']:+.2f}")
    print(f"[manual-calib] deltas forward={_fmt_vec(meta['forward_delta_w'], 3)} "
          f"{args.side}={_fmt_vec(meta['side_delta_w'], 3)}")

    print(f"\n[manual-calib] Calibration geometry is valid. Keep the robot "
          f"still for the final {args.drift_window:.1f}s drift check ...")
    still = _collect_pose(odom, "final_still", args.drift_window,
                          args.staleness, args.min_samples)
    print(f"[manual-calib] final still drift={still['window_drift_m']:.3f}m "
          f"over {still['span_s']:.2f}s")
    if float(still["window_drift_m"]) > args.max_stationary_drift:
        _fail(f"final stationary SLAM drift {still['window_drift_m']:.3f}m "
              f"> {args.max_stationary_drift:.3f}m; do not trust recover yet",
              args.force)

    out = Path(args.out).expanduser().resolve()
    data = {
        "type": "manual_body_slam_return_frame",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "net": args.net,
        "topic": args.topic,
        "body_convention": "+x forward, +y left",
        "side_push": args.side,
        "rot_cmd_to_world": rot.astype(float).tolist(),
        "yaw_ref": meta.get("yaw_ref"),
        "validation": {
            "min_axis_delta": args.min_axis_delta,
            "max_axis_cos": args.max_axis_cos,
            "max_capture_drift": args.max_capture_drift,
            "max_stationary_drift": args.max_stationary_drift,
            **meta,
            "final_stationary_drift_m": still["window_drift_m"],
        },
        "poses": {
            "start": start,
            "forward": forward,
            args.side: side,
            "final_still": still,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    print(f"[manual-calib] wrote {out}")
    print("[manual-calib] immediate deploy with the same running SLAM session:")
    print("  AUTO_START_SENSORS_BEFORE_DEPLOY=0 "
          f"RETURN_FRAME_CALIB={out} ./deploy.sh")
    print("[manual-calib] later deploy can also load the same file:")
    print(f"  RETURN_FRAME_CALIB={out} ./deploy.sh")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("\n[manual-calib] interrupted")
    except RuntimeError as exc:
        raise SystemExit(f"[manual-calib] FAIL {exc}")
