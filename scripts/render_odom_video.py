#!/usr/bin/env python3
"""Render an odometry JSONL path recording to MP4."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def load_records(path: Path):
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append({
                    "t": float(rec.get("t", 0.0)),
                    "x": float(rec.get("x", 0.0)),
                    "y": float(rec.get("y", 0.0)),
                    "yaw": float(rec.get("yaw", 0.0)),
                    "source": rec.get("source", "?"),
                })
            except Exception:
                continue
    return records


def world_to_px(x: float, y: float, cx: float, cy: float, scale: float, w: int, h: int):
    return int(round(w / 2 + (x - cx) * scale)), int(round(h / 2 - (y - cy) * scale))


def draw_text(img, text: str, xy, scale=0.55, color=(235, 235, 235), thickness=1):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_robot(img, px: int, py: int, yaw: float, color=(120, 255, 150)):
    pts = np.array([[16, 0], [-10, -7], [-10, 7]], dtype=np.float32)
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    out = pts @ rot.T + np.array([px, py], dtype=np.float32)
    cv2.fillConvexPoly(img, out.astype(np.int32), color)
    cv2.polylines(img, [out.astype(np.int32)], True, (20, 80, 40), 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--trail-seconds", type=float, default=0.0,
                   help="0 means draw the full path.")
    p.add_argument("--title", default="RHEA Dodge/Return SLAM Trace")
    args = p.parse_args()

    records = load_records(Path(args.input))
    if len(records) < 2:
        raise SystemExit(f"not enough odom records in {args.input}: {len(records)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_path = out_path
    if out_path.suffix.lower() == ".mp4" and shutil.which("ffmpeg"):
        write_path = out_path.with_name(out_path.stem + ".tmp_mp4v.mp4")
        if write_path.exists():
            write_path.unlink()

    xs = np.array([r["x"] for r in records], dtype=np.float32)
    ys = np.array([r["y"] for r in records], dtype=np.float32)
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    span_x = max(max_x - min_x, 0.5)
    span_y = max(max_y - min_y, 0.5)
    pad = 0.35
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    scale = min((args.width * 0.76) / (span_x + 2 * pad),
                (args.height * 0.70) / (span_y + 2 * pad))
    scale = min(scale, 260.0)

    start = records[0]
    end = records[-1]
    total_t = max(records[-1]["t"] - records[0]["t"], 1e-3)
    frame_count = max(2, int(math.ceil(total_t * args.fps)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(write_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise SystemExit(f"failed to open video writer: {write_path}")

    idx = 0
    t0 = records[0]["t"]
    for frame_i in range(frame_count):
        t = t0 + frame_i / args.fps
        while idx + 1 < len(records) and records[idx + 1]["t"] <= t:
            idx += 1
        current = records[idx]
        img = np.full((args.height, args.width, 3), 18, dtype=np.uint8)

        # Grid, 0.25 m minor and 1 m major.
        xmin = cx - args.width / (2 * scale)
        xmax = cx + args.width / (2 * scale)
        ymin = cy - args.height / (2 * scale)
        ymax = cy + args.height / (2 * scale)
        for gx in np.arange(math.floor(xmin / 0.25) * 0.25, xmax + 0.25, 0.25):
            px, _ = world_to_px(gx, 0, cx, cy, scale, args.width, args.height)
            color = (48, 48, 48) if abs((gx / 1.0) - round(gx / 1.0)) > 1e-4 else (72, 72, 72)
            cv2.line(img, (px, 0), (px, args.height), color, 1)
        for gy in np.arange(math.floor(ymin / 0.25) * 0.25, ymax + 0.25, 0.25):
            _, py = world_to_px(0, gy, cx, cy, scale, args.width, args.height)
            color = (48, 48, 48) if abs((gy / 1.0) - round(gy / 1.0)) > 1e-4 else (72, 72, 72)
            cv2.line(img, (0, py), (args.width, py), color, 1)

        # World axes.
        x0, y0 = world_to_px(0, 0, cx, cy, scale, args.width, args.height)
        cv2.line(img, (x0, 0), (x0, args.height), (95, 95, 95), 1)
        cv2.line(img, (0, y0), (args.width, y0), (95, 95, 95), 1)

        if args.trail_seconds > 0:
            first_idx = idx
            while first_idx > 0 and records[idx]["t"] - records[first_idx]["t"] < args.trail_seconds:
                first_idx -= 1
        else:
            first_idx = 0
        pts = []
        for rec in records[first_idx:idx + 1]:
            pts.append(world_to_px(rec["x"], rec["y"], cx, cy, scale, args.width, args.height))
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, (70, 105, 255), 3, cv2.LINE_AA)

        spx, spy = world_to_px(start["x"], start["y"], cx, cy, scale, args.width, args.height)
        epx, epy = world_to_px(end["x"], end["y"], cx, cy, scale, args.width, args.height)
        cpx, cpy = world_to_px(current["x"], current["y"], cx, cy, scale, args.width, args.height)
        cv2.circle(img, (spx, spy), 8, (255, 160, 70), -1, cv2.LINE_AA)
        cv2.circle(img, (epx, epy), 8, (80, 220, 255), 2, cv2.LINE_AA)
        draw_robot(img, cpx, cpy, current["yaw"])

        dx = current["x"] - start["x"]
        dy = current["y"] - start["y"]
        dyaw = wrap_pi(current["yaw"] - start["yaw"])
        final_dx = end["x"] - start["x"]
        final_dy = end["y"] - start["y"]
        final_dyaw = wrap_pi(end["yaw"] - start["yaw"])
        final_dist = math.hypot(final_dx, final_dy)
        draw_text(img, args.title, (24, 34), 0.78, (245, 245, 245), 2)
        draw_text(img, f"t={current['t'] - t0:5.1f}s  x={current['x']:+.3f} y={current['y']:+.3f} yaw={current['yaw']:+.3f}", (24, 66))
        draw_text(img, f"from start: dx={dx:+.3f} dy={dy:+.3f} dist={math.hypot(dx, dy):.3f} dyaw={dyaw:+.3f}", (24, 92))
        draw_text(img, f"final loop error: dx={final_dx:+.3f} dy={final_dy:+.3f} dist={final_dist:.3f} dyaw={final_dyaw:+.3f}", (24, 118), color=(120, 220, 255))
        draw_text(img, "blue=start  yellow=end ring  green=robot  red=SLAM path", (24, args.height - 26), 0.52, (220, 220, 220))

        writer.write(img)

    writer.release()
    if write_path != out_path:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(write_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True)
            write_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[render] h264 transcode failed ({exc}); keeping mp4v output")
            write_path.replace(out_path)
    print(f"[render] wrote {out_path} frames={frame_count} records={len(records)}")
    print(f"[render] final_loop_error dx={end['x'] - start['x']:+.3f} "
          f"dy={end['y'] - start['y']:+.3f} "
          f"dist={math.hypot(end['x'] - start['x'], end['y'] - start['y']):.3f} "
          f"dyaw={wrap_pi(end['yaw'] - start['yaw']):+.3f}")


if __name__ == "__main__":
    main()
