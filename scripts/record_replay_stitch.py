"""Sequential workflow:
   1. Record N seconds of real camera + YOLOv8m detections → cam video + npy trajectory
   2. Replay npy through MuJoCo sim → sim video (G1 dodges the "ghost" person)
   3. ffmpeg stitch the two side-by-side

No DDS concurrency — bridge runs alone in step 1 (full data captured to npy),
then sim runs alone in step 2 reading from the npy. World-frame replay so the
person stays put while the sim G1 dodges around them.

Assumes rgbd_publisher.py is already running on the robot (port 5005).

Usage:
    uv run --with "pillow==9.5.0" --with "ultralytics==8.4.51" --with "lap" \\
        --with "imageio==2.35.1" --with "imageio-ffmpeg==0.5.1" \\
        python scripts/record_replay_stitch.py --duration 50 \\
            --out vision_demo/outputs_real/demo_$(date +%s).mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def countdown(secs: int, msg: str = "GO"):
    for i in range(secs, 0, -1):
        print(f"  ...{i}", flush=True)
        time.sleep(1)
    print(f">>> {msg} <<<", flush=True)


def run(cmd: list[str], desc: str, timeout: float | None = None) -> int:
    print(f"\n[run] {desc}")
    print(f"      $ {' '.join(cmd)}")
    rc = subprocess.call(cmd, timeout=timeout)
    if rc != 0:
        print(f"[err] step failed (rc={rc})", file=sys.stderr)
    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=50.0,
                   help="Capture duration in seconds (default 50)")
    p.add_argument("--out", type=str,
                   default="vision_demo/outputs_real/sim_vs_camera_demo.mp4",
                   help="Final stitched mp4 path")
    p.add_argument("--keep-intermediates", action="store_true",
                   help="Keep /tmp cam_view.mp4, sim_view.mp4, yolo_traj.npy")
    p.add_argument("--scenario-msg", type=str,
                   default="远 → 近(<1m) → 退 → 再近 → 退",
                   help="Printed before countdown to remind operator of plan")
    # Pass-through for bridge
    p.add_argument("--robot-ip", default="192.168.123.164")
    p.add_argument("--depth-offset", type=float, default=0.20)
    # Pass-through for sim
    p.add_argument("--safety-distance", type=float, default=1.3)
    p.add_argument("--reset-offset", type=float, default=0.2)
    p.add_argument("--replay-max-gap", type=float, default=1.0,
                   help="Skip interpolation across YOLO gaps longer than this (s)")
    p.add_argument("--replay-lpf-alpha", type=float, default=0.3,
                   help="EMA alpha for smoothing ID-swap jumps (0=ignore raw, 1=no filter)")
    p.add_argument("--replay-kalman", action="store_true", default=True,
                   help="Use Kalman filter (CV model + outlier gate) — default on")
    p.add_argument("--no-replay-kalman", dest="replay_kalman", action="store_false")
    p.add_argument("--kf-meas-std", type=float, default=0.10)
    p.add_argument("--kf-meas-std-close", type=float, default=0.30,
                   help="Inflate meas R at close range (counter sensor noise bursts)")
    p.add_argument("--kf-gate", type=float, default=4.0)
    p.add_argument("--countdown", type=int, default=5)
    args = p.parse_args()

    cam_mp4  = "/tmp/cam_view.mp4"
    sim_mp4  = "/tmp/sim_view.mp4"
    traj_npy = "/tmp/yolo_traj.npy"
    for f in (cam_mp4, sim_mp4, traj_npy):
        try: os.remove(f)
        except FileNotFoundError: pass

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: record camera + YOLO ───────────────────────
    print("=" * 60)
    print(f" Step 1/3  Record {args.duration:.0f}s — {args.scenario_msg}")
    print("=" * 60)
    countdown(args.countdown, "GO")
    rc = run([
        "uv", "run",
        "--with", "pillow==9.5.0",
        "--with", "ultralytics==8.4.51",
        "--with", "lap",
        "--with", "imageio==2.35.1",
        "--with", "imageio-ffmpeg==0.5.1",
        "python", str(REPO / "scripts" / "yolo_to_dds_laptop.py"),
        "--duration", str(args.duration),
        "--record-video", cam_mp4,
        "--save-npy", traj_npy,
        "--depth-offset", str(args.depth_offset),
        "--robot-ip", args.robot_ip,
        "--print-every", "20",
    ], "record cam + npy", timeout=args.duration + 25)
    if rc != 0 or not os.path.exists(traj_npy):
        print(f"[err] step 1 failed (no npy at {traj_npy})", file=sys.stderr)
        sys.exit(2)

    # ─── Step 2: sim replay ─────────────────────────────────
    print("\n" + "=" * 60)
    print(" Step 2/3  Replay npy through MuJoCo sim")
    print("=" * 60)
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    sim_cmd = [
        "uv", "run", "python", str(REPO / "deploy_dodge_mujoco.py"),
        "--replay_yolo", traj_npy,
        "--duration", str(args.duration),
        "--safety_distance", str(args.safety_distance),
        "--reset_distance_offset", str(args.reset_offset),
        "--replay_max_gap", str(args.replay_max_gap),
        "--replay_lpf_alpha", str(args.replay_lpf_alpha),
        "--kf_meas_std", str(args.kf_meas_std),
        "--kf_meas_std_close", str(args.kf_meas_std_close),
        "--kf_gate", str(args.kf_gate),
        "--record", sim_mp4,
    ]
    if args.replay_kalman:
        sim_cmd.append("--replay_kalman")
    rc = subprocess.call(sim_cmd, env=env, timeout=args.duration + 35)
    if rc != 0 or not os.path.exists(sim_mp4):
        print(f"[err] step 2 failed (no sim mp4 at {sim_mp4})", file=sys.stderr)
        sys.exit(3)

    # ─── Step 3: ffmpeg stitch ──────────────────────────────
    print("\n" + "=" * 60)
    print(f" Step 3/3  Stitch side-by-side → {out}")
    print("=" * 60)
    # Figure out playback retiming factors so both fill `duration` seconds:
    #   sim: 1250 f @ 30 fps render every 2 ctrl steps over `duration` s
    #        → playback length = duration * (render_fps_actual / 30)
    #   cam: bridge writes annotated frames at ~3 fps capture, encoded at 10 fps
    #        → playback length = n_frames / 10
    # We use ffprobe to find each video's actual playback duration.
    # Both sim and cam are encoded at 25fps with 1:1 wall-clock mapping
    # (cam frames repeated per actual capture timing). No setpts needed.
    def probe_duration(path):
        out_str = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ]).decode().strip()
        return float(out_str)
    dur_sim = probe_duration(sim_mp4)
    dur_cam = probe_duration(cam_mp4)
    print(f"      sim playback {dur_sim:.2f}s,  cam playback {dur_cam:.2f}s")
    rc = run([
        "ffmpeg", "-y",
        "-i", sim_mp4, "-i", cam_mp4,
        "-filter_complex",
        "[0:v]scale=720:720:force_original_aspect_ratio=increase,crop=720:720[s];"
        "[1:v]scale=1280:720[c];"
        "[s][c]hstack",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        str(out),
    ], "ffmpeg stitch")
    if rc != 0:
        sys.exit(4)

    sz_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"\n[✓] Final video: {out}  ({sz_mb:.1f} MB)")

    if not args.keep_intermediates:
        for f in (cam_mp4, sim_mp4, traj_npy):
            try: os.remove(f)
            except FileNotFoundError: pass


if __name__ == "__main__":
    main()
