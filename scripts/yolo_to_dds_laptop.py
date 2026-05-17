"""Laptop-side: receive RGB+Depth from robot, run YOLOv8m + ByteTrack,
publish nearest-person position to DDS topic `rt/yolo/person`.

Pairs with `rgbd_publisher_robot.py`.

Output DDS topic format (`rt/yolo/person`, std_msgs::String_, JSON payload):
    {"frame_id": int, "ts_ms": int, "n": 0 or 1,
     "x_fwd": m, "y_left": m, "z": m, "dist": m, "bearing": deg,
     "track_id": int, "conf": float}

  - `n=0` means no person detected this frame (deploy_dodge_real.py should
    treat as "no obstacle").
  - Body frame convention: +x = robot forward, +y = robot left, +z = up.
    Camera frame is +y=right after the D435i extrinsic, so we negate here.

Run on laptop:
    uv run --with "pillow==9.5.0" --with "ultralytics==8.4.51" --with "lap" \\
        python scripts/yolo_to_dds_laptop.py
"""

import argparse, json, math, socket, struct, time
import numpy as np
import cv2

from ultralytics import YOLO
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionResetError("EOF")
        buf.extend(chunk)
    return bytes(buf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot-ip",  default="192.168.123.164")
    p.add_argument("--port",      type=int, default=5005)
    p.add_argument("--net",       default="eno1")
    p.add_argument("--topic",     default="rt/yolo/person")
    p.add_argument("--model",     default="yolov8m.pt")
    p.add_argument("--conf",      type=float, default=0.25)
    p.add_argument("--imgsz",     type=int, default=640)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--record-video", type=str, default="",
                   help="If set, save annotated camera frames to this mp4 path")
    p.add_argument("--duration",  type=float, default=0.0,
                   help="If > 0, stop after this many seconds (for paired recording)")
    p.add_argument("--depth-offset", type=float, default=0.20,
                   help="Subtract this many meters from raw RealSense depth.")
    p.add_argument("--depth-floor", type=float, default=0.05,
                   help="Minimum reported distance after offset (clamp).")
    p.add_argument("--save-npy", type=str, default="",
                   help="Save complete YOLO trajectory to .npy (rows: "
                        "t_sec, x_fwd, y_left, z, dist, bearing, track_id). "
                        "Rows skipped when no detection. Compatible with "
                        "deploy_dodge_mujoco.py --replay_yolo.")
    args = p.parse_args()

    print(f"[dds] init domain 0 on {args.net}")
    ChannelFactoryInitialize(0, args.net)
    pub = ChannelPublisher(args.topic, String_); pub.Init()
    msg = std_msgs_msg_dds__String_()

    print(f"[tcp] connecting to {args.robot_ip}:{args.port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.robot_ip, args.port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    info_len = struct.unpack("!I", recv_exact(sock, 4))[0]
    info = json.loads(recv_exact(sock, info_len).decode())
    fx, cx_px = info["fx"], info["cx"]
    print(f"[tcp] intrinsics: fx={fx:.1f} cx={cx_px:.1f}  source rate {info.get('rate_hz', '?')}Hz")

    print(f"[yolo] loading {args.model} ...")
    model = YOLO(args.model)

    annotated_frames = [] if args.record_video else None   # list of (frame_bgr, t_sec)
    traj_rows = [] if args.save_npy else None

    n_frames = 0
    n_detect = 0
    t_loop = time.time()
    try:
        while True:
            if args.duration > 0 and (time.time() - t_loop) > args.duration:
                print(f"[stop] duration {args.duration}s reached")
                break
            header = recv_exact(sock, 4 + 8 + 4 + 4)
            frame_id, ts_ms, c_len, d_len = struct.unpack("!IQII", header)
            color = cv2.imdecode(np.frombuffer(recv_exact(sock, c_len), np.uint8),
                                 cv2.IMREAD_COLOR)
            depth = cv2.imdecode(np.frombuffer(recv_exact(sock, d_len), np.uint8),
                                 cv2.IMREAD_UNCHANGED)

            t0 = time.time()
            results = model.track(color, classes=[0], conf=args.conf,
                                  verbose=False, device="cpu", imgsz=args.imgsz,
                                  persist=True, tracker="bytetrack.yaml")
            boxes = results[0].boxes

            nearest = None
            for b in boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                roi = depth[y1:y2, x1:x2]
                valid = roi[(roi > 0) & (roi < 8000)]
                if len(valid) < 20:
                    continue
                d_m_raw = float(np.median(valid)) / 1000.0
                d_m = max(args.depth_floor, d_m_raw - args.depth_offset)
                if (nearest is None) or (d_m < nearest["d"]):
                    cx_b = (x1 + x2) / 2
                    bearing = math.degrees(math.atan2(cx_b - cx_px, fx))
                    tid = int(b.id.item()) if b.id is not None else -1
                    nearest = dict(d=d_m, d_raw=d_m_raw, bearing=bearing, tid=tid,
                                   conf=float(b.conf.item()),
                                   x1=x1, y1=y1, x2=x2, y2=y2)

            if nearest:
                rad = math.radians(nearest["bearing"])
                payload = {
                    "frame_id": frame_id, "ts_ms": ts_ms, "n": 1,
                    "x_fwd":  nearest["d"] * math.cos(rad),
                    "y_left": -nearest["d"] * math.sin(rad),
                    "z": 0.85,
                    "dist": nearest["d"], "bearing": nearest["bearing"],
                    "track_id": nearest["tid"], "conf": nearest["conf"],
                }
                n_detect += 1
                if traj_rows is not None:
                    t_sec = time.time() - t_loop
                    traj_rows.append([t_sec, payload["x_fwd"], payload["y_left"],
                                      payload["z"], nearest["d"], nearest["bearing"],
                                      nearest["tid"]])
            else:
                payload = {"frame_id": frame_id, "ts_ms": ts_ms, "n": 0}

            msg.data = json.dumps(payload)
            pub.Write(msg)

            # Annotate frame for video recording
            if annotated_frames is not None:
                vis = color.copy()
                h, w = vis.shape[:2]
                cv2.line(vis, (w//2, 0), (w//2, h), (180,180,180), 1)
                cv2.line(vis, (0, h//2), (w, h//2), (180,180,180), 1)
                # all boxes (orange thin)
                for b in boxes:
                    x1_, y1_, x2_, y2_ = [int(v) for v in b.xyxy[0].tolist()]
                    cv2.rectangle(vis, (x1_, y1_), (x2_, y2_), (0,165,255), 1)
                # nearest (green thick + text)
                if nearest:
                    n = nearest
                    cv2.rectangle(vis, (n['x1'], n['y1']), (n['x2'], n['y2']),
                                  (0,255,0), 4)
                    for k, txt in enumerate([
                        f"id{n['tid']}  conf={n['conf']:.2f}",
                        f"DIST = {n['d']:.2f} m  (raw {n['d_raw']:.2f})",
                        f"BEAR = {n['bearing']:+.1f} deg",
                    ]):
                        yy = max(20, n['y1'] - 8 - 28 * (2-k))
                        cv2.putText(vis, txt, (n['x1'], yy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,0), 6)
                        cv2.putText(vis, txt, (n['x1'], yy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,0), 2)
                t_sec = time.time() - t_loop
                det_pct = 100 * n_detect / max(1, n_frames+1)
                hud = (f"t={t_sec:5.1f}s  f{n_frames+1}  det={det_pct:.0f}%  "
                       + (f"NEAREST  d={nearest['d']:.2f}m  bear={nearest['bearing']:+.1f}deg"
                          if nearest else "NO DETECTION"))
                cv2.rectangle(vis, (0, h-36), (w, h), (0,0,0), -1)
                cv2.putText(vis, hud, (10, h-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                annotated_frames.append((vis, time.time() - t_loop))

            n_frames += 1
            yolo_ms = (time.time() - t0) * 1000
            if n_frames % args.print_every == 0:
                fps = n_frames / (time.time() - t_loop)
                det_pct = 100 * n_detect / n_frames
                if nearest:
                    print(f"  f{frame_id:5d}  yolo={yolo_ms:.0f}ms  fps={fps:.1f}  "
                          f"det={det_pct:.0f}%  | nearest d={nearest['d']:.2f}m "
                          f"bear={nearest['bearing']:+.1f}° id{nearest['tid']}", flush=True)
                else:
                    print(f"  f{frame_id:5d}  yolo={yolo_ms:.0f}ms  fps={fps:.1f}  "
                          f"det={det_pct:.0f}%  | (no detection)", flush=True)
    except KeyboardInterrupt:
        print("\n[stop] interrupted")
    except ConnectionResetError as e:
        print(f"\n[stop] robot disconnected: {e}")
    finally:
        sock.close()
        print(f"[done] {n_frames} frames, {n_detect} with detection")

        if annotated_frames:
            import os
            os.makedirs(os.path.dirname(args.record_video) or ".", exist_ok=True)
            import imageio.v3 as iio
            # Time-correct encoding: each captured frame is repeated for its real-time
            # display duration so playback matches wall-clock. Gaps in capture →
            # last frame held until next capture. Encoded at fixed 25fps to match
            # the sim video's 25fps playback for perfect alignment.
            OUT_FPS = 25
            total_dur = annotated_frames[-1][1] + 0.5  # 0.5s tail past last frame
            n_out = int(round(total_dur * OUT_FPS))
            out_frames = []
            for k in range(n_out):
                t_out = k / OUT_FPS
                # find latest captured frame with t <= t_out
                idx = 0
                for i, (_, ts) in enumerate(annotated_frames):
                    if ts <= t_out:
                        idx = i
                    else:
                        break
                out_frames.append(cv2.cvtColor(annotated_frames[idx][0], cv2.COLOR_BGR2RGB))
            iio.imwrite(args.record_video, out_frames, fps=OUT_FPS,
                        codec="libx264", macro_block_size=1)
            sz = os.path.getsize(args.record_video) / (1024*1024)
            print(f"[encode] {len(annotated_frames)} captured → "
                  f"{len(out_frames)} output frames @ {OUT_FPS}fps "
                  f"({total_dur:.1f}s) → {args.record_video} ({sz:.1f} MB)")
        if traj_rows is not None and traj_rows:
            import os
            os.makedirs(os.path.dirname(args.save_npy) or ".", exist_ok=True)
            arr = np.array(traj_rows, dtype=np.float64)
            np.save(args.save_npy, arr)
            print(f"[npy] {len(arr)} detection rows -> {args.save_npy}  "
                  f"(t {arr[0,0]:.2f}→{arr[-1,0]:.2f}s, dist {arr[:,4].min():.2f}→{arr[:,4].max():.2f}m)")


if __name__ == "__main__":
    main()
