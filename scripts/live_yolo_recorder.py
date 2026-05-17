"""Record a video of the live RealSense feed with YOLOv8 person bbox +
real-time distance/bearing overlay (using true depth from RealSense).

Connects to rgbd_publisher_robot.py via TCP, runs YOLOv8m + ByteTrack +
bbox-median-depth on each frame, draws annotation, encodes mp4 at end.

Usage:
    uv run --with "pillow==9.5.0" --with "ultralytics==8.4.51" --with "lap" \\
        --with "imageio" --with "imageio-ffmpeg" \\
        python scripts/live_yolo_recorder.py --duration 60 \\
            --out vision_demo/outputs_real/live_distance.mp4
"""
import argparse, json, math, socket, struct, time, os
import numpy as np
import cv2
from ultralytics import YOLO


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk: raise ConnectionResetError("EOF")
        buf.extend(chunk)
    return bytes(buf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot-ip", default="192.168.123.164")
    p.add_argument("--port",     type=int, default=5005)
    p.add_argument("--out",      default="vision_demo/outputs_real/live_distance.mp4")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--model",    default="yolov8m.pt")
    p.add_argument("--conf",     type=float, default=0.25)
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--video-fps", type=float, default=10.0,
                   help="Output video playback fps (capture is ~3 fps)")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[tcp] connecting to {args.robot_ip}:{args.port} ...", flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.robot_ip, args.port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    info_len = struct.unpack("!I", recv_exact(sock, 4))[0]
    info = json.loads(recv_exact(sock, info_len).decode())
    fx, cx_px, cy_px = info["fx"], info["cx"], info["cy"]
    print(f"[tcp] intrinsics: fx={fx:.1f} cx={cx_px:.1f}")

    print(f"[yolo] loading {args.model} ...", flush=True)
    model = YOLO(args.model)

    frames = []
    distance_log = []   # (t, dist, bearing)
    t_start = time.time()
    n_det = 0
    n_total = 0
    print(f"[rec] capturing for {args.duration:.0f}s ...", flush=True)

    try:
        while (time.time() - t_start) < args.duration:
            header = recv_exact(sock, 4 + 8 + 4 + 4)
            frame_id, ts_ms, c_len, d_len = struct.unpack("!IQII", header)
            color = cv2.imdecode(np.frombuffer(recv_exact(sock, c_len), np.uint8),
                                 cv2.IMREAD_COLOR)
            depth = cv2.imdecode(np.frombuffer(recv_exact(sock, d_len), np.uint8),
                                 cv2.IMREAD_UNCHANGED)
            n_total += 1
            t_sec = time.time() - t_start

            results = model.track(color, classes=[0], conf=args.conf,
                                  verbose=False, device="cpu", imgsz=args.imgsz,
                                  persist=True, tracker="bytetrack.yaml")
            boxes = results[0].boxes
            vis = color.copy()
            h, w = vis.shape[:2]
            cv2.line(vis, (w//2, 0), (w//2, h), (180,180,180), 1)
            cv2.line(vis, (0, h//2), (w, h//2), (180,180,180), 1)

            nearest = None
            for b in boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                tid = int(b.id.item()) if b.id is not None else -1
                conf = float(b.conf.item())
                roi = depth[y1:y2, x1:x2]
                valid = roi[(roi > 0) & (roi < 8000)]
                if len(valid) < 20: continue
                d_m = float(np.median(valid)) / 1000.0
                if nearest is None or d_m < nearest["d"]:
                    cx_b = (x1 + x2) / 2
                    bearing = math.degrees(math.atan2(cx_b - cx_px, fx))
                    nearest = dict(d=d_m, bearing=bearing, tid=tid, conf=conf,
                                   x1=x1, y1=y1, x2=x2, y2=y2)
                # draw all (thin orange)
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,165,255), 1)

            # Highlight nearest in green
            if nearest:
                n_det += 1
                n = nearest
                cv2.rectangle(vis, (n['x1'], n['y1']), (n['x2'], n['y2']),
                              (0, 255, 0), 4)
                for k, txt in enumerate([
                    f"id{n['tid']}  conf={n['conf']:.2f}",
                    f"DIST = {n['d']:.2f} m",
                    f"BEAR = {n['bearing']:+.1f} deg",
                ]):
                    yy = max(20, n['y1'] - 8 - 28 * (2-k))
                    cv2.putText(vis, txt, (n['x1'], yy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,0), 6)
                    cv2.putText(vis, txt, (n['x1'], yy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,0), 2)
                distance_log.append((t_sec, n['d'], n['bearing']))

            # HUD strip
            det_pct = 100 * n_det / n_total if n_total else 0
            hud = (f"t={t_sec:5.1f}s  frame={n_total}  det={det_pct:.0f}%  "
                   + (f"NEAREST  d={nearest['d']:.2f}m  bear={nearest['bearing']:+.1f}deg"
                      if nearest else "NO DETECTION"))
            cv2.rectangle(vis, (0, h-36), (w, h), (0, 0, 0), -1)
            cv2.putText(vis, hud, (10, h-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

            frames.append(vis)
            if n_total % 10 == 0:
                if nearest:
                    print(f"  f{n_total:3d} t={t_sec:5.1f}s  d={nearest['d']:.2f}m  bear={nearest['bearing']:+.1f}°", flush=True)
                else:
                    print(f"  f{n_total:3d} t={t_sec:5.1f}s  (no detection)", flush=True)
    except (KeyboardInterrupt, ConnectionResetError):
        pass
    finally:
        sock.close()

    print(f"\n[encode] {len(frames)} frames -> {args.out} (playback {args.video_fps} fps)")
    import imageio.v3 as iio
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    iio.imwrite(args.out, rgb_frames, fps=args.video_fps,
                codec="libx264", macro_block_size=1)
    sz_mb = os.path.getsize(args.out) / (1024*1024)
    print(f"[done] {args.out}  {sz_mb:.1f} MB")
    if distance_log:
        ds = [r[1] for r in distance_log]
        bs = [r[2] for r in distance_log]
        print(f"  distance:  min={min(ds):.2f}m  max={max(ds):.2f}m  mean={sum(ds)/len(ds):.2f}m")
        print(f"  bearing:   min={min(bs):+.1f}°  max={max(bs):+.1f}°")


if __name__ == "__main__":
    main()
