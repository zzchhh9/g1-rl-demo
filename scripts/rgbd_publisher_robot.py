#!/usr/bin/env python3
"""Robot-side: read RealSense aligned color+depth, push pairs to laptop via TCP.

Stays running indefinitely. Laptop connects, receives intrinsics once, then
gets (frame_id, ts_ms, color_jpeg, depth_png16) tuples at ~10 Hz.

Why TCP not UDP: depth PNG (~200KB) + JPEG (~100KB) per frame > UDP MTU so
we'd need chunking. TCP handles it transparently on local network.

Requires: pyrealsense2 (in /home/unitree/.local) + librealsense2 in
/opt/ros/noetic/lib/aarch64-linux-gnu.

Run on robot:
    export LD_LIBRARY_PATH=/opt/ros/noetic/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
    python3 /tmp/rgbd_publisher_robot.py
"""

import os, sys, time, struct, socket, json
os.environ.setdefault("LD_LIBRARY_PATH", "/opt/ros/noetic/lib/aarch64-linux-gnu")
import numpy as np
import cv2
import pyrealsense2 as rs


LISTEN_PORT = 5005
RATE_HZ = 10.0
# 1280x720 needs USB 3.0; 640x480 fits USB 2.0 fine.
# Override via env: RGBD_W=1280 RGBD_H=720 python3 rgbd_publisher.py
import os as _os
RGB_W = int(_os.environ.get("RGBD_W", "640"))
RGB_H = int(_os.environ.get("RGBD_H", "480"))

pipe = rs.pipeline()
# 15fps to fit USB 2.0 bandwidth (USB 3.0 supports 30fps). Try 30 first, fall
# back to 15 then 6 if either stream config or warmup frames fail.
profile = None
align = None
for fps in (30, 15, 6):
    try:
        cfg2 = rs.config()
        cfg2.enable_stream(rs.stream.color, RGB_W, RGB_H, rs.format.bgr8, fps)
        cfg2.enable_stream(rs.stream.depth, RGB_W, RGB_H, rs.format.z16, fps)
        profile = pipe.start(cfg2)
        align = rs.align(rs.stream.color)
        for _ in range(10):          # warmup auto-exposure
            pipe.wait_for_frames(timeout_ms=5000)
        print(f"[rs] using fps={fps}", flush=True)
        break
    except RuntimeError as e:
        print(f"[rs] fps={fps} failed ({e}), trying lower", flush=True)
        try:
            pipe.stop()
        except Exception:
            pass
        profile = None
        align = None
        time.sleep(1.0)
if profile is None or align is None:
    raise SystemExit("[rs] no fps worked")

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
print(f"[rs] {intr.width}x{intr.height}  fx={intr.fx:.1f} fy={intr.fy:.1f} "
      f"cx={intr.ppx:.1f} cy={intr.ppy:.1f}  depth_scale={depth_scale}",
      flush=True)

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", LISTEN_PORT))
srv.listen(1)
print(f"[rs] listening on :{LISTEN_PORT} (accepts repeatedly) ...", flush=True)

info = json.dumps({"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
                   "w": intr.width, "h": intr.height,
                   "depth_scale_m": depth_scale, "rate_hz": RATE_HZ}).encode()
info_msg = struct.pack("!I", len(info)) + info

interval = 1.0 / RATE_HZ
try:
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[rs] client connected: {addr}", flush=True)
        try:
            conn.sendall(info_msg)
            frame_id = 0
            nxt = time.time()
            t_loop = time.time()
            while True:
                now = time.time()
                if now < nxt:
                    time.sleep(min(0.005, nxt - now)); continue
                nxt += interval
                frames = pipe.wait_for_frames()
                aligned = align.process(frames)
                cf = aligned.get_color_frame(); df = aligned.get_depth_frame()
                if not cf or not df: continue
                color = np.asanyarray(cf.get_data())
                depth = np.asanyarray(df.get_data())
                ok_c, color_enc = cv2.imencode(".jpg", color,
                                               [cv2.IMWRITE_JPEG_QUALITY, 80])
                ok_d, depth_enc = cv2.imencode(".png", depth)
                if not (ok_c and ok_d): continue
                ts_ms = int(time.time() * 1000)
                header = struct.pack("!IQII", frame_id, ts_ms,
                                     len(color_enc), len(depth_enc))
                conn.sendall(header + color_enc.tobytes() + depth_enc.tobytes())
                frame_id += 1
                if frame_id % 20 == 0:
                    fps = frame_id / (time.time() - t_loop)
                    print(f"  f{frame_id}  c={len(color_enc)}B d={len(depth_enc)}B "
                          f"avg fps={fps:.1f}", flush=True)
        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"[rs] client {addr} disconnected ({e}), waiting for next ...",
                  flush=True)
        finally:
            try: conn.close()
            except Exception: pass
finally:
    pipe.stop()
    srv.close()
    print("[rs] stopped")
