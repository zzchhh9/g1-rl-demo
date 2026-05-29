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
# Override via env. The dodge stack lowers these to keep the robot's CPU free for
# the locomotion controller (capture+encode here and the loco service share PC4;
# a heavy camera load jitters the balance loop and makes fsm=200 idle-drift):
#   RGBD_W/RGBD_H  resolution (default 640x480)
#   RGBD_FPS       cap on the RealSense capture fps (default 30)
#   RGBD_RATE      encode+send rate in Hz (default 10)
RGB_W = int(os.environ.get("RGBD_W", "640"))
RGB_H = int(os.environ.get("RGBD_H", "480"))
RATE_HZ = float(os.environ.get("RGBD_RATE", "10"))
_FPS_CAP = int(os.environ.get("RGBD_FPS", "30"))
_FPS_TRY = [f for f in (30, 15, 6) if f <= _FPS_CAP] or [6]

# Try the requested resolution first, then fall back across resolutions AND fps. The
# device can come up in a reduced mode (e.g. only 424x240/480x270 depth, no 640x480) after
# another process (Unitree video_hub) grabbed and released it; both color and depth must use
# a profile the device currently exposes.
_RES_TRY = []
for _wh in [(RGB_W, RGB_H), (640, 480), (424, 240)]:
    if _wh not in _RES_TRY:
        _RES_TRY.append(_wh)


def _open_camera(pipe):
    for (rw, rh) in _RES_TRY:
        for fps in _FPS_TRY:
            try:
                cfg2 = rs.config()
                cfg2.enable_stream(rs.stream.color, rw, rh, rs.format.bgr8, fps)
                cfg2.enable_stream(rs.stream.depth, rw, rh, rs.format.z16, fps)
                prof = pipe.start(cfg2)
                al = rs.align(rs.stream.color)
                for _ in range(10):          # warmup auto-exposure
                    pipe.wait_for_frames(timeout_ms=5000)
                print(f"[rs] using {rw}x{rh} fps={fps}", flush=True)
                return prof, al
            except RuntimeError as e:
                print(f"[rs] {rw}x{rh} fps={fps} failed ({e}), trying next", flush=True)
                try:
                    pipe.stop()
                except Exception:
                    pass
                time.sleep(0.5)
    return None, None


pipe = rs.pipeline()
profile, align = _open_camera(pipe)
if profile is None or align is None:
    # Device stuck in a reduced/half-claimed mode -> hardware reset and retry once. This
    # self-heals the "no 640x480 / Failed to resolve Z16" state seen after a reboot.
    print("[rs] open failed; hardware_reset + retry", flush=True)
    try:
        for _d in rs.context().query_devices():
            _d.hardware_reset()
    except Exception as e:
        print(f"[rs] hardware_reset failed: {e}", flush=True)
    time.sleep(12.0)
    pipe = rs.pipeline()
    profile, align = _open_camera(pipe)
if profile is None or align is None:
    raise SystemExit("[rs] no resolution/fps worked (even after hardware reset)")

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
