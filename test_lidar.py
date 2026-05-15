"""测试 G1 头顶 Livox Mid-360 LiDAR — 直接 UDP 路径，不经 Unitree DDS。

数据流：
    Mid-360 (.120) → robot .164:56301 (UDP) → Python forwarder → laptop .222:56301
    本脚本在笔记本上监听 :56301，按 Livox SDK 2 以太网协议解析点云。

前置条件 (一次性 bringup)：
    1. 在机器人上把 LiDAR 配置启动一次（让它开始往 .164 推流）
    2. 在机器人上启动 UDP 转发器（把 .164:56301 → .222:56301）
    见 README / scripts/start_lidar.sh

Usage:
    uv run python test_lidar.py                  # 默认 :56301，10s
    uv run python test_lidar.py --duration 30
    uv run python test_lidar.py --detect         # 障碍物检测，请站在前方 1-3m

参考:
    Livox-SDK2 协议: https://github.com/Livox-SDK/livox_wiki_en/blob/master/source/tutorials/new_product/mid360/livox_eth_protocol_mid360.md
"""

from __future__ import annotations

import argparse
import collections
import socket
import struct
import sys
import threading
import time

import numpy as np


# ───── Livox Mid-360 Ethernet packet layout ─────
# Header (36 bytes):
#   ver(1) length(2) time_interval(2) dot_num(2) udp_cnt(2) frame_cnt(1)
#   data_type(1) time_type(1) rsvd[12] crc32(4) timestamp[8]
# Then `dot_num` points; layout depends on data_type:
#   1 = CartesianHighRaw  (x,y,z int32 mm + refl + tag) = 14 B/pt
#   2 = CartesianLowRaw   (x,y,z int16 cm + refl + tag) =  8 B/pt
#   3 = SphericalRaw

_HEADER_FMT = "<BHHHHBBB"   # ver,length,time_interval,dot_num,udp_cnt,frame_cnt,data_type,time_type
_HEADER_SIZE = 36
_DT_HIGH = 1
_DT_LOW = 2


class LivoxLidarReceiver:
    """监听 UDP 接收 Livox Mid-360 数据，持续累计最近的点云。

    使用一个滚动 deque 保存最近 N 个点；`.points` 返回当前快照。
    """

    def __init__(self, port: int = 56301, buffer_pts: int = 50000):
        self._port = port
        self._buf: collections.deque[tuple[float, float, float]] = collections.deque(maxlen=buffer_pts)
        self._lock = threading.Lock()
        self._packets = 0
        self._points_recv = 0
        self._last_dot_num = 0
        self._last_data_type = 0
        self._stop = threading.Event()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self._sock.bind(("", port))
        self._sock.settimeout(0.5)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[LiDAR] 已绑定 UDP :{port}，等待数据 ...")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            if len(data) < _HEADER_SIZE:
                continue
            ver, length, _time_interval, dot_num, _udp_cnt, _frame_cnt, data_type, _time_type = \
                struct.unpack_from(_HEADER_FMT, data, 0)
            if ver != 0:
                continue
            self._packets += 1
            self._last_dot_num = dot_num
            self._last_data_type = data_type
            payload = data[_HEADER_SIZE:]
            if data_type == _DT_HIGH:
                stride = 14
                scale = 0.001
                fmt = "<iii"
            elif data_type == _DT_LOW:
                stride = 8
                scale = 0.01
                fmt = "<hhh"
            else:
                # IMU or unsupported — skip
                continue
            n_in_pkt = min(dot_num, len(payload) // stride)
            new_pts = []
            for i in range(n_in_pkt):
                x, y, z = struct.unpack_from(fmt, payload, i * stride)
                new_pts.append((x * scale, y * scale, z * scale))
            if new_pts:
                with self._lock:
                    self._buf.extend(new_pts)
                    self._points_recv += len(new_pts)

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except Exception:
            pass

    @property
    def packets(self) -> int:
        return self._packets

    @property
    def msg_count(self) -> int:
        """Compat alias — original API used `msg_count` (PointCloud2 message count).
        For UDP-direct, return packet count instead."""
        return self._packets

    @property
    def total_points(self) -> int:
        return self._points_recv

    @property
    def points(self) -> np.ndarray | None:
        with self._lock:
            if not self._buf:
                return None
            return np.array(self._buf, dtype=np.float32)

    @property
    def last_dot_num(self) -> int:
        return self._last_dot_num

    @property
    def last_data_type(self) -> int:
        return self._last_data_type


# Backwards-compat name (some other scripts import LidarTest)
LidarTest = LivoxLidarReceiver


# ───── Tests ─────

def test_basic(lidar: LivoxLidarReceiver, duration: float = 10.0) -> bool:
    print(f"\n{'=' * 60}")
    print(f"测试 1: 基础连接测试 ({duration}s)")
    print(f"{'=' * 60}")

    start = time.time()
    while time.time() - start < duration:
        time.sleep(1.0)
        pts = lidar.points
        if pts is not None and len(pts) > 0:
            n = len(pts)
            x_range = f"[{pts[:, 0].min():.2f}, {pts[:, 0].max():.2f}]"
            y_range = f"[{pts[:, 1].min():.2f}, {pts[:, 1].max():.2f}]"
            z_range = f"[{pts[:, 2].min():.2f}, {pts[:, 2].max():.2f}]"
            print(f"  收到 {lidar.packets:6d} 包  | 缓冲 {n:5d} 点  | "
                  f"X{x_range} Y{y_range} Z{z_range}")
        else:
            print(f"  收到 {lidar.packets:6d} 包  | 无点云数据")

    if lidar.packets == 0:
        print("\n❌ 失败: 没有收到任何 UDP 数据")
        print("   检查:")
        print("   1. 机器人侧 forward.py 是否在跑 (ssh unitree@192.168.123.164)")
        print("   2. LiDAR 是否启动过 (跑过 driver 一次让它进入推流状态)")
        print("   3. 防火墙是否拦截 UDP :56301")
        return False

    if lidar.points is None or len(lidar.points) == 0:
        print("\n⚠️  收到 UDP 包但解析不出点云 — 检查 data_type")
        print(f"   last data_type={lidar.last_data_type} dot_num={lidar.last_dot_num}")
        return False

    print(f"\n✅ 通过: 收到 {lidar.packets} 包，累计 {lidar.total_points} 点")
    return True


def test_frequency(lidar: LivoxLidarReceiver, duration: float = 5.0) -> float:
    print(f"\n{'=' * 60}")
    print(f"测试 2: 频率测试 ({duration}s)")
    print(f"{'=' * 60}")

    p0 = lidar.packets
    pt0 = lidar.total_points
    time.sleep(duration)
    pkt_hz = (lidar.packets - p0) / duration
    pt_hz = (lidar.total_points - pt0) / duration
    print(f"  包速率: {pkt_hz:.1f} pkt/s   (Mid-360 典型 ~2000)")
    print(f"  点速率: {pt_hz:.0f} pt/s     (典型 ~200k)")
    return pkt_hz


def test_obstacle_detection(lidar: LivoxLidarReceiver, duration: float = 15.0) -> bool:
    print(f"\n{'=' * 60}")
    print(f"测试 3: 障碍物检测 ({duration}s)")
    print(f"  请让一个人站在机器人前方 1-3m 处")
    print(f"{'=' * 60}")

    min_height = 0.3
    max_range = 5.0

    for i in range(int(duration)):
        time.sleep(1.0)
        pts = lidar.points
        if pts is None or len(pts) == 0:
            print(f"  {i + 1:2d}s: 无数据")
            continue
        above_floor = pts[pts[:, 2] > min_height]
        dists = np.linalg.norm(above_floor[:, :2], axis=1)
        mask = (dists > 0.3) & (dists < max_range)
        nearby = above_floor[mask]
        if len(nearby) == 0:
            print(f"  {i + 1:2d}s: buf={len(pts):5d} above_floor={len(above_floor)}  近距离: 0")
            continue
        centroid = nearby.mean(axis=0)
        nearest = dists[mask].min()
        print(f"  {i + 1:2d}s: buf={len(pts):5d}  obstacle pos=[{centroid[0]:+.2f},{centroid[1]:+.2f},{centroid[2]:.2f}]  "
              f"最近={nearest:.2f}m  n={len(nearby)}")

    pts = lidar.points
    if pts is not None:
        above = pts[pts[:, 2] > min_height]
        dists = np.linalg.norm(above[:, :2], axis=1)
        nearby = above[(dists > 0.3) & (dists < max_range)]
        if len(nearby) > 10:
            print(f"\n✅ 障碍物检测成功: {len(nearby)} 个点在 0.3-{max_range}m 范围内")
            return True
    print(f"\n⚠️  未能稳定检测到障碍物")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G1 Livox Mid-360 LiDAR 测试 (UDP 直连)")
    parser.add_argument("--port", type=int, default=56301,
                        help="本地监听 UDP 端口 (默认 56301)")
    parser.add_argument("--detect", action="store_true",
                        help="运行障碍物检测测试")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="每项测试时长 (秒)")
    args = parser.parse_args()

    lidar = LivoxLidarReceiver(port=args.port)
    time.sleep(0.5)

    ok = test_basic(lidar, args.duration)
    if not ok:
        sys.exit(1)

    test_frequency(lidar, min(args.duration, 5.0))

    if args.detect:
        test_obstacle_detection(lidar, args.duration)

    print(f"\n{'=' * 60}")
    print(f"测试完成。共收到 {lidar.packets} 包 / {lidar.total_points} 点。")
    print(f"{'=' * 60}")
    lidar.close()
