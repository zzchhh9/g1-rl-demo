"""实时测量 G1 Livox Mid-360 视野内最近障碍物的距离 — 仅 LiDAR，不动机器人。

依赖与 test_lidar.py 相同的数据链路：
    Mid-360 (.120) → robot .164:56301 (UDP) → forwarder → laptop :56301

机器人侧 bringup（一次性）:
    ./scripts/start_lidar.sh

笔记本侧 (本脚本):
    uv run python nearest_obstacle.py                # 默认配置
    uv run python nearest_obstacle.py --min-h 0.2 --max-r 8.0
    uv run python nearest_obstacle.py --window 0.2   # 200ms 滑动窗口
    uv run python nearest_obstacle.py --plot         # 画俯视图 (matplotlib)

输出（默认每 0.1s 刷新一次）:
    t=12.3s  pkts=24580  最近障碍: dist=1.83m  bearing=+15.4°  (x=+1.76, y=+0.48, z=+0.95)  n_pts=312
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time

import numpy as np

from test_lidar import LivoxLidarReceiver


ANSI_RED = "\033[91m"
ANSI_YEL = "\033[93m"
ANSI_GRN = "\033[92m"
ANSI_RST = "\033[0m"


def colorize(dist: float, warn: float, crit: float) -> str:
    if dist < crit:
        return ANSI_RED
    if dist < warn:
        return ANSI_YEL
    return ANSI_GRN


def main():
    p = argparse.ArgumentParser(description="实时测量 Mid-360 视野内最近障碍物距离")
    p.add_argument("--port", type=int, default=56301, help="UDP 监听端口")
    p.add_argument("--min-h", type=float, default=0.30,
                   help="最小高度过滤 (m, 排除地面)；LiDAR 自身坐标系 z 向上")
    p.add_argument("--max-h", type=float, default=2.00,
                   help="最大高度过滤 (m, 排除天花板/挂件)")
    p.add_argument("--min-r", type=float, default=0.30,
                   help="最小半径过滤 (m, 排除机器人自身/支架)")
    p.add_argument("--max-r", type=float, default=8.00,
                   help="最大半径过滤 (m)")
    p.add_argument("--window", type=float, default=0.10,
                   help="累积窗口 (s)，越大点云越密但响应越慢")
    p.add_argument("--rate", type=float, default=10.0, help="打印频率 Hz")
    p.add_argument("--warn", type=float, default=2.0, help="黄色警戒距离 (m)")
    p.add_argument("--crit", type=float, default=1.0, help="红色危险距离 (m)")
    p.add_argument("--plot", action="store_true",
                   help="额外开一个 matplotlib 俯视图实时显示 (慢，仅 debug)")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()

    # Buffer size: ~200k pts/sec × window
    buf_pts = max(10000, int(200_000 * args.window))
    lidar = LivoxLidarReceiver(port=args.port, buffer_pts=buf_pts)

    # plot setup
    plot_fig = None
    if args.plot:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.ion()
        plot_fig, ax = plt.subplots(figsize=(7, 7))
        sc = ax.scatter([], [], s=1, c="b")
        nearest_marker, = ax.plot([], [], "ro", markersize=10, label="nearest")
        ax.set_xlim(-args.max_r, args.max_r)
        ax.set_ylim(-args.max_r, args.max_r)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x (m, robot forward)")
        ax.set_ylabel("y (m, robot left)")
        ax.set_title("Mid-360 top-down (sensor frame)")
        circle1 = plt.Circle((0, 0), args.warn, fill=False, color="orange", ls="--", alpha=0.5)
        circle2 = plt.Circle((0, 0), args.crit, fill=False, color="red", ls="--", alpha=0.5)
        ax.add_patch(circle1); ax.add_patch(circle2)
        ax.legend(loc="upper right")

    # SIGINT 优雅退出
    stop = {"go": True}
    def handler(*_):
        stop["go"] = False
    signal.signal(signal.SIGINT, handler)

    print("[ready] 监听点云数据… Ctrl+C 退出\n")
    t_start = time.time()
    interval = 1.0 / args.rate
    next_print = time.time()

    while stop["go"]:
        now = time.time()
        if now < next_print:
            time.sleep(min(0.01, next_print - now))
            continue
        next_print = now + interval

        pts = lidar.points
        elapsed = now - t_start

        if pts is None or len(pts) == 0:
            sys.stdout.write(f"\rt={elapsed:6.1f}s  pkts={lidar.packets:6d}  (无数据){'':30s}")
            sys.stdout.flush()
            continue

        # Filter in SENSOR frame:
        #   sensor: x=forward, y=left, z=up
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r = np.sqrt(x * x + y * y)
        mask = (z > args.min_h) & (z < args.max_h) & (r > args.min_r) & (r < args.max_r)
        valid = pts[mask]
        if len(valid) == 0:
            sys.stdout.write(f"\rt={elapsed:6.1f}s  pkts={lidar.packets:6d}  buf={len(pts):5d}  "
                             f"过滤后=0  (没有视野内障碍物)         ")
            sys.stdout.flush()
            continue

        rv = np.sqrt(valid[:, 0]**2 + valid[:, 1]**2)
        idx = int(np.argmin(rv))
        nearest = valid[idx]
        dist = float(rv[idx])
        bearing = math.degrees(math.atan2(nearest[1], nearest[0]))

        color = "" if args.no_color else colorize(dist, args.warn, args.crit)
        rst = "" if args.no_color else ANSI_RST
        sys.stdout.write(
            f"\rt={elapsed:6.1f}s  pkts={lidar.packets:6d}  buf={len(pts):5d}  "
            f"{color}最近: {dist:5.2f}m  方位={bearing:+6.1f}°  "
            f"(x={nearest[0]:+5.2f}, y={nearest[1]:+5.2f}, z={nearest[2]:+5.2f}){rst}  "
            f"n={len(valid)}    "
        )
        sys.stdout.flush()

        if plot_fig is not None:
            # top-down all valid points
            import matplotlib.pyplot as plt
            sc.set_offsets(np.column_stack([valid[:, 0], valid[:, 1]]))
            nearest_marker.set_data([nearest[0]], [nearest[1]])
            plot_fig.canvas.draw_idle()
            plt.pause(0.001)

    print(f"\n\n[stop] 共收到 {lidar.packets} 包 / {lidar.total_points} 点  ({time.time()-t_start:.1f}s)")
    lidar.close()


if __name__ == "__main__":
    main()
