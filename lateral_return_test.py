#!/usr/bin/env python3
"""独立的“左右平移 + 回原点”运动测试 —— 不依赖 YOLO / dodge 策略 / checkpoint。

序列：  左移 D 米  ->  回原点  ->  右移 D 米  ->  回原点
- 用 Unitree LocoClient.SetVelocity 做 body 侧向(±y)平移；
- 用 SLAM 里程计 rt/dodge/odom 做闭环：到达 D 就停，回到原点附近就停。
目的：单独验证机器人能不能“按命令精确平移、再回到原点”，把 YOLO/相机/dodge 全部排除掉。

跑法（机器人先切蓝色行走模式、急停在手）：
    uv run python lateral_return_test.py --balance_mode 0
若发现“左/右”方向反了，加  --left_sign -1
急停：Ctrl-C（会 StopMove）或遥控器物理急停。命令 duration 很短，脚本一停机器人就停。

说明 / 局限：
- “回原点”是反向侧移到 odom 距原点 < done_dist；如果机器人侧移时有明显前后漂移，
  纯侧移回不准会超时 —— 那本身就说明运动/平衡有前后漂（正是要诊断的）。
- 全程打印离原点距离，方便看“走没走够 0.5m、回没回准”。
"""
import argparse
import json
import sys
import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


class Odom:
    """订阅 rt/dodge/odom (JSON String)，暴露最新 (x, y, yaw)。"""

    def __init__(self, topic: str):
        self._lock = threading.Lock()
        self._d = None
        self._t = 0.0
        self.sub = ChannelSubscriber(topic, String_)
        self.sub.Init(self._cb, 10)

    def _cb(self, msg):
        try:
            d = json.loads(msg.data)
            x = float(d["x"]); y = float(d["y"])
        except Exception:
            return
        if not (np.isfinite(x) and np.isfinite(y)):
            return
        with self._lock:
            self._d = (x, y, float(d.get("yaw", 0.0)))
            self._t = time.time()

    def get(self, max_age: float = 0.5):
        with self._lock:
            d = self._d
            age = (time.time() - self._t) if self._d else 1e9
        return d if (d is not None and age <= max_age) else None

    def wait(self, timeout: float = 10.0, max_age: float = 0.5):
        t0 = time.time()
        while time.time() - t0 < timeout:
            g = self.get(max_age)
            if g:
                return g
            time.sleep(0.05)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="eno1")
    ap.add_argument("--odom_topic", default="rt/dodge/odom")
    ap.add_argument("--move_dist", type=float, default=0.5, help="左/右平移距离 (m)")
    ap.add_argument("--speed", type=float, default=0.22, help="侧移速度上限 (m/s)")
    ap.add_argument("--min_vel", type=float, default=0.12, help="最小命令速度，克服死区")
    ap.add_argument("--kp", type=float, default=1.0, help="接近目标时的 P 减速增益")
    ap.add_argument("--reach_tol", type=float, default=0.04, help="到达 move_dist 容差 (m)")
    ap.add_argument("--done_dist", type=float, default=0.06, help="回到原点阈值 (m)")
    ap.add_argument("--phase_timeout", type=float, default=12.0, help="每段最长秒数")
    ap.add_argument("--settle", type=float, default=1.5, help="每段后停稳秒数")
    ap.add_argument("--left_sign", type=float, default=1.0, help="+1: vy>0 朝左；实机反了用 -1")
    ap.add_argument("--balance_mode", type=int, default=0, help="SetBalanceMode 值 (0=静态, 1=连续步态)")
    ap.add_argument("--cmd_hz", type=float, default=10.0)
    ap.add_argument("--loco_start_wait", type=float, default=1.0)
    args = ap.parse_args()

    ChannelFactoryInitialize(0, args.net)
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()  # 保持当前高层模式（和 deploy 一致，用的是高层 LocoClient，无需 ReleaseMode）
    odom = Odom(args.odom_topic)
    loco = LocoClient()
    loco.SetTimeout(2.0)
    loco.Init()

    print("[odom] 等待 rt/dodge/odom ...")
    if odom.wait(10.0) is None:
        print("[FATAL] 收不到里程计，先把 LIO / odom bridge 起好再跑")
        sys.exit(1)

    print(f"[loco] SetFsmId(200) Start -> code={loco.SetFsmId(200)}")
    time.sleep(args.loco_start_wait)
    print(f"[loco] SetBalanceMode({args.balance_mode}) -> code={loco.SetBalanceMode(args.balance_mode)}")
    time.sleep(0.3)

    g = odom.wait(5.0)
    if g is None:
        print("[FATAL] enable 后丢失里程计")
        sys.exit(1)
    origin = np.array([g[0], g[1]], dtype=float)
    print(f"[origin] 记录原点 x={origin[0]:.3f} y={origin[1]:.3f} yaw={g[2]:+.3f}")
    period = 1.0 / args.cmd_hz
    LS = args.left_sign

    def disp():
        c = odom.get(0.5)
        if c is None:
            return None
        return float(np.linalg.norm(np.array([c[0], c[1]]) - origin))

    def stop(tag):
        for _ in range(6):
            loco.SetVelocity(0.0, 0.0, 0.0, duration=0.1)
            time.sleep(0.02)
        loco.StopMove()
        time.sleep(args.settle)
        d = disp()
        print(f"[{tag}] 停稳, 离原点 {d:.3f} m" if d is not None else f"[{tag}] 停稳(无 odom)")

    def go(tag, vy_sign, target_out):
        # target_out=True: 走到 disp >= move_dist；False: 回到 disp <= done_dist
        print(f"[{tag}] {'外移' if target_out else '回原点'} 开始")
        t0 = time.time()
        step = 0
        while time.time() - t0 < args.phase_timeout:
            d = disp()
            if d is None:
                loco.SetVelocity(0.0, 0.0, 0.0, duration=0.2)
                time.sleep(period)
                continue
            if target_out and d >= args.move_dist - args.reach_tol:
                print(f"[{tag}] 到达 {d:.3f} m")
                return True
            if (not target_out) and d <= args.done_dist:
                print(f"[{tag}] 回到原点 {d:.3f} m")
                return True
            rem = (args.move_dist - d) if target_out else d
            mag = max(args.min_vel, min(args.speed, args.kp * rem))
            vy = vy_sign * mag
            loco.SetVelocity(0.0, float(vy), 0.0, duration=0.4)
            if step % 5 == 0:
                print(f"  [{tag}] disp={d:.3f}m  vy={vy:+.2f}")
            step += 1
            time.sleep(period)
        fd = disp()
        fds = f"{fd:.3f}" if fd is not None else "?"
        print(f"[{tag}] 超时! 最终 disp={fds} m (没走够 / 没回准)")
        return False

    try:
        go("LEFT",  +LS, True)
        stop("LEFT")
        go("RET1",  -LS, False)
        stop("RET1")
        go("RIGHT", -LS, True)
        stop("RIGHT")
        go("RET2",  +LS, False)
        stop("RET2")
        print("[DONE] 左移0.5 -> 回 -> 右移0.5 -> 回  全部跑完")
    except KeyboardInterrupt:
        print("\n[INT] 用户中断")
    finally:
        for _ in range(6):
            loco.SetVelocity(0.0, 0.0, 0.0, duration=0.1)
            time.sleep(0.02)
        loco.StopMove()
        print("[EXIT] 已 StopMove。需要彻底急停请用遥控器。")


if __name__ == "__main__":
    main()
