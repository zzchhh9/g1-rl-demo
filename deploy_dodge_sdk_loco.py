"""G1 dodge deployment using Unitree's built-in high-level locomotion.

This keeps the built-in G1 locomotion mode active and sends velocity commands
through unitree_sdk2py.g1.loco.LocoClient.SetVelocity(). It does not publish
rt/lowcmd and does not run the third-party motion.pt locomotion policy.

Usage:
    1. Put the robot in the blue high-level locomotion mode from the remote:
       hold L2 + UP until the controller light is blue.
    2. Run:
       uv run python deploy_dodge_sdk_loco.py eno1 --source yolo --max_vel 0.35

Controls:
    A      -> enable autonomous dodge commands if --wait_for_a is passed
    SELECT -> stop move and exit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.loco.g1_loco_api import (
    ROBOT_API_ID_LOCO_GET_BALANCE_MODE,
    ROBOT_API_ID_LOCO_GET_FSM_ID,
    ROBOT_API_ID_LOCO_GET_FSM_MODE,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG

_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO / "deploy"))
from dodge_policy import DodgePolicy

from deploy_dodge_real import (
    LOCO_LIN_CMD_DEADBAND,
    YoloDdsObstacleDetector,
    _block_toward_obstacle,
    _enforce_away_component,
    _escape_metrics,
    _fmt_vec,
    _motion_mode_name,
    shape_locomotion_velocity,
)
from common.remote_controller import KeyMap, RemoteController


def _quat_to_yaw(q) -> float:
    w, x, y, z = [float(v) for v in q]
    return float(np.arctan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z)))


def _motion_mode(msc: MotionSwitcherClient) -> str:
    code, info = msc.CheckMode()
    return _motion_mode_name(info) if code == 0 else "?"


def _select_ai_mode(msc: MotionSwitcherClient):
    print("[motion_switcher] selecting built-in 'ai' mode for SDK loco")
    code, _ = msc.SelectMode("ai")
    deadline = time.time() + 4.0
    while time.time() < deadline:
        mode = _motion_mode(msc)
        if mode:
            print(f"[motion_switcher] mode='{mode}'")
            return
        time.sleep(0.2)
    raise RuntimeError(f"SelectMode('ai') returned code={code}, mode still empty")


def _parse_loco_data(data):
    if data is None:
        return None
    try:
        parsed = json.loads(data)
    except Exception:
        return data
    if isinstance(parsed, dict) and "data" in parsed:
        return parsed["data"]
    return parsed


def _loco_get(client: LocoClient, api_id: int):
    code, data = client._Call(api_id, "{}")
    return code, _parse_loco_data(data)


def _loco_status_values(client: LocoClient) -> dict:
    fsm_code, fsm = _loco_get(client, ROBOT_API_ID_LOCO_GET_FSM_ID)
    mode_code, mode = _loco_get(client, ROBOT_API_ID_LOCO_GET_FSM_MODE)
    bal_code, bal = _loco_get(client, ROBOT_API_ID_LOCO_GET_BALANCE_MODE)
    return {
        "fsm": fsm,
        "fsm_code": fsm_code,
        "mode": mode,
        "mode_code": mode_code,
        "balance": bal,
        "balance_code": bal_code,
    }


def _loco_status_text(status: dict) -> str:
    return (f"fsm={status.get('fsm')}(code={status.get('fsm_code')}) "
            f"mode={status.get('mode')}(code={status.get('mode_code')}) "
            f"balance={status.get('balance')}(code={status.get('balance_code')})")


def _loco_status(client: LocoClient) -> str:
    return _loco_status_text(_loco_status_values(client))


def _loco_ready(status: dict) -> bool:
    """Best-effort check for the blue high-level locomotion mode.

    On the user's G1, SetVelocity ACKed with code=0 but produced no motion while
    fsm=0. After entering the blue remote locomotion mode (L2+UP), the sport
    service accepts velocity commands as real walking commands.
    """
    try:
        fsm = int(status.get("fsm"))
    except (TypeError, ValueError):
        return False
    return fsm != 0 and status.get("fsm_code") == 0


def _wait_for_blue_loco_mode(client: LocoClient, timeout_s: float) -> dict:
    status = _loco_status_values(client)
    if _loco_ready(status):
        return status

    print("[loco] not in blue high-level locomotion mode yet.")
    print("[loco] Hold L2 + UP on the remote until the controller light is blue.")
    print(f"[loco] waiting up to {timeout_s:.0f}s; current {_loco_status_text(status)}")
    deadline = time.time() + timeout_s
    last_print = 0.0
    while time.time() < deadline:
        status = _loco_status_values(client)
        if _loco_ready(status):
            print(f"[loco] blue locomotion mode ready: {_loco_status_text(status)}")
            return status
        now = time.time()
        if now - last_print > 2.0:
            print(f"[WAIT_BLUE] {_loco_status_text(status)}")
            last_print = now
        time.sleep(0.1)

    raise RuntimeError(
        "Loco FSM stayed at 0. SetVelocity may return code=0 here but it will not move. "
        "Enter blue mode with L2+UP first, then rerun."
    )


def _wait_for_loco_state(client: LocoClient,
                         timeout_s: float,
                         want_fsm: int | None = None,
                         want_mode: int | None = None,
                         want_balance: int | None = None) -> dict:
    deadline = time.time() + timeout_s
    last_print = 0.0
    status = _loco_status_values(client)
    while time.time() < deadline:
        status = _loco_status_values(client)
        ok = True
        for key, want in (("fsm", want_fsm), ("mode", want_mode),
                          ("balance", want_balance)):
            if want is None:
                continue
            try:
                got = int(status.get(key))
            except (TypeError, ValueError):
                ok = False
                break
            if got != int(want):
                ok = False
                break
        if ok:
            return status
        now = time.time()
        if now - last_print > 1.0:
            print(f"[WAIT-LOCO] {_loco_status_text(status)}")
            last_print = now
        time.sleep(0.1)
    return status


class LowStateMonitor:
    def __init__(self, topic: str = "rt/lowstate"):
        self.low_state = None
        self.remote = RemoteController()
        self.sub = ChannelSubscriber(topic, LowStateHG)
        self.sub.Init(self._callback, 10)

    def _callback(self, msg):
        self.low_state = msg
        self.remote.set(msg.wireless_remote)

    def wait(self, timeout_s: float = 5.0):
        deadline = time.time() + timeout_s
        while self.low_state is None and time.time() < deadline:
            time.sleep(0.02)
        if self.low_state is None:
            raise TimeoutError("no rt/lowstate received")

    @property
    def yaw(self) -> float:
        if self.low_state is None:
            return 0.0
        return _quat_to_yaw(self.low_state.imu_state.quaternion)


def _rate_limit(cmd: np.ndarray, target: np.ndarray,
                max_dlin: float, max_dang: float) -> np.ndarray:
    delta = target - cmd
    delta[:2] = np.clip(delta[:2], -max_dlin, max_dlin)
    delta[2] = np.clip(delta[2], -max_dang, max_dang)
    return (cmd + delta).astype(np.float32)


def _body_to_world_xy(v_b: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = float(v_b[0]), float(v_b[1])
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)


def _world_to_body_xy(v_w: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = float(v_w[0]), float(v_w[1])
    return np.array([c * x + s * y, -s * x + c * y], dtype=np.float32)


def _shape_return_velocity(back_to_origin_b: np.ndarray,
                           gain: float,
                           max_vel: float,
                           min_vel: float,
                           done_dist: float) -> np.ndarray:
    dist = float(np.linalg.norm(back_to_origin_b))
    out = np.zeros(3, dtype=np.float32)
    if dist <= done_dist or dist < 1e-6:
        return out
    xy = gain * np.asarray(back_to_origin_b, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(xy))
    max_norm = float(np.sqrt(2.0) * max_vel)
    if norm > max_norm:
        xy *= max_norm / max(norm, 1e-6)
    norm = float(np.linalg.norm(xy))
    if min_vel > 0.0 and norm < min_vel:
        xy *= min_vel / max(norm, 1e-6)
    out[:2] = np.clip(xy, -max_vel, max_vel)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="Network interface, e.g. eno1")
    parser.add_argument("--source", choices=["yolo"], default="yolo")
    parser.add_argument("--startup_mode", choices=["keep", "ai"], default="keep",
                        help="keep current built-in mode, or select 'ai'. Never releases low-level.")
    parser.add_argument("--wait_for_a", action="store_true",
                        help="Wait for remote A before enabling. Default enables immediately "
                             "to avoid the built-in high-level mode consuming A.")
    parser.add_argument("--no_loco_start", action="store_true",
                        help="Do not call LocoClient.Start() on enable.")
    parser.add_argument("--loco_start_wait", type=float, default=0.5,
                        help="Seconds to wait after LocoClient.Start() before velocity commands.")
    parser.add_argument("--balance_mode", type=int, default=1,
                        help="Call LocoClient.SetBalanceMode(value) after Start. "
                             "On this G1, balance=1 is the stable blue walking mode; "
                             "balance=0 has produced unstable in-place stepping. "
                             "Pass -1 to leave unchanged.")
    parser.add_argument("--require_loco_ready", action="store_true", default=True,
                        help="Wait for fsm=200, mode=1 and requested balance before dodge.")
    parser.add_argument("--no_require_loco_ready", dest="require_loco_ready",
                        action="store_false",
                        help="Do not wait for locomotion mode/balance readiness.")
    parser.add_argument("--max_vel", type=float, default=0.35,
                        help="Per-axis high-level velocity cap in m/s.")
    parser.add_argument("--max_ang_vel", type=float, default=0.0,
                        help="Yaw-rate cap rad/s. Default 0 keeps camera pointed at the person.")
    parser.add_argument("--safety_dist", type=float, default=1.5)
    parser.add_argument("--clear_margin", type=float, default=0.10,
                        help="Stop dodge after obstacle distance exceeds safety_dist + margin.")
    parser.add_argument("--cmd_hz", type=float, default=10.0,
                        help="High-level SetVelocity refresh rate.")
    parser.add_argument("--cmd_duration", type=float, default=0.30,
                        help="SetVelocity duration. Short duration prevents runaway if script dies.")
    parser.add_argument("--max_accel_lin", type=float, default=1.5,
                        help="Linear command slew limit in m/s^2.")
    parser.add_argument("--max_accel_ang", type=float, default=2.0,
                        help="Angular command slew limit in rad/s^2.")
    parser.add_argument("--min_loco_lin_vel", type=float, default=0.30,
                        help="Raise nonzero dodge command norm above this value if cap allows it.")
    parser.add_argument("--close_escape_speed", type=float, default=None,
                        help="Minimum radial away speed near the obstacle. Default uses the "
                             "largest feasible radial speed under max_vel.")
    parser.add_argument("--return_enable", action="store_true", default=True,
                        help="After dodge clears, estimate displacement from sent SDK "
                             "velocity commands and walk back toward the start pose.")
    parser.add_argument("--no_return", dest="return_enable", action="store_false",
                        help="Disable post-dodge return.")
    parser.add_argument("--return_max_vel", type=float, default=0.20,
                        help="Per-axis cap for post-dodge return velocity.")
    parser.add_argument("--return_min_vel", type=float, default=0.12,
                        help="Minimum return command norm while estimated displacement "
                             "is still above --return_done_dist.")
    parser.add_argument("--return_gain", type=float, default=0.8,
                        help="P gain from estimated displacement to return velocity.")
    parser.add_argument("--return_done_dist", type=float, default=0.10,
                        help="Estimated displacement below this is considered returned.")
    parser.add_argument("--return_timeout", type=float, default=5.0,
                        help="Maximum seconds spent in one RETURN phase.")
    parser.add_argument("--return_clear_delay", type=float, default=0.5,
                        help="Obstacle must stay clear/missing this long before RETURN starts.")
    parser.add_argument("--return_odom_scale", type=float, default=1.0,
                        help="Scale factor for integrating sent SDK velocity as rough odometry.")
    parser.add_argument("--dodge_ewma_alpha", type=float, default=0.2)
    parser.add_argument("--yolo_staleness", type=float, default=0.5)
    parser.add_argument("--yolo_hold_timeout", type=float, default=1.0)
    parser.add_argument("--yolo_no_kalman", action="store_true")
    parser.add_argument("--yolo_kf_meas_std", type=float, default=0.10)
    parser.add_argument("--yolo_kf_meas_std_close", type=float, default=0.30)
    parser.add_argument("--yolo_kf_gate", type=float, default=4.0)
    parser.add_argument("--debug_obs", action="store_true")
    parser.add_argument("--debug_every", type=int, default=5)
    parser.add_argument("--l2b_damp_hold", type=float, default=2.0,
                        help="Software damping shortcut: while this script is running, "
                             "holding remote L2+B for this many seconds sends zero "
                             "velocity then LocoClient.Damp() and exits. Pass 0 to disable.")
    args = parser.parse_args()

    args.cmd_hz = max(1.0, float(args.cmd_hz))
    if args.close_escape_speed is None:
        args.close_escape_speed = float(np.sqrt(2.0) * args.max_vel)
    else:
        args.close_escape_speed = max(0.0, float(args.close_escape_speed))

    ChannelFactoryInitialize(0, args.net)

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    mode = _motion_mode(msc)
    if args.startup_mode == "ai":
        _select_ai_mode(msc)
    elif not mode:
        raise SystemExit("[motion_switcher] current mode is empty. This script needs the "
                         "built-in high-level mode, not low-level/debug. Put G1 back in "
                         "Regular/AI mode or rerun with --startup_mode ai.")
    else:
        print(f"[motion_switcher] keeping current built-in mode='{mode}'")

    low = LowStateMonitor()
    low.wait()

    loco = LocoClient()
    loco.SetTimeout(2.0)
    loco.Init()
    print(f"[loco] initial {_loco_status(loco)}")

    dodge = DodgePolicy(str(_REPO / "checkpoints" / "dodge_v23b_54400.pt"))
    detector = YoloDdsObstacleDetector(
        staleness_threshold=args.yolo_staleness,
        hold_timeout=args.yolo_hold_timeout,
        use_kalman=(not args.yolo_no_kalman),
        kf_meas_std=args.yolo_kf_meas_std,
        kf_meas_std_close=args.yolo_kf_meas_std_close,
        kf_gate_sigma=args.yolo_kf_gate,
    )

    robot_pos = np.array([0.0, 0.0, 0.8], dtype=np.float32)
    robot_xy_est = np.zeros(2, dtype=np.float32)
    cmd = np.zeros(3, dtype=np.float32)
    target = np.zeros(3, dtype=np.float32)
    ewma = np.zeros(3, dtype=np.float32)
    active = False
    return_active = False
    return_start_t = 0.0
    clear_since = None
    enabled = not args.wait_for_a
    started = False
    counter = 0
    l2b_since = None
    max_dlin = args.max_accel_lin / args.cmd_hz
    max_dang = args.max_accel_ang / args.cmd_hz

    print("\n[SDK LOCO DODGE]")
    print(f"  backend=Unitree LocoClient.SetVelocity, no rt/lowcmd, no motion.pt")
    print(f"  max_vel={args.max_vel:.2f}m/s max_ang={args.max_ang_vel:.2f}rad/s "
          f"safety={args.safety_dist:.2f}m cmd_hz={args.cmd_hz:.1f}")
    print(f"  loco state: start={'off' if args.no_loco_start else 'fsm=200'} "
          f"balance={'keep' if args.balance_mode < 0 else args.balance_mode} "
          f"require_ready={args.require_loco_ready}")
    print(f"  close_escape min_away={args.close_escape_speed:.2f}m/s "
          f"(capped by per-axis max_vel and bearing)")
    print(f"  return: {'on' if args.return_enable else 'off'} "
          f"max_vel={args.return_max_vel:.2f}m/s done={args.return_done_dist:.2f}m "
          f"delay={args.return_clear_delay:.1f}s timeout={args.return_timeout:.1f}s")
    if args.wait_for_a:
        print("  Press A to enable. Press SELECT to StopMove and exit.")
    else:
        print("  Enabled immediately. Press SELECT to StopMove and exit.")
    if args.l2b_damp_hold > 0:
        print(f"  Software emergency: hold L2+B for {args.l2b_damp_hold:.1f}s "
              "to send Damp() and exit.\n")
    else:
        print()

    period = 1.0 / args.cmd_hz
    try:
        while True:
            t0 = time.time()
            counter += 1
            if args.l2b_damp_hold > 0:
                l2b_pressed = (low.remote.button[KeyMap.L2] == 1
                               and low.remote.button[KeyMap.B] == 1)
                if l2b_pressed:
                    if l2b_since is None:
                        l2b_since = time.time()
                        print("[EMERGENCY] L2+B detected, keep holding for "
                              f"{args.l2b_damp_hold:.1f}s to enter damping")
                    elif time.time() - l2b_since >= args.l2b_damp_hold:
                        print("[EMERGENCY] L2+B hold reached; sending zero velocity + Damp()")
                        for _ in range(10):
                            try:
                                loco.SetVelocity(0.0, 0.0, 0.0, duration=0.05)
                            except Exception as e:
                                print(f"[WARN] emergency zero failed: {e}")
                            time.sleep(0.02)
                        try:
                            loco.Damp()
                        except Exception as e:
                            print(f"[WARN] emergency Damp failed: {e}")
                        break
                else:
                    l2b_since = None
            if low.remote.button[KeyMap.select] == 1:
                break
            if not enabled and args.wait_for_a and low.remote.button[KeyMap.A] == 1:
                enabled = True
            if enabled and not started:
                if not args.no_loco_start:
                    print("[loco] SetFsmId(200)  # Start")
                    code = loco.SetFsmId(200)
                    print(f"[loco] SetFsmId(200) returned code={code}")
                    time.sleep(max(0.0, args.loco_start_wait))
                    print(f"[loco] after Start {_loco_status(loco)}")
                if args.balance_mode >= 0:
                    print(f"[loco] SetBalanceMode({args.balance_mode})")
                    code = loco.SetBalanceMode(int(args.balance_mode))
                    print(f"[loco] SetBalanceMode returned code={code}")
                    time.sleep(0.2)
                    print(f"[loco] after BalanceMode {_loco_status(loco)}")
                if args.require_loco_ready:
                    want_fsm = None if args.no_loco_start else 200
                    want_balance = None if args.balance_mode < 0 else int(args.balance_mode)
                    status = _wait_for_loco_state(
                        loco, timeout_s=4.0, want_fsm=want_fsm,
                        want_mode=1, want_balance=want_balance)
                    print(f"[loco] ready check {_loco_status_text(status)}")
                    if (want_fsm is not None and int(status.get("fsm", -1)) != want_fsm
                        or int(status.get("mode", -1)) != 1
                        or (want_balance is not None
                            and int(status.get("balance", -1)) != want_balance)):
                        raise RuntimeError(
                            "locomotion did not reach the stable walking state. "
                            "Do not send dodge velocities; put the robot in blue mode "
                            "and retry."
                        )
                dodge.reset(robot_pos[:2], low.yaw)
                ewma[:] = 0.0
                started = True
                print("[ENABLE] SDK loco dodge commands enabled")

            yaw = low.yaw
            obstacle_pos = detector.detect(robot_pos, yaw)
            dist = (float(np.linalg.norm(obstacle_pos[:2] - robot_pos[:2]))
                    if obstacle_pos is not None else float("inf"))
            ydbg = detector.debug_snapshot
            obstacle_vel_w = None
            if ydbg.get("kf_vxy") is not None:
                obstacle_vel_w = np.asarray(ydbg["kf_vxy"], dtype=np.float32)

            if enabled and obstacle_pos is not None and dist < args.safety_dist:
                if not active:
                    if return_active:
                        print(f"[RETURN INTERRUPT] obstacle dist={dist:.2f}m")
                    return_active = False
                    active = True
                    clear_since = None
                    dodge.reset(robot_pos[:2], yaw)
                    ewma[:] = 0.0
                    print(f"[DODGE START] dist={dist:.2f}m")

            safe_dbg = {"changed": False}
            if not enabled:
                target[:] = 0.0
                active = False
                return_active = False
            elif active and obstacle_pos is not None and dist <= args.safety_dist + args.clear_margin:
                obs18 = dodge.build_obs(robot_pos, yaw, obstacle_pos,
                                        dt=period, obstacle_vel_w=obstacle_vel_w)
                vel = dodge.get_velocity_command(obs18)
                ewma[:] = (1.0 - args.dodge_ewma_alpha) * ewma + args.dodge_ewma_alpha * vel
                target[:] = shape_locomotion_velocity(
                    ewma,
                    max_lin_vel=args.max_vel,
                    max_ang_vel=args.max_ang_vel,
                    lin_deadband=LOCO_LIN_CMD_DEADBAND,
                    min_lin_vel=args.min_loco_lin_vel,
                    max_yaw_when_translating=0.0,
                )
                obs_b = dodge._body_frame_xy(obstacle_pos[:2] - robot_pos[:2], yaw)
                target[:], _ = _block_toward_obstacle(target, obs_b, max_toward=0.0)
                target[:], safe_dbg = _enforce_away_component(
                    target, obs_b, args.close_escape_speed, args.max_vel)
                clear_since = None
            elif active:
                now = time.time()
                if clear_since is None:
                    clear_since = now
                    print(f"[DODGE CLEAR] dist={dist:.2f}m; waiting "
                          f"{args.return_clear_delay:.1f}s before return")
                target[:] = 0.0
                if now - clear_since >= args.return_clear_delay:
                    disp_mag = float(np.linalg.norm(robot_xy_est))
                    print(f"[DODGE STOP] dist={dist:.2f}m est_disp={disp_mag:.2f}m")
                    active = False
                    ewma[:] = 0.0
                    clear_since = None
                    if args.return_enable and disp_mag > args.return_done_dist:
                        return_active = True
                        return_start_t = now
                        print(f"[RETURN START] est_disp={disp_mag:.2f}m "
                              f"est_xy={_fmt_vec(robot_xy_est, 2)}")
                    else:
                        return_active = False
                        target[:] = 0.0
            elif return_active:
                now = time.time()
                disp_mag = float(np.linalg.norm(robot_xy_est))
                clear_enough = (obstacle_pos is None
                                or dist >= args.safety_dist + args.clear_margin)
                if not clear_enough:
                    target[:] = 0.0
                elif disp_mag <= args.return_done_dist:
                    print(f"[RETURN DONE] est_disp={disp_mag:.2f}m")
                    return_active = False
                    robot_xy_est[:] = 0.0
                    target[:] = 0.0
                elif now - return_start_t >= args.return_timeout:
                    print(f"[RETURN TIMEOUT] est_disp={disp_mag:.2f}m; zeroing estimate")
                    return_active = False
                    robot_xy_est[:] = 0.0
                    target[:] = 0.0
                else:
                    back_b = _world_to_body_xy(-robot_xy_est, yaw)
                    target[:] = _shape_return_velocity(
                        back_b,
                        gain=args.return_gain,
                        max_vel=args.return_max_vel,
                        min_vel=args.return_min_vel,
                        done_dist=args.return_done_dist,
                    )
            else:
                target[:] = 0.0

            cmd[:] = _rate_limit(cmd, target, max_dlin, max_dang)
            code = loco.SetVelocity(float(cmd[0]), float(cmd[1]), float(cmd[2]),
                                    duration=float(args.cmd_duration))
            if enabled and started and code == 0:
                robot_xy_est[:] = (robot_xy_est
                                   + _body_to_world_xy(cmd[:2], yaw)
                                   * period * float(args.return_odom_scale))
            if counter % max(1, int(args.cmd_hz)) == 0:
                if code != 0:
                    print(f"[WARN] SetVelocity returned code={code} "
                          f"{_loco_status(loco)}")
                elif args.debug_obs:
                    print(f"[DBG-LOCO] SetVelocity code=0 {_loco_status(loco)}")

            if counter % max(1, int(args.debug_every)) == 0:
                obs_b = None
                geom = None
                if obstacle_pos is not None:
                    obs_b = dodge._body_frame_xy(obstacle_pos[:2] - robot_pos[:2], yaw)
                    geom = _escape_metrics(obs_b, cmd[:2])
                state = ("DODGE" if active else
                         ("RETURN" if return_active else
                          ("READY" if enabled else "WAIT_A")))
                msg = (f"[{state:6s}] cmd={_fmt_vec(cmd, 2)} target={_fmt_vec(target, 2)} "
                       f"dist={'N/A' if not np.isfinite(dist) else f'{dist:.2f}'} "
                       f"obs_b={_fmt_vec(obs_b, 2)} "
                       f"est_xy={_fmt_vec(robot_xy_est, 2)}")
                if geom is not None:
                    msg += (f" dir={geom['verdict']} par={geom['parallel']:+.2f} "
                            f"perp={geom['perp']:+.2f}")
                print(msg)
                if args.debug_obs and active and safe_dbg.get("changed"):
                    before = safe_dbg.get("before") or {}
                    after = safe_dbg.get("after") or {}
                    print(f"  [DBG-SAFE] close_escape "
                          f"par {before.get('parallel', 0.0):+.3f}"
                          f"->{after.get('parallel', 0.0):+.3f} "
                          f"perp {before.get('perp', 0.0):+.3f}"
                          f"->{after.get('perp', 0.0):+.3f}")

            sleep_s = period - (time.time() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        print("[EXIT] zero velocity + StopMove")
        try:
            for _ in range(10):
                loco.SetVelocity(0.0, 0.0, 0.0, duration=0.05)
                time.sleep(0.02)
            loco.StopMove()
        except Exception as e:
            print(f"[WARN] StopMove failed: {e}")


if __name__ == "__main__":
    main()
