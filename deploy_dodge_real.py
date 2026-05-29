"""Real G1 dodge deployment: dodge policy + locomotion via unitree_sdk2.

Extends deploy_real.py with:
  1. LiDAR obstacle detection (unitree_sdk2py DDS 订阅 PointCloud2)
  2. Dodge policy overrides locomotion velocity command
  3. Return head + yaw P-controller for post-dodge return
  4. Safety limits with configurable max velocity

Usage:
    uv run python deploy_dodge_real.py eth0 configs/g1.yaml

Controls:
    START  → exit zero-torque, begin standing
    A      → enable walking + dodge mode
    SELECT → EMERGENCY STOP (any time)

LiDAR: 通过 unitree_sdk2py DDS 订阅 rt/utlidar/voxel_map (PointCloud2),
       与 test_lidar.py 使用完全相同的接口。不需要 ROS2。
"""

from __future__ import annotations

import json
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add deploy package
_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO / "deploy"))
from dodge_policy import DodgePolicy

from legged_gym import LEGGED_GYM_ROOT_DIR
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

sys.path.insert(0, str(Path(LEGGED_GYM_ROOT_DIR) / "deploy" / "deploy_real"))
from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, MotorMode
from common.rotation_helper import get_gravity_orientation
from common.remote_controller import RemoteController, KeyMap
from config import Config


LOCO_LIN_CMD_DEADBAND = 0.20  # legged_gym zeros sampled xy commands <= 0.2 m/s
BUILTIN_STARTUP_MODE_ALIASES = {
    "regular": ("regular", "normal", "ai"),
    "normal": ("normal", "regular", "ai"),
    "ai": ("ai",),
}


def _fmt_vec(v, ndigits: int = 2) -> str:
    if v is None:
        return "None"
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    return "[" + ",".join(f"{x:+.{ndigits}f}" for x in arr) + "]"


def _fmt_float(x, ndigits: int = 2) -> str:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return "None"
    if not np.isfinite(val):
        return "nan"
    return f"{val:+.{ndigits}f}"


def _fmt_range(v, ndigits: int = 2) -> str:
    if v is None:
        return "None"
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return "[]"
    return (f"min={float(np.min(arr)):+.{ndigits}f} "
            f"max={float(np.max(arr)):+.{ndigits}f} "
            f"rms={float(np.sqrt(np.mean(arr * arr))):.{ndigits}f}")


def _escape_metrics(obs_b_xy, cmd_xy):
    """Project a body-frame cmd onto the body-frame obstacle ray."""
    if obs_b_xy is None or cmd_xy is None:
        return None
    obs = np.asarray(obs_b_xy, dtype=np.float32).reshape(2)
    cmd = np.asarray(cmd_xy, dtype=np.float32).reshape(2)
    dist = float(np.linalg.norm(obs))
    if dist < 1e-4:
        return None
    unit = obs / dist
    par = float(np.dot(cmd, unit))
    perp = float(unit[0] * cmd[1] - unit[1] * cmd[0])
    if par > 0.05:
        verdict = "toward"
    elif par < -0.05:
        verdict = "away"
    else:
        verdict = "lateral"
    return {"dist": dist, "bearing": float(np.degrees(np.arctan2(obs[1], obs[0]))),
            "parallel": par, "perp": perp, "verdict": verdict}


def _block_toward_obstacle(cmd: np.ndarray, obs_b_xy, max_toward: float = 0.0):
    """Remove body-frame velocity component that points toward the obstacle."""
    metrics = _escape_metrics(obs_b_xy, cmd[:2])
    if metrics is None or metrics["parallel"] <= max_toward:
        return cmd.astype(np.float32), {"changed": False, "before": metrics, "after": metrics}
    out = np.asarray(cmd, dtype=np.float32).copy()
    obs = np.asarray(obs_b_xy, dtype=np.float32).reshape(2)
    unit = obs / max(float(np.linalg.norm(obs)), 1e-6)
    out[:2] -= (metrics["parallel"] - max_toward) * unit
    after = _escape_metrics(obs_b_xy, out[:2])
    return out.astype(np.float32), {"changed": True, "before": metrics, "after": after}


def _enforce_away_component(
    cmd: np.ndarray,
    obs_b_xy,
    min_away_speed: float,
    max_lin_vel: float,
):
    """Ensure a minimum velocity component directly away from the obstacle.

    `parallel` is positive toward the obstacle and negative away.  If the
    policy mostly asks for tangential motion, close-range separation can stall.
    This helper preserves as much tangential command as possible while making
    the away component at least `min_away_speed`.
    """
    metrics = _escape_metrics(obs_b_xy, cmd[:2])
    if metrics is None or min_away_speed <= 0.0:
        return cmd.astype(np.float32), {"changed": False, "before": metrics, "after": metrics}

    obs = np.asarray(obs_b_xy, dtype=np.float32).reshape(2)
    unit = obs / max(float(np.linalg.norm(obs)), 1e-6)
    perp_unit = np.array([-unit[1], unit[0]], dtype=np.float32)
    current_par = metrics["parallel"]
    current_perp = metrics["perp"]

    max_feasible_away = float(max_lin_vel) / max(float(np.max(np.abs(unit))), 1e-6)
    desired_away = min(float(min_away_speed), max_feasible_away)
    target_par = min(current_par, -desired_away)
    if current_par <= target_par + 1e-6:
        return cmd.astype(np.float32), {"changed": False, "before": metrics, "after": metrics}

    def build(perp_scale: float) -> np.ndarray:
        return target_par * unit + (current_perp * perp_scale) * perp_unit

    # Reduce tangential motion only as much as needed to respect per-axis caps.
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = (lo + hi) * 0.5
        xy = build(mid)
        if np.max(np.abs(xy)) <= max_lin_vel + 1e-6:
            lo = mid
        else:
            hi = mid
    out = np.asarray(cmd, dtype=np.float32).copy()
    out[:2] = build(lo)
    out[:2] = np.clip(out[:2], -max_lin_vel, max_lin_vel)
    after = _escape_metrics(obs_b_xy, out[:2])
    return out.astype(np.float32), {
        "changed": True,
        "before": metrics,
        "after": after,
        "target_par": target_par,
        "perp_scale": lo,
    }


def _motion_mode_name(info) -> str:
    return (info or {}).get("name", "") if isinstance(info, dict) else ""


def select_builtin_startup_mode(msc: MotionSwitcherClient, requested: str) -> str:
    aliases = BUILTIN_STARTUP_MODE_ALIASES.get(requested, (requested,))
    for name in aliases:
        print(f"[motion_switcher] selecting built-in '{name}' startup mode")
        code, _ = msc.SelectMode(name)
        deadline = time.time() + 4.0
        mode = ""
        while time.time() < deadline:
            chk_code, info = msc.CheckMode()
            mode = _motion_mode_name(info) if chk_code == 0 else ""
            if mode:
                print(f"[motion_switcher] mode='{mode}' ✓ (built-in startup)")
                return mode
            time.sleep(0.2)
        print(f"[motion_switcher] SelectMode('{name}') returned code={code}, mode still empty")
    raise RuntimeError(f"failed to select built-in startup mode from aliases {aliases}")


def release_motion_mode_for_low_level(msc: MotionSwitcherClient):
    code, info = msc.CheckMode()
    mode = _motion_mode_name(info) if code == 0 else "?"
    if mode:
        print(f"[motion_switcher] releasing '{mode}' for low-level control")
        msc.ReleaseMode()
        time.sleep(0.5)
        code, info = msc.CheckMode()
        mode = _motion_mode_name(info) if code == 0 else "?"
        if mode:
            print(f"❌ failed to release motion mode (still '{mode}'). "
                  f"Press L2+B on remote for 3s then retry. Aborting.",
                  file=sys.stderr)
            sys.exit(2)
    print(f"[motion_switcher] mode='{mode}' ✓ (low-level control enabled)")


def shape_locomotion_velocity(
    raw_cmd: np.ndarray,
    max_lin_vel: float,
    max_ang_vel: float,
    lin_deadband: float = LOCO_LIN_CMD_DEADBAND,
    min_lin_vel: float = 0.30,
    max_yaw_when_translating: float | None = 0.0,
    lin_intent_threshold: float = 0.05,
) -> np.ndarray:
    """Map dodge-policy velocity into the real G1 locomotion policy's usable range."""
    raw = np.asarray(raw_cmd, dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)

    out[:2] = np.clip(raw[:2], -max_lin_vel, max_lin_vel)
    raw_lin_norm = float(np.linalg.norm(raw[:2]))
    lin_norm = float(np.linalg.norm(out[:2]))

    if raw_lin_norm > lin_intent_threshold and lin_norm > 1e-6 and min_lin_vel > 0.0:
        max_possible_norm = float(np.sqrt(2.0) * max_lin_vel)
        target_norm = min(max(float(min_lin_vel), float(lin_deadband) + 1e-3),
                          max_possible_norm)
        if lin_norm < target_norm:
            out[:2] *= target_norm / lin_norm
            out[:2] = np.clip(out[:2], -max_lin_vel, max_lin_vel)

    yaw_cap = float(max_ang_vel)
    if raw_lin_norm > lin_intent_threshold and max_yaw_when_translating is not None:
        yaw_cap = min(yaw_cap, float(max_yaw_when_translating))
    out[2] = float(np.clip(raw[2], -yaw_cap, yaw_cap))
    return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# LiDAR obstacle detection (Livox Mid-360 UDP direct, 与 test_lidar.py 一致)
# ═══════════════════════════════════════════════════════════════

from test_lidar import LivoxLidarReceiver


class LidarObstacleDetector:
    """直接监听机器人转发过来的 Livox Mid-360 UDP 点云，做障碍物检测。

    数据链路 (见 test_lidar.py 文档头)：
      Mid-360 (.120) → robot .164:56301 (UDP) → forwarder → laptop :56301

    复用 test_lidar.LivoxLidarReceiver 做接收+解析，本类专注于"把点云转到
    world frame、过滤地面、返回最近障碍物质心"的检测逻辑。
    """

    def __init__(self, port: int = 56301,
                 min_height: float = 0.3, max_range: float = 5.0):
        self._min_height = min_height
        self._max_range = max_range
        self.receiver = LivoxLidarReceiver(port=port)
        print(f"[LiDAR] UDP receiver on :{port}, min_h={min_height}, max_r={max_range}")

    def detect(self, robot_pos: np.ndarray,
               robot_yaw: float) -> np.ndarray | None:
        """检测最近障碍物，返回 world frame 位置 [x,y,z] 或 None。"""
        pts = self.receiver.points
        if pts is None or len(pts) == 0:
            return None

        # LiDAR 点在 sensor frame (X-前, Y-左, Z-上)
        # 转换到 world frame
        cy, sy = np.cos(robot_yaw), np.sin(robot_yaw)
        wx = pts[:, 0] * cy - pts[:, 1] * sy + robot_pos[0]
        wy = pts[:, 0] * sy + pts[:, 1] * cy + robot_pos[1]
        wz = pts[:, 2] + robot_pos[2]

        # 过滤: 地面以上 + 范围内 + 排除自身
        mask = wz > self._min_height
        dists = np.sqrt((wx - robot_pos[0])**2 + (wy - robot_pos[1])**2)
        mask &= (dists < self._max_range) & (dists > 0.3)

        if mask.sum() == 0:
            return None

        filtered = np.stack([wx[mask], wy[mask], wz[mask]], axis=1)
        return filtered.mean(axis=0).astype(np.float32)

    @property
    def msg_count(self) -> int:
        return self.receiver.packets

    def stop(self):
        self.receiver.close()


# ═══════════════════════════════════════════════════════════════
# YOLO DDS obstacle detector (subscribes rt/yolo/person, published by
# scripts/yolo_to_dds_laptop.py). Drop-in replacement for LidarObstacleDetector.
# ═══════════════════════════════════════════════════════════════

class YoloDdsObstacleDetector:
    """Subscribes to `rt/yolo/person` (JSON-in-String_), runs constant-velocity
    Kalman filter (state = world xy + vxy), exposes filtered world-frame nearest
    person via `detect(robot_pos, robot_yaw)`.

    Three layers of robustness:
      (1) **Staleness**: msg > threshold sec old → treat as no detection
          (protects against laptop / network failure while robot walking)
      (2) **CV Kalman filter**: smooth body-frame YOLO+depth noise; predict
          at sim 50Hz between sparse YOLO updates (~3.5Hz)
      (3) **Mahalanobis outlier gate + adaptive R**: reject sudden jumps from
          ID swaps or depth median glitches; close-range measurements get
          inflated R because RealSense depth at <0.5m is much noisier than at 2m
    """

    def __init__(self,
                 topic: str = "rt/yolo/person",
                 staleness_threshold: float = 0.5,
                 hold_timeout: float = 1.0,
                 use_kalman: bool = True,
                 kf_process_pos_std: float = 0.02,
                 kf_process_vel_std: float = 0.4,
                 kf_meas_std: float = 0.10,
                 kf_meas_std_close: float = 0.30,
                 kf_gate_sigma: float = 4.0,
                 lock_first_track: bool = False,
                 lock_track_id: int | None = None,
                 commit_enable: bool = True,
                 commit_dist: float = 1.3,
                 commit_min_speed: float = 0.10,
                 commit_break_dist: float = 0.6,
                 commit_max_time: float = 3.0,
                 commit_min_accepts: int = 3):
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._latest_recv_time: float = 0.0
        self._msg_count = 0
        self._staleness = staleness_threshold
        # Hold-through: when YOLO publishes n=0 or staleness > threshold but
        # the KF still has a recent accepted measurement, keep returning
        # KF-predicted position for up to hold_timeout seconds. Prevents
        # DODGE↔RETURN flip-flop when YOLO intermittently loses person.
        self._hold_timeout = hold_timeout
        self._kf_last_accept_t: float | None = None
        self._last_z: float = -0.48     # remembered for hold-mode

        # KF config
        self._use_kalman = use_kalman
        self._kf_q_pos = kf_process_pos_std
        self._kf_q_vel = kf_process_vel_std
        self._kf_r_std = kf_meas_std
        self._kf_r_std_close = kf_meas_std_close
        self._kf_gate = kf_gate_sigma
        # KF state
        self._kf_x = None    # np.array shape (4,) = [wx, wy, vx, vy]
        self._kf_P = None    # np.array shape (4,4)
        self._kf_last_pred_t = None
        self._kf_last_frame_id = -1
        self._kf_n_rejected = 0
        self._kf_n_accepted = 0
        self._kf_consec_rej = 0
        self._kf_max_consec_rej = 5     # ≥5 rejections in a row → track loss → reset
        self._kf_n_held = 0             # ticks we returned predict-only state
        self._debug = {"status": "none"}
        self._last_nis: float | None = None
        self._lock_first_track = bool(lock_first_track)
        self._locked_track_id = (int(lock_track_id)
                                 if lock_track_id is not None and int(lock_track_id) >= 0
                                 else None)
        self._ignored_track_count = 0

        # Commit-and-project (input alignment): once a person is locked as
        # approaching, drive the policy with a SMOOTH constant-velocity projection
        # through the dodge instead of the jumpy close-range YOLO. real YOLO only
        # (a) triggers the lock and (b) seeds direction+speed from the smooth approach.
        self._commit_enable = bool(commit_enable)
        self._commit_dist = float(commit_dist)
        self._commit_min_speed = float(commit_min_speed)
        self._commit_break_dist = float(commit_break_dist)
        self._commit_max_time = float(commit_max_time)
        self._commit_min_accepts = int(commit_min_accepts)
        self._commit_active = False
        self._commit_p = None      # world xy at lock
        self._commit_v = None      # world vxy at lock
        self._commit_t0 = None
        self._commit_z = -0.48

        self.subscriber = ChannelSubscriber(topic, String_)
        self.subscriber.Init(self._callback, 10)
        kf_str = (f"KF q_pos={kf_process_pos_std} q_vel={kf_process_vel_std} "
                  f"R={kf_meas_std}/{kf_meas_std_close} gate={kf_gate_sigma}σ"
                  if use_kalman else "KF=OFF (raw passthrough)")
        print(f"[YOLO-DDS] subscribing to {topic}, stale>{staleness_threshold}s ⇒ no-detect")
        print(f"[YOLO-DDS] {kf_str}")
        if self._locked_track_id is not None:
            print(f"[YOLO-DDS] locked to track_id={self._locked_track_id}")

    def _callback(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        with self._lock:
            self._latest = data
            self._latest_recv_time = time.time()
            self._msg_count += 1

    def _reset_kf(self):
        self._kf_x = None
        self._kf_P = None
        self._kf_last_pred_t = None
        self._kf_last_frame_id = -1
        self._kf_consec_rej = 0
        self._kf_last_accept_t = None
        self._commit_active = False

    @property
    def debug_snapshot(self) -> dict:
        return dict(self._debug)

    def set_track_lock(self, track_id: int):
        track_id = int(track_id)
        if track_id < 0:
            return
        self._locked_track_id = track_id
        self._lock_first_track = False
        print(f"[YOLO-DDS] locked to track_id={track_id}")

    def clear_track_lock(self):
        self._locked_track_id = None
        self._ignored_track_count = 0
        self._reset_kf()

    def detect(self, robot_pos: np.ndarray, robot_yaw: float) -> np.ndarray | None:
        with self._lock:
            data = self._latest
            recv_t = self._latest_recv_time
            msg_count = self._msg_count
        now = time.time()
        age = (now - recv_t) if data is not None else float("inf")
        has_detection = (data is not None
                         and age <= self._staleness
                         and data.get("n", 0) >= 1)
        track_id = -1
        ignored_track_id = None
        if data is not None:
            try:
                track_id = int(data.get("track_id", -1))
            except (TypeError, ValueError):
                track_id = -1
        if has_detection and self._lock_first_track and self._locked_track_id is None:
            if track_id >= 0:
                self._locked_track_id = track_id
                self._lock_first_track = False
                print(f"[YOLO-DDS] locked first track_id={track_id}")
        if has_detection and self._locked_track_id is not None:
            if track_id != self._locked_track_id:
                ignored_track_id = track_id
                self._ignored_track_count += 1
                has_detection = False

        # ── COMMIT-AND-PROJECT: while locked on an approaching person, output a
        #    SMOOTH constant-velocity projection (p + v*t) so the dodge policy sees
        #    a clean ramp like the fake signal — independent of YOLO jitter/dropouts.
        #    Release if a fresh detection contradicts the projection or it times out.
        if self._commit_active:
            elapsed = now - self._commit_t0
            proj = self._commit_p + self._commit_v * elapsed
            proj_dist = float(np.linalg.norm(proj - robot_pos[:2]))
            release = None
            if elapsed > self._commit_max_time:
                release = "timeout"
            elif has_detection:
                x_b = float(data["x_fwd"]); y_b = float(data["y_left"])
                cy, sy = np.cos(robot_yaw), np.sin(robot_yaw)
                mwx = robot_pos[0] + x_b * cy - y_b * sy
                mwy = robot_pos[1] + x_b * sy + y_b * cy
                if float(np.hypot(mwx - proj[0], mwy - proj[1])) > self._commit_break_dist:
                    release = "deviation"
            if release is None:
                self._debug = {
                    "status": "commit_project",
                    "age": age,
                    "frame_id": data.get("frame_id", -1) if data else -1,
                    "raw_n": data.get("n", 0) if data else 0,
                    "msg_count": msg_count,
                    "track_id": track_id,
                    "locked_track_id": self._locked_track_id,
                    "commit_elapsed": elapsed,
                    "commit_vxy": (float(self._commit_v[0]), float(self._commit_v[1])),
                    "kf_dist": proj_dist,
                    "accepted": self._kf_n_accepted,
                    "rejected": self._kf_n_rejected,
                    "held": self._kf_n_held,
                    "nis": self._last_nis,
                }
                return np.array([proj[0], proj[1], self._commit_z], dtype=np.float32)
            # released: drop the projection and fall through to normal processing
            self._commit_active = False
            self._reset_kf()
            print(f"[YOLO-DDS] COMMIT release ({release}) after {elapsed:.2f}s")

        if not has_detection:
            # No fresh detection. Hold last KF state if recent enough.
            if (self._use_kalman and self._kf_x is not None
                and self._kf_last_accept_t is not None
                and (now - self._kf_last_accept_t) <= self._hold_timeout):
                # KF predict-only (no update). Keeps obstacle position stable
                # so DODGE state doesn't flip-flop on YOLO drops.
                dt = now - (self._kf_last_pred_t or now)
                self._kf_last_pred_t = now
                F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                              [0, 0, 1, 0],  [0, 0, 0, 1]])
                Q_pos = self._kf_q_pos ** 2
                Q_vel = (self._kf_q_vel * max(dt, 1e-3)) ** 2
                Q = np.diag([Q_pos, Q_pos, Q_vel, Q_vel])
                self._kf_x = F @ self._kf_x
                self._kf_P = F @ self._kf_P @ F.T + Q
                self._kf_n_held += 1
                self._debug = {
                    "status": "held_ignored_track" if ignored_track_id is not None else "held",
                    "age": age,
                    "frame_id": data.get("frame_id", -1) if data else -1,
                    "raw_n": data.get("n", 0) if data else 0,
                    "msg_count": msg_count,
                    "track_id": track_id,
                    "locked_track_id": self._locked_track_id,
                    "ignored_track_id": ignored_track_id,
                    "ignored_tracks": self._ignored_track_count,
                    "hold_age": now - self._kf_last_accept_t,
                    "kf_xy": (float(self._kf_x[0]), float(self._kf_x[1])),
                    "kf_vxy": (float(self._kf_x[2]), float(self._kf_x[3])),
                    "kf_dist": float(np.linalg.norm(self._kf_x[:2] - robot_pos[:2])),
                    "accepted": self._kf_n_accepted,
                    "rejected": self._kf_n_rejected,
                    "held": self._kf_n_held,
                    "nis": self._last_nis,
                }
                return np.array([self._kf_x[0], self._kf_x[1], self._last_z],
                                dtype=np.float32)
            status = "none"
            if data is not None:
                if ignored_track_id is not None:
                    status = "ignored_track"
                else:
                    status = "stale" if age > self._staleness else "empty"
            self._debug = {
                "status": status,
                "age": age,
                "frame_id": data.get("frame_id", -1) if data else -1,
                "raw_n": data.get("n", 0) if data else 0,
                "msg_count": msg_count,
                "track_id": track_id,
                "locked_track_id": self._locked_track_id,
                "ignored_track_id": ignored_track_id,
                "ignored_tracks": self._ignored_track_count,
                "accepted": self._kf_n_accepted,
                "rejected": self._kf_n_rejected,
                "held": self._kf_n_held,
                "nis": self._last_nis,
            }
            self._reset_kf()
            return None

        # body→world using CURRENT robot pose (latency error bounded by
        # max_vel × DDS latency ≈ 0.1 m/s × 0.15s ≈ 1.5cm — negligible)
        x_b = float(data["x_fwd"])
        y_b = float(data["y_left"])
        z = float(data["z"])
        self._last_z = z
        cy, sy = np.cos(robot_yaw), np.sin(robot_yaw)
        wx_obs = robot_pos[0] + x_b * cy - y_b * sy
        wy_obs = robot_pos[1] + x_b * sy + y_b * cy

        if not self._use_kalman:
            self._kf_last_accept_t = now
            self._debug = {
                "status": "fresh_raw",
                "age": age,
                "frame_id": int(data.get("frame_id", -1)),
                "raw_n": data.get("n", 0),
                "msg_count": msg_count,
                "track_id": track_id,
                "locked_track_id": self._locked_track_id,
                "ignored_tracks": self._ignored_track_count,
                "raw_body": (x_b, y_b, z),
                "raw_dist": float(data.get("dist", np.hypot(x_b, y_b))),
                "bearing": float(data.get("bearing", np.degrees(np.arctan2(y_b, x_b)))),
                "meas_world": (float(wx_obs), float(wy_obs), z),
                "kf_xy": None,
                "kf_vxy": None,
                "kf_dist": float(np.linalg.norm(np.array([wx_obs, wy_obs]) - robot_pos[:2])),
                "accepted": self._kf_n_accepted,
                "rejected": self._kf_n_rejected,
                "held": self._kf_n_held,
                "nis": None,
            }
            return np.array([wx_obs, wy_obs, z], dtype=np.float32)

        # ── KF PREDICT (every detect() call) ──
        if self._kf_x is None:
            # seed at first detection
            self._kf_x = np.array([wx_obs, wy_obs, 0.0, 0.0], dtype=np.float64)
            self._kf_P = np.diag([0.1, 0.1, 1.0, 1.0])
            self._kf_last_pred_t = now
            self._kf_last_accept_t = now
            self._last_nis = 0.0
        else:
            dt = now - self._kf_last_pred_t
            self._kf_last_pred_t = now
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                          [0, 0, 1, 0],  [0, 0, 0, 1]])
            Q_pos = self._kf_q_pos ** 2
            Q_vel = (self._kf_q_vel * max(dt, 1e-3)) ** 2
            Q = np.diag([Q_pos, Q_pos, Q_vel, Q_vel])
            self._kf_x = F @ self._kf_x
            self._kf_P = F @ self._kf_P @ F.T + Q

        # ── KF UPDATE (only on a new frame_id — don't update repeatedly with
        #               the same observation since detect() runs at 50Hz but
        #               YOLO publishes at ~3.5Hz) ──
        fid = int(data.get("frame_id", -1))
        if fid != self._kf_last_frame_id:
            self._kf_last_frame_id = fid
            z_obs = np.array([wx_obs, wy_obs])
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            # Adaptive R: inflate at close range
            pred_dist = float(np.linalg.norm(self._kf_x[:2] - robot_pos[:2]))
            blend = max(0.0, min(1.0, (2.0 - pred_dist) / 1.5))
            r_std = self._kf_r_std * (1 - blend) + self._kf_r_std_close * blend
            R = np.eye(2) * (r_std ** 2)
            y_innov = z_obs - H @ self._kf_x
            S = H @ self._kf_P @ H.T + R
            nis = float(y_innov @ np.linalg.solve(S, y_innov))
            self._last_nis = nis
            if nis > (self._kf_gate ** 2):
                self._kf_n_rejected += 1
                self._kf_consec_rej += 1
                # Track loss: too many consecutive rejections → state diverged.
                # Reset KF state to current measurement, force-accept, resume.
                if self._kf_consec_rej >= self._kf_max_consec_rej:
                    self._kf_x = np.array([wx_obs, wy_obs, 0.0, 0.0], dtype=np.float64)
                    self._kf_P = np.diag([0.1, 0.1, 1.0, 1.0])
                    self._kf_consec_rej = 0
                    self._kf_n_accepted += 1
                    self._kf_last_accept_t = now
                    self._last_nis = 0.0
            else:
                K = self._kf_P @ H.T @ np.linalg.inv(S)
                self._kf_x = self._kf_x + K @ y_innov
                self._kf_P = (np.eye(4) - K @ H) @ self._kf_P
                self._kf_n_accepted += 1
                self._kf_consec_rej = 0
                self._kf_last_accept_t = now

        self._debug = {
            "status": "fresh",
            "age": age,
            "frame_id": fid,
            "raw_n": data.get("n", 0),
            "msg_count": msg_count,
            "track_id": track_id,
            "locked_track_id": self._locked_track_id,
            "ignored_tracks": self._ignored_track_count,
            "raw_body": (x_b, y_b, z),
            "raw_dist": float(data.get("dist", np.hypot(x_b, y_b))),
            "bearing": float(data.get("bearing", np.degrees(np.arctan2(y_b, x_b)))),
            "meas_world": (float(wx_obs), float(wy_obs), z),
            "kf_xy": (float(self._kf_x[0]), float(self._kf_x[1])),
            "kf_vxy": (float(self._kf_x[2]), float(self._kf_x[3])),
            "kf_dist": float(np.linalg.norm(self._kf_x[:2] - robot_pos[:2])),
            "accepted": self._kf_n_accepted,
            "rejected": self._kf_n_rejected,
            "held": self._kf_n_held,
            "nis": self._last_nis,
        }

        # ── COMMIT TRIGGER: lock an approaching person for smooth projection.
        #    Seed direction+speed from the (smooth) approach KF state. We lock just
        #    outside the dodge trigger so the projection covers the whole dodge.
        if (self._commit_enable and not self._commit_active
                and self._kf_x is not None
                and self._kf_n_accepted >= self._commit_min_accepts):
            kf_pos = np.asarray(self._kf_x[:2], dtype=np.float64)
            kf_vel = np.asarray(self._kf_x[2:4], dtype=np.float64)
            kf_dist = float(np.linalg.norm(kf_pos - robot_pos[:2]))
            to_robot = robot_pos[:2] - kf_pos
            d = float(np.linalg.norm(to_robot))
            closing = float(kf_vel @ (to_robot / d)) if d > 1e-6 else 0.0
            if kf_dist < self._commit_dist and closing > self._commit_min_speed:
                self._commit_p = kf_pos.copy()
                self._commit_v = kf_vel.copy()
                self._commit_t0 = now
                self._commit_z = z
                self._commit_active = True
                print(f"[YOLO-DDS] COMMIT lock @ dist={kf_dist:.2f}m "
                      f"v=[{kf_vel[0]:+.2f},{kf_vel[1]:+.2f}] closing={closing:.2f}m/s")
        return np.array([self._kf_x[0], self._kf_x[1], z], dtype=np.float32)

    @property
    def msg_count(self) -> int:
        return self._msg_count

    @property
    def kf_stats(self) -> tuple[int, int]:
        return self._kf_n_accepted, self._kf_n_rejected

    @property
    def kf_held_count(self) -> int:
        return self._kf_n_held

    def stop(self):
        pass


# ═══════════════════════════════════════════════════════════════
# Dodge Controller (extends deploy_real.py's Controller)
# ═══════════════════════════════════════════════════════════════

class DodgeController:
    def __init__(self, config: Config, dodge_ckpt: str, return_head_ckpt: str,
                 max_lin_vel: float = 0.08, safety_distance: float = 2.0,
                 source: str = "yolo",
                 stability_max_tilt_rad: float = 0.30,
                 max_dcmd_lin: float = 0.04,        # m/s per tick (~50Hz → 2 m/s²)
                 max_dcmd_ang: float = 0.10,        # rad/s per tick
                 max_ang_vel: float = 0.30,         # absolute yaw-rate ceiling (rad/s)
                 min_loco_lin_vel: float = 0.30,    # escape locomotion training deadband
                 max_yaw_when_translating: float = 0.0,
                 lin_cmd_deadband: float = LOCO_LIN_CMD_DEADBAND,
                 dodge_lost_timeout: float = 2.0,
                 block_toward_until_clear: bool = True,
                 close_escape_speed: float = 0.35,
                 dodge_ewma_alpha: float = 0.2,     # EWMA smoothing on dodge policy output
                 debug_obs: bool = False,
                 debug_every: int = 10,
                 defer_lowcmd_init: bool = False,
                 yolo_staleness: float = 0.5,
                 yolo_hold_timeout: float = 1.0,
                 yolo_use_kalman: bool = True,
                 yolo_kf_meas_std: float = 0.10,
                 yolo_kf_meas_std_close: float = 0.30,
                 yolo_kf_gate: float = 4.0):
        self.config = config
        self.remote_controller = RemoteController()
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.min_loco_lin_vel = min_loco_lin_vel
        self.max_yaw_when_translating = max_yaw_when_translating
        self.lin_cmd_deadband = lin_cmd_deadband
        self.dodge_lost_timeout = dodge_lost_timeout
        self.block_toward_until_clear = block_toward_until_clear
        self.close_escape_speed = close_escape_speed
        self.debug_obs = debug_obs
        self.debug_every = max(1, int(debug_every))
        self.defer_lowcmd_init = defer_lowcmd_init
        self.safety_distance = safety_distance
        # Gentle-mode safety
        self.stability_max_tilt = stability_max_tilt_rad
        self.max_dcmd = np.array([max_dcmd_lin, max_dcmd_lin, max_dcmd_ang],
                                 dtype=np.float32)
        self._target_cmd = np.zeros(3, dtype=np.float32)   # ideal cmd (pre-ramp)
        self._unstable_streak = 0
        # EWMA on dodge policy output to give locomotion a stable direction.
        # Without this, raw policy flips per-tick at close range, rate-limit
        # can't follow, locomotion never commits to a step.
        self._dodge_ewma_alpha = dodge_ewma_alpha
        self._dodge_vel_ewma = np.zeros(3, dtype=np.float32)
        self._warned_loco_deadband = False

        if 0.0 < self.max_lin_vel <= self.lin_cmd_deadband:
            print(f"⚠️  max_lin_vel={self.max_lin_vel:.2f} is at/below the "
                  f"G1 locomotion training deadband ({self.lin_cmd_deadband:.2f}m/s). "
                  f"Use --max_vel {self.min_loco_lin_vel:.2f} for reliable displacement.",
                  file=sys.stderr)

        # ── Locomotion policy (TorchScript) ──
        self.loco_policy = torch.jit.load(config.policy_path)
        print(f"[Loco] Loaded {config.policy_path}")

        # ── Dodge policy (transformer) ──
        self.dodge_policy = DodgePolicy(dodge_ckpt)

        # ── Return head ──
        self.return_head = None
        if Path(return_head_ckpt).exists():
            rh_sd = torch.load(return_head_ckpt, map_location="cpu",
                               weights_only=False)["model_state_dict"]
            self.return_head = nn.Sequential(
                nn.Linear(2, 32), nn.ELU(), nn.Linear(32, 3), nn.Tanh())
            self.return_head.load_state_dict(
                {k.replace("return_head.", ""): v
                 for k, v in rh_sd.items() if "return_head" in k})
            self.return_head.eval()
            print("[ReturnHead] Loaded")

        # ── Obstacle source: YOLO+Depth via DDS (default), or legacy LiDAR-UDP ──
        if source == "yolo":
            self.lidar = YoloDdsObstacleDetector(
                staleness_threshold=yolo_staleness,
                hold_timeout=yolo_hold_timeout,
                use_kalman=yolo_use_kalman,
                kf_meas_std=yolo_kf_meas_std,
                kf_meas_std_close=yolo_kf_meas_std_close,
                kf_gate_sigma=yolo_kf_gate,
            )
        elif source == "lidar":
            self.lidar = LidarObstacleDetector()
        else:
            raise ValueError(f"unknown source: {source} (expected 'yolo' or 'lidar')")
        self._source = source

        # ── SDK communication ──
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0
        self.lowcmd_publisher_ = None
        self._low_level_initialized = False

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)

        self.wait_for_low_state()
        if not self.defer_lowcmd_init:
            self.initialize_low_level_control()
        else:
            print("[SDK] LowCmd publisher deferred until A.")

        # ── State ──
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.counter = 0

        # Dodge state
        self.dodge_active = False
        self.dodge_start_pos = None
        self.dodge_start_yaw = 0.0
        self.return_converged = False
        self._last_obstacle_pos: np.ndarray | None = None
        self._last_obstacle_dist = float("inf")
        self._last_obstacle_seen_t = 0.0
        self._last_dodge_debug: dict | None = None
        self._last_loco_debug: dict | None = None

        # Robot pose estimation (from IMU)
        self._robot_pos = np.array([0.0, 0.0, 0.8], dtype=np.float32)
        self._robot_yaw = 0.0
        self._prev_robot_pos = None

    def _low_state_handler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd):
        if self.lowcmd_publisher_ is None:
            raise RuntimeError("lowcmd publisher not initialized")
        cmd.mode_machine = self.mode_machine_
        cmd.mode_pr = self.mode_pr_
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def initialize_low_level_control(self):
        if self._low_level_initialized:
            return
        self.lowcmd_publisher_ = ChannelPublisher(self.config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        self._low_level_initialized = True
        print("[SDK] LowCmd publisher initialized.")

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("[SDK] Connected to G1.")

    def wait_for_builtin_regular_takeover(self):
        print("[STEP] Built-in Regular/Record mode. Press A to release and enable dodge mode...")
        while self.remote_controller.button[KeyMap.A] != 1:
            if self.remote_controller.button[KeyMap.select] == 1:
                raise KeyboardInterrupt
            time.sleep(self.config.control_dt)

    def snapshot_arm_waist_target_from_low_state(self):
        q = np.array(
            [self.low_state.motor_state[idx].q
             for idx in self.config.arm_waist_joint2motor_idx],
            dtype=np.float32,
        )
        self.config.arm_waist_target = q
        print(f"[regular] captured arm_waist_target from built-in mode: "
              f"{[round(float(x), 4) for x in q]}")

    def zero_torque_state(self):
        print("[STEP] Zero torque. Press START to continue...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("[STEP] Moving to default pos (2s)...")
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles,
                                      self.config.arm_waist_target))
        init_pos = np.array([self.low_state.motor_state[j].q for j in dof_idx],
                            dtype=np.float32)
        for i in range(num_step):
            alpha = i / num_step
            for j_idx, motor_idx in enumerate(dof_idx):
                self.low_cmd.motor_cmd[motor_idx].q = (
                    init_pos[j_idx] * (1 - alpha) + default_pos[j_idx] * alpha)
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j_idx]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j_idx]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("[STEP] Default pos. Press A to enable dodge mode...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i, motor_idx in enumerate(self.config.leg_joint2motor_idx):
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i, motor_idx in enumerate(self.config.arm_waist_joint2motor_idx):
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def _update_robot_pose(self):
        """Estimate robot world pose from IMU. Position is FIXED at the
        initial (0,0,0.8) — cmd-based dead-reckoning creates fake velocity
        in obs (base_lin_vel = cmd, not actual motion), which can drive the
        dodge policy into a 50 Hz oscillation feedback loop."""
        quat = self.low_state.imu_state.quaternion  # [w, x, y, z]
        w, x, y, z = quat
        self._robot_yaw = float(np.arctan2(2 * (w * z + x * y),
                                            1 - 2 * (y * y + z * z)))
        # _robot_pos intentionally not updated — without true odometry the
        # cmd-integration is open-loop and worse than assuming stationary.

    def run(self):
        self.counter += 1

        # Read joint state
        for i, motor_idx in enumerate(self.config.leg_joint2motor_idx):
            self.qj[i] = self.low_state.motor_state[motor_idx].q
            self.dqj[i] = self.low_state.motor_state[motor_idx].dq

        # IMU
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        gravity = get_gravity_orientation(quat)

        self._update_robot_pose()

        # ── Obstacle detection ──
        now = time.time()
        raw_obstacle_pos = self.lidar.detect(self._robot_pos, self._robot_yaw)
        obstacle_vel_w = None
        obstacle_vel_source = "fdiff"
        yolo_dbg = self.lidar.debug_snapshot if hasattr(self.lidar, "debug_snapshot") else {}
        obstacle_from_memory = False
        if raw_obstacle_pos is not None:
            obstacle_pos = raw_obstacle_pos
            kf_vxy = yolo_dbg.get("kf_vxy")
            if kf_vxy is not None:
                obstacle_vel_w = np.asarray(kf_vxy, dtype=np.float32)
                obstacle_vel_source = "kf"
            dist = float(np.linalg.norm(self._robot_pos[:2] - obstacle_pos[:2]))
            self._last_obstacle_pos = obstacle_pos.copy()
            self._last_obstacle_dist = dist
            self._last_obstacle_seen_t = now
        elif (self.dodge_active
              and self._last_obstacle_pos is not None
              and (now - self._last_obstacle_seen_t) <= self.dodge_lost_timeout):
            obstacle_pos = self._last_obstacle_pos.copy()
            obstacle_vel_w = np.zeros(2, dtype=np.float32)
            obstacle_vel_source = "memory_zero"
            dist = self._last_obstacle_dist
            obstacle_from_memory = True
        else:
            obstacle_pos = None
            dist = float("inf")

        # ── Dodge state machine ──
        if dist < self.safety_distance and not self.dodge_active and not self.return_converged:
            self.dodge_active = True
            self.dodge_start_pos = self._robot_pos[:2].copy()
            self.dodge_start_yaw = self._robot_yaw
            self.dodge_policy.reset(self._robot_pos[:2], self._robot_yaw)
            self._dodge_vel_ewma[:] = 0.0     # reset EWMA on new dodge episode
            print(f"  [DODGE START] dist={dist:.2f}m")

        # ── Build TARGET cmd (pre-ramp, pre-safety) ──
        if self.dodge_active:
            # Missing vision is unknown, not "person departed". Only a real
            # obstacle measurement outside the hysteresis band can enter RETURN.
            depart = (obstacle_pos is not None
                      and not obstacle_from_memory
                      and dist > self.safety_distance + 0.10)
            if obstacle_pos is None:
                self._target_cmd[:] = 0
                self._dodge_vel_ewma[:] = 0
                self.dodge_active = False
                self.return_converged = False
                print(f"  [DODGE LOST] no obstacle for "
                      f"{now - self._last_obstacle_seen_t:.2f}s; stopping, not returning")
            elif not depart:
                obs18 = self.dodge_policy.build_obs(
                    self._robot_pos, self._robot_yaw, obstacle_pos,
                    dt=self.config.control_dt,
                    obstacle_vel_w=obstacle_vel_w,
                    depart_mask=False)
                if self.debug_obs:
                    vel, act_norm, act_raw, obs18_model = \
                        self.dodge_policy.get_velocity_command_with_debug(obs18)
                else:
                    vel = self.dodge_policy.get_velocity_command(obs18)
                    act_norm = self.dodge_policy._last_action.copy()
                    act_raw = None
                    obs18_model = None
                # EWMA smoothing: stabilize direction so locomotion can step.
                a = self._dodge_ewma_alpha
                self._dodge_vel_ewma = (1.0 - a) * self._dodge_vel_ewma + a * vel
                vel_s = self._dodge_vel_ewma
                self._target_cmd[:] = shape_locomotion_velocity(
                    vel_s,
                    max_lin_vel=self.max_lin_vel,
                    max_ang_vel=self.max_ang_vel,
                    lin_deadband=self.lin_cmd_deadband,
                    min_lin_vel=self.min_loco_lin_vel,
                    max_yaw_when_translating=self.max_yaw_when_translating,
                )
                obs_pos_b_xy = self.dodge_policy._body_frame_xy(
                    obstacle_pos[:2] - self._robot_pos[:2], self._robot_yaw)
                barrier_debug = {
                    "changed": False,
                    "block": {"changed": False},
                    "escape": {"changed": False},
                }
                if (self.block_toward_until_clear
                    and dist <= self.safety_distance + 0.10):
                    self._target_cmd[:], block_debug = _block_toward_obstacle(
                        self._target_cmd, obs_pos_b_xy, max_toward=0.0)
                    self._target_cmd[:], escape_debug = _enforce_away_component(
                        self._target_cmd,
                        obs_pos_b_xy,
                        min_away_speed=self.close_escape_speed,
                        max_lin_vel=self.max_lin_vel)
                    barrier_debug = {
                        "changed": (block_debug.get("changed", False)
                                    or escape_debug.get("changed", False)),
                        "block": block_debug,
                        "escape": escape_debug,
                    }
                if (not self._warned_loco_deadband
                    and np.linalg.norm(self._target_cmd[:2]) <= self.lin_cmd_deadband
                    and np.linalg.norm(vel_s[:2]) > 0.05):
                    print(f"  ⚠️ dodge linear cmd norm={np.linalg.norm(self._target_cmd[:2]):.2f} "
                          f"<= trained standstill deadband {self.lin_cmd_deadband:.2f}; "
                          f"raise --max_vel for displacement")
                    self._warned_loco_deadband = True
                self._last_dodge_debug = {
                    "phase": "DODGE_MEM" if obstacle_from_memory else "DODGE",
                    "obs18": obs18.copy(),
                    "obs_pos_b": obs_pos_b_xy.copy(),
                    "obs_vel_w_used": obstacle_vel_w.copy() if obstacle_vel_w is not None else None,
                    "obs_vel_source": obstacle_vel_source,
                    "vel_raw": vel.copy(),
                    "act_norm": act_norm.copy() if act_norm is not None else None,
                    "act_raw": act_raw.copy() if act_raw is not None else None,
                    "obs18_model": obs18_model.copy() if obs18_model is not None else None,
                    "vel_smooth": vel_s.copy(),
                    "target": self._target_cmd.copy(),
                    "barrier": barrier_debug,
                    "target_escape": _escape_metrics(obs_pos_b_xy, self._target_cmd[:2]),
                }
            elif self.return_head is not None:
                disp_w = self._robot_pos[:2] - self.dodge_start_pos
                disp_b = self.dodge_policy._body_frame_xy(disp_w, self._robot_yaw)
                with torch.no_grad():
                    _d = torch.tensor(disp_b, dtype=torch.float32).unsqueeze(0)
                    _ret = self.return_head(_d).squeeze(0).numpy()
                yaw_err = self._robot_yaw - self.dodge_start_yaw
                yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
                self._target_cmd[0] = float(np.clip(
                    _ret[0] * self.dodge_policy.MAX_LIN_VEL,
                    -self.max_lin_vel, self.max_lin_vel))
                self._target_cmd[1] = float(np.clip(
                    _ret[1] * self.dodge_policy.MAX_LIN_VEL,
                    -self.max_lin_vel, self.max_lin_vel))
                self._target_cmd[2] = float(np.clip(-2.0 * yaw_err,
                                                     -self.max_ang_vel, self.max_ang_vel))
                self._last_dodge_debug = {
                    "phase": "RETURN",
                    "disp_b": disp_b.copy(),
                    "return_raw": _ret.copy(),
                    "yaw_err": yaw_err,
                    "target": self._target_cmd.copy(),
                }
                disp_mag = float(np.linalg.norm(disp_w))
                if disp_mag < 0.20 and abs(yaw_err) < 0.15:
                    self.dodge_active = False
                    self.return_converged = True
                    print(f"  [RETURN DONE] disp={disp_mag:.3f}m "
                          f"yaw={np.degrees(yaw_err):+.1f}°")
            else:
                self.dodge_active = False
                self.return_converged = True
                self._target_cmd[:] = 0
                self._last_dodge_debug = {"phase": "DONE_NO_RETURN_HEAD"}
        else:
            self._last_dodge_debug = None
            # Normal: remote controller (also clipped to max_lin_vel for gentle mode)
            self._target_cmd[0] = float(np.clip(self.remote_controller.ly,
                                                -self.max_lin_vel, self.max_lin_vel))
            self._target_cmd[1] = float(np.clip(self.remote_controller.lx * -1,
                                                -self.max_lin_vel, self.max_lin_vel))
            self._target_cmd[2] = float(np.clip(self.remote_controller.rx * -1,
                                                -self.max_ang_vel, self.max_ang_vel))
            if dist > self.safety_distance + 1.0:
                self.return_converged = False

        # ── SAFETY: IMU instability check ──
        # quat is [w, x, y, z]; roll/pitch from body-axis quaternion
        w, qx, qy, qz = quat
        roll = float(np.arctan2(2 * (w * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
        pitch_arg = 2 * (w * qy - qz * qx)
        pitch_arg = max(-1.0, min(1.0, pitch_arg))
        pitch = float(np.arcsin(pitch_arg))
        unstable = (abs(roll) > self.stability_max_tilt) or \
                   (abs(pitch) > self.stability_max_tilt)
        if unstable:
            self._unstable_streak += 1
            self._target_cmd[:] = 0
            if self._unstable_streak == 1 or self._unstable_streak % 25 == 0:
                print(f"  ⚠️ UNSTABLE roll={np.degrees(roll):+.1f}° "
                      f"pitch={np.degrees(pitch):+.1f}° (limit ±{np.degrees(self.stability_max_tilt):.0f}°) — cmd=0")
        else:
            if self._unstable_streak > 5:
                print(f"  ✓ recovered (after {self._unstable_streak} unstable ticks)")
            self._unstable_streak = 0

        # ── SAFETY: rate limit (cmd ramp) ──
        # Limits per-tick velocity change → prevents jerky 飞踢/手舞足蹈
        delta = self._target_cmd - self.cmd
        delta = np.clip(delta, -self.max_dcmd, self.max_dcmd)
        self.cmd = (self.cmd + delta).astype(np.float32)
        if self._last_dodge_debug is not None and obstacle_pos is not None:
            obs_b_xy = self._last_dodge_debug.get("obs_pos_b")
            if obs_b_xy is None:
                obs_b_xy = self.dodge_policy._body_frame_xy(
                    obstacle_pos[:2] - self._robot_pos[:2], self._robot_yaw)
            self._last_dodge_debug["cmd"] = self.cmd.copy()
            self._last_dodge_debug["cmd_escape"] = _escape_metrics(obs_b_xy, self.cmd[:2])
            self._last_dodge_debug["cmd_delta"] = delta.copy()

        # ── Build locomotion obs (47-dim) ──
        qj_obs = (self.qj - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = self.dqj * self.config.dof_vel_scale
        ang_vel_obs = ang_vel * self.config.ang_vel_scale

        period = 0.8
        phase = (self.counter * self.config.control_dt) % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        n = self.config.num_actions
        self.obs[:3] = ang_vel_obs
        self.obs[3:6] = gravity
        # self.cmd is already in physical units (m/s, m/s, rad/s) — unlike
        # deploy_real.py where cmd is normalized joystick [-1,+1] and gets
        # × max_cmd to convert. Multiplying by max_cmd here would double-scale
        # and per-axis distort (×0.5 lateral, ×1.57 yaw → robot only rotates).
        self.obs[6:9] = self.cmd * self.config.cmd_scale
        self.obs[9:9 + n] = qj_obs
        self.obs[9 + n:9 + 2 * n] = dqj_obs
        self.obs[9 + 2 * n:9 + 3 * n] = self.action
        self.obs[9 + 3 * n] = sin_phase
        self.obs[9 + 3 * n + 1] = cos_phase

        # ── Locomotion inference ──
        self.action = self.loco_policy(
            torch.from_numpy(self.obs).unsqueeze(0)).detach().numpy().squeeze()
        self.target_dof_pos = self.config.default_angles + self.action * self.config.action_scale
        self._last_loco_debug = {
            "roll": roll,
            "pitch": pitch,
            "gravity": gravity.copy(),
            "ang_vel_obs": ang_vel_obs.reshape(-1).copy(),
            "cmd_obs": self.obs[6:9].copy(),
            "qj_obs": qj_obs.copy(),
            "dqj_obs": dqj_obs.copy(),
            "loco_action": self.action.copy(),
            "target_dof": self.target_dof_pos.copy(),
            "phase": phase,
            "sin_phase": sin_phase,
            "cos_phase": cos_phase,
        }

        # ── Send motor commands ──
        for i, motor_idx in enumerate(self.config.leg_joint2motor_idx):
            self.low_cmd.motor_cmd[motor_idx].q = self.target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i, motor_idx in enumerate(self.config.arm_waist_joint2motor_idx):
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)

        # ── Log (每秒一次) ──
        if self.counter % 50 == 0:
            phase_str = ("UNSTABLE" if self._unstable_streak > 0
                         else ("DODGE" if self.dodge_active else
                               ("DONE" if self.return_converged else "IDLE")))
            obs_str = f"obs={dist:.2f}m" if dist < 100 else "obs=N/A"
            src_str = f"{self._source}_msgs={self.lidar.msg_count}"
            kf_str = ""
            if self._source == "yolo" and hasattr(self.lidar, "kf_stats"):
                acc, rej = self.lidar.kf_stats
                held = self.lidar.kf_held_count
                if acc + rej + held > 0:
                    kf_str = f" KF={acc}+/{rej}rej/{held}held"
            geom_str = ""
            if obstacle_pos is not None:
                obs_b_log = self.dodge_policy._body_frame_xy(
                    obstacle_pos[:2] - self._robot_pos[:2], self._robot_yaw)
                m = _escape_metrics(obs_b_log, self.cmd[:2])
                if m is not None:
                    geom_str = (f" dir={m['verdict']} par={m['parallel']:+.2f} "
                                f"perp={m['perp']:+.2f} bear={m['bearing']:+.0f}°")
            print(f"  [{phase_str:8s}] cmd=[{self.cmd[0]:+.2f},{self.cmd[1]:+.2f},"
                  f"{self.cmd[2]:+.2f}] target=[{self._target_cmd[0]:+.2f},"
                  f"{self._target_cmd[1]:+.2f},{self._target_cmd[2]:+.2f}] "
                  f"lin={np.linalg.norm(self.cmd[:2]):.2f}/"
                  f"{np.linalg.norm(self._target_cmd[:2]):.2f} "
                  f"{obs_str}{geom_str} {src_str}{kf_str}")

        if self.debug_obs and self.counter % self.debug_every == 0:
            ydbg = self.lidar.debug_snapshot if hasattr(self.lidar, "debug_snapshot") else {}
            status = ydbg.get("status", "?")
            age = ydbg.get("age", float("inf"))
            age_str = "inf" if not np.isfinite(age) else f"{age:.2f}"
            raw_body = ydbg.get("raw_body")
            kf_xy = ydbg.get("kf_xy")
            kf_vxy = ydbg.get("kf_vxy")
            kf_dist = ydbg.get("kf_dist", float("nan"))
            bearing = ydbg.get("bearing", float("nan"))
            nis = ydbg.get("nis", None)
            nis_str = "None" if nis is None else f"{nis:.1f}"
            obs_b = None
            kf_body = None
            if obstacle_pos is not None:
                obs_b = self.dodge_policy._body_frame_xy(
                    obstacle_pos[:2] - self._robot_pos[:2], self._robot_yaw)
            if kf_xy is not None:
                kf_body = self.dodge_policy._body_frame_xy(
                    np.asarray(kf_xy, dtype=np.float32) - self._robot_pos[:2],
                    self._robot_yaw)
            target_escape = _escape_metrics(obs_b, self._target_cmd[:2])
            cmd_escape = _escape_metrics(obs_b, self.cmd[:2])
            dist_str = "N/A" if not np.isfinite(dist) else f"{dist:.2f}"
            print(f"  [DBG-TICK] k={self.counter} low_mode={self.mode_machine_} "
                  f"active={self.dodge_active} returned={self.return_converged} "
                  f"mem={obstacle_from_memory} dist={dist_str} "
                  f"remote=[lx={self.remote_controller.lx:+.2f},"
                  f"ly={self.remote_controller.ly:+.2f},"
                  f"rx={self.remote_controller.rx:+.2f}]")
            print(f"  [DBG-YOLO] status={status} age={age_str}s "
                  f"frame={ydbg.get('frame_id', -1)} n={ydbg.get('raw_n', '?')} "
                  f"msg={ydbg.get('msg_count', '?')} "
                  f"raw_body={_fmt_vec(raw_body, 3)} "
                  f"raw_dist={ydbg.get('raw_dist', float('nan')):.3f} "
                  f"raw_bear={bearing:+.1f} meas_w={_fmt_vec(ydbg.get('meas_world'), 3)}")
            print(f"  [DBG-KF  ] kf_xy={_fmt_vec(kf_xy, 3)} "
                  f"kf_body={_fmt_vec(kf_body, 3)} kf_v_w={_fmt_vec(kf_vxy, 3)} "
                  f"kf_dist={kf_dist:.3f} nis={nis_str} "
                  f"acc={ydbg.get('accepted', '?')} rej={ydbg.get('rejected', '?')} "
                  f"held={ydbg.get('held', '?')}")
            if target_escape is not None and cmd_escape is not None:
                print(f"  [DBG-GEOM] obs_b={_fmt_vec(obs_b, 3)} "
                      f"bear={target_escape['bearing']:+.1f}deg "
                      f"target={target_escape['verdict']} "
                      f"par={target_escape['parallel']:+.3f} "
                      f"perp={target_escape['perp']:+.3f} | "
                      f"cmd={cmd_escape['verdict']} "
                      f"par={cmd_escape['parallel']:+.3f} "
                      f"perp={cmd_escape['perp']:+.3f}")
            else:
                print(f"  [DBG-GEOM] obs_b={_fmt_vec(obs_b, 3)} "
                      f"target={_fmt_vec(self._target_cmd, 3)} "
                      f"cmd={_fmt_vec(self.cmd, 3)}")
            loco_dbg = self._last_loco_debug or {}
            print(f"  [DBG-BASE] yaw={np.degrees(self._robot_yaw):+.1f}deg "
                  f"roll={np.degrees(loco_dbg.get('roll', 0.0)):+.1f}deg "
                  f"pitch={np.degrees(loco_dbg.get('pitch', 0.0)):+.1f}deg "
                  f"grav={_fmt_vec(loco_dbg.get('gravity'), 3)} "
                  f"omega_obs={_fmt_vec(loco_dbg.get('ang_vel_obs'), 3)}")
            if self._last_dodge_debug:
                dbg = self._last_dodge_debug
                if "obs18" in dbg:
                    obs18 = dbg["obs18"]
                    obs18_model = dbg.get("obs18_model")
                    barrier = dbg.get("barrier") or {}
                    block_debug = barrier.get("block", barrier) or {}
                    escape_debug = barrier.get("escape") or {}
                    print(f"  [DBG-POL0] phase={dbg.get('phase')} obs18_raw "
                          f"base_vel={_fmt_vec(obs18[0:3], 3)} "
                          f"obs_pos={_fmt_vec(obs18[3:6], 3)} "
                          f"obs_vel={_fmt_vec(obs18[6:9], 3)} "
                          f"obs_vel_src={dbg.get('obs_vel_source')} "
                          f"obs_vel_w={_fmt_vec(dbg.get('obs_vel_w_used'), 3)} "
                          f"disp={_fmt_vec(obs18[9:11], 3)} "
                          f"box={_fmt_vec(obs18[11:14], 3)} "
                          f"yaw={obs18[14]:+.3f} "
                          f"last_act={_fmt_vec(obs18[15:18], 3)}")
                    if obs18_model is not None:
                        max_abs = float(np.max(np.abs(obs18_model)))
                        print(f"  [DBG-POL1] obs18_norm maxabs={max_abs:.2f} "
                              f"base_vel={_fmt_vec(obs18_model[0:3], 2)} "
                              f"obs_pos={_fmt_vec(obs18_model[3:6], 2)} "
                              f"obs_vel={_fmt_vec(obs18_model[6:9], 2)} "
                              f"disp={_fmt_vec(obs18_model[9:11], 2)} "
                              f"yaw={obs18_model[14]:+.2f} "
                              f"last_act={_fmt_vec(obs18_model[15:18], 2)}")
                    print(f"  [DBG-POL2] logits={_fmt_vec(dbg.get('act_raw'), 3)} "
                          f"tanh_act={_fmt_vec(dbg.get('act_norm'), 3)} "
                          f"vel_raw={_fmt_vec(dbg.get('vel_raw'), 3)} "
                          f"ewma={_fmt_vec(dbg.get('vel_smooth'), 3)}")
                    print(f"  [DBG-CMD ] shaped_target={_fmt_vec(dbg.get('target'), 3)} "
                          f"rate_delta={_fmt_vec(dbg.get('cmd_delta'), 3)} "
                          f"cmd_after_rate={_fmt_vec(dbg.get('cmd'), 3)} "
                          f"loco_cmd_obs={_fmt_vec(loco_dbg.get('cmd_obs'), 3)}")
                    if block_debug.get("changed"):
                        barrier_before = block_debug.get("before") or {}
                        barrier_after = block_debug.get("after") or {}
                        print(f"  [DBG-SAFE] blocked_toward "
                              f"par {barrier_before.get('parallel', 0.0):+.3f}"
                              f"->{barrier_after.get('parallel', 0.0):+.3f} "
                              f"perp {barrier_before.get('perp', 0.0):+.3f}"
                              f"->{barrier_after.get('perp', 0.0):+.3f}")
                    if escape_debug.get("changed"):
                        escape_before = escape_debug.get("before") or {}
                        escape_after = escape_debug.get("after") or {}
                        print(f"  [DBG-SAFE] close_escape "
                              f"par {escape_before.get('parallel', 0.0):+.3f}"
                              f"->{escape_after.get('parallel', 0.0):+.3f} "
                              f"perp {escape_before.get('perp', 0.0):+.3f}"
                              f"->{escape_after.get('perp', 0.0):+.3f} "
                              f"perp_scale={escape_debug.get('perp_scale', 0.0):.2f}")
                else:
                    print(f"  [DBG-POL ] phase={dbg.get('phase')} "
                          f"disp_b={_fmt_vec(dbg.get('disp_b'), 3)} "
                          f"ret={_fmt_vec(dbg.get('return_raw'), 3)} "
                          f"yaw_err={dbg.get('yaw_err', 0.0):+.2f} "
                          f"target={_fmt_vec(dbg.get('target'), 3)}")
            print(f"  [DBG-LOCO] phase={loco_dbg.get('phase', 0.0):.3f} "
                  f"sin/cos={loco_dbg.get('sin_phase', 0.0):+.3f}/"
                  f"{loco_dbg.get('cos_phase', 0.0):+.3f} "
                  f"action={_fmt_vec(loco_dbg.get('loco_action'), 3)}")
            print(f"  [DBG-JOINT] qj_obs({_fmt_range(loco_dbg.get('qj_obs'), 3)}) "
                  f"dqj_obs({_fmt_range(loco_dbg.get('dqj_obs'), 3)}) "
                  f"target_dof={_fmt_vec(loco_dbg.get('target_dof'), 3)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="Network interface (e.g. eth0)")
    parser.add_argument("config", type=str, default="configs/g1.yaml",
                        help="Config file")
    parser.add_argument("--max_vel", type=float, default=0.08,
                        help="Max dodge/manual velocity m/s (default 0.08, very gentle). "
                             "For actual stepping, use >0.20 because the G1 locomotion "
                             "policy was trained with <=0.20m/s xy commands zeroed.")
    parser.add_argument("--safety_dist", type=float, default=2.5,
                        help="Dodge trigger distance m (default 2.5, early trigger)")
    parser.add_argument("--source", choices=["yolo", "lidar"], default="yolo",
                        help="Obstacle source: 'yolo' (DDS rt/yolo/person) or "
                             "'lidar' (LivoxLidarReceiver UDP forwarder). "
                             "yolo requires yolo_to_dds_laptop.py running.")
    parser.add_argument("--max_tilt_deg", type=float, default=17.0,
                        help="Body tilt (roll/pitch) limit; above this, cmd→0 "
                             "to prevent flailing while suspended (default 17° ≈ 0.30 rad)")
    parser.add_argument("--max_dcmd_lin", type=float, default=0.04,
                        help="Max per-tick velocity change m/s (~50Hz → 2 m/s² accel cap)")
    parser.add_argument("--max_dcmd_ang", type=float, default=0.10,
                        help="Max per-tick angular velocity change rad/s")
    parser.add_argument("--max_ang_vel", type=float, default=None,
                        help="Absolute yaw-rate ceiling rad/s. Default = max_vel. Pass 0 to "
                             "fully disable rotation (Phase 4 / stand-only).")
    parser.add_argument("--min_loco_lin_vel", type=float, default=0.30,
                        help="When dodge asks for translation, raise xy command norm to at "
                             "least this value if the max_vel cap allows it. This escapes "
                             "the G1 locomotion training deadband at <=0.20m/s. Pass 0 to disable.")
    parser.add_argument("--max_yaw_when_translating", type=float, default=0.0,
                        help="Yaw-rate cap while dodge has nonzero translation intent. "
                             "Default 0 keeps the RealSense pointed at the person. "
                             "Set negative to disable this extra cap.")
    parser.add_argument("--dodge_lost_timeout", type=float, default=2.0,
                        help="After DODGE starts, keep using the last known obstacle for "
                             "this many seconds when YOLO loses the person. Missing vision "
                             "does not trigger RETURN.")
    parser.add_argument("--allow_toward_until_clear", action="store_true",
                        help="Disable the safety projection that removes velocity toward "
                             "the obstacle while distance <= safety_dist+0.10. Default "
                             "keeps this projection on.")
    parser.add_argument("--close_escape_speed", type=float, default=None,
                        help="Minimum body-frame velocity component directly away from "
                             "the obstacle while distance <= safety_dist+0.10. This "
                             "keeps close-range dodge from becoming mostly tangential. "
                             "Default uses the largest feasible radial speed under the "
                             "per-axis --max_vel cap. Pass 0 to disable.")
    parser.add_argument("--debug_obs", action="store_true",
                        help="Print YOLO raw/KF state, dodge obs18, and policy command "
                             "debug lines in this terminal.")
    parser.add_argument("--debug_every", type=int, default=10,
                        help="Print --debug_obs every N control ticks (default 10 = 5Hz).")
    parser.add_argument("--startup_mode", type=str, default="keep",
                        help="Built-in MotionSwitcher startup mode before low-level "
                             "takeover. Default 'keep' leaves the robot in its current "
                             "built-in Regular/Record mode and waits for A. "
                             "Use 'ai' to explicitly select ai, or 'regular' to try "
                             "regular/normal/ai fallback. "
                             "Use 'none' for the old zero-torque START flow.")
    parser.add_argument("--yolo_staleness", type=float, default=0.5,
                        help="YOLO msg older than this → no-detect (s)")
    parser.add_argument("--yolo_hold_timeout", type=float, default=1.0,
                        help="When YOLO drops (n=0 or stale), hold KF predict-only "
                             "for this many seconds before giving up. Prevents "
                             "DODGE↔RETURN flip-flop on intermittent detection.")
    parser.add_argument("--dodge_ewma_alpha", type=float, default=0.2,
                        help="EWMA smoothing on dodge policy output (0=full smooth, "
                             "1=raw). α=0.2 → ~200ms time constant.")
    parser.add_argument("--yolo_no_kalman", action="store_true",
                        help="Disable Kalman filter (raw YOLO passthrough)")
    parser.add_argument("--yolo_kf_meas_std", type=float, default=0.10,
                        help="KF measurement R baseline std (m)")
    parser.add_argument("--yolo_kf_meas_std_close", type=float, default=0.30,
                        help="KF measurement R inflated std at close range (m)")
    parser.add_argument("--yolo_kf_gate", type=float, default=4.0,
                        help="KF Mahalanobis outlier gate (sigma)")
    args = parser.parse_args()

    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    ChannelFactoryInitialize(0, args.net)

    _msc = MotionSwitcherClient()
    _msc.SetTimeout(5.0)
    _msc.Init()
    startup_mode = args.startup_mode.lower()
    old_zero_torque_startup = startup_mode in ("", "none", "off", "false", "0")
    keep_current_startup = startup_mode in ("keep", "current", "manual")
    builtin_startup = not old_zero_torque_startup
    if keep_current_startup:
        code, info = _msc.CheckMode()
        mode = _motion_mode_name(info) if code == 0 else "?"
        print(f"[motion_switcher] keeping current built-in mode='{mode}' "
              f"(no SelectMode, no ReleaseMode before A)")
        if not mode:
            print("[motion_switcher] current mode is empty; robot is already in low-level/debug. "
                  "Switch it back to the built-in standing mode before pressing A.")
    elif builtin_startup:
        try:
            select_builtin_startup_mode(_msc, startup_mode)
        except RuntimeError as e:
            print(f"❌ {e}", file=sys.stderr)
            sys.exit(2)
    else:
        # Old flow: release immediately, then hold zero torque until START.
        # Low-level commands must never be sent while a high-level mode is active.
        release_motion_mode_for_low_level(_msc)

    # ── HARD GLOBAL VELOCITY CEILING. 0.20m/s is the locomotion training
    # standstill threshold, not a walking-speed ceiling.
    HARD_MAX_VEL = 0.50
    HARD_MAX_ANG = 0.50
    if args.max_vel > HARD_MAX_VEL:
        print(f"⚠️  --max_vel {args.max_vel} > hard ceiling {HARD_MAX_VEL} m/s — clamping",
              file=sys.stderr)
        args.max_vel = HARD_MAX_VEL
    if args.max_ang_vel is None:
        args.max_ang_vel = args.max_vel
    if args.max_ang_vel > HARD_MAX_ANG:
        print(f"⚠️  --max_ang_vel {args.max_ang_vel} > hard ceiling {HARD_MAX_ANG} rad/s — clamping",
              file=sys.stderr)
        args.max_ang_vel = HARD_MAX_ANG
    if args.max_yaw_when_translating < 0:
        args.max_yaw_when_translating = None
    if args.close_escape_speed is None:
        args.close_escape_speed = float(np.sqrt(2.0) * args.max_vel)
    else:
        args.close_escape_speed = max(0.0, float(args.close_escape_speed))
    if 0.0 < args.max_vel <= LOCO_LIN_CMD_DEADBAND:
        print(f"⚠️  --max_vel {args.max_vel:.2f} is at/below the G1 locomotion "
              f"training deadband ({LOCO_LIN_CMD_DEADBAND:.2f}m/s). "
              f"For displacement use --max_vel {args.min_loco_lin_vel:.2f}.",
              file=sys.stderr)

    controller = DodgeController(
        config,
        dodge_ckpt=str(_REPO / "checkpoints" / "dodge_v23b_54400.pt"),
        return_head_ckpt=str(_REPO / "checkpoints" / "return_head_v23b_v6.pt"),
        max_lin_vel=args.max_vel,
        safety_distance=args.safety_dist,
        source=args.source,
        stability_max_tilt_rad=np.radians(args.max_tilt_deg),
        max_dcmd_lin=args.max_dcmd_lin,
        max_dcmd_ang=args.max_dcmd_ang,
        max_ang_vel=args.max_ang_vel,
        min_loco_lin_vel=args.min_loco_lin_vel,
        max_yaw_when_translating=args.max_yaw_when_translating,
        dodge_lost_timeout=args.dodge_lost_timeout,
        block_toward_until_clear=(not args.allow_toward_until_clear),
        close_escape_speed=args.close_escape_speed,
        dodge_ewma_alpha=args.dodge_ewma_alpha,
        debug_obs=args.debug_obs,
        debug_every=args.debug_every,
        defer_lowcmd_init=builtin_startup,
        yolo_hold_timeout=args.yolo_hold_timeout,
        yolo_staleness=args.yolo_staleness,
        yolo_use_kalman=(not args.yolo_no_kalman),
        yolo_kf_meas_std=args.yolo_kf_meas_std,
        yolo_kf_meas_std_close=args.yolo_kf_meas_std_close,
        yolo_kf_gate=args.yolo_kf_gate,
    )

    if builtin_startup:
        controller.wait_for_builtin_regular_takeover()
        controller.snapshot_arm_waist_target_from_low_state()
        release_motion_mode_for_low_level(_msc)
        controller.initialize_low_level_control()
        controller.move_to_default_pos()
    else:
        controller.zero_torque_state()
        controller.move_to_default_pos()
        controller.default_pos_state()

    print(f"\n[DODGE MODE] source={args.source}  max_vel={args.max_vel}m/s  "
          f"max_ang={args.max_ang_vel}rad/s  "
          f"safety={args.safety_dist}m  max_tilt={args.max_tilt_deg}°")
    print(f"  loco shaping: deadband={LOCO_LIN_CMD_DEADBAND:.2f}m/s, "
          f"min_lin={args.min_loco_lin_vel:.2f}m/s, "
          f"yaw_cap_while_translating={args.max_yaw_when_translating}")
    print(f"  vision loss: hold={args.yolo_hold_timeout:.1f}s + "
          f"memory={args.dodge_lost_timeout:.1f}s, missing vision will stop instead of RETURN")
    print(f"  safety projection: "
          f"{'block toward until clear' if not args.allow_toward_until_clear else 'disabled'}")
    if not args.allow_toward_until_clear and args.close_escape_speed > 0:
        print(f"  close escape: min_away={args.close_escape_speed:.2f}m/s "
              f"(capped by per-axis max_vel and bearing)")
    if args.debug_obs:
        print(f"  debug_obs: every {args.debug_every} ticks")
    print(f"  rate-limit: Δlin={args.max_dcmd_lin}m/s per tick, "
          f"Δang={args.max_dcmd_ang}rad/s per tick (~50Hz)")
    print("  Walking with remote. Dodge activates when obstacle < safety_dist.")
    print("  Press SELECT to EMERGENCY STOP.\n")

    while True:
        try:
            controller.run()
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break

    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    controller.lidar.stop()
    print("\n[EXIT] Damping mode. Safe.")
