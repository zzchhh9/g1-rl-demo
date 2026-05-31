"""G1 dodge deployment using Unitree's built-in high-level locomotion.

This keeps the built-in G1 locomotion mode active and sends velocity commands
through unitree_sdk2py.g1.loco.LocoClient.SetVelocity(). It does not publish
rt/lowcmd and does not run the third-party motion.pt locomotion policy.

Usage:
    1. Put the robot in the blue high-level locomotion mode from the remote:
       hold L2 + UP until the controller light is blue.
    2. Run:
       uv run python deploy_dodge_sdk_loco.py eno1 --source yolo --max_vel 0.30

Controls:
    A      -> enable autonomous dodge commands if --wait_for_a is passed
    SELECT -> emergency damping and exit
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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


def _fmt_mat2(mat, ndigits: int = 2) -> str:
    if mat is None:
        return "None"
    arr = np.asarray(mat, dtype=np.float32)
    if arr.shape != (2, 2) or not np.all(np.isfinite(arr)):
        return "None"
    return ("["
            f"[{arr[0,0]:+.{ndigits}f},{arr[0,1]:+.{ndigits}f}],"
            f"[{arr[1,0]:+.{ndigits}f},{arr[1,1]:+.{ndigits}f}]"
            "]")


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


class ExternalOdomMonitor:
    """Subscribes to JSON odometry on DDS and exposes fresh x/y displacement.

    Expected topic payload, published by scripts/ros2_odom_to_dds.py:
        {"stamp": float, "x": m, "y": m, "z": m, "yaw": rad, ...}

    The controller starts with the first accepted x/y as a provisional origin
    and resets the origin again at DODGE START for the actual return target.
    """

    def __init__(self, topic: str = "rt/dodge/odom"):
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._recv_time = 0.0
        self._msg_count = 0
        self.sub = ChannelSubscriber(topic, String_)
        self.sub.Init(self._callback, 10)
        print(f"[ODOM-DDS] subscribing to {topic}")

    @staticmethod
    def _xy_from_payload(data: dict) -> tuple[float, float]:
        if "x" in data and "y" in data:
            return float(data["x"]), float(data["y"])
        pos = data.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return float(pos[0]), float(pos[1])
        if isinstance(pos, dict) and "x" in pos and "y" in pos:
            return float(pos["x"]), float(pos["y"])
        pose = data.get("pose")
        if isinstance(pose, (list, tuple)) and len(pose) >= 2:
            return float(pose[0]), float(pose[1])
        if isinstance(pose, dict):
            if "x" in pose and "y" in pose:
                return float(pose["x"]), float(pose["y"])
            pose_pos = pose.get("position")
            if isinstance(pose_pos, dict) and "x" in pose_pos and "y" in pose_pos:
                return float(pose_pos["x"]), float(pose_pos["y"])
        raise ValueError("odom payload has no x/y position")

    def _callback(self, msg):
        try:
            data = json.loads(msg.data)
            x, y = self._xy_from_payload(data)
        except Exception:
            return
        if not (np.isfinite(x) and np.isfinite(y)):
            return
        data["x"] = float(x)
        data["y"] = float(y)
        if "yaw" in data:
            try:
                yaw = float(data["yaw"])
                if np.isfinite(yaw):
                    data["yaw"] = yaw
                else:
                    data.pop("yaw", None)
            except Exception:
                data.pop("yaw", None)
        with self._lock:
            self._latest = data
            self._recv_time = time.time()
            self._msg_count += 1

    def snapshot(self, max_age_s: float) -> dict:
        with self._lock:
            data = dict(self._latest) if self._latest is not None else None
            recv_time = self._recv_time
            msg_count = self._msg_count
        age = time.time() - recv_time if data is not None else float("inf")
        if data is None:
            return {"fresh": False, "age": age, "msg_count": msg_count}
        data["fresh"] = age <= max_age_s
        data["age"] = age
        data["msg_count"] = msg_count
        return data

    def wait_fresh(self, timeout_s: float, max_age_s: float) -> dict | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            snap = self.snapshot(max_age_s)
            if snap.get("fresh"):
                return snap
            time.sleep(0.05)
        return None

    def wait_stable(self, timeout_s: float, max_age_s: float,
                    min_count: int = 5, min_span_s: float = 1.0) -> dict | None:
        deadline = time.time() + timeout_s
        first_count = None
        first_t = 0.0
        while time.time() < deadline:
            snap = self.snapshot(max_age_s)
            if snap.get("fresh"):
                msg_count = int(snap.get("msg_count", 0))
                if first_count is None:
                    first_count = msg_count
                    first_t = time.time()
                seen = msg_count - first_count + 1
                if seen >= max(1, min_count) and time.time() - first_t >= min_span_s:
                    return snap
            else:
                first_count = None
                first_t = 0.0
            time.sleep(0.05)
        return None


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


def _yaw_to_world_rot(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _world_to_body_xy(v_w: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = float(v_w[0]), float(v_w[1])
    return np.array([c * x + s * y, -s * x + c * y], dtype=np.float32)


class OnlineCommandOdomFrame:
    """Learns how SDK body-frame velocity commands appear in external odom.

    The MID-360/LIO yaw is not guaranteed to share the SDK command frame's zero
    heading. This estimator uses observed odom deltas under recently sent SDK
    commands, then converts odom displacement into the command/body frame used
    by the geometric return controller or by the optional return head.
    """

    def __init__(self,
                 window: int = 8,
                 min_samples: int = 3,
                 min_cmd_norm: float = 0.08,
                 min_delta_norm: float = 0.005,
                 max_delta_norm: float = 0.30):
        self.window = int(max(4, window))
        self.min_samples = int(max(2, min_samples))
        self.min_cmd_norm = float(min_cmd_norm)
        self.min_delta_norm = float(min_delta_norm)
        self.max_delta_norm = float(max_delta_norm)
        self.samples = deque(maxlen=self.window)
        self._last_xy = None
        self._last_msg_count = None
        self._last_cmd = np.zeros(2, dtype=np.float32)
        self._last_accept = None

    def reset(self,
              odom_xy: np.ndarray | None = None,
              msg_count: int | None = None,
              cmd_xy: np.ndarray | None = None):
        self.samples.clear()
        self._last_xy = (None if odom_xy is None
                         else np.asarray(odom_xy, dtype=np.float32).copy())
        self._last_msg_count = msg_count
        self._last_cmd = (np.zeros(2, dtype=np.float32) if cmd_xy is None
                          else np.asarray(cmd_xy, dtype=np.float32).copy())
        self._last_accept = None

    def update(self,
               odom_xy: np.ndarray | None,
               msg_count: int | None,
               cmd_xy: np.ndarray):
        if odom_xy is None or msg_count is None:
            return
        odom_xy = np.asarray(odom_xy, dtype=np.float32).reshape(2)
        cmd_xy = np.asarray(cmd_xy, dtype=np.float32).reshape(2)
        if self._last_xy is None:
            self._last_xy = odom_xy.copy()
            self._last_msg_count = msg_count
            self._last_cmd = cmd_xy.copy()
            return
        if msg_count == self._last_msg_count:
            self._last_cmd = cmd_xy.copy()
            return

        delta = odom_xy - self._last_xy
        cmd_used = self._last_cmd.copy()
        self._last_xy = odom_xy.copy()
        self._last_msg_count = msg_count
        self._last_cmd = cmd_xy.copy()

        cmd_norm = float(np.linalg.norm(cmd_used))
        delta_norm = float(np.linalg.norm(delta))
        if cmd_norm < self.min_cmd_norm:
            return
        if delta_norm < self.min_delta_norm or delta_norm > self.max_delta_norm:
            return
        if not (np.all(np.isfinite(cmd_used)) and np.all(np.isfinite(delta))):
            return
        self.samples.append((cmd_used, delta.astype(np.float32)))
        self._last_accept = {
            "cmd": cmd_used.copy(),
            "delta": delta.astype(np.float32).copy(),
            "cmd_norm": cmd_norm,
            "delta_norm": delta_norm,
        }

    def _fit(self) -> dict | None:
        if len(self.samples) < self.min_samples:
            return None
        cmds = np.stack([s[0] for s in self.samples], axis=0)
        deltas = np.stack([s[1] for s in self.samples], axis=0)
        try:
            # deltas ~= cmds @ map_t, so map maps command-frame xy to odom xy.
            map_t, *_ = np.linalg.lstsq(cmds, deltas, rcond=None)
        except np.linalg.LinAlgError:
            return None
        mapping = map_t.T.astype(np.float32)
        pred = cmds @ map_t
        fit_rms = float(np.sqrt(np.mean(np.sum((pred - deltas) ** 2, axis=1))))

        x_axis = mapping[:, 0]
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm < 1e-5:
            return None
        x_axis = x_axis / x_norm

        # The SDK command frame is a proper planar body frame: +x forward and
        # +y lateral. The lateral response from short dodge samples is noisy
        # enough to occasionally fit a reflected frame (det=-1), which sends
        # recovery commands in the wrong direction. Trust the measured forward
        # axis and construct the lateral axis as its proper +90 deg companion.
        y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)
        rot = np.column_stack([x_axis, y_axis]).astype(np.float32)
        last = self._last_accept
        last_pred = None
        if last is not None:
            last_pred = (mapping @ last["cmd"]).astype(np.float32)
        return {
            "samples": len(self.samples),
            "mapping": mapping,
            "rot": rot,
            "fit_rms": fit_rms,
            "scale_x": float(np.linalg.norm(mapping[:, 0])),
            "scale_y": float(np.linalg.norm(mapping[:, 1])),
            "det": float(np.linalg.det(rot)),
            "last": last,
            "last_pred": last_pred,
        }

    def rotation(self) -> np.ndarray | None:
        fit = self._fit()
        return None if fit is None else fit["rot"]

    def diagnostics(self) -> dict:
        fit = self._fit()
        if fit is None:
            return {
                "samples": len(self.samples),
                "mapping": None,
                "rot": None,
                "fit_rms": None,
                "scale_x": None,
                "scale_y": None,
                "det": None,
                "last": self._last_accept,
                "last_pred": None,
            }
        return fit

    @property
    def sample_count(self) -> int:
        return len(self.samples)


def _load_return_head(path: str | Path) -> nn.Module:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"return head checkpoint not found: {ckpt_path}")
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)["model_state_dict"]
    model = nn.Sequential(nn.Linear(2, 32), nn.ELU(), nn.Linear(32, 3), nn.Tanh())
    model.load_state_dict(
        {k.replace("return_head.", ""): v for k, v in sd.items()
         if "return_head" in k})
    model.eval()
    print(f"[ReturnHead] Loaded {ckpt_path}")
    return model


def _load_gated_return_head(path: str | Path) -> nn.Module:
    """Rebuild the 6-dim return_head MLP from a gated checkpoint.

    Mirrors eval_safe_recovery.py: Sequential(Linear, ELU, ..., Linear, Tanh)
    inferred from the sorted return_head.* weight shapes.
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"gated checkpoint not found: {ckpt_path}")
    md = torch.load(str(ckpt_path), map_location="cpu",
                    weights_only=False)["model_state_dict"]
    rsd = {k.replace("return_head.", ""): v for k, v in md.items()
           if k.startswith("return_head.")}
    if not rsd:
        raise ValueError(f"no return_head.* weights in {ckpt_path}")
    wk = sorted(k for k in rsd if k.endswith(".weight"))
    mods: list[nn.Module] = []
    for i, k in enumerate(wk):
        mods.append(nn.Linear(rsd[k].shape[1], rsd[k].shape[0]))
        mods.append(nn.ELU() if i < len(wk) - 1 else nn.Tanh())
    model = nn.Sequential(*mods)
    model.load_state_dict(rsd)
    model.eval()
    print(f"[GatedReturn] Loaded {ckpt_path} (input {rsd[wk[0]].shape[1]}D)")
    return model


def _return_head_velocity(return_head: nn.Module,
                          disp_b: np.ndarray,
                          lin_scale: float,
                          max_lin_vel: float,
                          max_ang_vel: float,
                          yaw_err: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        d = torch.tensor(np.asarray(disp_b, dtype=np.float32),
                         dtype=torch.float32).unsqueeze(0)
        raw = return_head(d).squeeze(0).numpy()
    out = np.zeros(3, dtype=np.float32)
    out[0] = float(np.clip(raw[0] * lin_scale, -max_lin_vel, max_lin_vel))
    out[1] = float(np.clip(raw[1] * lin_scale, -max_lin_vel, max_lin_vel))
    out[2] = float(np.clip(-2.0 * yaw_err, -max_ang_vel, max_ang_vel))
    return out, raw.astype(np.float32)


def _return_frame_rotation(yaw: float,
                           odom_yaw: float | None,
                           frame: OnlineCommandOdomFrame | None,
                           frozen_rot: np.ndarray | None = None,
                           frozen_yaw_ref: float | None = None) -> tuple[np.ndarray, str]:
    if frozen_rot is not None:
        rot = np.asarray(frozen_rot, dtype=np.float32)
        if frozen_yaw_ref is not None:
            yaw_now = odom_yaw if odom_yaw is not None else yaw
            yaw_delta = _wrap_pi(float(yaw_now) - float(frozen_yaw_ref))
            return (_yaw_to_world_rot(yaw_delta) @ rot).astype(np.float32), "online_frozen_yaw"
        return rot, "online_frozen"
    if frame is not None:
        rot = frame.rotation()
        if rot is not None:
            return rot, "online"
    if odom_yaw is not None:
        return _yaw_to_world_rot(odom_yaw), "odom_yaw"
    return _yaw_to_world_rot(yaw), "lowstate_yaw"


def _fused_return_yaw(origin_odom_yaw: float | None,
                      origin_low_yaw: float | None,
                      current_low_yaw: float,
                      current_odom_yaw: float | None,
                      source: str,
                      low_yaw_sign: float = -1.0) -> tuple[float | None, str, float | None, float | None, float | None]:
    """Yaw used for world->SDK return commands.

    FAST-LIO x/y is useful as a map displacement, but its live yaw can diverge
    during high-level gait transitions. The robot lowstate yaw is a better
    relative body-heading signal, so the default keeps SLAM's yaw at the dodge
    origin and advances it by lowstate's relative yaw change.
    """
    slam_delta = None
    low_delta = None
    mismatch = None
    if origin_odom_yaw is not None and current_odom_yaw is not None:
        slam_delta = _wrap_pi(float(current_odom_yaw) - float(origin_odom_yaw))
    if origin_low_yaw is not None:
        # lowstate IMU yaw rotates opposite to the FAST-LIO map frame on this
        # robot, so flip its delta before fusing with the SLAM origin yaw.
        low_delta = float(low_yaw_sign) * _wrap_pi(
            float(current_low_yaw) - float(origin_low_yaw))
    if slam_delta is not None and low_delta is not None:
        mismatch = abs(_wrap_pi(slam_delta - low_delta))

    if source == "slam":
        if current_odom_yaw is not None:
            return float(current_odom_yaw), "live_slam_yaw", slam_delta, low_delta, mismatch
        return float(current_low_yaw), "lowstate_abs_yaw", slam_delta, low_delta, mismatch
    if source == "lowstate":
        return float(current_low_yaw), "lowstate_abs_yaw", slam_delta, low_delta, mismatch
    if origin_odom_yaw is not None and low_delta is not None:
        return _wrap_pi(float(origin_odom_yaw) + low_delta), "slam_origin+lowstate_delta", slam_delta, low_delta, mismatch
    if current_odom_yaw is not None:
        return float(current_odom_yaw), "live_slam_yaw", slam_delta, low_delta, mismatch
    return float(current_low_yaw), "lowstate_abs_yaw", slam_delta, low_delta, mismatch


def _disp_world_to_return_frame(disp_w: np.ndarray,
                                yaw: float,
                                odom_yaw: float | None,
                                frame: OnlineCommandOdomFrame | None,
                                frozen_rot: np.ndarray | None = None,
                                frozen_yaw_ref: float | None = None) -> tuple[np.ndarray, str]:
    rot, src = _return_frame_rotation(yaw, odom_yaw, frame, frozen_rot, frozen_yaw_ref)
    return (rot.T @ np.asarray(disp_w, dtype=np.float32)).astype(np.float32), src


def _fit_probe_return_frame(delta_x_w: np.ndarray,
                            delta_y_w: np.ndarray,
                            min_delta: float,
                            max_axis_cos: float) -> tuple[np.ndarray | None, str]:
    """Fit command->world axes from short SDK probe motions.

    The SDK SetVelocity frame is a robot API contract, not a SLAM contract.  We
    measure the actual world displacement caused by +x and +y commands, then use
    those measured axes to convert the SLAM return vector into SDK commands.
    """
    dx = np.asarray(delta_x_w, dtype=np.float32).reshape(2)
    dy = np.asarray(delta_y_w, dtype=np.float32).reshape(2)
    nx = float(np.linalg.norm(dx))
    ny = float(np.linalg.norm(dy))
    if not np.all(np.isfinite(dx)) or not np.all(np.isfinite(dy)):
        return None, "non-finite probe delta"
    if nx < min_delta:
        return None, f"+x probe too small ({nx:.3f}m)"
    if ny < min_delta:
        return None, f"+y probe too small ({ny:.3f}m)"

    x_axis = dx / max(nx, 1e-6)
    axis_cos = float(np.dot(dx, dy) / max(nx * ny, 1e-6))
    if abs(axis_cos) > max_axis_cos:
        return None, (f"probe axes not independent "
                      f"(cos={axis_cos:+.2f}, max={max_axis_cos:.2f})")
    dy_orth = dy - float(np.dot(dy, x_axis)) * x_axis
    n_orth = float(np.linalg.norm(dy_orth))
    if n_orth < min_delta:
        return None, (f"+y probe too small "
                      f"(dy={ny:.3f}m orth={n_orth:.3f}m)")
    y_axis = dy_orth / n_orth
    rot = np.column_stack([x_axis, y_axis]).astype(np.float32)
    det = float(np.linalg.det(rot))
    if abs(det) < 0.5:
        return None, f"degenerate probe frame det={det:+.2f}"
    return rot, (f"dx={nx:.3f}m dy={ny:.3f}m "
                 f"orth={n_orth:.3f}m axis_cos={axis_cos:+.2f} "
                 f"det={det:+.2f}")


def _load_return_frame_calib(path: str) -> tuple[np.ndarray, float | None, dict]:
    data = json.loads(Path(path).expanduser().read_text())
    raw_rot = (
        data.get("rot_cmd_to_world")
        or data.get("return_frame_rot")
        or data.get("rot")
    )
    rot = np.asarray(raw_rot, dtype=np.float32)
    if rot.shape != (2, 2) or not np.all(np.isfinite(rot)):
        raise ValueError("calibration rot must be a finite 2x2 matrix")
    det = float(np.linalg.det(rot))
    if abs(det) < 0.5:
        raise ValueError(f"calibration rot is degenerate det={det:+.2f}")
    yaw_ref = data.get("yaw_ref")
    if yaw_ref is not None:
        yaw_ref = float(yaw_ref)
        if not np.isfinite(yaw_ref):
            yaw_ref = None
    return rot, yaw_ref, data


def _odom_xy_from_snap(snap: dict) -> np.ndarray:
    return np.array([snap["x"], snap["y"]], dtype=np.float32)


def _wait_external_stationary(odom: ExternalOdomMonitor,
                              staleness: float,
                              window_s: float,
                              max_disp: float,
                              timeout_s: float,
                              label: str) -> dict:
    deadline = time.time() + max(0.1, timeout_s)
    samples = deque()
    last_warn = 0.0
    last_snap = None
    while time.time() < deadline:
        snap = odom.snapshot(staleness)
        now = time.time()
        if snap.get("fresh"):
            xy = _odom_xy_from_snap(snap)
            samples.append((now, xy, snap))
            last_snap = snap
            while samples and now - samples[0][0] > window_s:
                samples.popleft()
            if len(samples) >= 2 and now - samples[0][0] >= window_s * 0.8:
                disp = float(np.linalg.norm(samples[-1][1] - samples[0][1]))
                if disp <= max_disp:
                    return snap
                if now - last_warn > 0.7:
                    print(f"[STARTUP CALIB] waiting stationary before {label}: "
                          f"window={disp:.3f}m/{window_s:.1f}s "
                          f"> {max_disp:.3f}m")
                    last_warn = now
        time.sleep(0.03)
    if last_snap is None:
        raise RuntimeError(f"no fresh odom while waiting stationary before {label}")
    raise RuntimeError(f"robot/SLAM did not settle before {label}")


def _command_axis_until_delta(loco: LocoClient,
                              odom: ExternalOdomMonitor,
                              staleness: float,
                              cmd_xy: tuple[float, float],
                              target_dist: float,
                              min_accept_dist: float,
                              max_time_s: float,
                              cmd_duration: float,
                              cmd_period: float,
                              label: str) -> np.ndarray:
    start = odom.wait_fresh(2.0, staleness)
    if start is None:
        raise RuntimeError(f"no fresh odom before {label}")
    start_xy = _odom_xy_from_snap(start)
    deadline = time.time() + max(0.1, max_time_s)
    best_delta = np.zeros(2, dtype=np.float32)
    best_dist = 0.0
    last_print = 0.0
    next_send = 0.0
    while time.time() < deadline:
        now = time.time()
        if now >= next_send:
            code = loco.SetVelocity(
                float(cmd_xy[0]),
                float(cmd_xy[1]),
                0.0,
                duration=float(cmd_duration),
            )
            if code != 0:
                raise RuntimeError(f"SetVelocity failed during {label}: code={code}")
            next_send = now + max(0.05, float(cmd_period))
        snap = odom.snapshot(staleness)
        if snap.get("fresh"):
            delta = _odom_xy_from_snap(snap) - start_xy
            dist = float(np.linalg.norm(delta))
            if dist > best_dist:
                best_dist = dist
                best_delta = delta.copy()
            if now - last_print > 0.5:
                print(f"[STARTUP CALIB] {label} progress "
                      f"{dist:.3f}/{target_dist:.3f}m "
                      f"delta={_fmt_vec(delta, 3)}")
                last_print = now
            if dist >= target_dist:
                print(f"[STARTUP CALIB] {label} reached "
                      f"{dist:.3f}m delta={_fmt_vec(delta, 3)}")
                return delta.astype(np.float32)
        time.sleep(0.05)
    if best_dist >= min_accept_dist:
        print(f"[STARTUP CALIB] {label} accepted partial "
              f"{best_dist:.3f}m delta={_fmt_vec(best_delta, 3)} "
              f"(target {target_dist:.3f}m not reached)")
        return best_delta.astype(np.float32)
    raise RuntimeError(
        f"{label} did not reach {target_dist:.3f}m; "
        f"best={best_dist:.3f}m delta={_fmt_vec(best_delta, 3)} "
        f"< min_accept={min_accept_dist:.3f}m")


def _zero_loco_velocity(loco: LocoClient,
                        repeats: int = 10,
                        stopmove: bool = False):
    for _ in range(max(1, int(repeats))):
        loco.SetVelocity(0.0, 0.0, 0.0, duration=0.05)
        time.sleep(0.02)
    if stopmove:
        loco.StopMove()


def _write_return_frame_calib(path: str,
                              rot: np.ndarray,
                              yaw_ref: float | None,
                              meta: dict):
    out = Path(path).expanduser().resolve()
    data = {
        "type": "startup_sdk_slam_return_frame",
        "created_unix": time.time(),
        "body_convention": "SDK SetVelocity command frame as measured",
        "rot_cmd_to_world": np.asarray(rot, dtype=float).tolist(),
        "yaw_ref": yaw_ref,
        "validation": meta,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"[STARTUP CALIB] wrote {out}")


def _run_startup_frame_calib(loco: LocoClient,
                             odom: ExternalOdomMonitor,
                             args) -> tuple[np.ndarray, float | None, dict]:
    if odom is None:
        raise RuntimeError("startup frame calibration requires external odom")
    if args.return_odom_source not in ("external", "auto"):
        raise RuntimeError("startup frame calibration requires --return_odom_source external/auto")

    print("[STARTUP CALIB] active SDK/SLAM calibration before YOLO dodge")
    print("[STARTUP CALIB] keep a clear area; YOLO dodge is not enabled yet")
    target_dist = float(args.startup_calib_distance)
    speed = abs(float(args.startup_calib_speed))
    y_sign = -1.0 if float(args.startup_calib_y_sign) < 0.0 else 1.0
    max_time = float(args.startup_calib_axis_timeout)
    min_delta = float(args.startup_calib_min_delta)

    try:
        _zero_loco_velocity(loco, args.startup_calib_zero_repeats, stopmove=True)
        start = _wait_external_stationary(
            odom,
            args.odom_staleness,
            args.startup_calib_stationary_time,
            args.startup_calib_stationary_disp,
            args.startup_calib_stationary_timeout,
            "+x",
        )
        yaw_ref = float(start["yaw"]) if "yaw" in start else None
        print(f"[STARTUP CALIB] start x={float(start['x']):+.3f} "
              f"y={float(start['y']):+.3f} yaw={yaw_ref}")

        dx = _command_axis_until_delta(
            loco,
            odom,
            args.odom_staleness,
            (speed, 0.0),
            target_dist,
            min_delta,
            max_time,
            args.startup_calib_cmd_duration,
            args.startup_calib_cmd_period,
            "+x",
        )
        _zero_loco_velocity(loco, args.startup_calib_zero_repeats, stopmove=True)
        _wait_external_stationary(
            odom,
            args.odom_staleness,
            args.startup_calib_stationary_time,
            args.startup_calib_stationary_disp,
            args.startup_calib_stationary_timeout,
            "+y",
        )

        dy_measured = _command_axis_until_delta(
            loco,
            odom,
            args.odom_staleness,
            (0.0, y_sign * speed),
            target_dist,
            min_delta,
            max_time,
            args.startup_calib_cmd_duration,
            args.startup_calib_cmd_period,
            "visible right" if y_sign < 0.0 else "+y",
        )
        _zero_loco_velocity(loco, args.startup_calib_zero_repeats, stopmove=True)
        end = _wait_external_stationary(
            odom,
            args.odom_staleness,
            args.startup_calib_stationary_time,
            args.startup_calib_stationary_disp,
            args.startup_calib_stationary_timeout,
            "finish",
        )
    finally:
        try:
            _zero_loco_velocity(loco, args.startup_calib_zero_repeats, stopmove=True)
        except Exception as exc:
            print(f"[STARTUP CALIB] final zero failed: {exc}")

    # _fit_probe_return_frame expects the measured delta for command +y.
    dy_plus_y = (dy_measured / y_sign).astype(np.float32)
    rot, reason = _fit_probe_return_frame(
        dx,
        dy_plus_y,
        min_delta,
        float(args.startup_calib_max_axis_cos),
    )
    if rot is None:
        raise RuntimeError(reason)

    end_yaw = float(end["yaw"]) if "yaw" in end else None
    yaw_drift = None
    if yaw_ref is not None and end_yaw is not None:
        yaw_drift = abs(_wrap_pi(end_yaw - yaw_ref))
        if yaw_drift > np.deg2rad(float(args.startup_calib_max_yaw_change_deg)):
            msg = (f"SLAM yaw changed {np.rad2deg(yaw_drift):.1f}deg during "
                   "startup calibration")
            if args.return_frame_rotate_with_slam_yaw:
                raise RuntimeError(
                    f"{msg}; cannot rotate the calibrated frame by live yaw")
            print(f"[STARTUP CALIB] WARN {msg}; keeping fixed calibrated frame")
    return_yaw_ref = (yaw_ref if args.return_frame_rotate_with_slam_yaw else None)

    meta = {
        "dx": dx.astype(float).tolist(),
        "dy_measured": dy_measured.astype(float).tolist(),
        "dy_plus_y": dy_plus_y.astype(float).tolist(),
        "y_cmd_sign": y_sign,
        "distance_target_m": target_dist,
        "speed_mps": speed,
        "fit": reason,
        "det": float(np.linalg.det(rot)),
        "yaw_ref": yaw_ref,
        "end_yaw": end_yaw,
        "yaw_drift_rad": yaw_drift,
        "return_frame_rotate_with_slam_yaw": bool(
            args.return_frame_rotate_with_slam_yaw),
    }
    print("[STARTUP CALIB] PASS "
          f"rot={_fmt_mat2(rot, 3)} yaw_ref={return_yaw_ref} {reason}")
    if args.startup_calib_out:
        _write_return_frame_calib(args.startup_calib_out, rot, return_yaw_ref, meta)
    return rot, return_yaw_ref, meta


def _shape_return_velocity(back_to_origin_b: np.ndarray,
                           gain: float,
                           lat_gain: float,
                           max_vel: float,
                           max_lat_vel: float,
                           min_vel: float,
                           done_dist: float) -> np.ndarray:
    dist = float(np.linalg.norm(back_to_origin_b))
    out = np.zeros(3, dtype=np.float32)
    if dist <= done_dist or dist < 1e-6:
        return out
    err = np.asarray(back_to_origin_b, dtype=np.float32).reshape(2)
    xy = np.array([gain * err[0], lat_gain * err[1]], dtype=np.float32)
    xy[0] = np.clip(xy[0], -max_vel, max_vel)
    xy[1] = np.clip(xy[1], -max_lat_vel, max_lat_vel)
    if min_vel > 0.0 and abs(err[0]) > done_dist and abs(xy[0]) < min_vel:
        xy[0] = np.copysign(min_vel, err[0])
    if abs(err[1]) <= done_dist:
        xy[1] = 0.0
    out[:2] = xy
    return out


def _stabilize_return_head_velocity(target: np.ndarray,
                                    disp_b: np.ndarray,
                                    max_vel: float,
                                    max_lat_vel: float,
                                    min_vel: float,
                                    done_dist: float) -> tuple[np.ndarray, bool]:
    """Keep the trained return head from losing the main return direction.

    The return head still chooses the command, but the SDK path has noisier
    odometry/command-frame alignment than MuJoCo. If the learned frame jitters,
    the raw head can briefly flip signs and spend the whole return phase moving
    sideways. This guard enforces progress on the dominant fore/aft error and
    caps lateral recovery so forward/back recovery remains the primary action.
    """
    out = np.asarray(target, dtype=np.float32).copy()
    disp_b = np.asarray(disp_b, dtype=np.float32).reshape(2)
    changed = False

    if abs(float(disp_b[0])) > done_dist:
        desired_x_sign = -np.sign(float(disp_b[0]))
        if desired_x_sign == 0.0:
            desired_x_sign = 1.0
        if out[0] == 0.0 or np.sign(float(out[0])) != desired_x_sign:
            out[0] = desired_x_sign * max(min_vel, min(max_vel, abs(float(out[0]))))
            changed = True
        elif abs(float(out[0])) < min_vel:
            out[0] = desired_x_sign * min_vel
            changed = True
    else:
        if out[0] != 0.0:
            out[0] = 0.0
            changed = True

    lat_before = float(out[1])
    if abs(float(disp_b[1])) <= done_dist:
        out[1] = 0.0
    else:
        out[1] = float(np.clip(out[1], -max_lat_vel, max_lat_vel))
    if abs(float(out[1]) - lat_before) > 1e-6:
        changed = True

    out[0] = float(np.clip(out[0], -max_vel, max_vel))
    out[1] = float(np.clip(out[1], -max_lat_vel, max_lat_vel))
    return out.astype(np.float32), changed


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
    parser.add_argument("--max_vel", type=float, default=0.30,
                        help="Per-axis high-level velocity cap in m/s.")
    parser.add_argument("--max_ang_vel", type=float, default=0.0,
                        help="Yaw-rate cap rad/s. Default 0 keeps camera pointed at the person.")
    parser.add_argument("--safety_dist", type=float, default=1.0)
    parser.add_argument("--clear_margin", type=float, default=0.00,
                        help="Stop dodge after obstacle distance exceeds safety_dist + margin.")
    parser.add_argument("--cmd_hz", type=float, default=10.0,
                        help="High-level SetVelocity refresh rate.")
    parser.add_argument("--cmd_duration", type=float, default=0.30,
                        help="SetVelocity duration. Short duration prevents runaway if script dies.")
    parser.add_argument("--idle_stopmove", dest="idle_stopmove",
                        action="store_true", default=True,
                        help="When READY/idle, send StopMove once and stop "
                             "streaming SetVelocity(0,0,0). This makes idle "
                             "behave like a standstill instead of keeping the "
                             "walking controller active.")
    parser.add_argument("--no_idle_stopmove", dest="idle_stopmove",
                        action="store_false",
                        help="Do not call StopMove when READY/idle.")
    parser.add_argument("--hold_zero_ready", action="store_true", default=False,
                        help="Legacy behavior: keep streaming SetVelocity(0,0,0) "
                             "while READY/idle. This can make G1 step in place.")
    parser.add_argument("--max_accel_lin", type=float, default=1.5,
                        help="Linear command slew limit in m/s^2.")
    parser.add_argument("--max_accel_ang", type=float, default=2.0,
                        help="Angular command slew limit in rad/s^2.")
    parser.add_argument("--min_loco_lin_vel", type=float, default=0.30,
                        help="Raise nonzero dodge command norm above this value if cap allows it.")
    parser.add_argument("--close_escape_speed", type=float, default=None,
                        help="Minimum radial away speed near the obstacle. Default uses the "
                             "largest feasible radial speed under max_vel.")
    parser.add_argument("--dodge_dir_latch", dest="dodge_dir_latch",
                        action="store_true", default=True,
                        help="Latch the lateral escape side at the start of a dodge so a "
                             "crossing obstacle cannot make the dodge wag left<->right "
                             "(net displacement cancels and the online return-frame fit "
                             "starves, which is what makes the recover walk the wrong way).")
    parser.add_argument("--no_dodge_dir_latch", dest="dodge_dir_latch",
                        action="store_false",
                        help="Allow the dodge lateral command to switch sides mid-dodge.")
    parser.add_argument("--dodge_latch_deadband", type=float, default=0.10,
                        help="Lateral command magnitude (m/s) needed to commit the latched "
                             "dodge side.")
    parser.add_argument("--max_dodge_dist", type=float, default=2.0,
                        help="Cap dodge displacement from the origin (m). At/above this the "
                             "dodge holds position instead of running further away. "
                             "<=0 disables the cap.")
    parser.add_argument("--min_dodge_dist", type=float, default=0.5,
                        help="Keep escaping in the latched direction until the dodge has "
                             "moved at least this far from origin (m), so every dodge is big "
                             "enough for a clean return-frame fit. <=0 disables.")
    parser.add_argument("--min_dodge_time", type=float, default=2.5,
                        help="Max seconds to spend reaching --min_dodge_dist before allowing "
                             "the return regardless.")
    parser.add_argument("--dodge_rear_bias", dest="dodge_rear_bias",
                        action="store_true", default=False,
                        help="Force the dodge direction into the REAR sector (straight-back / "
                             "diagonal-back), clamped to +/- --dodge_rear_angle around "
                             "straight-back. Makes the recover a pure FORWARD walk -- the most "
                             "accurate direction for the G1 and robust to lateral "
                             "return-frame degeneracy.")
    parser.add_argument("--no_dodge_rear_bias", dest="dodge_rear_bias",
                        action="store_false")
    parser.add_argument("--dodge_rear_angle", type=float, default=40.0,
                        help="Half-angle (deg) of the rear sector the dodge is clamped to: "
                             "direction held within [180-angle, 180+angle]. 0 = straight back.")
    parser.add_argument("--return_enable", action="store_true", default=True,
                        help="After dodge clears, estimate displacement from sent SDK "
                             "velocity commands and walk back toward the start pose.")
    parser.add_argument("--no_return", dest="return_enable", action="store_false",
                        help="Disable post-dodge return.")
    parser.add_argument("--return_mode", choices=["geo", "head", "p", "gated"], default="geo",
                        help="'geo' uses SLAM displacement plus live LIO yaw to command "
                             "directly toward the dodge origin. 'head' uses "
                             "checkpoints/return_head_v23b_v6.pt for debugging. "
                             "'p' keeps the old hand-written P controller. 'gated' uses "
                             "the learned return_head inside --gated_ckpt.")
    parser.add_argument("--return_head_ckpt",
                        default=str(_REPO / "checkpoints" / "return_head_v23b_v6.pt"),
                        help="Checkpoint for the trained return head.")
    parser.add_argument("--gated_ckpt",
                        default=str(_REPO / "checkpoints" / "v23b_gated_return.pt"),
                        help="Checkpoint holding the gated return_head (6-dim MLP). "
                             "Used when --return_mode gated.")
    parser.add_argument("--return_gated_lin_vel", type=float, default=0.5,
                        help="Linear velocity scale (m/s) applied to the gated "
                             "return_head XY action. 0.5 matches training; lower to "
                             "~0.25 to match the validated geo return speed.")
    parser.add_argument("--return_gated_yaw_sign", type=float, default=-1.0,
                        help="Sign applied to the heading error fed as the gated "
                             "return_head's yaw input, so its own v_rz output matches "
                             "this robot's turn direction. -1 matches geo; flip to +1 "
                             "if return yaw rotates the wrong way.")
    parser.add_argument("--return_max_vel", type=float, default=0.25,
                        help="Per-axis cap for post-dodge return velocity.")
    parser.add_argument("--return_min_vel", type=float, default=0.12,
                        help="Minimum return command norm while estimated displacement "
                             "is still above --return_done_dist.")
    parser.add_argument("--return_gain", type=float, default=0.8,
                        help="P gain from estimated displacement to return velocity.")
    parser.add_argument("--return_lat_gain", type=float, default=0.35,
                        help="P gain for lateral return correction. Keep lower than "
                             "--return_gain because SDK lateral tracking is less stable.")
    parser.add_argument("--return_done_dist", type=float, default=0.10,
                        help="Estimated displacement below this is considered returned.")
    parser.add_argument("--return_timeout", type=float, default=15.0,
                        help="Maximum seconds spent in one RETURN phase.")
    parser.add_argument("--return_no_progress_timeout", type=float, default=3.0,
                        help="Abort RETURN if real LIO displacement does not make "
                             "positive progress toward the origin for this many seconds.")
    parser.add_argument("--return_stationary_speed", type=float, default=0.03,
                        help="Average SLAM speed threshold over the stationary "
                        "window before RETURN/probe. Used with "
                             "--return_stationary_disp; both must pass.")
    parser.add_argument("--return_stationary_time", type=float, default=1.00,
                        help="Window length used to decide whether the robot has "
                             "settled before RETURN/probe starts.")
    parser.add_argument("--return_stationary_disp", type=float, default=0.03,
                        help="Maximum SLAM net displacement over "
                             "--return_stationary_time accepted as settled. This is "
                             "more robust than two-sample speed when LIO jitters.")
    parser.add_argument("--return_settle_max_wait", type=float, default=4.0,
                        help="Maximum extra seconds to hold zero waiting for the "
                             "stationary gate before aborting the return attempt. "
                             "Set <=0 to keep holding zero indefinitely.")
    parser.add_argument("--return_bad_progress_abort_count", type=int, default=3,
                        help="Abort RETURN after this many fresh odom updates move "
                             "clearly away from the origin. Set <=0 to disable.")
    parser.add_argument("--return_max_frame_flips", type=int, default=1,
                        help="When the return is detected walking AWAY from origin, the "
                             "online return frame is almost always fit ~180deg backwards "
                             "(det=+1 but reversed). Instead of aborting, flip the frame "
                             "180deg (negate disp_b) and retry up to this many times. "
                             "Set 0 to disable (abort immediately, old behavior).")
    parser.add_argument("--return_odom_stale_abort", type=float, default=1.5,
                        help="Abort RETURN if external odom stays stale this long. "
                             "Set <=0 to keep waiting with zero velocity.")
    parser.add_argument("--return_sdk_error_abort", type=int, default=3,
                        help="Abort RETURN after this many consecutive nonzero "
                             "SetVelocity return codes. Set <=0 to disable.")
    parser.add_argument("--return_prestop", dest="return_prestop",
                        action="store_true", default=True,
                        help="After dodge clears, send zero velocity + StopMove, "
                             "keep the controller alive, then wait for SLAM "
                             "settle before probe/recover.")
    parser.add_argument("--no_return_prestop", dest="return_prestop",
                        action="store_false",
                        help="Do not call StopMove before return/probe; only hold "
                             "zero velocity while waiting to settle.")
    parser.add_argument("--return_prestop_repeats", type=int, default=10,
                        help="Number of short zero-velocity commands to send "
                             "before pre-return StopMove.")
    parser.add_argument("--exit_on_return_abort", action="store_true", default=True,
                        help="Exit the controller after an unrecoverable RETURN "
                             "abort so the final StopMove path runs immediately.")
    parser.add_argument("--no_exit_on_return_abort", dest="exit_on_return_abort",
                        action="store_false",
                        help="After RETURN abort, stay in READY instead of "
                             "exiting. Normal READY idle handling still applies.")
    parser.add_argument("--return_clear_delay", type=float, default=1.0,
                        help="Obstacle must stay clear/missing this long before RETURN starts.")
    parser.add_argument("--return_probe", dest="return_probe",
                        action="store_true", default=True,
                        help="In geo mode, measure SDK +x/+y directions with short "
                             "SLAM-observed probe motions before returning.")
    parser.add_argument("--no_return_probe", dest="return_probe",
                        action="store_false",
                        help="Disable return SDK-frame probe and use LIO yaw directly.")
    parser.add_argument("--return_probe_speed", type=float, default=0.30,
                        help="Forward speed for geo return SDK-frame probe.")
    parser.add_argument("--return_probe_lat_speed", type=float, default=0.30,
                        help="Lateral speed for geo return SDK-frame probe.")
    parser.add_argument("--return_probe_duration", type=float, default=0.80,
                        help="Seconds for each geo return SDK-frame probe axis.")
    parser.add_argument("--return_probe_settle", type=float, default=0.30,
                        help="Zero-velocity settle time before/between return probe axes.")
    parser.add_argument("--return_probe_min_delta", type=float, default=0.080,
                        help="Minimum SLAM displacement for accepting one probe axis.")
    parser.add_argument("--return_probe_max_axis_cos", type=float, default=0.70,
                        help="Reject SDK frame probe if +x/+y measured SLAM deltas "
                             "are too collinear. Large values mean residual drift "
                             "polluted the probe.")
    parser.add_argument("--return_probe_retry_count", type=int, default=1,
                        help="Retry the SDK-frame probe this many times if one "
                             "axis displacement is too small to calibrate.")
    parser.add_argument("--return_probe_retry_scale", type=float, default=1.7,
                        help="Multiplier applied to probe speed and duration on "
                             "each too-small retry, capped by --max_vel for speed.")
    parser.add_argument("--return_frame_calib", default="",
                        help="Manual body/SLAM frame JSON from "
                             "scripts/manual_slam_frame_calib.py. In geo mode this "
                             "skips active SDK probing and uses the saved frame.")
    parser.add_argument("--return_frame_rotate_with_slam_yaw",
                        action="store_true", default=False,
                        help="Rotate a loaded/startup return frame by live SLAM "
                             "yaw. Off by default because the current LIO yaw has "
                             "shown large drift while the robot is stationary.")
    parser.add_argument("--return_yaw_source",
                        choices=("fused_lowstate", "slam", "lowstate"),
                        default="fused_lowstate",
                        help="Yaw source for geo world->SDK return conversion. "
                             "fused_lowstate keeps SLAM yaw at the dodge origin "
                             "and applies lowstate's relative yaw change.")
    parser.add_argument("--return_max_slam_yaw_drift", type=float, default=0.45,
                        help="With --return_yaw_source slam, abort RETURN if live "
                             "SLAM yaw drifts this many radians from the dodge "
                             "origin yaw. With fused_lowstate, this is only a "
                             "SLAM/lowstate yaw-mismatch warning threshold. "
                             "Set <=0 to disable.")
    parser.add_argument("--startup_frame_calib", action="store_true", default=False,
                        help="Before enabling YOLO dodge, actively calibrate the "
                             "SDK SetVelocity frame against SLAM by walking one "
                             "forward axis and one lateral axis.")
    parser.add_argument("--startup_calib_distance", type=float, default=0.50,
                        help="Target SLAM displacement per startup calibration axis.")
    parser.add_argument("--startup_calib_speed", type=float, default=0.30,
                        help="SDK speed used during startup frame calibration.")
    parser.add_argument("--startup_calib_y_sign", type=float, default=-1.0,
                        help="Lateral command sign for the second startup "
                             "calibration move. -1 usually moves robot-right; "
                             "the result is converted back to the +y command axis.")
    parser.add_argument("--startup_calib_axis_timeout", type=float, default=10.0,
                        help="Max seconds allowed for each startup calibration axis.")
    parser.add_argument("--startup_calib_cmd_duration", type=float, default=1.0,
                        help="SetVelocity duration used during startup "
                             "calibration. The Unitree examples issue roughly "
                             "1-second high-level move commands.")
    parser.add_argument("--startup_calib_cmd_period", type=float, default=0.25,
                        help="Refresh period for startup calibration velocity "
                             "commands. Lower is not necessarily better for "
                             "high-level SDK RPCs.")
    parser.add_argument("--startup_calib_stationary_time", type=float, default=1.0,
                        help="Stationary SLAM window before/between startup "
                             "calibration moves.")
    parser.add_argument("--startup_calib_stationary_disp", type=float, default=0.05,
                        help="Maximum SLAM displacement over the startup "
                             "stationary window.")
    parser.add_argument("--startup_calib_stationary_timeout", type=float, default=5.0,
                        help="Max seconds to wait for stationary SLAM before an "
                             "axis move.")
    parser.add_argument("--startup_calib_min_delta", type=float, default=0.35,
                        help="Minimum measured SLAM displacement accepted for "
                             "each startup calibration axis.")
    parser.add_argument("--startup_calib_max_axis_cos", type=float, default=0.75,
                        help="Reject startup calibration if the two measured axes "
                             "are too collinear.")
    parser.add_argument("--startup_calib_max_yaw_change_deg", type=float, default=20.0,
                        help="Warn if SLAM yaw changes this much during the two "
                             "moves. This is fatal only when "
                             "--return_frame_rotate_with_slam_yaw is enabled.")
    parser.add_argument("--startup_calib_zero_repeats", type=int, default=12,
                        help="Zero SetVelocity repeats before/between startup "
                             "calibration moves.")
    parser.add_argument("--startup_calib_out",
                        default=str(_REPO / "configs" / "startup_return_frame_calib.json"),
                        help="Where to save startup calibration JSON. Empty disables saving.")
    parser.add_argument("--return_lateral_sign", type=float, default=-1.0,
                        help="Sign applied to external-odom return lateral command after "
                             "rotating odom displacement into body frame.")
    parser.add_argument("--return_low_yaw_sign", type=float, default=-1.0,
                        help="Sign applied to the lowstate yaw delta before fusing with "
                             "the SLAM origin yaw. -1 matches the FAST-LIO map frame on "
                             "this robot; use +1 if lowstate and SLAM yaw rotate the same "
                             "way.")
    parser.add_argument("--return_max_ang_vel", type=float, default=0.3,
                        help="Yaw-rate cap rad/s for geo-return heading correction "
                             "(restores the dodge-start heading). 0 disables turning "
                             "during return. Independent of --max_ang_vel (dodge).")
    parser.add_argument("--return_yaw_gain", type=float, default=-1.5,
                        help="P gain on heading error for geo-return turning. Default "
                             "-1.5 matches this robot's SetVelocity-omega vs SLAM-yaw "
                             "sign (validated on hardware); flip to positive only if a "
                             "future setup turns the wrong way.")
    parser.add_argument("--return_max_lat_vel", type=float, default=0.08,
                        help="Per-axis cap for lateral return velocity. Lower values "
                             "avoid sideways overshoot during recovery.")
    parser.add_argument("--return_odom_source", choices=["cmd", "external", "auto"],
                        default="cmd",
                        help="'cmd' integrates sent SDK velocity. 'external' uses DDS odom "
                             "from --odom_topic and fails closed if unavailable. 'auto' uses "
                             "external only if it is fresh when dodge is enabled.")
    parser.add_argument("--odom_topic", default="rt/dodge/odom",
                        help="DDS std_msgs/String JSON odometry topic.")
    parser.add_argument("--odom_staleness", type=float, default=0.35,
                        help="Maximum DDS odom age before return is paused.")
    parser.add_argument("--odom_startup_timeout", type=float, default=8.0,
                        help="Seconds to wait for sustained fresh external odom before "
                             "locomotion is allowed to start.")
    parser.add_argument("--odom_startup_min_count", type=int, default=5,
                        help="Minimum fresh odom messages required at startup before "
                             "locomotion is allowed to start.")
    parser.add_argument("--odom_startup_stable_s", type=float, default=1.0,
                        help="Minimum continuous fresh-odom span required at startup "
                             "before locomotion is allowed to start.")
    parser.add_argument("--return_odom_scale", type=float, default=1.0,
                        help="Scale factor for integrating sent SDK velocity as rough odometry.")
    parser.add_argument("--return_map_window", type=int, default=8,
                        help="Recent SDK-command/odom-delta samples used only for online "
                             "frame diagnostics and non-geo return modes.")
    parser.add_argument("--return_map_min_samples", type=int, default=3,
                        help="Minimum online frame samples for diagnostics/non-geo frame use.")
    parser.add_argument("--no_return_require_online_frame",
                        dest="return_require_online_frame",
                        action="store_false",
                        default=True,
                        help="Allow non-geo learned-frame RETURN to fall back to yaw if the "
                             "online SDK-command/SLAM frame is unavailable. Geo mode always "
                             "uses LIO yaw directly.")
    parser.add_argument("--no_return_freeze_frame", dest="return_freeze_frame",
                        action="store_false", default=True,
                        help="Do not freeze the learned command/odom frame during RETURN.")
    parser.add_argument("--no_return_head_guard", dest="return_head_guard",
                        action="store_false", default=True,
                        help="Do not enforce forward progress and lateral cap on return head output.")
    parser.add_argument("--dodge_ewma_alpha", type=float, default=0.2)
    parser.add_argument("--yolo_staleness", type=float, default=0.5)
    parser.add_argument("--yolo_hold_timeout", type=float, default=1.0)
    parser.add_argument("--yolo_no_kalman", action="store_true")
    parser.add_argument("--yolo_kf_meas_std", type=float, default=0.10)
    parser.add_argument("--yolo_kf_meas_std_close", type=float, default=0.30)
    parser.add_argument("--yolo_kf_gate", type=float, default=4.0)
    parser.add_argument("--commit_project", dest="commit_project",
                        action="store_true", default=True,
                        help="Input alignment: once a person is locked as approaching, "
                             "drive the dodge policy with a SMOOTH constant-velocity "
                             "projection (seeded from the clean approach) instead of the "
                             "jumpy close-range YOLO — so real input looks like the fake. "
                             "Releases if a fresh detection contradicts it or it times out.")
    parser.add_argument("--no_commit_project", dest="commit_project",
                        action="store_false",
                        help="Disable commit-and-project; feed raw filtered YOLO to the dodge.")
    parser.add_argument("--commit_dist", type=float, default=1.3,
                        help="Lock the approaching person for projection when its filtered "
                             "distance drops below this (m); set just outside --safety_dist.")
    parser.add_argument("--commit_max_time", type=float, default=3.0,
                        help="Max seconds to run the smooth projection before releasing.")
    parser.add_argument("--commit_break_dist", type=float, default=0.6,
                        help="Release the projection if a fresh YOLO detection is this far "
                             "from the projected position (m) — guards against the person "
                             "deviating from the projected straight-line path.")
    parser.add_argument("--lock_first_yolo_track", action="store_true", default=True,
                        help="Lock to the YOLO/ByteTrack track_id that starts the first "
                             "DODGE. Later track_id changes are ignored.")
    parser.add_argument("--no_lock_first_yolo_track", dest="lock_first_yolo_track",
                        action="store_false",
                        help="Allow YOLO target switching to the currently published person.")
    parser.add_argument("--require_yolo_track_to_dodge",
                        action="store_true", default=True,
                        help="Require a valid non-held YOLO track_id before starting DODGE.")
    parser.add_argument("--no_require_yolo_track_to_dodge",
                        dest="require_yolo_track_to_dodge",
                        action="store_false")
    parser.add_argument("--dodge_start_confirm_frames", type=int, default=2,
                        help="Consecutive valid YOLO frames required before DODGE starts.")
    parser.add_argument("--debug_obs", action="store_true")
    parser.add_argument("--debug_every", type=int, default=5)
    parser.add_argument("--l2b_damp_hold", type=float, default=2.0,
                        help="Software damping shortcut: while this script is running, "
                             "holding remote L2+B for this many seconds sends zero "
                             "velocity then LocoClient.Damp() and exits. Pass 0 to disable.")
    args = parser.parse_args()

    args.cmd_hz = max(1.0, float(args.cmd_hz))
    if args.close_escape_speed is None:
        args.close_escape_speed = float(args.max_vel)
    else:
        args.close_escape_speed = max(0.0, float(args.close_escape_speed))
    manual_return_frame_rot = None
    manual_return_frame_yaw_ref = None
    manual_return_frame_meta = None
    if args.return_frame_calib:
        try:
            (manual_return_frame_rot,
             manual_return_frame_yaw_ref,
             manual_return_frame_meta) = _load_return_frame_calib(args.return_frame_calib)
        except Exception as exc:
            raise SystemExit(f"[RETURN FRAME] failed to load "
                             f"{args.return_frame_calib}: {exc}")
        if not args.return_frame_rotate_with_slam_yaw:
            manual_return_frame_yaw_ref = None
        if args.return_mode == "geo":
            args.return_probe = False

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
    odom = (ExternalOdomMonitor(args.odom_topic)
            if args.return_odom_source in ("external", "auto") else None)

    loco = LocoClient()
    loco.SetTimeout(2.0)
    loco.Init()
    print(f"[loco] initial {_loco_status(loco)}")

    dodge = DodgePolicy(str(_REPO / "checkpoints" / "dodge_v23b_54400.pt"))
    return_head = None
    if args.return_enable and args.return_mode == "head":
        print("[WARN] --return_mode head is for replay/debug comparison; "
              "use --return_mode geo for real SDK deploy unless you are "
              "intentionally testing the checkpoint return head.")
        return_head = _load_return_head(args.return_head_ckpt)
    gated_return_head = None
    if args.return_enable and args.return_mode == "gated":
        gated_return_head = _load_gated_return_head(args.gated_ckpt)
        print("[RETURN] gated mode: learned return_head drives return (dodge "
              "unchanged). Deterministic state machine gates dodge vs return; "
              "the checkpoint's learned gate is not used.")
    if args.return_enable and args.return_mode == "geo":
        print("[RETURN] geo mode uses SLAM displacement to return to the dodge "
              "origin; SDK velocity frame is measured with a short SLAM probe "
              "when enabled, or loaded from manual calibration.")
    detector = YoloDdsObstacleDetector(
        staleness_threshold=args.yolo_staleness,
        hold_timeout=args.yolo_hold_timeout,
        use_kalman=(not args.yolo_no_kalman),
        kf_meas_std=args.yolo_kf_meas_std,
        kf_meas_std_close=args.yolo_kf_meas_std_close,
        kf_gate_sigma=args.yolo_kf_gate,
        commit_enable=args.commit_project,
        commit_dist=args.commit_dist,
        commit_max_time=args.commit_max_time,
        commit_break_dist=args.commit_break_dist,
    )

    robot_pos = np.array([0.0, 0.0, 0.8], dtype=np.float32)
    robot_xy_est = np.zeros(2, dtype=np.float32)
    cmd = np.zeros(3, dtype=np.float32)
    target = np.zeros(3, dtype=np.float32)
    ewma = np.zeros(3, dtype=np.float32)
    active = False
    return_active = False
    return_start_t = 0.0
    return_last_progress_t = 0.0
    return_best_disp = float("inf")
    return_frame_src = "none"
    return_frame_rot = None
    return_frame_yaw_ref = None
    return_probe_phase = None
    return_probe_attempt = 0
    return_probe_stage_start_t = 0.0
    return_probe_stage_start_xy = None
    return_probe_dx_w = None
    return_odom_stale_since = None
    return_bad_progress_count = 0
    return_flip_sign = 1.0
    return_frame_flips = 0
    dodge_start_valid_count = 0
    last_dodge_gate_warn = 0.0
    return_prev_xy = None
    return_actual_delta_w = None
    return_actual_prog_dot = None
    return_actual_prog_cos = None
    stationary_samples = deque()
    stationary_last_xy = None
    stationary_last_t = 0.0
    stationary_since = None
    stationary_speed = float("inf")
    stationary_window_span = 0.0
    stationary_window_disp = float("inf")
    stationary_avg_speed = float("inf")
    stationary_ready = False
    return_prestop_sent = False
    last_stationary_wait_warn = 0.0
    last_probe_wait_warn = 0.0
    return_disp_b = None
    return_raw = None
    return_head_cmd = None
    return_guarded = False
    dodge_start_yaw = 0.0
    dodge_start_t = 0.0
    dodge_lat_sign = 0.0
    dodge_commit_b = np.zeros(2, dtype=np.float32)
    dodge_capped = False
    last_commit_warn = 0.0
    clear_since = None
    odom_origin_xy = None
    odom_origin_yaw = None
    odom_origin_low_yaw = None
    current_odom_xy = None
    current_odom_yaw = None
    current_return_yaw = None
    current_return_yaw_src = "none"
    current_slam_yaw_delta = None
    current_low_yaw_delta = None
    current_slam_low_yaw_mismatch = None
    current_odom_msg_count = None
    return_odom_mode = "cmd"
    external_odom_fresh = False
    external_odom_age = float("inf")
    last_odom_warn = 0.0
    last_return_block_warn = 0.0
    last_yaw_mismatch_warn = 0.0
    target_track_id = None
    odom_frame = OnlineCommandOdomFrame(
        window=args.return_map_window,
        min_samples=args.return_map_min_samples,
    )
    enabled = not args.wait_for_a
    started = False
    loco_stop_needed = False
    damping_exit = False
    counter = 0
    sdk_error_count = 0
    l2b_since = None
    idle_stop_sent = False
    max_dlin = args.max_accel_lin / args.cmd_hz
    max_dang = args.max_accel_ang / args.cmd_hz

    print("\n[SDK LOCO DODGE]")
    print("  backend=Unitree LocoClient.SetVelocity, no rt/lowcmd, no motion.pt")
    print(f"  max_vel={args.max_vel:.2f}m/s max_ang={args.max_ang_vel:.2f}rad/s "
          f"safety={args.safety_dist:.2f}m clear_margin={args.clear_margin:.2f}m "
          f"cmd_hz={args.cmd_hz:.1f}")
    print(f"  loco state: start={'off' if args.no_loco_start else 'fsm=200'} "
          f"balance={'keep' if args.balance_mode < 0 else args.balance_mode} "
          f"require_ready={args.require_loco_ready}")
    if args.hold_zero_ready:
        idle_mode = "stream SetVelocity(0,0,0)"
    elif args.idle_stopmove:
        idle_mode = "StopMove once, then no SetVelocity stream"
    else:
        idle_mode = "no SetVelocity stream"
    print(f"  idle: {idle_mode}")
    print(f"  close_escape min_away={args.close_escape_speed:.2f}m/s "
          f"(capped by per-axis max_vel and bearing)")
    print(f"  return: {'on' if args.return_enable else 'off'} mode={args.return_mode} "
          f"max_vel={args.return_max_vel:.2f}m/s done={args.return_done_dist:.2f}m "
          f"delay={args.return_clear_delay:.1f}s timeout={args.return_timeout:.1f}s "
          f"no_progress={args.return_no_progress_timeout:.1f}s "
          f"online_map={args.return_map_min_samples}/{args.return_map_window}")
    settle_abort_s = (
        "never" if args.return_settle_max_wait <= 0.0
        else f"{args.return_settle_max_wait:.1f}s"
    )
    print(f"  return settle: window={args.return_stationary_time:.1f}s "
          f"disp<={args.return_stationary_disp:.2f}m and "
          f"avg_speed<={args.return_stationary_speed:.2f}m/s; "
          f"abort_after={settle_abort_s}")
    print("  return prestop: "
          f"{'zero velocity + StopMove before settle/probe' if args.return_prestop else 'off'}")
    print("  return abort: "
          f"{'exit + StopMove' if args.exit_on_return_abort else 'READY + zero velocity'}")
    yaw_abort_s = (
        "off" if args.return_max_slam_yaw_drift <= 0.0
        else f"{args.return_max_slam_yaw_drift:.2f}rad"
    )
    print("  return frame yaw: "
          f"source={args.return_yaw_source} "
          f"calib={'rotate' if args.return_frame_rotate_with_slam_yaw else 'fixed'}; "
          f"slam_yaw_check={yaw_abort_s}")
    if args.return_mode == "geo":
        print("  return geo: direct point-to-origin control from SLAM x/y + fused body yaw; "
              "online command/SLAM frame is diagnostic only; "
              f"sdk_probe={'on' if args.return_probe else 'off'} "
              f"probe_min={args.return_probe_min_delta:.2f}m "
              f"probe_axis_cos<={args.return_probe_max_axis_cos:.2f} "
              f"probe_retry={args.return_probe_retry_count}x"
              f"@{args.return_probe_retry_scale:.1f}")
        print(f"  return geo lateral: sdk_y_sign={args.return_lateral_sign:+.0f} "
              f"lat_gain={args.return_lat_gain:.2f} "
              f"lat_cap={args.return_max_lat_vel:.2f}m/s")
        if manual_return_frame_rot is not None:
            yaw_ref_s = ("None" if manual_return_frame_yaw_ref is None
                         else f"{manual_return_frame_yaw_ref:+.2f}")
            print("  return manual frame: "
                  f"{args.return_frame_calib} "
                  f"rot={_fmt_mat2(manual_return_frame_rot, 3)} "
                  f"yaw_ref={yaw_ref_s}")
        print("  startup frame calib: "
              f"{'on' if args.startup_frame_calib else 'off'} "
              f"dist={args.startup_calib_distance:.2f}m "
              f"speed={args.startup_calib_speed:.2f}m/s "
              f"y_sign={args.startup_calib_y_sign:+.0f}")
    if args.return_mode == "head":
        print(f"  return head guard={'on' if args.return_head_guard else 'off'} "
              f"freeze_frame={'on' if args.return_freeze_frame else 'off'} "
              f"lat_cap={args.return_max_lat_vel:.2f}m/s")
    if args.return_mode == "gated":
        print(f"  return gated: ckpt={Path(args.gated_ckpt).name} "
              f"lin_vel={args.return_gated_lin_vel:.2f}m/s "
              f"yaw_clamp=0.50rad/s xy_amp=2x (gate=state-machine, dodge unchanged)")
    if args.return_mode == "p":
        print(f"  return P-controller: lat_sign={args.return_lateral_sign:+.0f} "
              f"lat_gain={args.return_lat_gain:.2f} "
              f"lat_cap={args.return_max_lat_vel:.2f}m/s")
    print(f"  return odom={args.return_odom_source} "
          f"topic={args.odom_topic if odom is not None else 'N/A'} "
          f"stale>{args.odom_staleness:.2f}s "
          f"startup={args.odom_startup_min_count}msgs/"
          f"{args.odom_startup_stable_s:.1f}s before loco")
    print(f"  yolo target lock={'first-dodge-track' if args.lock_first_yolo_track else 'off'}")
    print(f"  yolo start gate: "
          f"{'valid non-held track_id required' if args.require_yolo_track_to_dodge else 'off'} "
          f"confirm={args.dodge_start_confirm_frames} frames")
    if args.wait_for_a:
        print("  Press A to enable. Press SELECT to send Damp() and exit.")
    else:
        print("  Enabled immediately. Press SELECT to send Damp() and exit.")
    if args.l2b_damp_hold > 0:
        print(f"  Software emergency: hold L2+B for {args.l2b_damp_hold:.1f}s "
              "to send Damp() and exit.\n")
    else:
        print()

    period = 1.0 / args.cmd_hz

    def send_loco_stopmove(reason: str):
        nonlocal idle_stop_sent
        cmd[:] = 0.0
        target[:] = 0.0
        repeats = max(1, int(args.return_prestop_repeats))
        print(f"[STOPMOVE] {reason}: zero velocity x{repeats} + StopMove")
        for _ in range(repeats):
            try:
                loco.SetVelocity(0.0, 0.0, 0.0, duration=0.05)
            except Exception as e:
                print(f"[WARN] stop zero failed: {e}")
            time.sleep(0.02)
        try:
            loco.StopMove()
        except Exception as e:
            print(f"[WARN] StopMove failed: {e}")
        idle_stop_sent = True

    def send_loco_damp(reason: str):
        nonlocal idle_stop_sent, loco_stop_needed, damping_exit
        cmd[:] = 0.0
        target[:] = 0.0
        repeats = max(10, int(args.return_prestop_repeats))
        print(f"[EMERGENCY] {reason}: zero velocity x{repeats} + Damp(SetFsmId(1))")
        for _ in range(repeats):
            try:
                loco.SetVelocity(0.0, 0.0, 0.0, duration=0.05)
            except Exception as e:
                print(f"[WARN] emergency zero failed: {e}")
            time.sleep(0.02)
        try:
            code = loco.SetFsmId(1)
            print(f"[EMERGENCY] Damp code={code}")
            if code == 0:
                damping_exit = True
                loco_stop_needed = False
            else:
                print(f"[WARN] Damp returned non-zero code={code}; final StopMove will run")
        except Exception as e:
            print(f"[WARN] emergency Damp failed: {e}")
        idle_stop_sent = True

    def clear_episode_target_lock(reason: str):
        nonlocal target_track_id
        if target_track_id is None:
            return
        if hasattr(detector, "clear_track_lock"):
            detector.clear_track_lock()
        print(f"[TARGET LOCK] cleared track_id={target_track_id} ({reason})")
        target_track_id = None

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
                        send_loco_damp("L2+B hold reached")
                        break
                else:
                    l2b_since = None
            if low.remote.button[KeyMap.select] == 1:
                send_loco_damp("SELECT pressed")
                break
            if not enabled and args.wait_for_a and low.remote.button[KeyMap.A] == 1:
                enabled = True
            if enabled and not started:
                dodge.reset(robot_pos[:2], low.yaw)
                if args.return_odom_source in ("external", "auto"):
                    snap = odom.wait_stable(
                        args.odom_startup_timeout,
                        args.odom_staleness,
                        args.odom_startup_min_count,
                        args.odom_startup_stable_s)
                    if snap is None and args.return_odom_source == "external":
                        raise RuntimeError(
                            f"external odometry requested but no sustained fresh "
                            f"{args.odom_topic} messages arrived before locomotion "
                            f"startup ({args.odom_startup_min_count} msgs over "
                            f"{args.odom_startup_stable_s:.1f}s within "
                            f"{args.odom_startup_timeout:.1f}s). "
                            "Not starting loco; fix LiDAR odometry bridge first."
                        )
                    if snap is not None:
                        odom_origin_xy = np.array([snap["x"], snap["y"]], dtype=np.float32)
                        odom_origin_yaw = float(snap["yaw"]) if "yaw" in snap else None
                        odom_origin_low_yaw = float(low.yaw)
                        current_odom_xy = odom_origin_xy.copy()
                        current_odom_yaw = odom_origin_yaw
                        current_odom_msg_count = int(snap.get("msg_count", 0))
                        odom_frame.reset(current_odom_xy, current_odom_msg_count, cmd[:2])
                        robot_xy_est[:] = 0.0
                        robot_pos[:2] = 0.0
                        stationary_samples.clear()
                        stationary_last_xy = robot_xy_est.copy()
                        stationary_last_t = time.time()
                        stationary_since = None
                        stationary_ready = False
                        return_odom_mode = "external"
                        print(f"[ODOM] sustained external origin set before loco start at "
                              f"{_fmt_vec(odom_origin_xy, 2)} "
                              f"age={snap.get('age', 0.0):.2f}s "
                              f"msgs={snap.get('msg_count', 0)}")
                    else:
                        return_odom_mode = "cmd"
                        print("[ODOM] no sustained external odom at enable; using cmd integration")
                else:
                    return_odom_mode = "cmd"

                if not args.no_loco_start:
                    print("[loco] SetFsmId(200)  # Start")
                    code = loco.SetFsmId(200)
                    loco_stop_needed = True
                    print(f"[loco] SetFsmId(200) returned code={code}")
                    time.sleep(max(0.0, args.loco_start_wait))
                    print(f"[loco] after Start {_loco_status(loco)}")
                if args.balance_mode >= 0:
                    print(f"[loco] SetBalanceMode({args.balance_mode})")
                    code = loco.SetBalanceMode(int(args.balance_mode))
                    loco_stop_needed = True
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
                if args.startup_frame_calib:
                    if return_odom_mode != "external":
                        raise RuntimeError(
                            "startup frame calibration requires fresh external SLAM odom")
                    (manual_return_frame_rot,
                     manual_return_frame_yaw_ref,
                     manual_return_frame_meta) = _run_startup_frame_calib(
                        loco, odom, args)
                    args.return_probe = False
                    snap = odom.wait_stable(
                        3.0,
                        args.odom_staleness,
                        args.odom_startup_min_count,
                        args.odom_startup_stable_s)
                    if snap is None:
                        raise RuntimeError(
                            "no stable odom after startup frame calibration")
                    odom_origin_xy = np.array([snap["x"], snap["y"]], dtype=np.float32)
                    odom_origin_yaw = float(snap["yaw"]) if "yaw" in snap else None
                    odom_origin_low_yaw = float(low.yaw)
                    current_odom_xy = odom_origin_xy.copy()
                    current_odom_yaw = odom_origin_yaw
                    current_odom_msg_count = int(snap.get("msg_count", 0))
                    odom_frame.reset(current_odom_xy, current_odom_msg_count, cmd[:2])
                    robot_xy_est[:] = 0.0
                    robot_pos[:2] = 0.0
                    stationary_samples.clear()
                    stationary_last_xy = robot_xy_est.copy()
                    stationary_last_t = time.time()
                    stationary_since = None
                    stationary_ready = False
                    print("[STARTUP CALIB] odom origin reset after calibration at "
                          f"{_fmt_vec(odom_origin_xy, 2)} "
                          f"yaw={0.0 if odom_origin_yaw is None else odom_origin_yaw:+.2f}")
                ewma[:] = 0.0
                started = True
                loco_stop_needed = True
                print("[ENABLE] SDK loco dodge commands enabled")

            yaw = low.yaw
            if return_odom_mode == "external":
                snap = odom.snapshot(args.odom_staleness)
                external_odom_fresh = bool(snap.get("fresh"))
                external_odom_age = float(snap.get("age", float("inf")))
                if external_odom_fresh:
                    current_odom_xy = np.array([snap["x"], snap["y"]],
                                               dtype=np.float32)
                    current_odom_yaw = float(snap["yaw"]) if "yaw" in snap else None
                    current_odom_msg_count = int(snap.get("msg_count", 0))
                    if odom_origin_xy is None:
                        odom_origin_xy = current_odom_xy.copy()
                        odom_origin_yaw = current_odom_yaw
                        odom_origin_low_yaw = float(yaw)
                        print(f"[ODOM] external origin set at "
                              f"{_fmt_vec(odom_origin_xy, 2)}")
                    robot_xy_est[:] = current_odom_xy - odom_origin_xy
                    robot_pos[:2] = robot_xy_est
                    if enabled and started and not return_active:
                        odom_frame.update(current_odom_xy, current_odom_msg_count, cmd[:2])
            else:
                external_odom_fresh = False
                external_odom_age = float("inf")
                robot_pos[:2] = robot_xy_est

            if return_odom_mode == "external":
                (current_return_yaw,
                 current_return_yaw_src,
                 current_slam_yaw_delta,
                 current_low_yaw_delta,
                 current_slam_low_yaw_mismatch) = _fused_return_yaw(
                    odom_origin_yaw,
                    odom_origin_low_yaw,
                    float(yaw),
                    current_odom_yaw,
                    args.return_yaw_source,
                    low_yaw_sign=args.return_low_yaw_sign,
                )
            else:
                current_return_yaw = float(yaw)
                current_return_yaw_src = "lowstate_abs_yaw"
                current_slam_yaw_delta = None
                current_low_yaw_delta = None
                current_slam_low_yaw_mismatch = None

            now_still = time.time()
            stationary_ready = False
            stationary_window_span = 0.0
            stationary_window_disp = float("inf")
            stationary_avg_speed = float("inf")
            if return_odom_mode == "external" and external_odom_fresh:
                if stationary_last_xy is not None:
                    dt_still = max(1e-3, now_still - stationary_last_t)
                    dxy_still = robot_xy_est - stationary_last_xy
                    stationary_speed = float(np.linalg.norm(dxy_still) / dt_still)
                stationary_samples.append((now_still, robot_xy_est.copy()))
                keep_s = max(
                    1.0,
                    float(args.return_stationary_time)
                    + max(0.0, float(args.return_settle_max_wait))
                    + 1.0,
                    float(args.return_probe_settle)
                    + max(0.0, float(args.return_settle_max_wait))
                    + 1.0,
                )
                while (stationary_samples
                       and now_still - stationary_samples[0][0] > keep_s):
                    stationary_samples.popleft()
                if args.return_stationary_time <= 0.0:
                    stationary_ready = True
                    stationary_window_span = 0.0
                    stationary_window_disp = 0.0
                    stationary_avg_speed = 0.0
                    if stationary_since is None:
                        stationary_since = now_still
                else:
                    cutoff_t = now_still - float(args.return_stationary_time)
                    start_sample = None
                    for sample_t, sample_xy in stationary_samples:
                        if sample_t <= cutoff_t:
                            start_sample = (sample_t, sample_xy)
                        else:
                            break
                    if start_sample is not None:
                        stationary_window_span = max(1e-3, now_still - start_sample[0])
                        stationary_window_disp = float(np.linalg.norm(
                            robot_xy_est - start_sample[1]))
                        stationary_avg_speed = (
                            stationary_window_disp / stationary_window_span)
                        stationary_ready = (
                            stationary_window_disp <= args.return_stationary_disp
                            and stationary_avg_speed <= args.return_stationary_speed
                        )
                    if stationary_ready:
                        if stationary_since is None:
                            stationary_since = now_still - stationary_window_span
                    else:
                        stationary_since = None
                stationary_last_xy = robot_xy_est.copy()
                stationary_last_t = now_still
            else:
                stationary_samples.clear()
                stationary_since = None
                stationary_speed = float("inf")
                stationary_avg_speed = float("inf")
                stationary_window_disp = float("inf")
                stationary_window_span = 0.0

            obstacle_pos = detector.detect(robot_pos, yaw)
            dist = (float(np.linalg.norm(obstacle_pos[:2] - robot_pos[:2]))
                    if obstacle_pos is not None else float("inf"))
            ydbg = detector.debug_snapshot
            obstacle_vel_w = None
            if ydbg.get("kf_vxy") is not None:
                obstacle_vel_w = np.asarray(ydbg["kf_vxy"], dtype=np.float32)

            yolo_track_id = -1
            try:
                yolo_track_id = int(ydbg.get("track_id", -1))
            except (TypeError, ValueError):
                yolo_track_id = -1
            yolo_status = str(ydbg.get("status", ""))
            yolo_track_ok = (
                yolo_track_id >= 0
                and yolo_status.startswith("fresh")
                and int(ydbg.get("raw_n", 0) or 0) >= 1
            )
            dodge_trigger_valid = True
            if args.require_yolo_track_to_dodge:
                dodge_trigger_valid = yolo_track_ok
            if (enabled and not active
                    and obstacle_pos is not None and dist < args.safety_dist):
                if dodge_trigger_valid:
                    dodge_start_valid_count += 1
                else:
                    dodge_start_valid_count = 0
                    now_gate = time.time()
                    if now_gate - last_dodge_gate_warn > 1.0:
                        print("[DODGE GATE] ignoring inside-safety YOLO target "
                              f"dist={dist:.2f} status={yolo_status} "
                              f"track={yolo_track_id} raw_n={ydbg.get('raw_n', '?')}")
                        last_dodge_gate_warn = now_gate
            elif not active:
                dodge_start_valid_count = 0

            if (enabled and obstacle_pos is not None and dist < args.safety_dist
                    and (active or not args.require_yolo_track_to_dodge
                         or dodge_start_valid_count
                         >= max(1, int(args.dodge_start_confirm_frames)))):
                if not active:
                    was_returning = return_active
                    if return_active:
                        print(f"[RETURN INTERRUPT] obstacle dist={dist:.2f}m")
                    return_active = False
                    return_frame_rot = None
                    return_frame_yaw_ref = None
                    return_probe_phase = None
                    return_probe_attempt = 0
                    return_probe_stage_start_xy = None
                    return_probe_dx_w = None
                    return_odom_stale_since = None
                    return_prev_xy = None
                    return_last_progress_t = 0.0
                    active = True
                    dodge_start_valid_count = 0
                    clear_since = None
                    if (return_odom_mode == "external" and external_odom_fresh
                            and current_odom_xy is not None and not was_returning):
                        # Return should go back to the pose at dodge trigger, not the
                        # pose where the script happened to be enabled. Keep the old
                        # origin only when a return is interrupted by a new obstacle.
                        odom_origin_xy = current_odom_xy.copy()
                        odom_origin_yaw = current_odom_yaw
                        odom_origin_low_yaw = float(yaw)
                        robot_xy_est[:] = 0.0
                        robot_pos[:2] = 0.0
                        odom_frame.reset(current_odom_xy, current_odom_msg_count, cmd[:2])
                        stationary_samples.clear()
                        stationary_last_xy = robot_xy_est.copy()
                        stationary_last_t = time.time()
                        stationary_since = None
                        stationary_ready = False
                        print(f"[ODOM] external dodge origin reset at "
                              f"{_fmt_vec(odom_origin_xy, 2)} "
                              f"yaw={0.0 if odom_origin_yaw is None else odom_origin_yaw:+.2f}")
                    dodge_start_yaw = (
                        current_return_yaw if current_return_yaw is not None else yaw)
                    dodge_start_t = time.time()
                    dodge_lat_sign = 0.0
                    dodge_commit_b[:] = 0.0
                    dodge_capped = False
                    dodge.reset(robot_pos[:2], yaw)
                    if args.lock_first_yolo_track and target_track_id is None:
                        try:
                            tid_raw = ydbg.get("locked_track_id")
                            if tid_raw is None:
                                tid_raw = ydbg.get("track_id", -1)
                            tid = int(tid_raw)
                        except (TypeError, ValueError):
                            tid = -1
                        if tid >= 0 and hasattr(detector, "set_track_lock"):
                            detector.set_track_lock(tid)
                            target_track_id = tid
                            print(f"[TARGET LOCK] using first dodge YOLO track_id={tid}; "
                                  "later tracks ignored")
                        else:
                                print("[TARGET LOCK] no valid YOLO track_id on first dodge; "
                                      "cannot lock target")
                    ewma[:] = 0.0
                    return_prestop_sent = False
                    print(f"[DODGE START] dist={dist:.2f}m")

            safe_dbg = {"changed": False}
            return_disp_b = None
            return_raw = None
            return_head_cmd = None
            return_guarded = False
            return_actual_delta_w = None
            return_actual_prog_dot = None
            return_actual_prog_cos = None
            if not enabled:
                target[:] = 0.0
                active = False
                return_active = False
                return_frame_rot = None
                return_frame_yaw_ref = None
                return_probe_phase = None
                return_probe_stage_start_xy = None
                return_probe_dx_w = None
                return_odom_stale_since = None
                return_prev_xy = None
                return_last_progress_t = 0.0
                clear_episode_target_lock("disabled")
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
                # Force the dodge into the REAR sector (straight-back / diagonal-back):
                # clamp its direction to within +/- dodge_rear_angle of straight-back, keeping
                # the speed. This makes the recover a pure FORWARD walk -- the most accurate
                # direction for the G1, and robust to lateral return-frame degeneracy.
                if args.dodge_rear_bias:
                    _spd = float(np.linalg.norm(target[:2]))
                    if _spd > 1e-3:
                        _th = float(np.arctan2(float(target[1]), float(target[0])))
                        _dev = (_th % (2.0 * np.pi)) - np.pi   # deviation from straight-back (pi)
                        _A = float(np.radians(args.dodge_rear_angle))
                        _dev = float(np.clip(_dev, -_A, _A))
                        target[0] = -_spd * float(np.cos(_dev))
                        target[1] = -_spd * float(np.sin(_dev))
                # Latch the lateral escape side: once the dodge commits to a side, a
                # crossing obstacle can't make the lateral command reverse (the wag
                # that cancels displacement and ruins the online return-frame fit).
                if args.dodge_dir_latch:
                    _lat = float(target[1])
                    if dodge_lat_sign == 0.0:
                        if abs(_lat) >= args.dodge_latch_deadband:
                            dodge_lat_sign = 1.0 if _lat > 0.0 else -1.0
                    elif _lat * dodge_lat_sign < 0.0:
                        target[1] = 0.0
                # Remember the committed body-frame escape direction (for min-commit).
                _tnorm = float(np.linalg.norm(target[:2]))
                if _tnorm > 1e-3:
                    dodge_commit_b[:] = np.asarray(target[:2], dtype=np.float32) / _tnorm
                # Cap: never let the dodge run past --max_dodge_dist from the origin.
                _disp_now = float(np.linalg.norm(robot_xy_est))
                if args.max_dodge_dist > 0.0 and _disp_now >= args.max_dodge_dist:
                    target[:] = 0.0
                    if not dodge_capped:
                        print(f"[DODGE CAP] est_disp={_disp_now:.2f}m >= "
                              f"{args.max_dodge_dist:.2f}m; holding (no further escape)")
                        dodge_capped = True
                clear_since = None
                return_prestop_sent = False
            elif (active and args.min_dodge_dist > 0.0
                  and float(np.linalg.norm(dodge_commit_b)) > 1e-3
                  and float(np.linalg.norm(robot_xy_est)) < args.min_dodge_dist
                  and (args.max_dodge_dist <= 0.0
                       or float(np.linalg.norm(robot_xy_est)) < args.max_dodge_dist)
                  and (time.time() - dodge_start_t) < args.min_dodge_time):
                # Obstacle cleared but the dodge is still too small for a reliable
                # return-frame fit: keep escaping in the latched direction until it
                # has committed at least --min_dodge_dist before allowing the return.
                target[0] = float(dodge_commit_b[0]) * float(args.close_escape_speed)
                target[1] = float(dodge_commit_b[1]) * float(args.close_escape_speed)
                target[2] = 0.0
                clear_since = None
                return_prestop_sent = False
                _now_c = time.time()
                if _now_c - last_commit_warn > 1.0:
                    print(f"[DODGE COMMIT] est_disp="
                          f"{float(np.linalg.norm(robot_xy_est)):.2f}m < "
                          f"{args.min_dodge_dist:.2f}m; continuing latched escape")
                    last_commit_warn = _now_c
            elif active:
                now = time.time()
                if clear_since is None:
                    clear_since = now
                    print(f"[DODGE CLEAR] dist={dist:.2f}m; waiting "
                          f"{args.return_clear_delay:.1f}s before return")
                    if args.return_prestop and not return_prestop_sent:
                        send_loco_stopmove("pre-return after dodge clear")
                        return_prestop_sent = True
                        now = time.time()
                        clear_since = now
                        stationary_samples.clear()
                        stationary_last_xy = robot_xy_est.copy()
                        stationary_last_t = now
                        stationary_since = None
                        stationary_ready = False
                target[:] = 0.0
                if now - clear_since >= args.return_clear_delay:
                    settle_elapsed = max(0.0, now - clear_since
                                         - args.return_clear_delay)
                    settle_timed_out = (
                        args.return_settle_max_wait > 0.0
                        and settle_elapsed >= args.return_settle_max_wait
                    )
                    if return_odom_mode == "external" and not external_odom_fresh:
                        if now - last_odom_warn > 1.0:
                            print(f"[RETURN WAIT] external odom stale "
                                  f"age={external_odom_age:.2f}s; holding zero")
                            last_odom_warn = now
                    elif (return_odom_mode == "external"
                          and args.return_stationary_time > 0.0
                          and not stationary_ready
                          and not settle_timed_out):
                        target[:] = 0.0
                        if now - last_stationary_wait_warn > 1.0:
                            still_for = (0.0 if stationary_since is None
                                         else now - stationary_since)
                            print("[RETURN WAIT] robot still moving after dodge "
                                  f"speed={stationary_speed:.3f}m/s "
                                  f"window={stationary_window_disp:.3f}m/"
                                  f"{stationary_window_span:.2f}s "
                                  f"avg={stationary_avg_speed:.3f}m/s "
                                  f"stable={still_for:.2f}/"
                                  f"{args.return_stationary_time:.2f}s; holding zero")
                            last_stationary_wait_warn = now
                    elif (return_odom_mode == "external"
                          and args.return_stationary_time > 0.0
                          and not stationary_ready
                          and settle_timed_out):
                        target[:] = 0.0
                        active = False
                        ewma[:] = 0.0
                        clear_since = None
                        return_active = False
                        return_frame_rot = None
                        return_frame_yaw_ref = None
                        return_probe_phase = None
                        return_probe_stage_start_xy = None
                        return_probe_dx_w = None
                        return_prev_xy = None
                        return_last_progress_t = 0.0
                        print("[RETURN ABORT] robot did not settle after dodge; "
                              "zeroing instead of starting return/probe "
                              f"window={stationary_window_disp:.3f}m/"
                              f"{stationary_window_span:.2f}s "
                              f"avg={stationary_avg_speed:.3f}m/s")
                        clear_episode_target_lock("return settle timeout")
                        if args.exit_on_return_abort:
                            print("[RETURN ABORT] exiting controller to StopMove")
                            break
                    else:
                        disp_mag = float(np.linalg.norm(robot_xy_est))
                        print(f"[DODGE STOP] dist={dist:.2f}m est_disp={disp_mag:.2f}m")
                        active = False
                        ewma[:] = 0.0
                        clear_since = None
                        geo_needs_online = (
                            args.return_mode not in ("geo", "gated")
                            and return_odom_mode == "external"
                            and args.return_require_online_frame)
                        online_ready = (odom_frame.rotation() is not None)
                        if (args.return_enable and disp_mag > args.return_done_dist
                                and geo_needs_online and not online_ready):
                            return_active = False
                            return_frame_rot = None
                            return_frame_yaw_ref = None
                            return_probe_phase = None
                            return_probe_stage_start_xy = None
                            return_probe_dx_w = None
                            return_odom_stale_since = None
                            return_prev_xy = None
                            return_last_progress_t = 0.0
                            target[:] = 0.0
                            print("[RETURN FRAME] no online SDK-command/SLAM frame; "
                                  "not starting learned-frame return")
                            clear_episode_target_lock("no online return frame")
                        elif args.return_enable and disp_mag > args.return_done_dist:
                            return_active = True
                            return_start_t = now
                            return_last_progress_t = now
                            return_best_disp = disp_mag
                            return_bad_progress_count = 0
                            return_flip_sign = 1.0
                            return_frame_flips = 0
                            if args.return_mode == "geo":
                                return_frame_rot = None
                                return_frame_yaw_ref = None
                                return_odom_stale_since = None
                                if manual_return_frame_rot is not None:
                                    return_frame_rot = manual_return_frame_rot.copy()
                                    return_frame_yaw_ref = manual_return_frame_yaw_ref
                                    return_probe_phase = None
                                    return_probe_attempt = 0
                                    return_probe_stage_start_xy = None
                                    return_probe_dx_w = None
                                    print("[RETURN FRAME] geo using manual body/SLAM "
                                          f"frame rot={_fmt_mat2(return_frame_rot, 3)} "
                                          f"yaw_ref={return_frame_yaw_ref}")
                                elif args.return_probe and return_odom_mode == "external":
                                    return_probe_phase = "settle_x"
                                    return_probe_attempt = 0
                                    return_probe_stage_start_t = now
                                    return_probe_stage_start_xy = robot_xy_est.copy()
                                    return_probe_dx_w = None
                                    print("[RETURN PROBE] measuring SDK frame with "
                                          f"+x {args.return_probe_speed:.2f}m/s and "
                                          f"+y {args.return_probe_lat_speed:.2f}m/s "
                                          f"for {args.return_probe_duration:.2f}s each "
                                          f"after {args.return_probe_settle:.2f}s settles")
                                else:
                                    return_probe_phase = None
                                    return_probe_attempt = 0
                                    return_probe_stage_start_xy = None
                                    return_probe_dx_w = None
                                    print("[RETURN FRAME] geo using SLAM x/y + "
                                          f"{args.return_yaw_source} yaw for SDK SetVelocity")
                            elif args.return_freeze_frame:
                                return_frame_rot = odom_frame.rotation()
                                if return_frame_rot is not None:
                                    return_frame_yaw_ref = (
                                        current_return_yaw if current_return_yaw is not None else yaw)
                                    diag = odom_frame.diagnostics()
                                    det = diag.get("det")
                                    rms = diag.get("fit_rms")
                                    sx = diag.get("scale_x")
                                    sy = diag.get("scale_y")
                                    det_s = "None" if det is None else f"{det:+.2f}"
                                    rms_s = "None" if rms is None else f"{rms:.3f}"
                                    sx_s = "None" if sx is None else f"{sx:.3f}"
                                    sy_s = "None" if sy is None else f"{sy:.3f}"
                                    print("[RETURN FRAME] frozen online command/odom frame "
                                          f"rot={_fmt_mat2(return_frame_rot, 2)} "
                                          f"yaw_ref={return_frame_yaw_ref:+.2f} "
                                          f"det={det_s} rms={rms_s} "
                                          f"scale=[{sx_s},{sy_s}]")
                                else:
                                    return_frame_yaw_ref = None
                                    if args.return_mode == "geo" and geo_needs_online:
                                        return_active = False
                                        return_frame_yaw_ref = None
                                        return_probe_phase = None
                                        return_probe_stage_start_xy = None
                                        return_probe_dx_w = None
                                        return_odom_stale_since = None
                                        return_prev_xy = None
                                        return_last_progress_t = 0.0
                                        target[:] = 0.0
                                        print("[RETURN FRAME] no online frame yet; "
                                              "not starting geo return")
                                        clear_episode_target_lock("no online return frame")
                                    else:
                                        print("[RETURN FRAME] no online frame yet; "
                                              "falling back to odom/lowstate yaw")
                            else:
                                return_frame_rot = None
                                return_frame_yaw_ref = None
                                print("[RETURN FRAME] learned frame not frozen; "
                                      "using live yaw/frame fallback")
                            if return_active:
                                return_prev_xy = robot_xy_est.copy()
                                print(f"[RETURN START] est_disp={disp_mag:.2f}m "
                                      f"est_xy={_fmt_vec(robot_xy_est, 2)}")
                        else:
                            return_active = False
                            return_frame_rot = None
                            return_frame_yaw_ref = None
                            return_probe_phase = None
                            return_probe_stage_start_xy = None
                            return_probe_dx_w = None
                            return_odom_stale_since = None
                            return_prev_xy = None
                            return_last_progress_t = 0.0
                            target[:] = 0.0
                            clear_episode_target_lock("no return needed")
            elif return_active:
                now = time.time()
                disp_mag = float(np.linalg.norm(robot_xy_est))
                if return_prev_xy is not None:
                    prev_xy = np.asarray(return_prev_xy, dtype=np.float32)
                    return_actual_delta_w = (robot_xy_est - prev_xy).astype(np.float32)
                    prev_back_w = (-prev_xy).astype(np.float32)
                    denom = float(np.linalg.norm(return_actual_delta_w)
                                  * np.linalg.norm(prev_back_w))
                    return_actual_prog_dot = float(np.dot(return_actual_delta_w, prev_back_w))
                    return_actual_prog_cos = (
                        0.0 if denom < 1e-6 else return_actual_prog_dot / denom)
                    if return_probe_phase is None:
                        delta_norm = float(np.linalg.norm(return_actual_delta_w))
                        if return_actual_prog_dot < -0.015 and delta_norm > 0.015:
                            return_bad_progress_count += 1
                        elif return_actual_prog_dot > 0.003:
                            return_bad_progress_count = 0
                return_prev_xy = robot_xy_est.copy()
                clear_enough = (obstacle_pos is None
                                or dist >= args.safety_dist + args.clear_margin)
                return_slam_yaw_drift = None
                if (args.return_max_slam_yaw_drift > 0.0
                        and return_probe_phase is None
                        and return_odom_mode == "external"
                        and args.return_yaw_source == "slam"
                        and current_odom_yaw is not None
                        and odom_origin_yaw is not None):
                    return_slam_yaw_drift = abs(
                        _wrap_pi(float(current_odom_yaw) - float(odom_origin_yaw)))
                elif (args.return_max_slam_yaw_drift > 0.0
                      and return_probe_phase is None
                      and return_odom_mode == "external"
                      and args.return_yaw_source == "fused_lowstate"
                      and current_slam_low_yaw_mismatch is not None
                      and current_slam_low_yaw_mismatch
                      > float(args.return_max_slam_yaw_drift)
                      and now - last_yaw_mismatch_warn > 1.0):
                    slam_s = ("None" if current_slam_yaw_delta is None
                              else f"{current_slam_yaw_delta:+.2f}")
                    low_s = ("None" if current_low_yaw_delta is None
                             else f"{current_low_yaw_delta:+.2f}")
                    print(f"[RETURN WARN] SLAM yaw disagrees with lowstate by "
                          f"{current_slam_low_yaw_mismatch:.2f}rad "
                          f"(slam_delta={slam_s} low_delta={low_s}); "
                          "using fused lowstate yaw and checking actual progress")
                    last_yaw_mismatch_warn = now
                if return_odom_mode == "external" and not external_odom_fresh:
                    if return_odom_stale_since is None:
                        return_odom_stale_since = now
                    target[:] = 0.0
                    return_start_t += period
                    return_last_progress_t += period
                    if now - last_odom_warn > 1.0:
                        print(f"[RETURN WAIT] external odom stale "
                              f"age={external_odom_age:.2f}s; holding zero")
                        last_odom_warn = now
                    if (args.return_odom_stale_abort > 0.0
                            and now - return_odom_stale_since
                            >= args.return_odom_stale_abort):
                        print(f"[RETURN ABORT] external odom stale for "
                              f"{now - return_odom_stale_since:.2f}s "
                              f"(age={external_odom_age:.2f}s); zeroing")
                        return_active = False
                        return_frame_rot = None
                        return_frame_yaw_ref = None
                        return_probe_phase = None
                        return_probe_stage_start_xy = None
                        return_probe_dx_w = None
                        return_odom_stale_since = None
                        return_prev_xy = None
                        return_last_progress_t = 0.0
                        target[:] = 0.0
                        clear_episode_target_lock("return odom stale")
                        if args.exit_on_return_abort:
                            print("[RETURN ABORT] exiting controller to StopMove")
                            break
                elif (return_slam_yaw_drift is not None
                      and return_slam_yaw_drift
                      > float(args.return_max_slam_yaw_drift)):
                    return_odom_stale_since = None
                    print(f"[RETURN ABORT] live SLAM yaw drift "
                          f"{return_slam_yaw_drift:.2f}rad exceeds "
                          f"{float(args.return_max_slam_yaw_drift):.2f}rad; "
                          "return_yaw_source=slam is not reliable enough for recover")
                    return_active = False
                    return_frame_rot = None
                    return_frame_yaw_ref = None
                    return_probe_phase = None
                    return_probe_attempt = 0
                    return_probe_stage_start_xy = None
                    return_probe_dx_w = None
                    return_prev_xy = None
                    return_last_progress_t = 0.0
                    return_bad_progress_count = 0
                    target[:] = 0.0
                    clear_episode_target_lock("return slam yaw drift")
                    if args.exit_on_return_abort:
                        print("[RETURN ABORT] exiting controller to StopMove")
                        break
                elif not clear_enough:
                    return_odom_stale_since = None
                    target[:] = 0.0
                    return_start_t += period
                    return_last_progress_t += period
                    if now - last_return_block_warn > 1.0:
                        print(f"[RETURN WAIT] obstacle dist={dist:.2f}m inside "
                              f"clear margin; holding zero")
                        last_return_block_warn = now
                elif disp_mag <= args.return_done_dist:
                    return_odom_stale_since = None
                    print(f"[RETURN DONE] est_disp={disp_mag:.2f}m")
                    return_active = False
                    return_frame_rot = None
                    return_frame_yaw_ref = None
                    return_probe_phase = None
                    return_probe_stage_start_xy = None
                    return_probe_dx_w = None
                    return_prev_xy = None
                    return_last_progress_t = 0.0
                    robot_xy_est[:] = 0.0
                    if (return_odom_mode == "external" and current_odom_xy is not None):
                        odom_origin_xy = current_odom_xy.copy()
                        odom_origin_yaw = current_odom_yaw
                        odom_origin_low_yaw = float(yaw)
                        stationary_samples.clear()
                        stationary_last_xy = robot_xy_est.copy()
                        stationary_last_t = time.time()
                        stationary_since = None
                        stationary_ready = False
                    target[:] = 0.0
                    clear_episode_target_lock("return done")
                elif now - return_start_t >= args.return_timeout:
                    return_odom_stale_since = None
                    print(f"[RETURN TIMEOUT] est_disp={disp_mag:.2f}m; "
                          "holding residual estimate")
                    return_active = False
                    return_frame_rot = None
                    return_frame_yaw_ref = None
                    return_probe_phase = None
                    return_probe_stage_start_xy = None
                    return_probe_dx_w = None
                    return_prev_xy = None
                    return_last_progress_t = 0.0
                    target[:] = 0.0
                    clear_episode_target_lock("return timeout")
                    if args.exit_on_return_abort:
                        print("[RETURN ABORT] exiting controller to StopMove")
                        break
                elif (args.return_bad_progress_abort_count > 0
                      and return_probe_phase is None
                      and return_bad_progress_count
                      >= args.return_bad_progress_abort_count
                      and return_frame_flips < args.return_max_frame_flips):
                    # Walking AWAY from origin: the online return frame is almost always
                    # fit ~180deg backwards here (det=+1 but reversed). Flip it (negate
                    # disp_b) and retry instead of aborting.
                    return_flip_sign = -return_flip_sign
                    return_frame_flips += 1
                    return_bad_progress_count = 0
                    return_last_progress_t = now
                    return_best_disp = disp_mag
                    return_prev_xy = robot_xy_est.copy()
                    target[:] = 0.0
                    print(f"[RETURN FLIP] moving away from origin "
                          f"(actual_cos={0.0 if return_actual_prog_cos is None else return_actual_prog_cos:+.2f}); "
                          f"flipped return frame 180deg and retrying "
                          f"({return_frame_flips}/{args.return_max_frame_flips})")
                elif (args.return_bad_progress_abort_count > 0
                      and return_probe_phase is None
                      and return_bad_progress_count
                      >= args.return_bad_progress_abort_count):
                    print(f"[RETURN ABORT] moving away from origin "
                          f"{return_bad_progress_count}x "
                          f"actual_dot={0.0 if return_actual_prog_dot is None else return_actual_prog_dot:+.3f} "
                          f"actual_cos={0.0 if return_actual_prog_cos is None else return_actual_prog_cos:+.2f}; zeroing")
                    return_active = False
                    return_frame_rot = None
                    return_frame_yaw_ref = None
                    return_probe_phase = None
                    return_probe_stage_start_xy = None
                    return_probe_dx_w = None
                    return_odom_stale_since = None
                    return_bad_progress_count = 0
                    return_prev_xy = None
                    return_last_progress_t = 0.0
                    target[:] = 0.0
                    clear_episode_target_lock("return bad progress")
                    if args.exit_on_return_abort:
                        print("[RETURN ABORT] exiting controller to StopMove")
                        break
                elif (args.return_no_progress_timeout > 0.0
                      and now - return_last_progress_t
                      >= args.return_no_progress_timeout
                      and not (disp_mag + 0.05 < return_best_disp)
                      and not (return_actual_prog_dot is not None
                               and return_actual_prog_dot > 0.003)):
                    actual_dot = (0.0 if return_actual_prog_dot is None
                                  else return_actual_prog_dot)
                    actual_cos = (0.0 if return_actual_prog_cos is None
                                  else return_actual_prog_cos)
                    print(f"[RETURN NO PROGRESS] est_disp={disp_mag:.2f}m "
                          f"best={return_best_disp:.2f}m "
                          f"stalled={now - return_last_progress_t:.1f}s "
                          f"actual_dot={actual_dot:+.3f} "
                          f"actual_cos={actual_cos:+.2f}; holding residual estimate")
                    return_active = False
                    return_frame_rot = None
                    return_frame_yaw_ref = None
                    return_probe_phase = None
                    return_probe_stage_start_xy = None
                    return_probe_dx_w = None
                    return_odom_stale_since = None
                    return_prev_xy = None
                    return_last_progress_t = 0.0
                    target[:] = 0.0
                    clear_episode_target_lock("return no progress")
                    if args.exit_on_return_abort:
                        print("[RETURN ABORT] exiting controller to StopMove")
                        break
                else:
                    return_odom_stale_since = None
                    if return_probe_phase is not None:
                        return_start_t += period
                        return_last_progress_t += period
                        probe_scale = max(
                            1.0,
                            float(args.return_probe_retry_scale)
                            ** max(0, int(return_probe_attempt)),
                        )
                        probe_x_speed = min(
                            float(args.max_vel),
                            float(args.return_probe_speed) * probe_scale,
                        )
                        probe_y_speed = min(
                            float(args.max_vel),
                            float(args.return_probe_lat_speed) * probe_scale,
                        )
                        probe_duration = (
                            float(args.return_probe_duration) * probe_scale)
                        if return_probe_stage_start_xy is None:
                            return_probe_stage_start_xy = robot_xy_est.copy()
                            return_probe_stage_start_t = now
                        if return_probe_phase == "settle_x":
                            target[:] = 0.0
                            probe_stage_elapsed = now - return_probe_stage_start_t
                            probe_settle_timed_out = (
                                args.return_settle_max_wait > 0.0
                                and probe_stage_elapsed >= (
                                    args.return_probe_settle
                                    + args.return_settle_max_wait)
                            )
                            probe_stationary_ready = (
                                return_odom_mode != "external"
                                or args.return_stationary_time <= 0.0
                                or stationary_ready
                            )
                            if (now - return_probe_stage_start_t >= args.return_probe_settle
                                    and probe_stationary_ready):
                                return_probe_phase = "x"
                                return_probe_stage_start_t = now
                                return_probe_stage_start_xy = robot_xy_est.copy()
                                print("[RETURN PROBE] settled; probing +x")
                            elif probe_settle_timed_out:
                                print("[RETURN ABORT] robot did not settle before +x probe; "
                                      "zeroing instead of probing "
                                      f"window={stationary_window_disp:.3f}m/"
                                      f"{stationary_window_span:.2f}s "
                                      f"avg={stationary_avg_speed:.3f}m/s")
                                return_active = False
                                return_frame_rot = None
                                return_frame_yaw_ref = None
                                return_probe_phase = None
                                return_probe_attempt = 0
                                return_probe_stage_start_xy = None
                                return_probe_dx_w = None
                                return_prev_xy = None
                                target[:] = 0.0
                                clear_episode_target_lock("return probe settle timeout")
                                if args.exit_on_return_abort:
                                    print("[RETURN ABORT] exiting controller to StopMove")
                                    break
                            elif now - last_probe_wait_warn > 1.0:
                                still_for = (0.0 if stationary_since is None
                                             else now - stationary_since)
                                print("[RETURN PROBE WAIT] before +x "
                                      f"speed={stationary_speed:.3f}m/s "
                                      f"window={stationary_window_disp:.3f}m/"
                                      f"{stationary_window_span:.2f}s "
                                      f"avg={stationary_avg_speed:.3f}m/s "
                                      f"stable={still_for:.2f}/"
                                      f"{args.return_stationary_time:.2f}s")
                                last_probe_wait_warn = now
                        elif return_probe_phase == "x":
                            target[:] = np.array(
                                [probe_x_speed, 0.0, 0.0],
                                dtype=np.float32)
                            if now - return_probe_stage_start_t >= probe_duration:
                                return_probe_dx_w = (
                                    robot_xy_est
                                    - np.asarray(return_probe_stage_start_xy,
                                                 dtype=np.float32)
                                ).astype(np.float32)
                                print(f"[RETURN PROBE] +x delta_w="
                                      f"{_fmt_vec(return_probe_dx_w, 3)}")
                                return_probe_phase = "settle_y"
                                return_probe_stage_start_t = now
                                return_probe_stage_start_xy = robot_xy_est.copy()
                                target[:] = 0.0
                        elif return_probe_phase == "settle_y":
                            target[:] = 0.0
                            probe_stage_elapsed = now - return_probe_stage_start_t
                            probe_settle_timed_out = (
                                args.return_settle_max_wait > 0.0
                                and probe_stage_elapsed >= (
                                    args.return_probe_settle
                                    + args.return_settle_max_wait)
                            )
                            probe_stationary_ready = (
                                return_odom_mode != "external"
                                or args.return_stationary_time <= 0.0
                                or stationary_ready
                            )
                            if (now - return_probe_stage_start_t >= args.return_probe_settle
                                    and probe_stationary_ready):
                                return_probe_phase = "y"
                                return_probe_stage_start_t = now
                                return_probe_stage_start_xy = robot_xy_est.copy()
                                print("[RETURN PROBE] settled; probing +y")
                            elif probe_settle_timed_out:
                                print("[RETURN ABORT] robot did not settle before +y probe; "
                                      "zeroing instead of probing "
                                      f"window={stationary_window_disp:.3f}m/"
                                      f"{stationary_window_span:.2f}s "
                                      f"avg={stationary_avg_speed:.3f}m/s")
                                return_active = False
                                return_frame_rot = None
                                return_frame_yaw_ref = None
                                return_probe_phase = None
                                return_probe_attempt = 0
                                return_probe_stage_start_xy = None
                                return_probe_dx_w = None
                                return_prev_xy = None
                                target[:] = 0.0
                                clear_episode_target_lock("return probe settle timeout")
                                if args.exit_on_return_abort:
                                    print("[RETURN ABORT] exiting controller to StopMove")
                                    break
                            elif now - last_probe_wait_warn > 1.0:
                                still_for = (0.0 if stationary_since is None
                                             else now - stationary_since)
                                print("[RETURN PROBE WAIT] before +y "
                                      f"speed={stationary_speed:.3f}m/s "
                                      f"window={stationary_window_disp:.3f}m/"
                                      f"{stationary_window_span:.2f}s "
                                      f"avg={stationary_avg_speed:.3f}m/s "
                                      f"stable={still_for:.2f}/"
                                      f"{args.return_stationary_time:.2f}s")
                                last_probe_wait_warn = now
                        elif return_probe_phase == "y":
                            target[:] = np.array(
                                [0.0, probe_y_speed, 0.0],
                                dtype=np.float32)
                            if now - return_probe_stage_start_t >= probe_duration:
                                return_probe_dy_w = (
                                    robot_xy_est
                                    - np.asarray(return_probe_stage_start_xy,
                                                 dtype=np.float32)
                                ).astype(np.float32)
                                print(f"[RETURN PROBE] +y delta_w="
                                      f"{_fmt_vec(return_probe_dy_w, 3)}")
                                rot, reason = _fit_probe_return_frame(
                                    return_probe_dx_w,
                                    return_probe_dy_w,
                                    args.return_probe_min_delta,
                                    args.return_probe_max_axis_cos)
                                if rot is None:
                                    too_small = "too small" in reason
                                    can_retry = (
                                        too_small
                                        and return_probe_attempt
                                        < max(0, int(args.return_probe_retry_count))
                                    )
                                    if can_retry:
                                        return_probe_attempt += 1
                                        retry_scale = max(
                                            1.0,
                                            float(args.return_probe_retry_scale)
                                            ** int(return_probe_attempt),
                                        )
                                        retry_x = min(
                                            float(args.max_vel),
                                            float(args.return_probe_speed)
                                            * retry_scale,
                                        )
                                        retry_y = min(
                                            float(args.max_vel),
                                            float(args.return_probe_lat_speed)
                                            * retry_scale,
                                        )
                                        retry_dur = (
                                            float(args.return_probe_duration)
                                            * retry_scale)
                                        return_probe_phase = "settle_x"
                                        return_probe_stage_start_t = now
                                        return_probe_stage_start_xy = robot_xy_est.copy()
                                        return_probe_dx_w = None
                                        target[:] = 0.0
                                        print("[RETURN PROBE] retry "
                                              f"{return_probe_attempt}/"
                                              f"{max(0, int(args.return_probe_retry_count))} "
                                              f"because {reason}; next "
                                              f"+x {retry_x:.2f}m/s +y {retry_y:.2f}m/s "
                                              f"for {retry_dur:.2f}s each")
                                    else:
                                        print(f"[RETURN ABORT] SDK frame probe failed: "
                                              f"{reason}; zeroing")
                                        return_active = False
                                        return_frame_rot = None
                                        return_frame_yaw_ref = None
                                        return_probe_phase = None
                                        return_probe_attempt = 0
                                        return_probe_stage_start_xy = None
                                        return_probe_dx_w = None
                                        return_prev_xy = None
                                        target[:] = 0.0
                                        clear_episode_target_lock("return probe failed")
                                        if args.exit_on_return_abort:
                                            print("[RETURN ABORT] exiting controller to StopMove")
                                            break
                                else:
                                    return_frame_rot = rot
                                    return_frame_yaw_ref = None
                                    return_probe_phase = None
                                    return_probe_attempt = 0
                                    return_probe_stage_start_xy = None
                                    return_probe_dx_w = None
                                    return_prev_xy = robot_xy_est.copy()
                                    target[:] = 0.0
                                    print("[RETURN FRAME] geo calibrated SDK/world "
                                          f"frame rot={_fmt_mat2(return_frame_rot, 3)} "
                                          f"{reason}")
                    if return_probe_phase is not None or not return_active:
                        pass
                    else:
                        if disp_mag + 0.05 < return_best_disp:
                            return_best_disp = disp_mag
                            return_last_progress_t = now
                        if (return_actual_prog_dot is not None
                                and return_actual_prog_dot > 0.003):
                            return_last_progress_t = now
                        disp_w = robot_xy_est.copy()
                        if args.return_mode == "geo":
                            disp_b, return_frame_src = _disp_world_to_return_frame(
                                disp_w,
                                yaw=yaw,
                                odom_yaw=current_return_yaw,
                                frame=None,
                                frozen_rot=return_frame_rot,
                                frozen_yaw_ref=return_frame_yaw_ref,
                            )
                            if return_frame_rot is None:
                                return_frame_src = current_return_yaw_src
                        else:
                            disp_b, return_frame_src = _disp_world_to_return_frame(
                                disp_w,
                                yaw=yaw,
                                odom_yaw=current_return_yaw,
                                frame=odom_frame if return_odom_mode == "external" else None,
                                frozen_rot=return_frame_rot,
                                frozen_yaw_ref=return_frame_yaw_ref,
                            )
                        # Auto-flip: if a 180deg-reversed frame was detected (below),
                        # negate disp_b so the controller drives toward home instead of away.
                        if return_flip_sign < 0.0:
                            disp_b = -np.asarray(disp_b, dtype=np.float32)
                        return_disp_b = disp_b.copy()
                        if args.return_mode == "head" and return_head is not None:
                            yaw_ref = current_return_yaw if current_return_yaw is not None else yaw
                            yaw_err = yaw_ref - dodge_start_yaw
                            yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
                            target[:], return_raw = _return_head_velocity(
                                return_head,
                                disp_b,
                                lin_scale=dodge.MAX_LIN_VEL,
                                max_lin_vel=args.return_max_vel,
                                max_ang_vel=args.max_ang_vel,
                                yaw_err=yaw_err,
                            )
                            return_head_cmd = target.copy()
                            return_guarded = False
                            if args.return_head_guard:
                                target[:], return_guarded = _stabilize_return_head_velocity(
                                    target,
                                    disp_b,
                                    max_vel=args.return_max_vel,
                                    max_lat_vel=args.return_max_lat_vel,
                                    min_vel=args.return_min_vel,
                                    done_dist=args.return_done_dist,
                                )
                        elif args.return_mode == "geo":
                            return_raw = None
                            return_head_cmd = None
                            return_guarded = False
                            back_b = -disp_b
                            back_b[1] *= float(args.return_lateral_sign)
                            target[:] = _shape_return_velocity(
                                back_b,
                                gain=args.return_gain,
                                lat_gain=args.return_lat_gain,
                                max_vel=args.return_max_vel,
                                max_lat_vel=args.return_max_lat_vel,
                                min_vel=args.return_min_vel,
                                done_dist=args.return_done_dist,
                            )
                            if (args.return_max_ang_vel > 0.0
                                    and current_return_yaw is not None):
                                # Turn back toward the dodge-start heading while
                                # walking home; once heading is restored the home
                                # vector becomes mostly forward instead of lateral.
                                yaw_err = _wrap_pi(current_return_yaw - dodge_start_yaw)
                                if abs(yaw_err) > 0.10:
                                    target[2] = float(np.clip(
                                        -args.return_yaw_gain * yaw_err,
                                        -args.return_max_ang_vel,
                                        args.return_max_ang_vel))
                        elif args.return_mode == "gated" and gated_return_head is not None:
                            return_guarded = False
                            # Return-head input uses GEO's ONLINE-calibrated body frame
                            # (`disp_b`, fitted from SDK-command vs odom-delta) — the raw
                            # lowstate/LIO yaw is too unreliable here. The yaw input is the
                            # heading error (geo's yaw_err) with --return_gated_yaw_sign, so
                            # the return_head's OWN v_rz comes out matching geo's validated
                            # turn direction. We correct the INPUT, not override the output,
                            # so a_ret stays a full learned 3-DOF action.
                            _yaw_disp = 0.0
                            if current_return_yaw is not None:
                                _yaw_disp = (float(args.return_gated_yaw_sign)
                                             * _wrap_pi(current_return_yaw - dodge_start_yaw))
                            _raw6 = np.array([disp_b[0], disp_b[1], _yaw_disp, 0.0, 0.0, 0.0],
                                             dtype=np.float32)
                            with torch.no_grad():
                                _ga = gated_return_head(
                                    torch.from_numpy(_raw6).float().unsqueeze(0)
                                ).squeeze(0).numpy()
                            _ga = np.clip(_ga, -1.0, 1.0)
                            # eval_safe_recovery.py:2870-2880: yaw clamp ±0.5, DEPART XY 2x amp
                            _ga[2] = float(np.clip(_ga[2], -0.5, 0.5))
                            _ga[0] = float(np.clip(2.0 * _ga[0], -1.0, 1.0))
                            _ga[1] = float(np.clip(2.0 * _ga[1], -1.0, 1.0))
                            target[0] = _ga[0] * float(args.return_gated_lin_vel)
                            target[1] = (_ga[1] * float(args.return_gated_lin_vel)
                                         * float(args.return_lateral_sign))
                            # Yaw = the return_head's own v_rz (capped by --return_max_ang_vel).
                            target[2] = 0.0
                            if args.return_max_ang_vel > 0.0:
                                target[2] = float(np.clip(
                                    _ga[2] * float(dodge.MAX_ANG_VEL),
                                    -args.return_max_ang_vel, args.return_max_ang_vel))
                            return_raw = _ga
                            return_head_cmd = target.copy()
                        else:
                            return_raw = None
                            return_head_cmd = None
                            return_guarded = False
                            back_b = -disp_b
                            back_b[1] *= float(args.return_lateral_sign)
                            target[:] = _shape_return_velocity(
                                back_b,
                                gain=args.return_gain,
                                lat_gain=args.return_lat_gain,
                                max_vel=args.return_max_vel,
                                max_lat_vel=args.return_max_lat_vel,
                                min_vel=args.return_min_vel,
                                done_dist=args.return_done_dist,
                            )
            else:
                target[:] = 0.0

            cmd[:] = _rate_limit(cmd, target, max_dlin, max_dang)
            cmd_norm = float(max(abs(cmd[0]), abs(cmd[1]), abs(cmd[2])))
            target_norm = float(max(abs(target[0]), abs(target[1]), abs(target[2])))
            motion_requested = cmd_norm > 1e-4 or target_norm > 1e-4
            should_send_velocity = (
                enabled and started
                and (motion_requested or args.hold_zero_ready)
            )
            code = None
            if should_send_velocity:
                idle_stop_sent = False
                code = loco.SetVelocity(float(cmd[0]), float(cmd[1]), float(cmd[2]),
                                        duration=float(args.cmd_duration))
                if code == 0:
                    sdk_error_count = 0
                else:
                    sdk_error_count += 1
                    if (return_active and args.return_sdk_error_abort > 0
                            and sdk_error_count >= args.return_sdk_error_abort):
                        print(f"[RETURN ABORT] SetVelocity failed {sdk_error_count}x "
                              f"(last code={code}); zeroing and leaving RETURN")
                        return_active = False
                        return_frame_rot = None
                        return_frame_yaw_ref = None
                        return_probe_phase = None
                        return_probe_stage_start_xy = None
                        return_probe_dx_w = None
                        return_odom_stale_since = None
                        return_prev_xy = None
                        target[:] = 0.0
                        cmd[:] = 0.0
                        clear_episode_target_lock("return sdk error")
                        try:
                            loco.SetVelocity(0.0, 0.0, 0.0, duration=0.05)
                        except Exception as e:
                            print(f"[WARN] return sdk-error zero failed: {e}")
                        if args.exit_on_return_abort:
                            print("[RETURN ABORT] exiting controller to StopMove")
                            break
            else:
                sdk_error_count = 0
                if enabled and started and args.idle_stopmove and not idle_stop_sent:
                    send_loco_stopmove("READY idle")
            if enabled and started and code == 0 and return_odom_mode == "cmd":
                robot_xy_est[:] = (robot_xy_est
                                   + _body_to_world_xy(cmd[:2], yaw)
                                   * period * float(args.return_odom_scale))
                robot_pos[:2] = robot_xy_est
            if counter % max(1, int(args.cmd_hz)) == 0:
                if code is None:
                    if args.debug_obs:
                        print(f"[DBG-LOCO] idle; no SetVelocity {_loco_status(loco)}")
                elif code != 0:
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
                       f"est_xy={_fmt_vec(robot_xy_est, 2)} "
                       f"odom={return_odom_mode}")
                if return_odom_mode == "external":
                    msg += (f"/{'fresh' if external_odom_fresh else 'stale'}"
                            f"({external_odom_age:.2f}s)")
                if ydbg.get("track_id") is not None or ydbg.get("locked_track_id") is not None:
                    msg += (f" track={ydbg.get('track_id', '?')}"
                            f"/lock={ydbg.get('locked_track_id', None)}")
                if return_active:
                    msg += (f" rmode={args.return_mode} "
                            f"rframe={return_frame_src}({odom_frame.sample_count})")
                    if return_disp_b is not None:
                        msg += f" disp_b={_fmt_vec(return_disp_b, 2)}"
                    if return_raw is not None:
                        msg += f" ret={_fmt_vec(return_raw, 2)}"
                    if return_guarded:
                        msg += " guard=on"
                if geom is not None:
                    msg += (f" dir={geom['verdict']} par={geom['parallel']:+.2f} "
                            f"perp={geom['perp']:+.2f}")
                print(msg)
                if args.debug_obs and return_odom_mode == "external":
                    lio_yaw_s = "None" if current_odom_yaw is None else f"{current_odom_yaw:+.2f}"
                    origin_yaw_s = "None" if odom_origin_yaw is None else f"{odom_origin_yaw:+.2f}"
                    ret_yaw_s = ("None" if current_return_yaw is None
                                 else f"{current_return_yaw:+.2f}")
                    mismatch_s = ("None" if current_slam_low_yaw_mismatch is None
                                  else f"{current_slam_low_yaw_mismatch:.2f}")
                    print(f"  [DBG-ODOM] raw={_fmt_vec(current_odom_xy, 2)} "
                          f"origin={_fmt_vec(odom_origin_xy, 2)} "
                          f"disp_w={_fmt_vec(robot_xy_est, 2)} "
                          f"lio_yaw={lio_yaw_s} origin_yaw={origin_yaw_s} "
                          f"low_yaw={yaw:+.2f} ret_yaw={ret_yaw_s}"
                          f"/{current_return_yaw_src} yaw_mismatch={mismatch_s} "
                          f"msg={current_odom_msg_count} "
                          f"age={external_odom_age:.2f}s")
                if args.debug_obs and (active or return_active):
                    diag = odom_frame.diagnostics()
                    det = diag.get("det")
                    rms = diag.get("fit_rms")
                    sx = diag.get("scale_x")
                    sy = diag.get("scale_y")
                    det_s = "None" if det is None else f"{det:+.2f}"
                    rms_s = "None" if rms is None else f"{rms:.3f}"
                    sx_s = "None" if sx is None else f"{sx:.3f}"
                    sy_s = "None" if sy is None else f"{sy:.3f}"
                    last = diag.get("last")
                    last_cmd = None if last is None else last.get("cmd")
                    last_delta = None if last is None else last.get("delta")
                    rot_dbg = return_frame_rot if return_frame_rot is not None else diag.get("rot")
                    print(f"  [DBG-FRAME] samples={diag.get('samples', 0)} "
                          f"rot={_fmt_mat2(rot_dbg, 2)} "
                          f"map={_fmt_mat2(diag.get('mapping'), 3)} "
                          f"det={det_s} rms={rms_s} scale=[{sx_s},{sy_s}] "
                          f"last_cmd={_fmt_vec(last_cmd, 2)} "
                          f"last_dodom={_fmt_vec(last_delta, 3)} "
                          f"last_pred={_fmt_vec(diag.get('last_pred'), 3)}")
                if args.debug_obs and return_active:
                    if args.return_mode == "geo":
                        rot, src = _return_frame_rotation(
                            yaw=yaw,
                            odom_yaw=current_return_yaw,
                            frame=None,
                            frozen_rot=return_frame_rot,
                            frozen_yaw_ref=return_frame_yaw_ref,
                        )
                        if return_frame_rot is None:
                            src = current_return_yaw_src
                    else:
                        rot, src = _return_frame_rotation(
                            yaw=yaw,
                            odom_yaw=current_return_yaw,
                            frame=odom_frame if return_odom_mode == "external" else None,
                            frozen_rot=return_frame_rot,
                            frozen_yaw_ref=return_frame_yaw_ref,
                        )
                    cmd_dbg = np.asarray(cmd[:2], dtype=np.float32).copy()
                    target_dbg = np.asarray(target[:2], dtype=np.float32).copy()
                    if args.return_mode == "geo":
                        cmd_dbg[1] *= float(args.return_lateral_sign)
                        target_dbg[1] *= float(args.return_lateral_sign)
                    cmd_odom = (rot @ cmd_dbg).astype(np.float32)
                    target_odom = (rot @ target_dbg).astype(np.float32)
                    back_w = (-robot_xy_est).astype(np.float32)
                    denom = float(np.linalg.norm(cmd_odom) * np.linalg.norm(back_w))
                    prog_dot = float(np.dot(cmd_odom, back_w))
                    prog_cos = 0.0 if denom < 1e-6 else prog_dot / denom
                    print(f"  [DBG-RETURN] frame={src} "
                          f"disp_w={_fmt_vec(robot_xy_est, 2)} "
                          f"disp_b={_fmt_vec(return_disp_b, 2)} "
                          f"back_w={_fmt_vec(back_w, 2)} "
                          f"cmd_odom={_fmt_vec(cmd_odom, 2)} "
                          f"target_odom={_fmt_vec(target_odom, 2)} "
                          f"prog_dot={prog_dot:+.3f} prog_cos={prog_cos:+.2f} "
                          f"dodom_w={_fmt_vec(return_actual_delta_w, 3)} "
                          f"actual_dot={0.0 if return_actual_prog_dot is None else return_actual_prog_dot:+.3f} "
                          f"actual_cos={0.0 if return_actual_prog_cos is None else return_actual_prog_cos:+.2f} "
                          f"head_cmd={_fmt_vec(return_head_cmd, 2)} "
                          f"final_target={_fmt_vec(target, 2)}")
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
        if damping_exit:
            print("[EXIT] damping requested; skip final StopMove")
        elif loco_stop_needed:
            print("[EXIT] zero velocity + StopMove")
            try:
                for _ in range(10):
                    loco.SetVelocity(0.0, 0.0, 0.0, duration=0.05)
                    time.sleep(0.02)
                loco.StopMove()
            except Exception as e:
                print(f"[WARN] StopMove failed: {e}")
        else:
            print("[EXIT] loco was not started; no StopMove sent")


if __name__ == "__main__":
    main()
