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

import struct
import sys
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

sys.path.insert(0, str(Path(LEGGED_GYM_ROOT_DIR) / "deploy" / "deploy_real"))
from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, MotorMode
from common.rotation_helper import get_gravity_orientation
from common.remote_controller import RemoteController, KeyMap
from config import Config


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
# Dodge Controller (extends deploy_real.py's Controller)
# ═══════════════════════════════════════════════════════════════

class DodgeController:
    def __init__(self, config: Config, dodge_ckpt: str, return_head_ckpt: str,
                 max_lin_vel: float = 0.15, safety_distance: float = 2.0):
        self.config = config
        self.remote_controller = RemoteController()
        self.max_lin_vel = max_lin_vel
        self.safety_distance = safety_distance

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

        # ── LiDAR (unitree_sdk2py DDS, 与 test_lidar.py 一致) ──
        self.lidar = LidarObstacleDetector()

        # ── SDK communication ──
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)

        self.wait_for_low_state()
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

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

        # Robot pose estimation (from IMU)
        self._robot_pos = np.array([0.0, 0.0, 0.8], dtype=np.float32)
        self._robot_yaw = 0.0
        self._prev_robot_pos = None

    def _low_state_handler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("[SDK] Connected to G1.")

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
        """Estimate robot world pose from IMU."""
        quat = self.low_state.imu_state.quaternion  # [w, x, y, z]
        w, x, y, z = quat
        self._robot_yaw = float(np.arctan2(2 * (w * z + x * y),
                                            1 - 2 * (y * y + z * z)))
        # Position from dead-reckoning (integrate velocity × dt)
        # Crude but sufficient for ~20s dodge episodes
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        # Use cmd as velocity proxy (actual velocity tracking)
        cy, sy = np.cos(self._robot_yaw), np.sin(self._robot_yaw)
        vx_w = self.cmd[0] * cy - self.cmd[1] * sy
        vy_w = self.cmd[0] * sy + self.cmd[1] * cy
        self._robot_pos[0] += vx_w * self.config.control_dt
        self._robot_pos[1] += vy_w * self.config.control_dt

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

        # ── LiDAR obstacle detection ──
        obstacle_pos = self.lidar.detect(self._robot_pos, self._robot_yaw)
        if obstacle_pos is not None:
            dist = float(np.linalg.norm(self._robot_pos[:2] - obstacle_pos[:2]))
        else:
            dist = float("inf")

        # ── Dodge state machine ──
        if dist < self.safety_distance and not self.dodge_active and not self.return_converged:
            self.dodge_active = True
            self.dodge_start_pos = self._robot_pos[:2].copy()
            self.dodge_start_yaw = self._robot_yaw
            self.dodge_policy.reset(self._robot_pos[:2], self._robot_yaw)
            print(f"  [DODGE START] dist={dist:.2f}m")

        if self.dodge_active:
            depart = dist > self.safety_distance + 0.10
            if not depart:
                obs18 = self.dodge_policy.build_obs(
                    self._robot_pos, self._robot_yaw, obstacle_pos,
                    dt=self.config.control_dt, depart_mask=False)
                vel = self.dodge_policy.get_velocity_command(obs18)
                self.cmd[0] = float(np.clip(vel[0], -self.max_lin_vel, self.max_lin_vel))
                self.cmd[1] = float(np.clip(vel[1], -self.max_lin_vel, self.max_lin_vel))
                self.cmd[2] = float(np.clip(vel[2], -0.3, 0.3))
            elif self.return_head is not None:
                disp_w = self._robot_pos[:2] - self.dodge_start_pos
                disp_b = self.dodge_policy._body_frame_xy(disp_w, self._robot_yaw)
                with torch.no_grad():
                    _d = torch.tensor(disp_b, dtype=torch.float32).unsqueeze(0)
                    _ret = self.return_head(_d).squeeze(0).numpy()
                yaw_err = self._robot_yaw - self.dodge_start_yaw
                yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
                self.cmd[0] = float(np.clip(
                    _ret[0] * self.dodge_policy.MAX_LIN_VEL,
                    -self.max_lin_vel, self.max_lin_vel))
                self.cmd[1] = float(np.clip(
                    _ret[1] * self.dodge_policy.MAX_LIN_VEL,
                    -self.max_lin_vel, self.max_lin_vel))
                self.cmd[2] = float(np.clip(-2.0 * yaw_err, -0.5, 0.5))
                disp_mag = float(np.linalg.norm(disp_w))
                if disp_mag < 0.20 and abs(yaw_err) < 0.15:
                    self.dodge_active = False
                    self.return_converged = True
                    print(f"  [RETURN DONE] disp={disp_mag:.3f}m "
                          f"yaw={np.degrees(yaw_err):+.1f}°")
            else:
                self.dodge_active = False
                self.return_converged = True
                self.cmd[:] = 0
        else:
            # Normal: remote controller
            self.cmd[0] = self.remote_controller.ly
            self.cmd[1] = self.remote_controller.lx * -1
            self.cmd[2] = self.remote_controller.rx * -1
            # Reset dodge state when no obstacle
            if dist > self.safety_distance + 1.0:
                self.return_converged = False

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
        self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        self.obs[9:9 + n] = qj_obs
        self.obs[9 + n:9 + 2 * n] = dqj_obs
        self.obs[9 + 2 * n:9 + 3 * n] = self.action
        self.obs[9 + 3 * n] = sin_phase
        self.obs[9 + 3 * n + 1] = cos_phase

        # ── Locomotion inference ──
        self.action = self.loco_policy(
            torch.from_numpy(self.obs).unsqueeze(0)).detach().numpy().squeeze()
        self.target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

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
            phase_str = "DODGE" if self.dodge_active else (
                "DONE" if self.return_converged else "IDLE")
            lidar_str = f"lidar={dist:.2f}m" if dist < 100 else "lidar=N/A"
            lidar_frames = self.lidar.msg_count
            print(f"  [{phase_str:5s}] cmd=[{self.cmd[0]:+.2f},{self.cmd[1]:+.2f},"
                  f"{self.cmd[2]:+.2f}] {lidar_str} lidar_frames={lidar_frames}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="Network interface (e.g. eth0)")
    parser.add_argument("config", type=str, default="configs/g1.yaml",
                        help="Config file")
    parser.add_argument("--max_vel", type=float, default=0.10,
                        help="Max dodge velocity m/s (default 0.10, very conservative)")
    parser.add_argument("--safety_dist", type=float, default=2.5,
                        help="Dodge trigger distance (default 2.5m, very early trigger)")
    args = parser.parse_args()

    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    ChannelFactoryInitialize(0, args.net)

    controller = DodgeController(
        config,
        dodge_ckpt=str(_REPO / "checkpoints" / "dodge_v23b_54400.pt"),
        return_head_ckpt=str(_REPO / "checkpoints" / "return_head_v23b_v6.pt"),
        max_lin_vel=args.max_vel,
        safety_distance=args.safety_dist,
    )

    controller.zero_torque_state()
    controller.move_to_default_pos()
    controller.default_pos_state()

    print(f"\n[DODGE MODE] max_vel={args.max_vel} m/s, safety={args.safety_dist}m")
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
