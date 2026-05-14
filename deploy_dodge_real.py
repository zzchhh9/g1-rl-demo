"""Real G1 dodge deployment: dodge policy + locomotion via unitree_sdk2.

Extends deploy_real.py with:
  1. LiDAR obstacle detection (Livox Mid-360 via Livox SDK 2 UDP)
  2. Dodge policy overrides locomotion velocity command
  3. Return head + yaw P-controller for post-dodge return
  4. Safety limits with configurable max velocity

Usage:
    uv run python deploy_dodge_real.py eth0 configs/g1.yaml

Controls:
    START  → exit zero-torque, begin standing
    A      → enable walking + dodge mode
    SELECT → EMERGENCY STOP (any time)

LiDAR: connects to Livox Mid-360 via UDP multicast (no ROS2 needed).
       Requires livox_lidar_sdk2.so on the system. If not available,
       falls back to a dummy detector that always returns no obstacle.
"""

from __future__ import annotations

import sys
import time
import threading
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
# LiDAR obstacle detection (Livox Mid-360 via UDP, no ROS2)
# ═══════════════════════════════════════════════════════════════

class LivoxObstacleDetector:
    """Detect nearest obstacle from Livox Mid-360 point cloud via UDP.

    Uses raw UDP socket to receive Livox point cloud packets on the
    multicast group. No ROS2, no livox_ros_driver2 needed.

    The Livox Mid-360 broadcasts point cloud on UDP multicast 224.1.1.5:56301.
    Each packet contains N points in Cartesian format (x,y,z in mm as int32).

    If the Livox SDK shared library is not available, this falls back to
    a dummy that always returns None (no obstacle detected). This allows
    testing the rest of the pipeline without hardware.
    """

    def __init__(self, host_ip: str = "192.168.123.222",
                 lidar_ip: str = "192.168.123.120",
                 point_port: int = 56301,
                 min_height: float = 0.3,
                 max_range: float = 5.0):
        self._min_height = min_height
        self._max_range = max_range
        self._latest_points: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False

        try:
            import socket
            import struct
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,
                                       socket.IPPROTO_UDP)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind(("", point_port))
            # Join multicast group
            mreq = struct.pack("4s4s",
                               socket.inet_aton("224.1.1.5"),
                               socket.inet_aton(host_ip))
            self._sock.setsockopt(socket.IPPROTO_IP,
                                  socket.IP_ADD_MEMBERSHIP, mreq)
            self._sock.settimeout(0.1)
            self._running = True
            self._thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._thread.start()
            print(f"[LiDAR] Listening on UDP {host_ip}:{point_port} "
                  f"(multicast 224.1.1.5)")
        except Exception as e:
            print(f"[LiDAR] WARNING: UDP init failed ({e}). "
                  f"Using dummy detector (no obstacle detection).")
            self._sock = None

    def _recv_loop(self):
        """Background thread: receive UDP packets and parse point cloud."""
        import struct
        buf = bytearray(65536)
        points_accum = []
        last_flush = time.time()

        while self._running:
            try:
                n = self._sock.recv_into(buf)
                if n < 24:
                    continue
                # Livox SDK2 packet header: version(1) + length(2) + ... + data_type(1)
                # Data starts after header. For Cartesian High (data_type=1):
                # each point = x(int32) + y(int32) + z(int32) + reflectivity(uint8)
                # + tag(uint8) = 14 bytes
                # Simplified parsing: try to extract xyz from fixed offset
                data_type = buf[18]
                dot_num = struct.unpack_from("<H", buf, 20)[0]
                offset = 24  # approximate header size

                if data_type == 1:  # Cartesian High
                    for _ in range(dot_num):
                        if offset + 14 > n:
                            break
                        x, y, z = struct.unpack_from("<iii", buf, offset)
                        points_accum.append([x / 1000.0, y / 1000.0, z / 1000.0])
                        offset += 14
                elif data_type == 2:  # Cartesian Low
                    for _ in range(dot_num):
                        if offset + 8 > n:
                            break
                        x, y, z = struct.unpack_from("<hhh", buf, offset)
                        points_accum.append([x / 100.0, y / 100.0, z / 100.0])
                        offset += 8

                # Flush accumulated points as a "frame" every 100ms
                now = time.time()
                if now - last_flush > 0.1 and len(points_accum) > 100:
                    with self._lock:
                        self._latest_points = np.array(points_accum,
                                                        dtype=np.float32)
                    points_accum = []
                    last_flush = now

            except TimeoutError:
                continue
            except Exception:
                continue

    def detect(self, robot_pos: np.ndarray,
               robot_yaw: float) -> np.ndarray | None:
        """Return obstacle position in WORLD frame, or None.

        Args:
            robot_pos: [x, y, z] robot world position
            robot_yaw: robot yaw in radians
        """
        if self._sock is None:
            return None

        with self._lock:
            pts = self._latest_points
        if pts is None or len(pts) == 0:
            return None

        # Points from Livox are in SENSOR frame (Z-up, X-forward).
        # Transform to world frame using robot pose.
        cy, sy = np.cos(robot_yaw), np.sin(robot_yaw)
        # Rotate sensor-frame points to world frame + translate
        wx = pts[:, 0] * cy - pts[:, 1] * sy + robot_pos[0]
        wy = pts[:, 0] * sy + pts[:, 1] * cy + robot_pos[1]
        wz = pts[:, 2] + robot_pos[2]

        # Filter: above floor, within range, not self
        mask = wz > self._min_height
        dists = np.sqrt((wx - robot_pos[0])**2 + (wy - robot_pos[1])**2)
        mask &= dists < self._max_range
        mask &= dists > 0.3  # exclude self-reflections

        if mask.sum() == 0:
            return None

        # Nearest cluster centroid
        filtered = np.stack([wx[mask], wy[mask], wz[mask]], axis=1)
        centroid = filtered.mean(axis=0)
        return centroid.astype(np.float32)

    def stop(self):
        self._running = False
        if self._sock:
            self._sock.close()


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

        # ── LiDAR (Livox SDK UDP, no ROS2) ──
        self.lidar = LivoxObstacleDetector()

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

        # ── Log ──
        if self.counter % 50 == 0:
            phase_str = "DODGE" if self.dodge_active else (
                "DONE" if self.return_converged else "IDLE")
            print(f"  [{phase_str:5s}] cmd=[{self.cmd[0]:+.2f},{self.cmd[1]:+.2f},"
                  f"{self.cmd[2]:+.2f}] dist={dist:.2f}")


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
