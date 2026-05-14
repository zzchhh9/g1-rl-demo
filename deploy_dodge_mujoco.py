"""Sim2sim dodge demo: dodge policy → unitree_rl_gym locomotion → MuJoCo G1.

Extends the unitree_rl_gym deploy_mujoco.py with:
  1. Simulated obstacle (red sphere geom in MuJoCo scene)
  2. LiDAR-based obstacle detection (mj_ray from head)
  3. Dodge policy velocity commands override the locomotion cmd
  4. Return head for post-dodge return + yaw P-controller

Usage:
    cd g1-rl-demo
    uv run python deploy_dodge_mujoco.py --record sim2sim_dodge.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

# Add parent deploy package
sys.path.insert(0, str(Path(__file__).parent.parent / "deploy"))
from dodge_policy import DodgePolicy
from lidar_sim import LidarSim, LidarConfig

from legged_gym import LEGGED_GYM_ROOT_DIR

# ── Paths ──
LOCO_CONFIG = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/g1.yaml"
DODGE_CKPT = str(Path(__file__).parent.parent / "h1_loco" / "logs" / "rsl_rl" /
    "h1_dodge_base_vel" / "2026-05-03_07-35-03_v23b_BC_fzyaw_revdep_latdodge50_vy100_drop0.3_ret80_lr5e-05_warmstart" / "model_54400.pt")
RETURN_HEAD_CKPT = str(Path(__file__).parent.parent / "checkpoints" / "return_head_v23b_v6.pt")


def get_gravity_orientation(quat):
    qw, qx, qy, qz = quat
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ])


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def build_scene_with_obstacle(base_xml_path: str) -> mujoco.MjModel:
    """Load the G1 scene XML and inject an obstacle mocap body."""
    import os
    xml_dir = os.path.dirname(os.path.abspath(base_xml_path))

    with open(base_xml_path) as f:
        xml = f.read()

    obstacle_xml = """
    <!-- Dodge obstacle (mocap = freely positionable) -->
    <body name="obstacle" mocap="true" pos="0 3 0.75">
      <geom name="obstacle_geom" type="sphere" size="0.25"
            rgba="0.9 0.15 0.15 0.7" contype="1" conaffinity="1"
            mass="0.001"/>
    </body>
    """
    xml = xml.replace("</worldbody>", obstacle_xml + "\n  </worldbody>")

    if "<global" in xml:
        xml = xml.replace("<global", '<global offwidth="1280" offheight="720"')
    else:
        xml = xml.replace("<visual>", '<visual>\n    <global offwidth="1280" offheight="720"/>')

    # Load from the XML directory so <include> relative paths resolve
    old_cwd = os.getcwd()
    os.chdir(xml_dir)
    try:
        model = mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(old_cwd)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=str, default="",
                        help="Output video path (empty = viewer only)")
    parser.add_argument("--dodge_ckpt", default=DODGE_CKPT)
    parser.add_argument("--obstacle_speed", type=float, default=0.18)
    parser.add_argument("--duration", type=float, default=25.0)
    args = parser.parse_args()

    # ── Load locomotion config ──
    with open(LOCO_CONFIG) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    sim_dt = cfg["simulation_dt"]
    decimation = cfg["control_decimation"]
    control_dt = sim_dt * decimation

    kps = np.array(cfg["kps"], dtype=np.float32)
    kds = np.array(cfg["kds"], dtype=np.float32)
    default_angles = np.array(cfg["default_angles"], dtype=np.float32)

    ang_vel_scale = cfg["ang_vel_scale"]
    dof_pos_scale = cfg["dof_pos_scale"]
    dof_vel_scale = cfg["dof_vel_scale"]
    action_scale = cfg["action_scale"]
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
    num_actions = cfg["num_actions"]
    num_obs = cfg["num_obs"]

    # ── Load MuJoCo model with obstacle ──
    m = build_scene_with_obstacle(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt

    obstacle_mocap_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
    pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    print(f"[Scene] obstacle body={obstacle_mocap_id}, pelvis body={pelvis_id}")

    # ── Load policies ──
    loco_policy = torch.jit.load(policy_path)
    print(f"[Loco] Loaded {policy_path}")

    dodge = DodgePolicy(args.dodge_ckpt)

    # Return head
    import torch.nn as nn
    return_head = None
    if Path(RETURN_HEAD_CKPT).exists():
        rh_sd = torch.load(RETURN_HEAD_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
        return_head = nn.Sequential(nn.Linear(2, 32), nn.ELU(), nn.Linear(32, 3), nn.Tanh())
        return_head.load_state_dict(
            {k.replace("return_head.", ""): v for k, v in rh_sd.items() if "return_head" in k})
        return_head.eval()
        print("[ReturnHead] Loaded")

    # ── LiDAR setup ──
    # Find torso body for LiDAR mount (in 12dof model it might be "torso" or just pelvis)
    torso_name = "pelvis"  # 12dof model may not have torso_link
    for candidate in ["torso_link", "torso", "pelvis"]:
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, candidate) >= 0:
            torso_name = candidate
            break
    lidar_cfg = LidarConfig(
        n_horizontal=72, n_vertical=5, noise_std=0.02,
        mount_body_name=torso_name,
        mount_offset=np.array([0.0, 0.0, 0.35]),  # above pelvis
    )
    lidar = LidarSim(m, d, lidar_cfg)

    # ── State ──
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    cmd = np.zeros(3, dtype=np.float32)

    dodge_active = False
    dodge_start_pos = None
    dodge_start_yaw = 0.0
    return_converged = False
    safety_distance = 1.5
    lidar_detected_pos = None

    # Obstacle trajectory
    start_pos = d.qpos[:2].copy()
    cross_dir = np.array([1.0, 0.0])
    obs_gt_pos = np.array([start_pos[0] - 2.0, start_pos[1] + 0.8, 0.75])
    obstacle_phase = "CROSSING"

    # Recording
    recording = bool(args.record)
    frames = []
    renderer = None
    if recording:
        renderer = mujoco.Renderer(m, 720, 1280)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = pelvis_id
        cam.distance = 5.0
        cam.elevation = -25
        cam.azimuth = 135

    FPS = 30
    render_every = max(1, round(1.0 / (FPS * control_dt)))
    lidar_every = max(1, round(1.0 / (10.0 * control_dt)))  # 10 Hz
    max_steps = int(args.duration / control_dt)
    counter = 0

    print(f"Running {max_steps} control steps ({args.duration}s)...")

    def step_sim():
        nonlocal counter
        for _ in range(decimation):
            tau = pd_control(target_dof_pos, d.qpos[7:], kps,
                             np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            counter += 1

    # ── Main loop ──
    for ctrl_step in range(max_steps):
        robot_pos = d.qpos[:3].copy()
        quat = d.qpos[3:7].copy()
        w, x, y, z = quat
        robot_yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

        # Move obstacle
        if obstacle_phase == "CROSSING":
            obs_gt_pos[:2] += cross_dir * args.obstacle_speed * control_dt
            if np.linalg.norm(obs_gt_pos[:2] - start_pos) > 5.0:
                obstacle_phase = "GONE"
        # Update mocap position
        mocap_idx = m.body_mocapid[obstacle_mocap_id]
        if mocap_idx >= 0:
            d.mocap_pos[mocap_idx] = obs_gt_pos

        # LiDAR scan
        if ctrl_step % lidar_every == 0:
            detected = lidar.detect_obstacle(robot_pos, max_dist=5.0, min_height=0.3)
            if detected is not None:
                lidar_detected_pos = detected.copy()

        if lidar_detected_pos is not None:
            dist = float(np.linalg.norm(robot_pos[:2] - lidar_detected_pos[:2]))
        else:
            dist = float("inf")

        # ── Dodge state machine ──
        if dist < safety_distance and not dodge_active and not return_converged:
            dodge_active = True
            dodge_start_pos = robot_pos[:2].copy()
            dodge_start_yaw = robot_yaw
            dodge.reset(robot_pos[:2], robot_yaw)
            print(f"  step={ctrl_step:4d} [DODGE START] dist={dist:.2f}m")

        if dodge_active:
            depart = dist > safety_distance + 0.10
            if not depart:
                obs18 = dodge.build_obs(robot_pos, robot_yaw,
                                         lidar_detected_pos if lidar_detected_pos is not None
                                         else np.array([10, 10, 0.75]),
                                         dt=control_dt, depart_mask=False)
                vel = dodge.get_velocity_command(obs18)
                cmd[:] = vel
            elif return_head is not None:
                disp_w = robot_pos[:2] - dodge_start_pos
                disp_b = dodge._body_frame_xy(disp_w, robot_yaw)
                with torch.no_grad():
                    _d = torch.tensor(disp_b, dtype=torch.float32).unsqueeze(0)
                    _ret = return_head(_d).squeeze(0).numpy()
                yaw_err = robot_yaw - dodge_start_yaw
                yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
                cmd[0] = float(_ret[0]) * dodge.MAX_LIN_VEL
                cmd[1] = float(_ret[1]) * dodge.MAX_LIN_VEL
                cmd[2] = float(np.clip(-2.0 * yaw_err, -1, 1)) * dodge.MAX_ANG_VEL

                disp_mag = float(np.linalg.norm(disp_w))
                if disp_mag < 0.20 and abs(yaw_err) < 0.15:
                    dodge_active = False
                    return_converged = True
                    print(f"  step={ctrl_step:4d} [RETURN DONE] disp={disp_mag:.3f}m "
                          f"yaw_err={np.degrees(yaw_err):+.1f}°")
            else:
                dodge_active = False
                return_converged = True
                cmd[:] = 0
        else:
            cmd[:] = 0

        # ── Build locomotion obs ──
        qj = d.qpos[7:]
        dqj = d.qvel[6:]
        omega = d.qvel[3:6]

        qj_obs = (qj - default_angles) * dof_pos_scale
        dqj_obs = dqj * dof_vel_scale
        gravity = get_gravity_orientation(quat)
        omega_obs = omega * ang_vel_scale

        period = 0.8
        phase = (counter * sim_dt) % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        obs[:3] = omega_obs
        obs[3:6] = gravity
        obs[6:9] = cmd * cmd_scale
        obs[9:9 + num_actions] = qj_obs
        obs[9 + num_actions:9 + 2 * num_actions] = dqj_obs
        obs[9 + 2 * num_actions:9 + 3 * num_actions] = action
        obs[9 + 3 * num_actions] = sin_phase
        obs[9 + 3 * num_actions + 1] = cos_phase

        obs_t = torch.from_numpy(obs).unsqueeze(0)
        action = loco_policy(obs_t).detach().numpy().squeeze()
        target_dof_pos = action * action_scale + default_angles

        step_sim()

        # ── Render / Record ──
        if recording and ctrl_step % render_every == 0:
            renderer.update_scene(d, cam)
            frames.append(renderer.render().copy())

        if ctrl_step % 100 == 0:
            phase_str = "DODGE" if dodge_active else ("DONE" if return_converged else "IDLE")
            print(f"  step={ctrl_step:4d} [{phase_str:5s}] x={robot_pos[0]:+.2f} "
                  f"y={robot_pos[1]:+.2f} z={robot_pos[2]:.2f} dist={dist:.2f}")

    if recording and frames:
        out = Path(args.record)
        out.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v3 as iio
        iio.imwrite(str(out), np.stack(frames), fps=FPS, codec="libx264", quality=8)
        print(f"\nSaved {len(frames)} frames to {out}")
        renderer.close()

    if return_converged:
        print("✓ IDLE → DODGE (LiDAR) → RETURN → STOP")
    else:
        print("✗ Return did not converge")


if __name__ == "__main__":
    main()
