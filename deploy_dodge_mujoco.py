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

# Add deploy package (inside this repo)
_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "deploy"))
from dodge_policy import DodgePolicy
from lidar_sim import LidarSim, LidarConfig

from legged_gym import LEGGED_GYM_ROOT_DIR

# ── Paths (all relative to repo root, works after fresh clone) ──
LOCO_CONFIG = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/g1.yaml"
DODGE_CKPT = str(_REPO_ROOT / "checkpoints" / "dodge_v23b_54400.pt")
RETURN_HEAD_CKPT = str(_REPO_ROOT / "checkpoints" / "return_head_v23b_v6.pt")


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
    parser.add_argument("--replay_yolo", type=str, default="",
                        help="Path to YOLO trajectory .npy "
                             "(rows: t, x_fwd, y_left, z, dist, bearing, id). "
                             "When set, replaces synthetic obstacle + LidarSim "
                             "with replayed person position (body→world transform "
                             "applied per frame).")
    parser.add_argument("--live_yolo", action="store_true",
                        help="Subscribe to LIVE DDS topic rt/yolo/person "
                             "(published by scripts/yolo_to_dds_laptop.py). "
                             "Lets the real RealSense + YOLO drive the sim G1 — "
                             "you stand in front of the real robot, sim G1 reacts. "
                             "Real robot is NEVER commanded; only sim moves.")
    parser.add_argument("--live_yolo_net", type=str, default="eno1",
                        help="Network interface for live YOLO DDS (default eno1)")
    parser.add_argument("--live_yolo_staleness", type=float, default=0.5,
                        help="If no YOLO msg in this many seconds, treat as no obstacle")
    parser.add_argument("--safety_distance", type=float, default=1.2)
    parser.add_argument("--replay_max_gap", type=float, default=1.0,
                        help="In replay: skip detection (lidar=None) when adjacent "
                             "samples are farther apart than this (sec). Prevents "
                             "linear-interp-through-tracking-loss artifacts. "
                             "Default 1.0s.")
    parser.add_argument("--replay_lpf_alpha", type=float, default=0.3,
                        help="EMA low-pass filter on (x_fwd, y_left) during replay. "
                             "0.0 = no filter, 1.0 = no smoothing. Default 0.3 = "
                             "smooth out per-frame jumps from YOLO ID swaps.")
    parser.add_argument("--replay_kalman", action="store_true",
                        help="Use constant-velocity Kalman filter for state estimation "
                             "(supersedes --replay_lpf_alpha). Predicts at sim 50Hz "
                             "between sparse YOLO samples, rejects outliers via "
                             "Mahalanobis gate. RECOMMENDED for noisy YOLO+depth data.")
    parser.add_argument("--kf_process_pos_std", type=float, default=0.02,
                        help="KF process noise on position (m per dt)")
    parser.add_argument("--kf_process_vel_std", type=float, default=0.4,
                        help="KF process noise on velocity (m/s per dt). "
                             "Higher = more responsive to person accel")
    parser.add_argument("--kf_meas_std", type=float, default=0.10,
                        help="KF measurement noise on YOLO position (m, baseline). "
                             "Higher = trust YOLO less, more smoothing")
    parser.add_argument("--kf_meas_std_close", type=float, default=0.30,
                        help="KF measurement noise inflated to this value when "
                             "person is at min range. Linear interpolation between "
                             "min_dist and 2m. Default 0.30m to counter close-range "
                             "depth noise bursts.")
    parser.add_argument("--kf_gate", type=float, default=4.0,
                        help="Mahalanobis distance gate to reject outliers (sigma). "
                             "Smaller = reject more aggressively. Default 4.0.")
    parser.add_argument("--reset_distance_offset", type=float, default=0.2,
                        help="After RETURN converges, reset converged=False when "
                             "dist > safety_distance + offset. Default 0.2 (= "
                             "safety+0.2, allows quick re-trigger). Old behavior "
                             "was 1.0 (= must back off well past dodge zone).")
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
    safety_distance = args.safety_distance
    lidar_detected_pos = None

    # Obstacle trajectory
    start_pos = d.qpos[:2].copy()
    cross_dir = np.array([1.0, 0.0])
    obs_gt_pos = np.array([start_pos[0] - 2.0, start_pos[1] + 0.8, 0.75])
    obstacle_phase = "CROSSING"

    # ── Optional: YOLO trajectory replay (overrides synthetic obstacle + LidarSim) ──
    replay_traj = None
    replay_t0_sim = None      # sim time when replay clock starts
    replay_lpf = {"xy": None}    # EMA state for x_fwd, y_left smoothing
    # KF state: [x, y, vx, vy], P 4x4, last_sample_idx (last consumed obs)
    replay_kf = {"x": None, "P": None, "last_sim_t": None, "last_sample_idx": -1,
                 "n_rejected": 0, "n_accepted": 0}
    if args.replay_yolo:
        replay_traj = np.load(args.replay_yolo)
        print(f"[Replay] {args.replay_yolo}: {len(replay_traj)} samples "
              f"over {replay_traj[0,0]:.2f}→{replay_traj[-1,0]:.2f}s "
              f"(dist {replay_traj[:,4].min():.2f}→{replay_traj[:,4].max():.2f}m)")
        # diagnostics
        dts = np.diff(replay_traj[:, 0]) * 1000
        max_dt = dts.max() if len(dts) else 0
        print(f"          sample dt: median={np.median(dts):.0f}ms max={max_dt:.0f}ms "
              f"(gap_thresh={args.replay_max_gap*1000:.0f}ms) lpf_alpha={args.replay_lpf_alpha}")
        replay_t0_sim = 0.0   # was 2.0; 0 → sim plays npy data on the same wall-clock as cam recording

    # ── Optional: LIVE YOLO over DDS (real RealSense + YOLO drives sim G1) ──
    live_yolo_state = None
    if args.live_yolo:
        if replay_traj is not None:
            raise SystemExit("--live_yolo and --replay_yolo are mutually exclusive")
        import json, threading
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize as _CFI, ChannelSubscriber as _CS,
        )
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_ as _String
        _CFI(0, args.live_yolo_net)
        live_yolo_state = dict(latest=None, recv_t=0.0, lock=threading.Lock(),
                               count=0, staleness=args.live_yolo_staleness)
        def _yolo_cb(msg, _s=live_yolo_state):
            try: d = json.loads(msg.data)
            except Exception: return
            with _s["lock"]:
                _s["latest"] = d
                _s["recv_t"] = time.time()
                _s["count"] += 1
            # Debug: log every msg with detection (rate-limited 5 Hz)
            if d.get("n", 0) >= 1:
                now = time.time()
                if not hasattr(_yolo_cb, "_last_log") or (now - _yolo_cb._last_log) > 0.2:
                    print(f"  [yolo-rx #{_s['count']}] d={d['dist']:.2f}m bear={d['bearing']:+5.1f}° id{d['track_id']}", flush=True)
                    _yolo_cb._last_log = now
        _sub = _CS("rt/yolo/person", _String)
        _sub.Init(_yolo_cb, 10)
        live_yolo_state["_sub"] = _sub
        print(f"[LiveYOLO] subscribed rt/yolo/person on {args.live_yolo_net}, "
              f"staleness={args.live_yolo_staleness}s")

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

    FPS = 25  # matches cam recording's OUT_FPS for 1:1 wall-clock alignment in stitch
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

        sim_t = ctrl_step * control_dt

        if live_yolo_state is not None:
            # LIVE: poll latest YOLO msg, drop if stale, convert body→world
            import time as _t
            with live_yolo_state["lock"]:
                _ymsg = live_yolo_state["latest"]
                recv_t = live_yolo_state["recv_t"]
            fresh = (_ymsg is not None) and ((_t.time() - recv_t) <= live_yolo_state["staleness"])
            has_det = fresh and _ymsg.get("n", 0) >= 1
            if has_det:
                x_b, y_b = float(_ymsg["x_fwd"]), float(_ymsg["y_left"])
                z_b = float(_ymsg.get("z", 0.85))
                cy_, sy_ = np.cos(robot_yaw), np.sin(robot_yaw)
                wx = robot_pos[0] + x_b * cy_ - y_b * sy_
                wy = robot_pos[1] + x_b * sy_ + y_b * cy_
                lidar_detected_pos = np.array([wx, wy, z_b], dtype=np.float64)
                obs_gt_pos[:] = wx, wy, z_b
            else:
                lidar_detected_pos = None
                obs_gt_pos[:] = robot_pos[0] + 4.0, robot_pos[1], 0.85
        elif replay_traj is not None and args.replay_kalman:
            # ── KF replay: constant-velocity model, predict at 50Hz, update on new samples ──
            # State = [x, y, vx, vy]. Each sim tick we predict; if a new YOLO sample
            # is available we update (with Mahalanobis gate to reject outliers).
            replay_t = sim_t - replay_t0_sim
            ts = replay_traj[:, 0]
            xs_obs = replay_traj[:, 1]
            ys_obs = replay_traj[:, 2]
            z_const = float(replay_traj[0, 3])

            if replay_t < ts[0] or replay_t > ts[-1] + args.replay_max_gap:
                lidar_detected_pos = None
                obs_gt_pos[:] = 4.0, 0.0, 0.85
                replay_kf["x"] = None
            else:
                # 1) PREDICT (CV model)
                if replay_kf["x"] is None:
                    replay_kf["last_sim_t"] = sim_t
                else:
                    dt = sim_t - replay_kf["last_sim_t"]
                    replay_kf["last_sim_t"] = sim_t
                    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                                  [0, 0, 1, 0],  [0, 0, 0, 1]])
                    Q_pos = (args.kf_process_pos_std) ** 2
                    Q_vel = (args.kf_process_vel_std * max(dt, 1e-3)) ** 2
                    Q = np.diag([Q_pos, Q_pos, Q_vel, Q_vel])
                    replay_kf["x"] = F @ replay_kf["x"]
                    replay_kf["P"] = F @ replay_kf["P"] @ F.T + Q

                # 2) UPDATE if a new YOLO sample crossed
                # Find current sample idx (most recent sample with ts[idx] <= replay_t)
                idx = int(np.searchsorted(ts, replay_t, side="right")) - 1
                if idx > replay_kf["last_sample_idx"] and idx >= 0:
                    # Gap check against previous valid sample
                    prev_idx = replay_kf["last_sample_idx"]
                    if prev_idx >= 0:
                        gap = ts[idx] - ts[prev_idx]
                    else:
                        gap = 0
                    if gap > args.replay_max_gap and replay_kf["x"] is not None:
                        # reset KF across gap
                        replay_kf["x"] = None
                        replay_kf["P"] = None
                    z_obs = np.array([xs_obs[idx], ys_obs[idx]])
                    if replay_kf["x"] is None:
                        # seed
                        replay_kf["x"] = np.array([z_obs[0], z_obs[1], 0.0, 0.0])
                        replay_kf["P"] = np.diag([0.1, 0.1, 1.0, 1.0])
                    else:
                        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
                        # Adaptive R: at close range RealSense depth + bbox
                        # ambiguity is noisier. Linearly interpolate between
                        # baseline (far) and close (near). Use predicted dist
                        # rather than raw obs to avoid letting one bad obs
                        # increase trust in itself.
                        pred_dist = float(np.sqrt(replay_kf["x"][0]**2
                                                   + replay_kf["x"][1]**2))
                        # blend: at dist≤0.5 → close, at dist≥2.0 → baseline
                        blend = max(0.0, min(1.0, (2.0 - pred_dist) / 1.5))
                        meas_std_adp = (args.kf_meas_std * (1 - blend)
                                        + args.kf_meas_std_close * blend)
                        R = np.eye(2) * (meas_std_adp ** 2)
                        y_innov = z_obs - H @ replay_kf["x"]
                        S = H @ replay_kf["P"] @ H.T + R
                        # Mahalanobis (NIS) for outlier gate
                        nis = float(y_innov @ np.linalg.solve(S, y_innov))
                        if nis > (args.kf_gate ** 2):
                            replay_kf["n_rejected"] += 1
                            if replay_kf["n_rejected"] % 5 == 1:
                                print(f"  [KF reject @t={sim_t:.2f}s] obs=({z_obs[0]:.2f},{z_obs[1]:.2f}) "
                                      f"vs pred=({replay_kf['x'][0]:.2f},{replay_kf['x'][1]:.2f}) "
                                      f"nis={nis:.1f} > gate²={args.kf_gate**2:.0f}", flush=True)
                        else:
                            K = replay_kf["P"] @ H.T @ np.linalg.inv(S)
                            replay_kf["x"] = replay_kf["x"] + K @ y_innov
                            replay_kf["P"] = (np.eye(4) - K @ H) @ replay_kf["P"]
                            replay_kf["n_accepted"] += 1
                    replay_kf["last_sample_idx"] = idx

                if replay_kf["x"] is not None:
                    wx = float(replay_kf["x"][0])
                    wy = float(replay_kf["x"][1])
                    lidar_detected_pos = np.array([wx, wy, z_const], dtype=np.float64)
                    obs_gt_pos[:] = wx, wy, z_const
                else:
                    lidar_detected_pos = None
                    obs_gt_pos[:] = 4.0, 0.0, 0.85
        elif replay_traj is not None:
            # YOLO replay: WORLD-frame fixed (assumes recording was with robot near origin/stationary).
            # Person stays at the world position they had at capture time,
            # so when sim G1 dodges the person doesn't chase it.
            #
            # Two safety nets:
            #   1) gap handling: if neighbouring samples > replay_max_gap apart,
            #      treat as "no detection" (avoids fake interpolated drift)
            #   2) low-pass filter: EMA on (x,y) smooths YOLO ID-swap jumps
            replay_t = sim_t - replay_t0_sim
            ts = replay_traj[:, 0]
            if replay_t < ts[0] or replay_t > ts[-1]:
                lidar_detected_pos = None
                obs_gt_pos[:] = 4.0, 0.0, 0.85
            else:
                # Find bracketing samples — gap-aware
                i_right = int(np.searchsorted(ts, replay_t))
                i_left = max(0, i_right - 1)
                gap = ts[i_right] - ts[i_left] if i_right < len(ts) else float('inf')
                if gap > args.replay_max_gap:
                    lidar_detected_pos = None
                    obs_gt_pos[:] = 4.0, 0.0, 0.85
                    replay_lpf["xy"] = None   # reset filter across gaps
                else:
                    wx_raw = float(np.interp(replay_t, ts, replay_traj[:, 1]))
                    wy_raw = float(np.interp(replay_t, ts, replay_traj[:, 2]))
                    z = float(np.interp(replay_t, ts, replay_traj[:, 3]))
                    # EMA low-pass filter on (x, y) — re-seeds after gap or first detect
                    if replay_lpf["xy"] is None:
                        replay_lpf["xy"] = np.array([wx_raw, wy_raw], dtype=np.float64)
                    else:
                        a = args.replay_lpf_alpha
                        replay_lpf["xy"] = (a * np.array([wx_raw, wy_raw])
                                            + (1 - a) * replay_lpf["xy"])
                    wx, wy = float(replay_lpf["xy"][0]), float(replay_lpf["xy"][1])
                    lidar_detected_pos = np.array([wx, wy, z], dtype=np.float64)
                    obs_gt_pos[:] = wx, wy, z
        else:
            # Original synthetic obstacle + LidarSim path
            if obstacle_phase == "CROSSING":
                obs_gt_pos[:2] += cross_dir * args.obstacle_speed * control_dt
                if np.linalg.norm(obs_gt_pos[:2] - start_pos) > 5.0:
                    obstacle_phase = "GONE"
            if ctrl_step % lidar_every == 0:
                detected = lidar.detect_obstacle(robot_pos, max_dist=5.0, min_height=0.3)
                if detected is not None:
                    lidar_detected_pos = detected.copy()

        # Update mocap position (for visualization in both modes)
        mocap_idx = m.body_mocapid[obstacle_mocap_id]
        if mocap_idx >= 0:
            d.mocap_pos[mocap_idx] = obs_gt_pos

        if lidar_detected_pos is not None:
            dist = float(np.linalg.norm(robot_pos[:2] - lidar_detected_pos[:2]))
        else:
            dist = float("inf")

        # ── DEBUG: print every tick where dist < safety+0.3 (track trigger conditions) ──
        if dist < safety_distance + 0.3:
            print(f"  [debug t={sim_t:.2f}s step={ctrl_step}] dist={dist:.2f}m "
                  f"safety={safety_distance:.2f} | active={dodge_active} "
                  f"converged={return_converged}", flush=True)

        # ── Dodge state machine ──
        if dist < safety_distance and not dodge_active and not return_converged:
            dodge_active = True
            dodge_start_pos = robot_pos[:2].copy()
            dodge_start_yaw = robot_yaw
            dodge.reset(robot_pos[:2], robot_yaw)
            print(f"  step={ctrl_step:4d} [DODGE START] dist={dist:.2f}m", flush=True)

        if dodge_active:
            depart = dist > safety_distance + 0.10
            if not depart:
                obs18 = dodge.build_obs(robot_pos, robot_yaw,
                                         lidar_detected_pos if lidar_detected_pos is not None
                                         else np.array([10, 10, 0.75]),
                                         dt=control_dt, depart_mask=False)
                vel = dodge.get_velocity_command(obs18)
                # HARD CAP: 0.2 m/s linear, 0.5 rad/s angular (matches real-robot
                # deploy ceiling so sim shows realistic-magnitude behavior)
                cmd[0] = float(np.clip(vel[0], -0.20, 0.20))
                cmd[1] = float(np.clip(vel[1], -0.20, 0.20))
                cmd[2] = float(np.clip(vel[2], -0.50, 0.50))
            elif return_head is not None:
                disp_w = robot_pos[:2] - dodge_start_pos
                disp_b = dodge._body_frame_xy(disp_w, robot_yaw)
                with torch.no_grad():
                    _d = torch.tensor(disp_b, dtype=torch.float32).unsqueeze(0)
                    _ret = return_head(_d).squeeze(0).numpy()
                yaw_err = robot_yaw - dodge_start_yaw
                yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
                cmd[0] = float(np.clip(_ret[0] * dodge.MAX_LIN_VEL, -0.20, 0.20))
                cmd[1] = float(np.clip(_ret[1] * dodge.MAX_LIN_VEL, -0.20, 0.20))
                cmd[2] = float(np.clip(-2.0 * yaw_err, -0.5, 0.5))

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
            # Reset converged when person leaves "post-dodge cool-off" zone
            # → allows re-trigger if they come close again
            if return_converged and dist > safety_distance + args.reset_distance_offset:
                return_converged = False
                print(f"  step={ctrl_step:4d} [RESET] dist={dist:.2f}m > "
                      f"{safety_distance + args.reset_distance_offset:.2f}m, "
                      f"converged → False (ready to re-trigger)", flush=True)

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
