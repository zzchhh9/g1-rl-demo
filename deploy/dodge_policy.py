"""Standalone dodge policy inference — no BiGym/mjlab dependency.

Loads a v23b-style transformer checkpoint and provides:
  - build_obs(): construct 18-dim obs from raw sensor data
  - get_action(): returns [vx_b, vy_b, vrz] in [-1, 1]

Usage:
    policy = DodgePolicy("path/to/model_54400.pt", device="cpu")
    policy.reset(robot_xy=np.array([0, 0]), robot_yaw=0.0)
    obs = policy.build_obs(
        robot_pos=np.array([x, y, z]),
        robot_yaw=yaw,
        obstacle_pos_w=np.array([ox, oy, oz]),
        dt=0.02,
    )
    action = policy.get_action(obs)  # [vx_b, vy_b, vrz] in [-1, 1]
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class DodgePolicy:
    """Transformer dodge policy for sim2sim and real deployment.

    Reconstructs the transformer actor from checkpoint state_dict
    (same logic as eval_safe_recovery.py:_init_transformer, but standalone).
    """

    MAX_LIN_VEL = 0.5   # m/s (training scale)
    MAX_ANG_VEL = 1.0   # rad/s
    CONTROL_DT = 0.02   # 50 Hz

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"]

        self._build_model(state_dict)
        self._load_normalizer(state_dict)

        self._last_action = np.zeros(3, dtype=np.float32)
        self._prev_robot_pos: np.ndarray | None = None
        self._prev_obs_pos_w: np.ndarray | None = None
        self._start_pos: np.ndarray | None = None
        self._start_yaw: float = 0.0

    def _build_model(self, state_dict: dict):
        """Reconstruct transformer actor from checkpoint keys."""
        ah_keys = sorted(
            k for k in state_dict
            if k.startswith("action_head.") and k.endswith(".weight") and "norm" not in k
        )
        action_dim = state_dict[ah_keys[-1]].shape[0]
        embed_dim = state_dict["cls_token"].shape[-1]

        # Recover obs layout from token embedder key insertion order.
        obs_groups = []
        for k in state_dict:
            if k.startswith("token_embedders.") and k.endswith(".weight"):
                name = k.split(".")[1]
                dim = state_dict[k].shape[1]
                obs_groups.append((name, dim))

        self.obs_dim = sum(d for _, d in obs_groups)
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self._obs_layout = obs_groups

        scale_factor = 2.0 / math.sqrt(embed_dim)

        class _Scale(nn.Module):
            def __init__(self, factor):
                super().__init__()
                self.factor = factor
            def forward(self, x):
                return x * self.factor

        class _TFActor(nn.Module):
            def __init__(self, obs_layout, embed_dim, action_dim):
                super().__init__()
                self.embed_dim = embed_dim
                self.token_embedders = nn.ModuleDict()
                self._slices = []
                offset = 0
                for name, dim in obs_layout:
                    self.token_embedders[name] = nn.Linear(dim, embed_dim)
                    self._slices.append((name, offset, dim))
                    offset += dim
                self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
                n_tok = len(obs_layout) + 1
                self.pos_embed = nn.Parameter(torch.randn(1, n_tok, embed_dim) * 0.02)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=4 if embed_dim >= 64 else 2,
                    dim_feedforward=embed_dim * 2, dropout=0.0,
                    activation="gelu", batch_first=True, norm_first=True,
                )
                self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

                ah_w0_shape = state_dict.get("action_head.0.weight", torch.zeros(1)).shape
                if len(ah_w0_shape) == 1:
                    self.action_head = nn.Sequential(
                        nn.LayerNorm(embed_dim),
                        _Scale(scale_factor),
                        nn.Linear(embed_dim, 128), nn.ELU(),
                        nn.Linear(128, 128), nn.ELU(),
                        nn.Linear(128, action_dim),
                    )
                else:
                    self.action_head = nn.Sequential(
                        nn.Linear(embed_dim, embed_dim), nn.ELU(),
                        nn.Linear(embed_dim, action_dim),
                    )

            def forward(self, obs):
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)
                B = obs.shape[0]
                tokens = [self.cls_token.expand(B, -1, -1)]
                for name, start, dim in self._slices:
                    if start + dim <= obs.shape[-1]:
                        g = obs[:, start:start + dim]
                    else:
                        g = torch.zeros(B, dim, device=obs.device)
                    tokens.append(self.token_embedders[name](g).unsqueeze(1))
                tokens = torch.cat(tokens, dim=1)
                tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
                out = self.transformer(tokens)
                cls_out = out[:, 0, :]
                return self.action_head(cls_out)

            def get_cls_features(self, obs):
                """Return CLS token output (for arm_bc head)."""
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)
                B = obs.shape[0]
                tokens = [self.cls_token.expand(B, -1, -1)]
                for name, start, dim in self._slices:
                    if start + dim <= obs.shape[-1]:
                        g = obs[:, start:start + dim]
                    else:
                        g = torch.zeros(B, dim, device=obs.device)
                    tokens.append(self.token_embedders[name](g).unsqueeze(1))
                tokens = torch.cat(tokens, dim=1)
                tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
                out = self.transformer(tokens)
                return out[:, 0, :]

        self._model = _TFActor(obs_groups, embed_dim, action_dim)
        model_sd = self._model.state_dict()
        loaded = 0
        for k, v in state_dict.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                loaded += 1
        self._model.load_state_dict(model_sd)
        self._model.to(self.device).eval()
        print(f"[DodgePolicy] Loaded transformer: embed={embed_dim}, act={action_dim}, "
              f"obs={self.obs_dim}, tokens={len(obs_groups)}, weights={loaded}/{len(state_dict)}")

    def _load_normalizer(self, state_dict: dict):
        self._norm_mean = state_dict.get("actor_obs_normalizer._mean")
        self._norm_std = state_dict.get("actor_obs_normalizer._std")
        if self._norm_mean is not None:
            self._norm_mean = self._norm_mean.to(self.device)
            self._norm_std = self._norm_std.to(self.device)
            print(f"[DodgePolicy] Normalizer loaded: mean shape={self._norm_mean.shape}")
        else:
            print("[DodgePolicy] WARNING: no normalizer found in checkpoint")

    def reset(self, robot_xy: np.ndarray, robot_yaw: float = 0.0):
        """Call when entering DODGE mode. Records start position for displacement obs."""
        self._last_action = np.zeros(3, dtype=np.float32)
        self._prev_robot_pos = None
        self._prev_obs_pos_w = None
        self._start_pos = robot_xy[:2].copy()
        self._start_yaw = robot_yaw

    @staticmethod
    def _body_frame_xy(vec_w: np.ndarray, yaw: float) -> np.ndarray:
        cy, sy = np.cos(yaw), np.sin(yaw)
        x_b = vec_w[0] * cy + vec_w[1] * sy
        y_b = -vec_w[0] * sy + vec_w[1] * cy
        return np.array([x_b, y_b], dtype=np.float32)

    def build_obs(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
        obstacle_pos_w: np.ndarray,
        dt: float | None = None,
        static_box_aabb: tuple[float, float, float, float] | None = None,
        depart_mask: bool = False,
    ) -> np.ndarray:
        """Build observation vector from raw sensor data.

        Args:
            robot_pos: [x, y, z] world frame.
            robot_yaw: scalar yaw in radians.
            obstacle_pos_w: [x, y, z] obstacle position, world frame.
            dt: timestep (default CONTROL_DT).
            static_box_aabb: (x_min, x_max, y_min, y_max) of static obstacle.
                             None → zeros for box obs.
            depart_mask: if True, zero obstacle XY obs (DEPART phase).
        """
        if dt is None:
            dt = self.CONTROL_DT

        # Base linear velocity (finite-diff, body frame)
        if self._prev_robot_pos is not None:
            vel_w = (robot_pos - self._prev_robot_pos) / dt
        else:
            vel_w = np.zeros(3, dtype=np.float32)
        bxy = self._body_frame_xy(vel_w[:2], robot_yaw)
        base_lin_vel = np.array([bxy[0], bxy[1], vel_w[2]], dtype=np.float32)
        self._prev_robot_pos = robot_pos.copy()

        # Obstacle position in body frame
        obs_rel_w = obstacle_pos_w[:2] - robot_pos[:2]
        obs_pos_b_xy = self._body_frame_xy(obs_rel_w, robot_yaw)
        obs_pos_b = np.array([obs_pos_b_xy[0], obs_pos_b_xy[1], -0.48], dtype=np.float32)

        # Obstacle velocity in body frame (finite-diff)
        if self._prev_obs_pos_w is not None:
            obs_vel_w = (obstacle_pos_w[:2] - self._prev_obs_pos_w[:2]) / dt
        else:
            obs_vel_w = np.zeros(2, dtype=np.float32)
        obs_vel_b_xy = self._body_frame_xy(obs_vel_w, robot_yaw)
        obs_vel_b = np.array([obs_vel_b_xy[0], obs_vel_b_xy[1], 0.0], dtype=np.float32)
        self._prev_obs_pos_w = obstacle_pos_w.copy()

        # Displacement from start (body frame)
        disp_w = robot_pos[:2] - self._start_pos
        displacement_b = self._body_frame_xy(disp_w, robot_yaw)

        # Static box relative position (body frame)
        if static_box_aabb is not None:
            xmin, xmax, ymin, ymax = static_box_aabb
            rx, ry = robot_pos[0], robot_pos[1]
            nearest_x = np.clip(rx, xmin, xmax)
            nearest_y = np.clip(ry, ymin, ymax)
            box_rel_w = np.array([nearest_x - rx, nearest_y - ry], dtype=np.float32)
            box_rel_b_xy = self._body_frame_xy(box_rel_w, robot_yaw)
            box_rel_b = np.array([box_rel_b_xy[0], box_rel_b_xy[1], 0.0], dtype=np.float32)
        else:
            box_rel_b = np.zeros(3, dtype=np.float32)

        # Yaw displacement
        yaw_disp = robot_yaw - self._start_yaw
        yaw_disp = (yaw_disp + np.pi) % (2 * np.pi) - np.pi
        yaw_obs = np.array([yaw_disp], dtype=np.float32)

        # DEPART masking: zero obstacle XY, keep Z=-0.48
        if depart_mask:
            obs_pos_b[:2] = 0.0
            obs_vel_b[:2] = 0.0

        # Concatenate: 18-dim = vel(3) + obs_pos(3) + obs_vel(3) + disp(2) + box(3) + yaw(1) + act(3)
        obs = np.concatenate([
            base_lin_vel,         # 3
            obs_pos_b,            # 3
            obs_vel_b,            # 3
            displacement_b,       # 2
            box_rel_b,            # 3
            yaw_obs,              # 1
            self._last_action,    # 3
        ])
        return obs.astype(np.float32)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Run inference. Returns [vx_b, vy_b, vrz] in [-1, 1]."""
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        if self._norm_mean is not None:
            obs_t = (obs_t - self._norm_mean) / (self._norm_std + 1e-8)
        raw = self._model(obs_t).cpu().numpy()[0]
        # Transformer action_head has no built-in tanh — apply here to break
        # corner-lock saturation (matches eval_safe_recovery.py:701).
        action = np.clip(np.tanh(raw), -1.0, 1.0)
        self._last_action = action.copy()
        return action

    def get_velocity_command(self, obs: np.ndarray) -> np.ndarray:
        """Like get_action but returns scaled velocity [vx, vy, vrz] in m/s and rad/s."""
        action = self.get_action(obs)
        return np.array([
            action[0] * self.MAX_LIN_VEL,
            action[1] * self.MAX_LIN_VEL,
            action[2] * self.MAX_ANG_VEL,
        ], dtype=np.float32)
