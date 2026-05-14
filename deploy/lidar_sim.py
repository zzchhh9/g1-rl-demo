"""Simulated Livox Mid-360 LiDAR for G1 sim2sim deployment.

Specs (from Livox datasheet):
    - FOV: 360° horizontal × 59° vertical (-7° to +52°)
    - Point rate: 200K pts/s, 10 Hz frame rate → 20K pts/frame
    - Range: 40m @10% reflectivity, min 0.1m
    - Precision: ≤2cm @10m (1σ)
    - Mount: G1 head top, on torso_link body

For sim2sim, we cast a reduced set of rays (configurable, default 500)
using MuJoCo's mj_ray(). This is sufficient to detect a human-sized
obstacle within the safety distance.

Usage:
    lidar = LidarSim(mj_model, mj_data)
    points, hits = lidar.scan()  # returns hit points in world frame
    obstacle_pos = lidar.detect_obstacle(robot_pos)  # nearest cluster center
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class LidarConfig:
    # Livox Mid-360 specs
    h_fov_deg: float = 360.0
    v_fov_min_deg: float = -7.0
    v_fov_max_deg: float = 52.0
    max_range: float = 40.0
    min_range: float = 0.1
    noise_std: float = 0.02        # 2cm Gaussian noise (1σ @10m)

    # Sim parameters (reduced from real 20K for performance)
    n_horizontal: int = 72         # rays per horizontal ring (every 5°)
    n_vertical: int = 7            # vertical rings across FOV
    frame_rate: float = 10.0       # Hz (real LiDAR update rate)

    # Mount position: offset from torso_link body origin.
    # head_link geom is at [0.004, 0, -0.054] from torso_link,
    # but the real LiDAR sits on TOP of the head (~15cm above head_link).
    mount_body_name: str = "torso_link"
    mount_offset: np.ndarray | None = None  # [x, y, z] in body frame

    # Geom group filtering: only hit collision geoms (group 0).
    # group=1 are visual-only meshes that shouldn't reflect LiDAR.
    geom_group: np.ndarray | None = None

    def __post_init__(self):
        if self.mount_offset is None:
            # LiDAR on top of head: head_link offset + ~15cm up
            self.mount_offset = np.array([0.004, 0.0, 0.10])
        if self.geom_group is None:
            # Enable group 0 (collision geoms), disable groups 1-5
            self.geom_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)


class LidarSim:
    """Simulated Livox Mid-360 using MuJoCo ray casting."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 cfg: LidarConfig | None = None):
        self.model = model
        self.data = data
        self.cfg = cfg or LidarConfig()

        # Find mount body
        self._mount_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.mount_body_name)
        if self._mount_body_id < 0:
            # Try with robot/ prefix (mjlab compiled model)
            self._mount_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{self.cfg.mount_body_name}")
        if self._mount_body_id < 0:
            raise ValueError(f"Mount body '{self.cfg.mount_body_name}' not found in model")

        # Pre-compute ray directions in LiDAR local frame
        self._ray_dirs = self._build_ray_pattern()
        self._n_rays = len(self._ray_dirs)

        # Robot body ID for exclusion (don't detect self)
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis_id < 0:
            pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
        self._robot_body_id = pelvis_id if pelvis_id >= 0 else -1

        print(f"[LidarSim] {self._n_rays} rays, mount=body[{self._mount_body_id}] "
              f"({self.cfg.mount_body_name}), FOV={self.cfg.h_fov_deg}°×"
              f"[{self.cfg.v_fov_min_deg}°,{self.cfg.v_fov_max_deg}°]")

    def _build_ray_pattern(self) -> np.ndarray:
        """Pre-compute ray directions in LiDAR local frame (Z-up, X-forward)."""
        h_angles = np.linspace(0, 2 * np.pi, self.cfg.n_horizontal, endpoint=False)
        v_min = np.radians(self.cfg.v_fov_min_deg)
        v_max = np.radians(self.cfg.v_fov_max_deg)
        v_angles = np.linspace(v_min, v_max, self.cfg.n_vertical)

        dirs = []
        for v in v_angles:
            cos_v = np.cos(v)
            sin_v = np.sin(v)
            for h in h_angles:
                dx = cos_v * np.cos(h)
                dy = cos_v * np.sin(h)
                dz = sin_v
                dirs.append([dx, dy, dz])
        return np.array(dirs, dtype=np.float64)

    def _get_lidar_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get LiDAR world position and rotation matrix from current sim state."""
        body_pos = self.data.xpos[self._mount_body_id].copy()
        body_mat = self.data.xmat[self._mount_body_id].reshape(3, 3).copy()

        # Apply local offset
        world_offset = body_mat @ self.cfg.mount_offset
        lidar_pos = body_pos + world_offset

        return lidar_pos, body_mat

    def scan(self) -> tuple[np.ndarray, np.ndarray]:
        """Cast rays and return hit points.

        Returns:
            points: [N_hits, 3] world-frame hit positions
            distances: [N_rays] distance per ray (-1 = no hit)
        """
        lidar_pos, rot_mat = self._get_lidar_pose()
        distances = np.full(self._n_rays, -1.0)
        points = []

        geomid = np.array([-1], dtype=np.int32)

        for i, local_dir in enumerate(self._ray_dirs):
            # Rotate ray direction to world frame
            world_dir = rot_mat @ local_dir
            norm = np.linalg.norm(world_dir)
            if norm < 1e-6:
                continue
            world_dir = world_dir / norm

            dist = mujoco.mj_ray(
                self.model, self.data,
                lidar_pos, world_dir,
                self.cfg.geom_group,
                1,  # flg_static: include static geoms (floor, obstacles)
                self._robot_body_id,  # exclude robot's own body
                geomid,
            )

            if dist >= self.cfg.min_range and dist <= self.cfg.max_range:
                # Add Gaussian noise
                if self.cfg.noise_std > 0:
                    dist += np.random.normal(0, self.cfg.noise_std)
                    dist = max(self.cfg.min_range, dist)

                distances[i] = dist
                hit_point = lidar_pos + world_dir * dist
                points.append(hit_point)

        if len(points) > 0:
            return np.array(points), distances
        return np.zeros((0, 3)), distances

    def detect_obstacle(self, robot_pos: np.ndarray,
                        max_dist: float = 5.0,
                        min_height: float = 0.3) -> np.ndarray | None:
        """Detect nearest non-floor obstacle from LiDAR scan.

        Filters out floor hits (Z < min_height) and returns the centroid
        of the nearest cluster of hit points.

        Args:
            robot_pos: [3] robot world position (for distance filtering)
            max_dist: only consider points within this distance
            min_height: ignore hits below this Z (floor filtering)

        Returns:
            [3] obstacle world position, or None if no obstacle detected
        """
        points, _ = self.scan()
        if len(points) == 0:
            return None

        # Filter: above floor + within range
        mask = points[:, 2] > min_height
        dists = np.linalg.norm(points[:, :2] - robot_pos[:2], axis=1)
        mask &= dists < max_dist
        # Exclude points very close to robot (self-reflections from floor near feet)
        mask &= dists > 0.2

        filtered = points[mask]
        if len(filtered) == 0:
            return None

        # Simple nearest-cluster: find the closest non-floor point group
        # For a single obstacle, centroid of all filtered points works.
        # For multiple obstacles, would need DBSCAN or similar.
        centroid = filtered.mean(axis=0)
        return centroid.astype(np.float32)

    def detect_obstacle_nearest(self, robot_pos: np.ndarray,
                                min_height: float = 0.3) -> np.ndarray | None:
        """Return the nearest non-floor hit point (simpler than centroid)."""
        points, _ = self.scan()
        if len(points) == 0:
            return None

        mask = points[:, 2] > min_height
        filtered = points[mask]
        if len(filtered) == 0:
            return None

        dists = np.linalg.norm(filtered[:, :2] - robot_pos[:2], axis=1)
        nearest_idx = np.argmin(dists)
        return filtered[nearest_idx].astype(np.float32)
