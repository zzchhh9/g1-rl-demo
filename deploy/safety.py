"""Safety supervisor for real-robot and sim2sim deployment.

Monitors robot state and enforces safety constraints:
  - E-stop on obstacle proximity, excessive tilt, or comm timeout
  - Velocity clamping (gradual ramp-up during testing)
  - Workspace boundary enforcement
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyConfig:
    max_lin_vel: float = 0.2       # start conservative, increase after validation
    max_ang_vel: float = 0.5
    estop_distance: float = 0.15   # obstacle closer than this → immediate stop
    estop_tilt_deg: float = 15.0   # IMU tilt beyond this → stop
    comm_timeout_s: float = 0.10   # no IMU data for this long → stop
    workspace_radius: float = 2.0  # meters from start position
    workspace_margin: float = 0.3  # ramp-down zone near boundary


class SafetySupervisor:

    def __init__(self, cfg: SafetyConfig | None = None):
        self.cfg = cfg or SafetyConfig()
        self._start_pos: np.ndarray | None = None
        self._estop = False
        self._estop_reason = ""
        self._last_imu_time = 0.0

    def reset(self, start_pos: np.ndarray):
        self._start_pos = start_pos[:2].copy()
        self._estop = False
        self._estop_reason = ""

    @property
    def is_estopped(self) -> bool:
        return self._estop

    @property
    def estop_reason(self) -> str:
        return self._estop_reason

    def check(
        self,
        robot_pos: np.ndarray,
        obstacle_dist: float,
        imu_tilt_deg: float = 0.0,
        current_time: float = 0.0,
    ) -> bool:
        """Check safety conditions. Returns True if safe, False if e-stopped."""
        if self._estop:
            return False

        if obstacle_dist < self.cfg.estop_distance:
            self._estop = True
            self._estop_reason = f"obstacle too close: {obstacle_dist:.3f}m"
            return False

        if imu_tilt_deg > self.cfg.estop_tilt_deg:
            self._estop = True
            self._estop_reason = f"excessive tilt: {imu_tilt_deg:.1f}deg"
            return False

        if (current_time - self._last_imu_time) > self.cfg.comm_timeout_s and current_time > 0.5:
            self._estop = True
            self._estop_reason = f"comm timeout: {current_time - self._last_imu_time:.3f}s"
            return False

        return True

    def update_imu_time(self, t: float):
        self._last_imu_time = t

    def clamp_velocity(
        self,
        vel_cmd: np.ndarray,
        robot_pos: np.ndarray,
    ) -> np.ndarray:
        """Clamp velocity command within safety limits + workspace bounds."""
        if self._estop:
            return np.zeros(3, dtype=np.float32)

        out = vel_cmd.copy()

        # Velocity magnitude clamp
        lin_speed = np.linalg.norm(out[:2])
        if lin_speed > self.cfg.max_lin_vel:
            out[:2] *= self.cfg.max_lin_vel / lin_speed
        out[2] = np.clip(out[2], -self.cfg.max_ang_vel, self.cfg.max_ang_vel)

        # Workspace boundary: ramp down velocity near edge
        if self._start_pos is not None:
            disp = np.linalg.norm(robot_pos[:2] - self._start_pos)
            boundary = self.cfg.workspace_radius
            margin = self.cfg.workspace_margin
            if disp > boundary - margin:
                # Direction toward center
                to_center = self._start_pos - robot_pos[:2]
                to_center_norm = to_center / (np.linalg.norm(to_center) + 1e-8)
                # Project velocity onto outward direction
                vel_xy = out[:2]
                outward_component = np.dot(vel_xy, -to_center_norm)
                if outward_component > 0:
                    # Scale down outward velocity
                    scale = max(0.0, 1.0 - (disp - (boundary - margin)) / margin)
                    out[:2] += to_center_norm * outward_component * (1.0 - scale)

        return out.astype(np.float32)
