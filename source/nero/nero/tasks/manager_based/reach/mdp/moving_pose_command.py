# Copyright 2025 Enactic, Inc.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class MovingPoseCommand(UniformPoseCommand):
    """Command generator for moving pose commands.
    
    This command generator starts at a fixed Y position and moves along the negative Y axis
    of the robot base frame. It resets when the Y position crosses a threshold.
    """

    cfg: MovingPoseCommandCfg

    def __init__(self, cfg: MovingPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def _resample_command(self, env_ids):
        # Call parent to sample X, Z, Roll, Pitch, Yaw
        super()._resample_command(env_ids)
        
        # Override Y to start position
        if len(env_ids) > 0:
            self.pose_command_b[env_ids, 1] = self.cfg.start_pos_y

    def _update_command(self):
        # Move along negative Y
        self.pose_command_b[:, 1] -= self.cfg.velocity * self._env.step_dt
        
        # Check limit
        reset_ids = (self.pose_command_b[:, 1] < self.cfg.reset_pos_y_limit).nonzero(as_tuple=False).flatten()
        
        if len(reset_ids) > 0:
             self._resample_command(reset_ids)

@configclass
class MovingPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for moving pose command generator."""
    
    class_type = MovingPoseCommand

    velocity: float = 0.1
    """Velocity of the target along the y-axis (m/s)."""

    start_pos_y: float = 0.4
    """Starting y-position of the target."""

    reset_pos_y_limit: float = -0.4
    """Limit for y-position to trigger reset."""
