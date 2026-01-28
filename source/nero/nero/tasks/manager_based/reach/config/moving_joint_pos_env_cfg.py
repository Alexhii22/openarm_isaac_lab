# Copyright 2025 Enactic, Inc.

from isaaclab.utils import configclass
from .joint_pos_env_cfg import NEROReachEnvCfg
from .. import mdp

@configclass
class NEROMovingReachEnvCfg(NEROReachEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override command generator（与 reward 一致：末端 body 为 link7）
        self.commands.ee_pose = mdp.MovingPoseCommandCfg(
            asset_name="robot",
            body_name="link7",
            resampling_time_range=(1.0, 1.0), # Resample X, Z, R, P, Y
            debug_vis=True,
            velocity=0.2,
            start_pos_y=0.4,
            reset_pos_y_limit=-0.4,
            ranges=mdp.MovingPoseCommandCfg.Ranges(
                pos_x=(0.0, 0.0),
                pos_y=(0.0, 0.0), # Starting Y
                pos_z=(0.0, 0.0),
                roll=(0.0, 0.0),
                pitch=(1.57, 1.57),
                yaw=(0.0, 0.0),
            ),
        )

@configclass
class NEROMovingReachEnvCfg_PLAY(NEROMovingReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
