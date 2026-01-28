# Copyright 2025 Enactic, Inc.

from isaaclab.utils import configclass
from isaaclab.assets.articulation import ArticulationCfg
from .. import mdp
from ..reach_env_cfg import ReachEnvCfg

from nero.assets.nero import NERO_IDEAL_CFG

@configclass
class NEROReachEnvCfg(ReachEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to Nero
        self.scene.robot = NERO_IDEAL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        # override rewards（末端 body 为 link7）
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link7"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link7"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link7"]
        self.rewards.end_effector_orientation_tracking_fine_grained.params["asset_cfg"].body_names = ["link7"]
        self.rewards.reach_success.params["asset_cfg"].body_names = ["link7"]

        # override actions: joint-space learning (joint position targets)
        # target_joint_pos = offset(default_joint_pos) + scale * action
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint.*"],
            scale=0.5,
            use_default_offset=True,
        )

        # override command generator body
        self.commands.ee_pose.body_name = "link7"

        # joint_pos_limits override removed: limits are now relaxed via startup event


@configclass
class NEROReachEnvCfg_PLAY(NEROReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
