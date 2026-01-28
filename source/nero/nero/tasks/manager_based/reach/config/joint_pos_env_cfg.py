# Copyright 2025 Enactic, Inc.

from isaaclab.utils import configclass
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from .. import mdp
from ..reach_env_cfg import ReachEnvCfg

from nero.assets.nero import NERO_IDEAL_CFG

@configclass
class NEROReachEnvCfg(ReachEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to FR5
        self.scene.robot = NERO_IDEAL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        # override rewards（末端 body 为 link7）
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link7"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link7"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link7"]

        # override actions: Differential IK in task-space (end-effector)
        # - action is delta position in robot root frame (3D), scaled by `scale`
        # - IK solver computes joint targets each step
        self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint.*"],
            body_name="link7",
            scale=0.05,  # meters per step for action in [-1, 1]
            controller=DifferentialIKControllerCfg(
                command_type="position",
                use_relative_mode=True,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
            ),
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
