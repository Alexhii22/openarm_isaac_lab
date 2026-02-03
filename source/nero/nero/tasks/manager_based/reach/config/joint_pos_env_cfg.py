# Copyright 2025 Enactic, Inc.

from isaaclab.utils import configclass
from isaaclab.assets.articulation import ArticulationCfg
from .. import mdp
from ..reach_env_cfg import ReachEnvCfg

from nero.assets.nero import NERO_CFG

@configclass
class NEROReachEnvCfg(ReachEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to Nero
        self.scene.robot = NERO_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "joint1": 0.0,
                    "joint2": 0.0,
                    "joint3": 0.0,
                    "joint4": 0.0,
                    "joint5": 0.0,
                    "joint6": 0.0,
                    "joint7": 0.0,
                }
            ),
        )

        # override rewards（末端 body 为 link7）
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link7"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link7"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link7"]


        # override actions: joint-space learning (joint position targets)
        # target_joint_pos = offset(default_joint_pos) + scale * action
        # scale 降低至 0.5 以减小单步动作幅度，配合更强的平滑性奖励
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint7",
            ],
            scale=0.5,  # 折中：便于收敛又利于 sim2real（每步约 ±1 rad）
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
