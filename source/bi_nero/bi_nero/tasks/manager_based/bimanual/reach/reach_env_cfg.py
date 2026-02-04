# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import MISSING
from turtle import left

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

import math

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.3),
            pos_y=(0.15, 0.25),
            pos_z=(0.3, 0.4),
            roll=(-math.pi / 9, math.pi / 9), #-30-30
            pitch=(math.pi+3 * math.pi / 2 + math.pi/2, math.pi + 3 * math.pi / 2 + math.pi/2),#恒定朝向
            yaw=(9.6 * math.pi / 10, 10.4 * math.pi / 10),
        ),
    )

    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.3),
            pos_y=(-0.25, -0.15),
            pos_z=(0.3, 0.4),
            roll=(-math.pi / 9, math.pi / 9),#恒定朝向 （-30 30)
            pitch=(3 * math.pi / 2+math.pi/2, 3 * math.pi / 2+math.pi/2),#360
            yaw=(math.pi+9.6 * math.pi / 10, math.pi+10.4 * math.pi / 10),#351-369
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """观测：目标与当前末端位姿误差 + 关节速度 + 上一步动作，降低维度、直接给误差。"""

        # 左臂：位置误差 (3) + 姿态误差 (1)
        left_pos_error = ObsTerm(
            func=mdp.obs_position_error,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "command_name": "left_ee_pose",
            },
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        left_orient_error = ObsTerm(
            func=mdp.obs_orientation_error,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "command_name": "left_ee_pose",
            },
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        # 右臂：位置误差 (3) + 姿态误差 (1)
        right_pos_error = ObsTerm(
            func=mdp.obs_position_error,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "command_name": "right_ee_pose",
            },
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        right_orient_error = ObsTerm(
            func=mdp.obs_orientation_error,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "command_name": "right_ee_pose",
            },
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        # 关节速度 + 上一步动作（利于平滑与时序）
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["left_joint1",
                                                                  "left_joint2",
                                                                  "left_joint3",
                                                                  "left_joint4",
                                                                  "left_joint5",
                                                                  "left_joint6",
                                                                  "left_joint7",
                                                                  ])
            },
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["right_joint1",
                                                                  "right_joint2",
                                                                  "right_joint3",
                                                                  "right_joint4",
                                                                  "right_joint5",
                                                                  "right_joint6",
                                                                  "right_joint7",
                                                                  ])
            },
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        left_actions = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "left_arm_action"},
        )
        right_actions = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "right_arm_action"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    left_end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.24,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "left_ee_pose",
        },
    )

    right_end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.24,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "right_ee_pose",
        },
    )

    left_end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.235,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.1,
            "command_name": "left_ee_pose",
        },
    )

    right_end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.233,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.1,
            "command_name": "right_ee_pose",
        },
    )

    left_end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.09,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "left_ee_pose",
        },
    )

    right_end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.09,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "right_ee_pose",
        },
    )

    # 稀疏成功奖励（课程在 2000 步启用）：位置 < 1cm 且姿态 < 5° 给 1，否则 0
    left_reach_success_sparse = RewTerm(
        func=mdp.reach_success_sparse,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "left_ee_pose",
            "pos_thresh": 0.01,
            "orient_thresh_deg": 5.0,
        },
    )
    right_reach_success_sparse = RewTerm(
        func=mdp.reach_success_sparse,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "right_ee_pose",
            "pos_thresh": 0.01,
            "orient_thresh_deg": 5.0,
        },
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_joint1",
                                                                    "left_joint2",
                                                                    "left_joint3",
                                                                    "left_joint4",
                                                                    "left_joint5",
                                                                    "left_joint6",
                                                                    "left_joint7",
                                                                  ])},
    )
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["right_joint1",
                                                                    "right_joint2",
                                                                    "right_joint3",
                                                                    "right_joint4",
                                                                    "right_joint5",
                                                                    "right_joint6",
                                                                    "right_joint7"
                                                                  ])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass  # 课程配置：5000 步加平滑惩罚，2000 步启用稀疏成功奖励
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.005, "num_steps": 3000},
    )

    left_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_joint_vel", "weight": -0.001, "num_steps": 3000},
    )

    right_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_joint_vel", "weight": -0.001, "num_steps": 3000},
    )

    # 2000 步时启用稀疏成功奖励（位置 < 1cm 且姿态 < 5° 给奖励）
    left_reach_success_sparse = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_reach_success_sparse", "weight": 3.0, "num_steps": 2000},
    )
    right_reach_success_sparse = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_reach_success_sparse", "weight": 3.0, "num_steps": 2000},
    )


##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 24.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0
