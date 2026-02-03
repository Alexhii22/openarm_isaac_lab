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

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)
        ),
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

    # 目标每 2.5–3.5 s 重采样，促使策略跟踪「当前目标」而非停在旧目标附近
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.10, -0.10),  # 扩大x轴范围  nero：580mm 
            pos_y=(-0.10, 0.10),  # 扩大y轴范围
            pos_z=(0.60, 0.70),   # 稍微扩大z轴范围
            roll=(math.pi-math.pi/2, math.pi-math.pi/2),  # 扩大roll范围  30-150
            pitch=(math.pi+math.pi/2, math.pi+math.pi/2),  # 扩大pitch范围 
            yaw=(-math.pi/6, math.pi/6),  # 允许yaw自由旋转  150-210
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                        "joint7",
                    ],
                )
            },
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                        "joint7",
                    ],
                )
            },
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "ee_pose"}
        )

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    apply_joint_limits = EventTerm(
        func=mdp.apply_joint_position_limits,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
            ),
            "joint_limits": {
                # from nero_description.urdf
                "joint1": (-2.740167, 2.740167),
                "joint2": (-1.731799, 3.316126),
                "joint3": (-2.792527, 2.792527),
                "joint4": (-1.047198, 2.181662),
                "joint5": (-2.792527, 2.792527),
                "joint6": (-0.750492, 1.012291),
                "joint7": (-1.570796, 1.570796),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
#事件配置重置机械臂位置

@configclass
class RewardsCfg:
    """Reward terms (goal-aligned, dense, tunable).

    说明：每个 term 单独输出，权重可在此处直接调，不要在函数里“硬编码权重”。
    """

    # --- Task: Position tracking ---
    # [DISABLED] Conflicting with Tanh reward. L2 penalty is too harsh at long range.
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "ee_pose",
        },
    )
    
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.3,  # 放大梯度感知（平替 scale），同位移产生更大奖励变化
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.3,  # 略减小使曲线更陡，同位移梯度更强
            "command_name": "ee_pose",
        },
    )

    
    # --- Task: Orientation tracking ---
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=0.1, # Increased penalty (from -0.1) to force orientation alignment
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "ee_pose",
        },
    )


    # --- Penalties: Smoothness ---
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=0.0)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                    "joint7",
                ],
            )
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


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

    def __post_init__(self):
        """Post initialization."""
        # general settings
        # decimation=4 → step_dt=1/15 s，单步关节约 0.143 rad（等效放大单步位移，不改 scale）
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 24.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0
