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

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import os

# The USD asset is located in fairino_description/urdf/fairino5_v6
# We assume the script is running from the project root or the path is reachable
# For robustness, we can try to find the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
NERO_USD_PATH = os.path.join(PROJECT_ROOT, "nero_description/urdf/nero.usd")

NERO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=NERO_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
        },
    ),
    actuators={
        "nero_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-7]"],
            velocity_limit_sim=3.0,#最大关节角速度
            effort_limit_sim={
                "joint[1-7]": 28.0,
            },
            stiffness=60.0,#P值 刚度 越大 → 越“硬”，越想马上到目标 越小 → 越“软”，动作慢但稳定
            damping=4.0,#抑制速度，防止震荡
        ),
    },
    soft_joint_pos_limit_factor=1.0,#完全使用URDF限位
)

"""Configuration of NERO robot."""

NERO_HIGH_PD_CFG = NERO_CFG.copy()
NERO_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
NERO_HIGH_PD_CFG.actuators["nero_arm"].stiffness = 400.0
NERO_HIGH_PD_CFG.actuators["nero_arm"].damping = 80.0
"""Configuration of neor robot with stiffer PD control."""

NERO_IDEAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=NERO_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # 1. 理想补偿：直接禁用重力，模拟机械臂完美的重力补偿算法
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            # 增加线性阻尼和角阻尼的微调，设为极小值以模拟极低摩擦
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # 2. 提高求解器精度：增加位置和速度迭代次数，减少关节“软”或“抖动”的现象
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            # 启用固定基座
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
        },
    ),
    actuators={
        "nero_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-7]"],
            velocity_limit_sim=2.15,
            effort_limit_sim={
                "joint[1-7]": 28.0,
            },
            # 3. 控制增益：理想环境通常需要极高的响应速度
            # 较高的 stiffness (Kp) 保证路径跟随，适中的 damping (Kd) 保证稳定性
            stiffness=80.0, 
            damping=4.0,
            # 4. 摩擦力补偿：在仿真层面将电机的摩擦力和电枢惯量设为 0
            friction=0.0,
            armature=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
