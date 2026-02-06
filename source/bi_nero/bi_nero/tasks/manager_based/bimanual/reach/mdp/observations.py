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

"""观测：以相对 SE(3) 几何误差为核心，与关键点奖励一致；弱化绝对世界量。"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_mul

from .rewards import compute_keypoint_distance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    """单位四元数求逆 (共轭)：q=(w,x,y,z) -> (w,-x,-y,-z)。"""
    return torch.cat([q[..., 0:1], -q[..., 1:4]], dim=-1)


def _quat_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    """四元数 q=(w,x,y,z) 转为轴角 (3D)：方向=旋转轴，模长=弧度。shape (N, 3)。"""
    w = q[..., 0:1].clamp(-1.0, 1.0)
    xyz = q[..., 1:4]
    angle = 2.0 * torch.acos(w)
    sin_half = torch.sin(angle / 2.0).clamp(min=1e-8)
    scale = torch.where(angle.abs() < 1e-6, torch.ones_like(sin_half) * 2.0, angle / sin_half)
    return xyz * scale


def obs_position_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """位置误差 (3D)：目标位置 - 当前末端位置（世界系），单位 m。shape (num_envs, 3)。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return des_pos_w - curr_pos_w


def obs_orientation_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """姿态误差 (3D 轴角)：从当前末端到目标姿态的旋转，轴角表示 (弧度)。
    向量方向=旋转轴，模长=旋转角。shape (num_envs, 3)。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    # 误差四元数：从当前到目标 = des * curr^{-1}
    err_quat = quat_mul(des_quat_w, _quat_inv(curr_quat_w))
    return _quat_to_axis_angle(err_quat)


def last_action_scaled(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """返回上一步动作的 processed_actions，即已乘以 scale 的实际关节偏移量（rad）。

    与 joint_pos_rel 尺度一致：
    - joint_pos_rel: 当前关节位置 - 默认位置 (rad)
    - last_action_scaled: 上一步期望的关节偏移量 (rad) = raw_action × scale

    这样策略可以直接比较"上一步想去哪"和"现在在哪"。
    shape (num_envs, action_dim)，如 (N, 7)。
    """
    term = env.action_manager.get_term(action_name)
    return term.processed_actions


def obs_keypoint_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    keypoint_scale: float = 1.0,
    add_cube_center_kp: bool = True,
) -> torch.Tensor:
    """相对 SE(3) 几何误差：当前与目标末端对应关键点的 L2 距离，与关键点奖励一致。

    无绝对世界位姿，仅相对几何。shape (num_envs, num_keypoints)，如 (N, 7)。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return compute_keypoint_distance(
        curr_pos_w, curr_quat_w, des_pos_w, des_quat_w,
        keypoint_scale=keypoint_scale, add_cube_center_kp=add_cube_center_kp,
    )
