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

"""Reach rewards (dense, goal-aligned, weight-tunable).

设计原则（对应你的 4 条）：
- 目标对齐：核心就是末端与目标（command）的距离/姿态误差。
- 避免稀疏：提供 tanh kernel 的稠密奖励（远处也有梯度，近处更精细）。
- 惩罚冗余：动作变化、关节速度等惩罚项建议在 cfg 里用很小权重启用。
- 数值平衡：每个 term 单独输出，权重全部放在 env cfg（`reach_env_cfg.py`）里调。

注意：
- 末端点取 `asset.data.body_pos_w[:, body_id]`，即该 body（如 link7）坐标系原点在世界系的位置。
  若你希望奖励驱动“工具尖/TCP”，应在 IK action 里设置 `body_offset` 或在机器人 USD 里提供 TCP frame。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of position error using L2-norm (meters, world frame)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    Returns in [0, 1]: 1 - tanh(d / std), where d is the L2 position error in meters.
    """
    distance = position_command_error(env, command_name, asset_cfg)
    return 1.0 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path (radians, unitless magnitude).

    Uses `quat_error_magnitude(curr_quat_w, des_quat_w)`.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the orientation using a tanh kernel (returns in [0, 1])."""
    err = orientation_command_error(env, command_name, asset_cfg)
    return 1.0 - torch.tanh(err / std)


__all__ = [
    "position_command_error",
    "position_command_error_tanh",
    "orientation_command_error",
    "orientation_command_error_tanh",
]
