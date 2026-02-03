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


from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Position error: link7 in world vs target in world (L2 norm).

    Command stores target in robot base frame; we transform to world with root pose,
    then compare to link7's position in world. Both in world frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]  # target in robot base frame
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        des_pos_b,
    )  # target in world frame
    curr_pos_w = asset.data.body_pos_w[:, body_id]  # link7 in world frame
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Dense reward: 1 - tanh(distance/std), distance = link7_w vs target_w (world frame)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        des_pos_b,
    )
    curr_pos_w = asset.data.body_pos_w[:, body_id]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std) 

def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Orientation error in world frame: command quat in base frame -> world, vs link7 quat in world."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)  # target orientation in world
    curr_quat_w = asset.data.body_quat_w[:, body_id]  # link7 orientation in world
    return quat_error_magnitude(curr_quat_w, des_quat_w)




def reach_pose_success(
    env: ManagerBasedRLEnv,
    pos_threshold: float,
    rot_threshold: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Sparse reward: 1.0 when end-effector is within position AND orientation threshold, else 0.0.

    Encourages the policy to reach the target pose. Use with a positive weight (e.g. 10.0).

    Args:
        pos_threshold: Max position error in meters (e.g. 0.03 = 3cm).
        rot_threshold: Max orientation error in rad (e.g. 0.2 ≈ 11.5°).
    """
    pos_err = position_command_error(env, command_name, asset_cfg)
    rot_err = orientation_command_error(env, command_name, asset_cfg)
    success = (pos_err < pos_threshold) & (rot_err < rot_threshold)
    return success.float()

