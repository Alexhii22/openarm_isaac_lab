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

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_keypoint_offsets_full_6d(
    add_cube_center_kp: bool = True, device: torch.device | None = None
) -> torch.Tensor:
    """Keypoint offsets for pose alignment: axis-aligned points, pose aligned if all axes align.

    Returns:
        Keypoint offsets (num_keypoints, 3). With center: 7 points; without: 6 points.
    """
    if add_cube_center_kp:
        corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        corners = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    keypoints = torch.tensor(corners, device=device, dtype=torch.float32)
    keypoints = torch.cat((keypoints, -keypoints[-3:]), dim=0)
    return keypoints
#获取得到坐标系 正负关键点

def compute_keypoint_distance(
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    keypoint_scale: float = 1.0,
    add_cube_center_kp: bool = True,
) -> torch.Tensor:
    """L2 distance between corresponding keypoints of current and target poses.

    Keypoints are axis-aligned offsets transformed to world by each pose.
    Returns shape (num_envs, num_keypoints).
    """
    device = current_pos.device
    num_envs = current_pos.shape[0]
    offsets = get_keypoint_offsets_full_6d(add_cube_center_kp, device) * keypoint_scale
    num_kp = offsets.shape[0]
    identity_quat = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32
    ).unsqueeze(0).expand(num_envs, 4)

    keypoints_curr = torch.zeros((num_envs, num_kp, 3), device=device, dtype=current_pos.dtype)
    keypoints_tgt = torch.zeros((num_envs, num_kp, 3), device=device, dtype=target_pos.dtype)

    for i in range(num_kp):
        offset = offsets[i].unsqueeze(0).expand(num_envs, 3)
        keypoints_curr[:, i], _ = combine_frame_transforms(
            current_pos, current_quat, offset, identity_quat
        )
        keypoints_tgt[:, i], _ = combine_frame_transforms(
            target_pos, target_quat, offset, identity_quat
        )

    return torch.norm(keypoints_tgt - keypoints_curr, p=2, dim=-1)
#计算各点之间的误差

def keypoint_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    keypoint_scale: float = 1.0,
    add_cube_center_kp: bool = True,
) -> torch.Tensor:
    """Mean keypoint distance between current and desired pose (L2 per keypoint, then mean).

    Pose aligned if all axis-aligned keypoints align. Returns (num_envs,) for use as penalty.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # typeignore末端当前世界坐标位置
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # 末端当前世界坐标位置

    dist_sep = compute_keypoint_distance(
        curr_pos_w, curr_quat_w, des_pos_w, des_quat_w,
        keypoint_scale=keypoint_scale, add_cube_center_kp=add_cube_center_kp,
    )
    return dist_sep.mean(dim=-1)#当前与目标的关键点距离误差


def keypoint_command_reward_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    a: float = 10.0,
    b: float = 0.1,
    keypoint_scale: float = 1.0,
    add_cube_center_kp: bool = True,
) -> torch.Tensor:
    """Exponential keypoint reward: mean over keypoints of 1/(exp(a*d)+b+exp(-a*d)).
        距离 d 越小 → exp_per_kp 越接近 1

        距离 d 越大 → exp_per_kp 越接近 0，reward 越小      
    Returns (num_envs,) in (0, 1] when d small.
    """
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
    dist_sep = compute_keypoint_distance(
        curr_pos_w, curr_quat_w, des_pos_w, des_quat_w,
        keypoint_scale=keypoint_scale, add_cube_center_kp=add_cube_center_kp,
    )
    exp_per_kp = 1.0 / (
        torch.exp(a * dist_sep) + b + torch.exp(-a * dist_sep)
    )#指数平滑函数 
    return exp_per_kp.mean(dim=-1) 


def reach_success_sparse(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    pos_thresh: float = 0.01,
    orient_thresh_deg: float = 5.0,
) -> torch.Tensor:
    """稀疏成功奖励：位置误差 < pos_thresh(m)，姿态误差 < orient_thresh_deg(度) 时给 1，否则 0。

    用于课程：可在 num_steps 后通过 modify_reward_weight 将本项 weight 从 0 设为正值启用。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    pos_err = torch.norm(curr_pos_w - des_pos_w, dim=1)#绝对值平方差
    orient_err_rad = quat_error_magnitude(curr_quat_w, des_quat_w)
    orient_thresh_rad = orient_thresh_deg * math.pi / 180.0 #四元数误差
    success = (pos_err < pos_thresh) & (orient_err_rad < orient_thresh_rad)
    return success.float()


def orientation_error_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """显式姿态误差惩罚：四元数误差的弧度值，用于加强姿态学习信号。
    
    关键点方法虽能统一位姿，但姿态梯度较弱。此项显式惩罚姿态偏差。
    Returns (num_envs,) 标量，单位弧度。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def action_rate_penalty_when_reached(
    env: ManagerBasedRLEnv,
    command_name: str,
    body_asset_cfg: SceneEntityCfg,
    action_name: str,
) -> torch.Tensor:
    """仅当该臂达到 sparse success（位姿在阈值内）时，对该臂的 action_rate 做惩罚。
    左臂到达只惩罚左臂动作变化，右臂到达只惩罚右臂动作变化；未到达时返回 0。
    抖动多由策略输出动作频繁变化导致，惩罚 action_rate 更直接。"""
    reached = reach_success_sparse(
        env, command_name, body_asset_cfg, pos_thresh=0.01, orient_thresh_deg=5.0
    )
    term = env.action_manager.get_term(action_name)
    curr = term.raw_actions
    prev_full = env.action_manager.prev_action
    # 计算该 term 在 prev_action 中的起始索引（按 action manager 中 term 顺序）
    term_names = list(env.action_manager._terms.keys())
    idx = term_names.index(action_name)
    offset = sum(
        env.action_manager.get_term(n).action_dim for n in term_names[:idx]
    )
    dim = term.action_dim
    prev = prev_full[:, offset : offset + dim]
    rate_sq = torch.sum(torch.square(curr - prev), dim=1)
    return reached * rate_sq
