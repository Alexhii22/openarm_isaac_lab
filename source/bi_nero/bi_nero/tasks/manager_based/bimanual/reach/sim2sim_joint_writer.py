# Copyright 2025 Enactic, Inc.
# sim2sim: 将 policy 输出的 actions 解码为 Bi-Nero 关节位置（rad）并写入文件，供 Gazebo/ROS 桥接读取。

from __future__ import annotations

import numpy as np
import torch

# Bi-Nero 默认关节位置（与 joint_pos_env_cfg 一致）
DEFAULT_LEFT = np.array([1.6, 1.2, 0.52, 0.52, -0.6, 0.0, 0.0], dtype=np.float32)
DEFAULT_RIGHT = np.array([-1.6, 1.2, -0.52, 0.52, 0.6, 0.0, 0.0], dtype=np.float32)
SCALE = 0.5


def write_joints_to_file_from_actions(actions: torch.Tensor, filepath: str) -> None:
    """将 policy 输出的 actions 解码为关节位置，写入文件（每行 14 个值：left7 + right7）。"""
    if actions is None or actions.numel() == 0:
        return
    a = actions.detach().cpu().numpy()
    if a.ndim == 1:
        a = a.reshape(1, -1)
    # action 顺序：left_arm_action(7) + right_arm_action(7)
    left_action = a[0, :7]
    right_action = a[0, 7:14]
    joint_pos = np.concatenate([
        DEFAULT_LEFT + SCALE * left_action,
        DEFAULT_RIGHT + SCALE * right_action,
    ])
    with open(filepath, "w") as f:
        f.write(" ".join(f"{x:.6f}" for x in joint_pos) + "\n")
