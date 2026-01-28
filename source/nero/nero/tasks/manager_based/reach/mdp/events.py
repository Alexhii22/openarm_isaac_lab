from __future__ import annotations

from typing import Dict, Tuple

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def apply_joint_position_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    joint_limits: Dict[str, Tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply explicit joint position limits to the simulator.

    This is used to override USD limits (e.g., when URDF limits were updated
    but the USD was not regenerated yet).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)
    # Use provided joint_names if available; otherwise fall back to dict keys.
    joint_name_keys = asset_cfg.joint_names if asset_cfg.joint_names is not None else list(joint_limits.keys())
    joint_ids, joint_names = asset.find_joints(joint_name_keys, preserve_order=True)

    limits = torch.zeros((len(env_ids), len(joint_ids), 2), device=asset.device)
    for idx, joint_name in enumerate(joint_names):
        if joint_name not in joint_limits:
            raise ValueError(f"Missing joint limit for '{joint_name}'. Available keys: {list(joint_limits)}")
        lower, upper = joint_limits[joint_name]
        limits[:, idx, 0] = lower
        limits[:, idx, 1] = upper

    asset.write_joint_position_limit_to_sim(
        limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
    )
