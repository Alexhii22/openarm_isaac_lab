# Reach 奖励引用检查与双臂协同设计说明

## 一、Reward 引用检查结果

### 1. 命名与引用一致性 ✅
- **CurriculumCfg** 中 `term_name` 与 **RewardsCfg** 属性名一一对应：
  - `action_rate`, `left_joint_vel`, `right_joint_vel`, `left_reach_success_sparse`, `right_reach_success_sparse`
- **joint_pos_env_cfg** 中覆盖的 reward 项与 reach_env_cfg 中定义一致：
  - 关键点类：`left_keypoint_error`, `left_keypoint_reward_exp`, `right_keypoint_error`, `right_keypoint_reward_exp` → 均设置 `asset_cfg.body_names`
  - 稀疏成功：`left_reach_success_sparse`, `right_reach_success_sparse` → `asset_cfg.body_names`
  - 到位后惩罚：`left_action_rate_penalty_when_reached`, `right_action_rate_penalty_when_reached` → `body_asset_cfg.body_names`

### 2. 参数与 MDP 函数签名 ✅
- 关键点类 reward：`asset_cfg` + `command_name` + `keypoint_scale` + `add_cube_center_kp`（exp 还有 `a`, `b`）与 `rewards.py` 中 `keypoint_command_error` / `keypoint_command_reward_exp` 一致。
- `reach_success_sparse`：`asset_cfg` + `command_name` + `pos_thresh` + `orient_thresh_deg` 一致。
- `action_rate_penalty_when_reached`：`command_name` + `body_asset_cfg` + `action_name` 一致；子配置中正确使用 **body_asset_cfg** 而非 asset_cfg。

### 3. 唯一需注意点
- **ReachEnvCfg** 中所有带 `body_names=MISSING` 的 `SceneEntityCfg` 必须在 **BiNeroReachEnvCfg.__post_init__** 里被赋值为具体 body（如 `left_link7` / `right_link7`），否则运行时解析 `body_ids` 会失败。当前 joint_pos_env_cfg 已对上述 reward/obs 全部赋值，无遗漏。

---

## 二、双臂协同与“逆运动学解”设计

### 1. 当前设计可实现什么
- **双目标、双臂独立 reach**：左/右各自有 `left_ee_pose` / `right_ee_pose`，奖励与观测均为左/右独立（关键点距离、关节状态、last action）。  
- **等效为两个独立 IK 型任务**：策略学的是「给定当前状态 + 目标位姿（通过关键点距离与 command 隐含），输出关节位置」，即用 RL 近似两个独立的“末端位姿 → 关节角”映射，**可以实现**，且已在当前 reward/obs 设计下支持。

### 2. “双臂协同”的两种含义与对应设计
- **协同含义一：两臂同时到达各自目标（当前已支持）**  
  - 左/右独立 keypoint reward + 独立 sparse success，无显式耦合。  
  - 若希望“两臂都到才给大奖励”，可补充：**双臂联合 sparse**（例如仅当 `left_reach_success_sparse & right_reach_success_sparse` 时再给一项 bonus），或提高双臂都成功时的课程权重。
- **协同含义二：两臂相对位姿/相对运动约束（典型协同 IK）**  
  - 例如：保持左右末端相对位姿、或避免碰撞。  
  - 可补充：
    - **相对位姿 reward**：左末端相对右末端（或反过来的）位置/姿态误差的 reward；
    - **双臂距离/碰撞惩罚**：两末端或两臂连杆间最小距离 < 阈值时惩罚，使策略学会避障与协同构型。

### 3. 可补充/修改的点（建议）

| 序号 | 类型 | 建议 | 说明 |
|------|------|------|------|
| 1 | 稀疏成功与关键点一致 | 增加「关键点版」稀疏成功 | 当前 sparse 仍为「位置 + 姿态」阈值；若希望与 keypoint 奖励完全一致，可新增 `reach_success_sparse_keypoint`（如 mean keypoint dist < ε），并用于课程或替代现有 sparse。 |
| 2 | 双臂联合成功 | 可选 joint bonus | 当 `left_reach_success_sparse & right_reach_success_sparse` 时给额外 reward，鼓励“双臂同时到位”的协同。 |
| 3 | 双臂防碰撞/距离 | 可选 collision 或 min_dist 惩罚 | 两臂末端（或指定 body）间距离 < 安全距离时负奖励，避免自碰撞、更符合真实双臂 IK 约束。 |
| 4 | 相对位姿（强协同） | 可选 relative_pose reward | 若任务需要固定相对抓取等，可加左末端相对右末端的位姿误差 reward，使策略显式学相对位姿约束。 |
| 5 | 课程与权重 | 先单臂后双臂 | 若训练不稳，可考虑课程：前期仅左臂或仅右臂 keypoint 有权重，后期再同时启用双臂 + 可选 joint bonus / collision。 |

---

## 三、小结
- **Reward 引用**：命名、参数、body_names/body_asset_cfg 在 reach_env_cfg 与 joint_pos_env_cfg 中一致，无发现错误；仅需保证所有 `body_names=MISSING` 在子类 __post_init__ 中被正确赋值（当前已满足）。
- **双臂协同逆运动学**：当前设计可实现「双目标、双臂独立 reach」的 RL 近似 IK；若需更强协同（联合成功、防碰、相对位姿），可按上表在现有 reward 上做增量补充与课程设计。
