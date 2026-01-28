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

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from isaaclab.utils import configclass


@configclass
class NEROReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64  #
    max_iterations = 1000
    save_interval = 50
    experiment_name = "nero_reach"
    run_name = ""
    resume = False
    empirical_normalization = True
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3, #降低初始探索噪声，提高学习稳定性
        actor_hidden_dims=[128, 128], #增加网络容量
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,   #值函数损失系数 未来与现在奖励对比
        use_clipped_value_loss=True,#剪裁机制 PPO核心
        clip_param=0.2,#稍微降低剪裁参数，提高更新幅度
        entropy_coef=0.001,#降低熵系数，减少随机探索
        num_learning_epochs=10, #增加学习轮次
        num_mini_batches=4,
        learning_rate=3.0e-4,    #提高学习率
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008, #稍微降低KL目标
        max_grad_norm=0.8,#降低梯度裁切
    )
    #调参指南：更新稳定性clip kl散度、学习速度 学习率 回合数、数据质量 step、探索强度 （entropy熵系数）

    #tensorboard内容： train/mean_reawrd 散度train/approx_kl  策略商loss/entropy  策略损失loss/policy_loss  
    #值函数损失loss/value_loss   1️⃣ 先看 KL，再看 reward  2️⃣ reward 不涨，先动 learning rate  3️⃣ 不稳定，优先降“更新幅度”，不是降 reward

#
#kl三度回答 新策略相对旧策略 到底变了多少 目前设置自适应0.008

    