# Nero Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

This repository provides simulation and learning environments for the **OpenArm robotic platform**, built on **NVIDIA Isaac Sim** and **Isaac Lab**.
It enables research and development in **reinforcement learning (RL)**, **imitation learning (IL)**, **teleoperation**, and **sim-to-real transfer** for both **unimanual (single-arm)** and **bimanual (dual-arm)** robotic systems.

### What this repo offers
- **Isaac Sim models** for OpenArm robots.
- **Isaac Lab training environments** for RL tasks (reach, lift a cube, open a drawer).
- **Imitation learning**, **teleoperation interfaces**, and **sim-to-sim / sim-to-real transfer pipelines** are currently under development and will be available soon.

This repository has been tested with:
- **Ubuntu 22.04**
- **Isaac Sim v5.1.0**
- **Isaac Lab v2.3.0**
- **Python 3.11**

---

## Index
- [OpenArm Isaac Lab](#openarm-isaac-lab)
  - [Overview](#overview)
    - [What this repo offers](#what-this-repo-offers)
  - [Index](#index)
  - [Installation Guide](#installation-guide)
    - [(Option 1) Docker installation (linux only)](#option-1-docker-installation-linux-only)
    - [(Option 2) Local installation](#option-2-local-installation)
  - [Reinforcement Learning (RL)](#reinforcement-learning-rl)
    - [Training Model](#training-model)
    - [Replay Trained Model](#replay-trained-model)
    - [Analyze logs](#analyze-logs)
  - [Sim2sim](#sim2sim)
  - [Sim2Real Deployment using OpenArm](#sim2real-deployment-using-openarm)
  - [Related links](#related-links)
  - [License](#license)
  - [Code of Conduct](#code-of-conduct)

## Installation Guide


### Local installation
根据OPENARM单双臂构建的isaaclab ppo rl训练库 实现 单双臂 reach功能 

1. Clone git at your HOME directory
```bash
cd ~
git clone git@github.com:enactic/openarm_isaac_lab.git
```

2. Activate your virtual env which contains Isaac Lab package
```bash
conda activate env_isaaclab
```

3. Install python package with
```bash
cd openarm_isaac_lab
python -m pip install -e source/bi_nero
```

4. With this command, you can verify that OpenArm package has been properly installed and check all the environments where it can be executed.
```bash
python ./scripts/tools/list_envs.py
```

### Training Model

```bash
python ./scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-BiNero-Bi-v0 --headless
```

### Replay Trained Model

```bash
python ./scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Reach-BiNero-Bi-v0 --num_envs 8
```

### Analyze logs

```bash
python -m tensorboard.main --logdir=logs
```

And open the google and go to `http://localhost:6006/`

## Related links

* Read the [documentation](https://docs.openarm.dev/)
* Join the community on [Discord](https://discord.gg/FsZaZ4z3We)
* Contact us through <openarm@enactic.ai>

## License

[Apache License 2.0](LICENSE.txt)

Copyright 2025 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
