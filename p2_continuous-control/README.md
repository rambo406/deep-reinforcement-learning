# Project 2: Continuous Control

![Trained Agent](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)

## Environment Description

This project uses the **Reacher** Unity environment, where a double-jointed arm must move to and maintain its position at a target location.

- **State Space**: 33 continuous variables (position, rotation, velocity, and angular velocities of the arm)
- **Action Space**: 4 continuous actions (torque applied to two joints), each value between -1 and 1
- **Reward**: +0.1 for each timestep the agent's hand is in the goal location

### Solving Criteria

Two environment versions are provided:

| Version | Agents | Solving Criteria |
|---------|--------|-----------------|
| **Version 1** | 1 agent | Average score ≥ +30 over 100 consecutive episodes |
| **Version 2** | 20 agents | Average score ≥ +30 over 100 consecutive episodes (averaged across all 20 agents) |

## Algorithm

This project implements **DDPG (Deep Deterministic Policy Gradient)** with the following enhancements:
- Batch normalization in actor and critic networks
- Gradient clipping on the critic network
- Learning every 20 steps with 10 update passes per learning step
- Ornstein-Uhlenbeck noise for exploration

See [Report.md](Report.md) for a detailed description of the algorithm and results.

## Project Structure

```
p2_continuous-control/
├── model.py                  # Actor and Critic neural network architectures
├── ddpg_agent.py             # DDPG Agent, OUNoise, ReplayBuffer
├── train.py                  # Command-line training script (both versions)
├── Continuous_Control.ipynb  # Jupyter notebook for interactive training
├── Report.md                 # Detailed project report
├── README.md                 # This file
├── Reacher_Version_1/        # Unity environment (single agent)
│   └── Reacher.exe
└── Reacher_Version_2/        # Unity environment (20 agents)
    └── Reacher.exe
```

## Getting Started

### Dependencies

- Python 3.6+
- PyTorch (with CUDA support for GPU training — tested on GTX 1070)
- NumPy
- Matplotlib
- Unity ML-Agents (`unityagents`)

Install the Unity ML-Agents package from the course repository:

```bash
cd ../python
pip install .
```

### Unity Environments

The Unity environments are already included:
- Version 1 (single agent): `Reacher_Version_1/Reacher.exe`
- Version 2 (20 agents): `Reacher_Version_2/Reacher.exe`

## Training Instructions

### Option 1: Command Line (train.py)

```bash
# Train Version 1 (single agent)
python train.py --version 1 --n_episodes 300 --max_t 1000

# Train Version 2 (20 agents)
python train.py --version 2 --n_episodes 300 --max_t 1000
```

### Option 2: Jupyter Notebook

Open `Continuous_Control.ipynb` and follow the sections to train interactively. The notebook supports both Version 1 and Version 2 training.

```bash
jupyter notebook Continuous_Control.ipynb
```

### GPU Note

This project was developed and tested on an NVIDIA GTX 1070 GPU. The DDPG agent automatically uses CUDA if available. Training Version 2 (20 agents) benefits significantly from GPU acceleration due to the larger batch of experiences collected per step.

## Results

After training, the following files are generated:
- `checkpoint_actor.pth` / `checkpoint_critic.pth` — Final model weights
- `checkpoint_actor_v1.pth` / `checkpoint_critic_v1.pth` — Version-specific weights
- `scores_plot_v1.png` / `scores_plot_v2.png` — Training reward plots
- `scores_v1.npy` / `scores_v2.npy` — Raw scores for further analysis

