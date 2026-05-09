# Project 3: Collaboration and Competition — Report

## Overview

This project solves the **Tennis** Unity environment, where two agents control rackets to bounce a ball over a net. Each agent receives its own local observation of 8 variables corresponding to the position and velocity of the ball and racket. The environment stacks 3 consecutive observations, resulting in a state vector of 24 values per agent. Each agent produces 2 continuous actions (movement toward/away from the net and jumping).

**Reward structure**:
- **+0.1** if an agent hits the ball over the net
- **-0.01** if an agent lets the ball hit the ground or hits it out of bounds

**Solve criterion**: The environment is considered solved when the average score over 100 consecutive episodes reaches **+0.5**, where each episode's score is the **maximum** of the two agents' cumulative rewards.

## Learning Algorithm

This project uses **Shared DDPG (Deep Deterministic Policy Gradient)**, where **both agents share the same actor and critic networks**. This approach exploits the inherent symmetry of the Tennis environment: both agents have identical observation and action spaces, and the task is symmetric (both are trying to keep the ball in play). By sharing weights, both agents use the same policy, which halves the number of parameters and doubles the effective amount of training data per network update (each agent's experience trains the same network).

### How It Works

A single `DDPGAgent` instance controls both players. At each timestep, both agents observe their own local state (24 dimensions), pass it through the **same** actor network to select an action (2 dimensions), and store their individual transitions in a **shared replay buffer**. The critic evaluates state-action pairs from a single agent's perspective (no joint observations needed), making the approach simpler than multi-agent methods that require centralized critics.

### Key Techniques

**Self-Play with Shared Weights**: Both agents use identical network weights, creating a natural self-play dynamic. As the shared policy improves, both agents simultaneously become better opponents and better cooperators.

**Shared Replay Buffer**: Both agents store transitions into a single replay buffer (1M capacity). This effectively doubles the training data available per network update, improving sample efficiency.

**Soft Target Updates**: Target networks are updated slowly using Polyak averaging: $\theta_{target} = \tau \cdot \theta_{local} + (1 - \tau) \cdot \theta_{target}$ with $\tau = 0.01$.

**Layer Normalization**: Applied after each fully connected layer (before activation) in both actor and critic networks. LayerNorm normalizes across features within each sample, providing training stability without the batch-size sensitivity of BatchNorm — particularly useful for reinforcement learning where batch statistics can be noisy.

**Gradient Clipping**: The critic's gradients are clipped to a maximum norm of 1 during training, preventing large gradient updates that could destabilize learning.

**Ornstein-Uhlenbeck Noise with Decay**: Temporally correlated OU noise ($\theta = 0.15$, $\sigma = 0.2$) is added to the actor's output for exploration. The noise scale starts at 1.0 and decays by a factor of 0.9999 per episode (minimum 0.1), gradually shifting from exploration to exploitation.

**Delayed Learning**: The agent only performs learning updates every 2 timesteps (with 3 learning passes per update), allowing more diverse experiences to accumulate before updating.

## Model Architecture

### Actor Network (shared by both agents)

```
Input: agent observation (state_size=24)
  → Linear(24, 256) → LayerNorm(256) → ReLU
  → Linear(256, 128) → LayerNorm(128) → ReLU
  → Linear(128, 2) → Tanh
Output: actions (action_size=2, range [-1, 1])
```

### Critic Network (shared by both agents)

```
Input: agent observation (state_size=24)
  → Linear(24, 256) → LayerNorm(256) → ReLU
  → Concatenate with action (256 + 2 = 258)
  → Linear(258, 128) → LayerNorm(128) → ReLU
  → Linear(128, 1)
Output: Q-value (scalar)
```

**Weight initialization**:
- Hidden layers: Uniform distribution $[-\frac{1}{\sqrt{fan\_in}}, \frac{1}{\sqrt{fan\_in}}]$
- Output layers: Uniform distribution $[-3 \times 10^{-3}, 3 \times 10^{-3}]$

## Hyperparameters

| Hyperparameter | Value | Description |
|---------------|-------|-------------|
| BUFFER_SIZE | 1,000,000 | Shared replay buffer size |
| BATCH_SIZE | 256 | Mini-batch size |
| GAMMA | 0.99 | Discount factor |
| TAU | 0.01 | Soft update interpolation parameter |
| LR_ACTOR | 1e-4 | Actor learning rate (Adam) |
| LR_CRITIC | 1e-3 | Critic learning rate (Adam) |
| UPDATE_EVERY | 2 | Learn every N timesteps |
| NUM_UPDATES | 3 | Learning passes per update |
| noise_decay | 0.9999 | OU noise scale decay per episode |
| noise_min | 0.1 | Minimum noise scale |
| OU θ (theta) | 0.15 | OU noise mean reversion rate |
| OU σ (sigma) | 0.2 | OU noise volatility |
| Random seed | 42 | Random seed for reproducibility |

## Scoring Methodology

In the Tennis environment, each episode produces **two scores** — one per agent — representing the cumulative (undiscounted) reward each agent received during the episode. The **episode score** is defined as the **maximum** of these two agent scores:

$$\text{episode\_score} = \max(\text{score}_{\text{agent}_0}, \text{score}_{\text{agent}_1})$$

The environment is considered **solved** when the **average episode score over 100 consecutive episodes** reaches +0.5.

## Results

The Shared DDPG agent solved the environment in **1322 episodes** with a 100-episode average score of **0.5017**.

![Shared DDPG Training Scores](scores_shared_plot.png)

*The plot shows the per-episode score (max over both agents) and the 100-episode moving average. The red dashed line marks the +0.5 solve threshold.*

## Ideas for Future Work

1. **MADDPG (Multi-Agent DDPG)**: Implement centralized training with decentralized execution, where each agent has its own actor/critic and the critic receives joint observations and actions. This could improve performance in asymmetric or competitive multi-agent settings.

2. **Prioritized Experience Replay**: Instead of sampling uniformly from the replay buffer, prioritize experiences with higher TD error. This can significantly improve sample efficiency by focusing learning on the most informative transitions.

3. **Twin Delayed DDPG (TD3)**: Address overestimation bias by using twin critics, delayed policy updates, and target policy smoothing. TD3 could yield more stable and higher-performing policies.

4. **PPO with Centralized Critic (MAPPO)**: Replace the DDPG-based actor with PPO while retaining a centralized critic. MAPPO has shown strong performance in cooperative multi-agent tasks and avoids many of the stability issues inherent to off-policy methods.

5. **Parameter Space Noise**: Replace action-space noise (OU noise) with noise applied directly to the network parameters. This can provide more consistent and state-dependent exploration, which may be particularly beneficial in the symmetric self-play setting.

6. **Curriculum Learning**: Start training with an easier opponent (e.g., a scripted agent or a weaker policy) and gradually increase difficulty. This can help the agents escape early-training equilibria where neither agent learns to hit the ball, bootstrapping the learning process.
