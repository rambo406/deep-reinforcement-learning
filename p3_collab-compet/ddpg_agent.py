import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.01
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
UPDATE_EVERY = 2
NUM_UPDATES = 3

# Device selection: MPS > CPU (no CUDA)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model with LayerNorm for MPS compatibility."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.ln1 = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.ln2 = nn.LayerNorm(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model with LayerNorm for MPS compatibility."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.ln1 = nn.LayerNorm(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.ln2 = nn.LayerNorm(fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.relu(self.ln1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class DDPGAgent:
    """Shared DDPG Agent — one agent used for both Tennis players."""

    def __init__(self, state_size=24, action_size=2, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.t_step = 0

        # Actor networks
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic networks
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Hard copy weights to targets
        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)

        # Noise process
        self.noise = OUNoise(action_size, seed)
        self.noise_scale = 1.0

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def act(self, state, noise_scale=None, add_noise=True):
        """Returns action for a single agent's observation (24-dim)."""
        if noise_scale is None:
            noise_scale = self.noise_scale
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state_t).cpu().data.numpy().squeeze(0)
        self.actor_local.train()
        if add_noise:
            action += noise_scale * self.noise.sample()
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        """Store experience and learn periodically."""
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1

        if self.t_step % UPDATE_EVERY == 0 and len(self.memory) > BATCH_SIZE * 10:
            for _ in range(NUM_UPDATES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """Update actor and critic using sampled experiences."""
        states, actions, rewards, next_states, dones = experiences

        # --- Update Critic --- #
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- Update Actor --- #
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft Update Targets --- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def reset(self):
        self.noise.reset()

    def save(self, actor_path='checkpoint_shared_actor.pth', critic_path='checkpoint_shared_critic.pth'):
        torch.save(self.actor_local.state_dict(), actor_path)
        torch.save(self.critic_local.state_dict(), critic_path)

    def load(self, actor_path='checkpoint_shared_actor.pth', critic_path='checkpoint_shared_critic.pth'):
        self.actor_local.load_state_dict(
            torch.load(actor_path, map_location=device, weights_only=True))
        self.critic_local.load_state_dict(
            torch.load(critic_path, map_location=device, weights_only=True))


class OUNoise:
    """Ornstein-Uhlenbeck process with decay support."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [np.random.standard_normal() for _ in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size replay buffer for single-agent DDPG experiences."""

    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
