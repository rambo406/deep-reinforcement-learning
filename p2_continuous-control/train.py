"""Training script for DDPG on the Reacher continuous control environment.

Supports both Version 1 (single agent) and Version 2 (20 agents).

Usage:
    python train.py --version 1 --n_episodes 300 --max_t 1000
    python train.py --version 2 --n_episodes 300 --max_t 1000
"""

import argparse
import os
import sys
import numpy as np
from collections import deque

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
from ddpg_agent import Agent


def train_ddpg(env, agent, brain_name, num_agents, n_episodes=300, max_t=1000, version=1):
    """Train DDPG agent(s) in the Reacher environment.

    Params
    ======
        env: Unity environment
        agent: DDPG agent
        brain_name: brain name for the environment
        num_agents: number of agents in the environment
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        version (int): 1 for single agent, 2 for 20 agents

    Returns
    =======
        scores_all (list): scores from each episode
    """
    scores_all = []
    scores_window = deque(maxlen=100)
    solved = False
    solved_episode = None

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        scores = np.zeros(num_agents)

        for t in range(max_t):
            # Get actions for all agents
            if num_agents == 1:
                actions = agent.act(states)
            else:
                actions = agent.act(states)

            # Take action in environment
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # Agent step (handles both single and multi-agent)
            agent.step(states, actions, rewards, next_states, dones)

            states = next_states
            scores += np.array(rewards)

            if np.any(dones):
                break

        # Record score
        episode_score = np.mean(scores)
        agent.decay_noise()
        scores_all.append(episode_score)
        scores_window.append(episode_score)
        avg_score = np.mean(scores_window)

        # Print progress
        if i_episode % 10 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {avg_score:.2f}\tLast Score: {episode_score:.2f}')

        # Check if solved
        if avg_score >= 30.0 and len(scores_window) >= 100 and not solved:
            solved = True
            solved_episode = i_episode
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {avg_score:.2f}')
            # Save checkpoint when solved
            torch.save(agent.actor_local.state_dict(), f'checkpoint_actor_v{version}_solved.pth')
            torch.save(agent.critic_local.state_dict(), f'checkpoint_critic_v{version}_solved.pth')
            break  # Early stop — no need to continue training

    return scores_all, solved_episode


def plot_scores(scores, version, filename=None):
    """Plot and save training scores."""
    if filename is None:
        filename = f'scores_plot_v{version}.png'

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(1, len(scores) + 1), scores, label='Score per Episode', alpha=0.6)

    # Rolling average
    if len(scores) >= 100:
        rolling_avg = [np.mean(scores[max(0, i - 100):i + 1]) for i in range(len(scores))]
        ax.plot(np.arange(1, len(scores) + 1), rolling_avg, label='100-Episode Average', linewidth=2)

    ax.axhline(y=30.0, color='r', linestyle='--', label='Solved Threshold (+30)')
    ax.set_xlabel('Episode #')
    ax.set_ylabel('Score')
    ax.set_title(f'DDPG Training Scores - Reacher Version {version}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f'Scores plot saved to {filename}')


def main():
    parser = argparse.ArgumentParser(description='Train DDPG agent on Reacher environment')
    parser.add_argument('--version', type=int, default=1, choices=[1, 2],
                        help='Environment version: 1 (single agent) or 2 (20 agents)')
    parser.add_argument('--n_episodes', type=int, default=None,
                        help='Maximum number of training episodes (default: 1000 for v1, 300 for v2)')
    parser.add_argument('--max_t', type=int, default=1000,
                        help='Maximum number of timesteps per episode')
    parser.add_argument('--worker_id', type=int, default=0,
                        help='Worker ID for Unity environment (use different IDs for parallel envs)')
    args = parser.parse_args()

    # Default n_episodes based on version
    if args.n_episodes is None:
        args.n_episodes = 1000 if args.version == 1 else 300

    # Set environment path based on version
    if args.version == 1:
        env_path = 'Reacher_Version_1/Reacher.exe'
    else:
        env_path = 'Reacher_Version_2/Reacher.exe'

    print(f'Training DDPG on Reacher Version {args.version}')
    print(f'Environment path: {env_path}')
    print(f'Episodes: {args.n_episodes}, Max timesteps: {args.max_t}')
    print(f'Device: {torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}')
    print()

    # Create environment
    env = UnityEnvironment(file_name=env_path, worker_id=args.worker_id)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Get environment info
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]

    print(f'Number of agents: {num_agents}')
    print(f'State size: {state_size}')
    print(f'Action size: {action_size}')
    print()

    # Create agent
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=42, num_agents=num_agents)

    # Train
    scores, solved_episode = train_ddpg(
        env, agent, brain_name, num_agents,
        n_episodes=args.n_episodes,
        max_t=args.max_t,
        version=args.version
    )

    # Save final checkpoints
    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
    torch.save(agent.actor_local.state_dict(), f'checkpoint_actor_v{args.version}.pth')
    torch.save(agent.critic_local.state_dict(), f'checkpoint_critic_v{args.version}.pth')
    print('Final checkpoints saved.')

    # Save scores
    np.save(f'scores_v{args.version}.npy', np.array(scores))

    # Plot scores
    plot_scores(scores, args.version)

    # Summary
    print('\n' + '=' * 60)
    print('TRAINING SUMMARY')
    print('=' * 60)
    print(f'Version:          {args.version} ({"single agent" if args.version == 1 else "20 agents"})')
    print(f'Total Episodes:   {len(scores)}')
    print(f'Final Avg Score:  {np.mean(scores[-100:]):.2f} (last 100 episodes)')
    print(f'Best Avg Score:   {max([np.mean(scores[max(0,i-99):i+1]) for i in range(len(scores))]):.2f}')
    if solved_episode:
        print(f'Solved at:        Episode {solved_episode}')
    else:
        print(f'Solved:           Not yet (need avg >= 30 over 100 episodes)')
    print('=' * 60)

    env.close()


if __name__ == '__main__':
    main()
