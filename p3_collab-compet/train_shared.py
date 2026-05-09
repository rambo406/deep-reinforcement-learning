import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
from unityagents import UnityEnvironment

from ddpg_agent import DDPGAgent


def train(n_episodes=5000, max_t=1000):
    """Train a shared DDPG agent in the Tennis environment.

    Both Tennis players share the same DDPGAgent (same actor/critic weights).
    """
    # Load environment
    env = UnityEnvironment(file_name="Tennis.app")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size

    print(f'Number of agents: {num_agents}')
    print(f'State size: {state_size}')
    print(f'Action size: {action_size}')

    # Single shared agent
    agent = DDPGAgent(state_size=state_size, action_size=action_size, seed=42)

    scores_all = []
    scores_window = deque(maxlen=100)
    best_avg_score = -np.inf
    solved = False
    noise_scale = 1.0

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations  # (2, 24)
        agent.reset()
        scores = np.zeros(num_agents)

        for t in range(max_t):
            # Both agents use the same shared agent
            actions = np.array([agent.act(states[i], noise_scale=noise_scale)
                                for i in range(num_agents)])

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # Store both agents' experiences into the same buffer
            for i in range(num_agents):
                agent.step(states[i], actions[i], rewards[i], next_states[i], float(dones[i]))

            states = next_states
            scores += rewards

            if any(dones):
                break

        # Episode score = max of both agents
        episode_score = np.max(scores)
        scores_all.append(episode_score)
        scores_window.append(episode_score)
        avg_score = np.mean(scores_window)

        # Decay noise
        noise_scale = max(0.1, noise_scale * 0.9999)

        print(f'\rEpisode {i_episode}\tAverage Score: {avg_score:.4f}\tNoise: {noise_scale:.4f}', end='')
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {avg_score:.4f}\tNoise: {noise_scale:.4f}')

        # Save best checkpoint
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            agent.save()

        if avg_score >= 0.5 and not solved:
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {avg_score:.4f}')
            solved = True
            agent.save()
            break

    if not solved:
        print(f'\nTraining complete. Best average score: {best_avg_score:.4f}')

    # Save scores
    scores_arr = np.array(scores_all)
    np.save('scores_shared.npy', scores_arr)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(1, len(scores_arr) + 1), scores_arr, alpha=0.3, label='Score')
    if len(scores_arr) >= 100:
        moving_avg = np.convolve(scores_arr, np.ones(100) / 100, mode='valid')
        ax.plot(np.arange(100, len(scores_arr) + 1), moving_avg, label='100-episode avg')
    ax.axhline(y=0.5, color='r', linestyle='--', label='Target (0.5)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Shared DDPG Tennis Training')
    ax.legend()
    fig.savefig('scores_shared_plot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved scores_shared_plot.png and scores_shared.npy')

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train shared DDPG agent for Tennis')
    parser.add_argument('--n_episodes', type=int, default=5000,
                        help='maximum number of training episodes (default: 5000)')
    parser.add_argument('--max_t', type=int, default=1000,
                        help='maximum number of timesteps per episode (default: 1000)')
    args = parser.parse_args()

    train(n_episodes=args.n_episodes, max_t=args.max_t)
