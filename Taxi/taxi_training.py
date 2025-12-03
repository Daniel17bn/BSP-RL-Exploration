import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import gymnasium as gym
from tqdm import tqdm
from typing import NamedTuple

from params import Params

sns.set_theme()


# ===========================================================
# Q-LEARNING
# ===========================================================

class Qlearning:
    def __init__(self, lr, gamma, state_size, action_size):
        self.lr = lr
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.reset_qtable()

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))

    def update(self, state, action, reward, new_state):
        best_next = np.max(self.qtable[new_state])
        td_error = reward + self.gamma * best_next - self.qtable[state, action]
        self.qtable[state, action] += self.lr * td_error

# ===========================================================
# EPSILON-GREEDY POLICY
# ===========================================================

class EpsilonGreedy:
    def __init__(self, epsilon, epsilon_min, epsilon_decay, rng=None):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = rng if rng is not None else np.random.default_rng()

    def choose_action(self, action_space, state, qtable):
        if self.rng.random() < self.epsilon:
            return action_space.sample()
        max_ids = np.where(qtable[state] == np.max(qtable[state]))[0]
        return int(self.rng.choice(max_ids))

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ===========================================================
# TRAINING FUNCTION
# ===========================================================

def run_single_config(env, learner, explorer, params, config_name):
    """Train a single configuration and return logs."""
    rewards_log = np.zeros(params.total_episodes)
    lengths_log = np.zeros(params.total_episodes)
    eps_log = np.zeros(params.total_episodes)
    q_change_log = np.zeros(params.total_episodes)

    # Seed the action space for reproducibility
    env.action_space.seed(params.seed)
    
    episodes = np.arange(params.total_episodes)

    learner.reset_qtable()
    explorer.epsilon = params.epsilon_start

    for ep in tqdm(episodes, desc=f"{config_name}", leave=True):
        eps_log[ep] = explorer.epsilon
        q_prev = learner.qtable.copy()

        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = explorer.choose_action(env.action_space, state, learner.qtable)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            learner.update(state, action, reward, next_state)
            total_reward += reward
            steps += 1
            state = next_state

        rewards_log[ep] = total_reward
        lengths_log[ep] = steps
        q_change_log[ep] = np.max(np.abs(learner.qtable - q_prev))

        explorer.decay()

    return rewards_log, lengths_log, eps_log, q_change_log, learner.qtable

# ===========================================================
# MOVING AVERAGE
# ===========================================================

def moving_average(x, window=200):
    if window <= 1:
        return x
    return np.convolve(x, np.ones(window) / window, mode="same")

# ===========================================================
# PLOTTING
# ===========================================================

def plot_multi_config_comparison(results_dict):
    """Plot overlay comparison of multiple configurations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    # 1. Smoothed rewards
    ax = axes[0, 0]
    for idx, (name, data) in enumerate(results_dict.items()):
        smoothed = moving_average(data['rewards'], window=200)
        ax.plot(smoothed, label=name, color=colors[idx], linewidth=2, alpha=0.8)
    ax.set_title("Smoothed Reward per Episode (window=200)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Episode lengths
    ax = axes[0, 1]
    for idx, (name, data) in enumerate(results_dict.items()):
        smoothed_lengths = moving_average(data['lengths'], window=200)
        ax.plot(smoothed_lengths, label=name, color=colors[idx], linewidth=2, alpha=0.8)
    ax.set_title("Smoothed Episode Lengths (window=200)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Epsilon decay
    ax = axes[1, 0]
    for idx, (name, data) in enumerate(results_dict.items()):
        ax.plot(data['epsilons'], label=name, color=colors[idx], linewidth=2, alpha=0.8)
    ax.set_title("Epsilon Decay", fontsize=12, fontweight='bold')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Q-value changes (convergence)
    ax = axes[1, 1]
    for idx, (name, data) in enumerate(results_dict.items()):
        smoothed_q = moving_average(data['q_changes'], window=200)
        ax.plot(smoothed_q, label=name, color=colors[idx], linewidth=2, alpha=0.8)
    ax.set_title("Smoothed Q-value Changes (window=200)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Max Q-change")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('saved_agents/config_comparison.png', dpi=200, bbox_inches='tight')
    plt.show()

# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":
    # Define multiple configurations to compare
    configs = [
        {
            "name": "Config-1 (baseline)",
            "params": Params(
                total_episodes=50_000,
                learning_rate=0.1,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_decay=0.99995,
                epsilon_min=0.05,
                seed=42,
                is_rainy=False,
                fickle_passenger=False,
                n_runs=1,
                action_size=6,
                state_size=500,
            )
        },
        {
            "name": "Config-2 (faster decay)",
            "params": Params(
                total_episodes=50_000,
                learning_rate=0.1,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_decay=0.9999,
                epsilon_min=0.05,
                seed=42,
                is_rainy=False,
                fickle_passenger=False,
                n_runs=1,
                action_size=6,
                state_size=500,
            )
        },
        {
            "name": "Config-3 (higher LR)",
            "params": Params(
                total_episodes=50_000,
                learning_rate=0.3,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_decay=0.99995,
                epsilon_min=0.05,
                seed=42,
                is_rainy=False,
                fickle_passenger=False,
                n_runs=1,
                action_size=6,
                state_size=500,
            )
        },
    ]

    # Train each configuration
    results = {}
    qtables_all = {}
    
    for idx, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Training {config['name']}")
        print(f"{'='*60}")
        
        params = config['params']
        
        # Set numpy random seed for reproducibility
        np.random.seed(params.seed)
        rng = np.random.default_rng(params.seed)

        env = gym.make(
            "Taxi-v3",
            is_rainy=params.is_rainy,
            fickle_passenger=params.fickle_passenger
        )

        learner = Qlearning(params.learning_rate, params.gamma, params.state_size, params.action_size)
        explorer = EpsilonGreedy(params.epsilon_start, params.epsilon_min, params.epsilon_decay, rng=rng)

        rewards, lengths, epsilons, q_changes, qtable = run_single_config(
            env, learner, explorer, params, config['name']
        )
        
        env.close()
        
        # Store results
        results[config['name']] = {
            'rewards': rewards,
            'lengths': lengths,
            'epsilons': epsilons,
            'q_changes': q_changes,
            'params': params,
        }
        qtables_all[config['name']] = qtable
    
    # Plot comparison across all configs
    print(f"\n{'='*60}")
    print("Generating comparison plots...")
    print(f"{'='*60}")
    plot_multi_config_comparison(results)

    # Save ALL configurations
    import os
    os.makedirs("saved_agents", exist_ok=True)
    
    for idx, config in enumerate(configs):
        config_name = config['name']
        safe_name = config_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        
        save_data = {
            "qtable": qtables_all[config_name],
            "params": results[config_name]['params'],
            "rewards": results[config_name]['rewards'],
            "lengths": results[config_name]['lengths'],
            "epsilons": results[config_name]['epsilons'],
            "q_changes": results[config_name]['q_changes'],
            "config_name": config_name,
        }
        
        save_path = f"saved_agents/{safe_name}.pkl"
        with open(save_path, "wb") as file:
            pickle.dump(save_data, file)
        
        print(f"Saved {config_name} to {save_path}")
    
    # Also save the first config as the default (for backward compatibility)
    first_config_name = configs[0]['name']
    save_data = {
        "qtable": qtables_all[first_config_name],
        "params": results[first_config_name]['params'],
        "rewards": results[first_config_name]['rewards'],
        "lengths": results[first_config_name]['lengths'],
        "epsilons": results[first_config_name]['epsilons'],
        "q_changes": results[first_config_name]['q_changes'],
        "config_name": first_config_name,
    }
    
    with open("saved_agents/taxi_qtable.pkl", "wb") as file:
        pickle.dump(save_data, file)

    print(f"\nTraining complete! All configs saved to saved_agents/")
    print(f"Default config ({first_config_name}) saved to saved_agents/taxi_qtable.pkl")
    print(f"Comparison plot saved to saved_agents/config_comparison.png")
