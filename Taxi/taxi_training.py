import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import gymnasium as gym
from tqdm import tqdm

from configs import configs

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
    
    episodes = np.arange(params.total_episodes)

    learner.reset_qtable()
    explorer.epsilon = params.epsilon_start

    for ep in tqdm(episodes, desc=f"{config_name}", leave=True):
        eps_log[ep] = explorer.epsilon
        q_prev = learner.qtable.copy()

        # No seed - random starting position each episode
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

def plot_multi_config_comparison(results_dict, save_dir='saved_agents'):
    """Plot overlay comparison of multiple configurations as separate files."""
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    # 1. Smoothed rewards
    plt.figure(figsize=(10, 6))
    for idx, (name, data) in enumerate(results_dict.items()):
        smoothed = moving_average(data['rewards'], window=400)
        plt.plot(smoothed, label=name, color=colors[idx], linewidth=2.5, alpha=0.8)
    plt.title("Smoothed Reward per Episode (window=400)", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Smoothed Reward", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/1_smoothed_rewards.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Episode lengths
    plt.figure(figsize=(10, 6))
    for idx, (name, data) in enumerate(results_dict.items()):
        smoothed_lengths = moving_average(data['lengths'], window=400)
        plt.plot(smoothed_lengths, label=name, color=colors[idx], linewidth=2.5, alpha=0.8)
    plt.title("Smoothed Episode Lengths (window=400)", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Steps", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/2_episode_lengths.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Epsilon decay
    plt.figure(figsize=(10, 6))
    for idx, (name, data) in enumerate(results_dict.items()):
        plt.plot(data['epsilons'], label=name, color=colors[idx], linewidth=2.5, alpha=0.8)
    plt.title("Epsilon Decay", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Epsilon", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/3_epsilon_decay.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Q-value changes (convergence)
    plt.figure(figsize=(10, 6))
    for idx, (name, data) in enumerate(results_dict.items()):
        smoothed_q = moving_average(data['q_changes'], window=400)
        plt.plot(smoothed_q, label=name, color=colors[idx], linewidth=2.5, alpha=0.8)
    plt.title("Smoothed Q-value Changes (window=400)", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Max Q-change", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/4_q_value_changes.png', dpi=300, bbox_inches='tight')
    plt.show()


def rolling_success_rate(rewards, window=100, success_threshold=0):
    """Calculate rolling success rate based on reward threshold."""
    success = (rewards > success_threshold).astype(float)
    if len(success) < window:
        return np.cumsum(success) / (np.arange(len(success)) + 1)
    kernel = np.ones(window) / window
    return np.convolve(success, kernel, mode='same')


def plot_advanced_analysis(results_dict, save_dir='saved_agents'):
    """Generate comprehensive analysis plots for final report as separate files."""
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    config_names = list(results_dict.keys())
    
    # 1. Smoothed learning curves with confidence intervals
    plt.figure(figsize=(10, 6))
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['rewards']
        smoothed = moving_average(rewards, window=400)
        
        # Calculate rolling std for confidence band
        rewards_padded = np.pad(rewards, (200, 200), mode='edge')
        rolling_std = np.array([np.std(rewards_padded[i:i+400]) for i in range(len(rewards))])
        
        plt.plot(smoothed, label=name, color=colors[idx], linewidth=2.5)
        plt.fill_between(range(len(smoothed)), 
                         smoothed - rolling_std, 
                         smoothed + rolling_std,
                         color=colors[idx], alpha=0.2)
    
    plt.title('Learning Curves with ±1 Std Dev', fontsize=14, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Smoothed Reward', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/5_learning_curves_confidence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Boxplot of final-window rewards (last 2000 episodes)
    plt.figure(figsize=(10, 6))
    final_rewards = []
    labels = []
    for name, data in results_dict.items():
        final_window = data['rewards'][-2000:]
        final_rewards.append(final_window)
        labels.append(name)
    
    bp = plt.boxplot(final_rewards, labels=labels, patch_artist=True,
                     medianprops=dict(color='red', linewidth=2),
                     showfliers=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    plt.title('Final 2000 Episodes Reward Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Reward', fontsize=12)
    plt.xticks(rotation=15)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/6_boxplot_final_rewards.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Convergence plot: episodes-to-threshold
    plt.figure(figsize=(10, 6))
    threshold = 0 
    window_size = 1000
    episodes_to_converge = []
    
    for name, data in results_dict.items():
        rewards = data['rewards']
        rolling_mean = moving_average(rewards, window=window_size)
        converged = np.where(rolling_mean > threshold)[0]
        
        if len(converged) > 0:
            episodes_to_converge.append(converged[0])
        else:
            episodes_to_converge.append(len(rewards))
    
    bars = plt.bar(range(len(config_names)), episodes_to_converge, color=colors, alpha=0.7, edgecolor='black')
    plt.xticks(range(len(config_names)), config_names, rotation=15, ha='right')
    plt.title(f'Episodes to Convergence (threshold={threshold})', fontsize=14, fontweight='bold')
    plt.ylabel('Episodes', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, episodes_to_converge):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/7_convergence_speed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Q-change convergence (log scale) - smoothed and downsampled
    plt.figure(figsize=(10, 6))
    downsample_factor = 10  # Plot every 10th point
    for idx, (name, data) in enumerate(results_dict.items()):
        q_changes = data['q_changes']
        # Smooth first to reduce noise
        q_smoothed = moving_average(q_changes, window=200)
        q_changes_safe = np.maximum(q_smoothed, 1e-18)
        # Downsample for faster rendering
        plt.plot(q_changes_safe[::downsample_factor], label=name, color=colors[idx], linewidth=2.5, alpha=0.8)
    
    plt.title('Q-value Changes (Convergence Stability)', fontsize=14, fontweight='bold')
    plt.xlabel('Episode (downsampled)', fontsize=12)
    plt.ylabel('Smoothed Max Q-change (log scale)', fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/8_q_change_stability.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Success rate rolling plot
    plt.figure(figsize=(10, 6))
    for idx, (name, data) in enumerate(results_dict.items()):
        success_rate = rolling_success_rate(data['rewards'], window=200, success_threshold=0)
        plt.plot(success_rate, label=name, color=colors[idx], linewidth=2.5, alpha=0.8)
    
    plt.title('Rolling Success Rate (window=200)', fontsize=14, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/9_success_rate.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Violin plot alternative for final performance
    plt.figure(figsize=(10, 6))
    positions = range(len(config_names))
    parts = plt.violinplot(final_rewards, positions=positions, showmeans=True, showmedians=True)
    
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    
    plt.xticks(positions, labels, rotation=15, ha='right')
    plt.title('Final Performance Distribution (Violin Plot)', fontsize=14, fontweight='bold')
    plt.ylabel('Reward', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/10_violin_final_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_heatmap_grid_search(results_dict, save_dir='saved_agents'):
    """Generate heatmap for grid search results (learning_rate vs gamma)."""
    
    # Extract learning rates and gammas from configs
    lr_gamma_rewards = {}
    
    for name, data in results_dict.items():
        params = data['params']
        lr = params.learning_rate
        gamma = params.gamma
        # Use mean of last 10k episodes as final performance
        final_perf = np.mean(data['rewards'][-10000:])
        
        if lr not in lr_gamma_rewards:
            lr_gamma_rewards[lr] = {}
        lr_gamma_rewards[lr][gamma] = final_perf
    
    # Create grid
    learning_rates = sorted(lr_gamma_rewards.keys())
    gammas = sorted(set(g for lr_dict in lr_gamma_rewards.values() for g in lr_dict.keys()))
    
    # Build matrix
    matrix = np.full((len(learning_rates), len(gammas)), np.nan)
    for i, lr in enumerate(learning_rates):
        for j, gamma in enumerate(gammas):
            if gamma in lr_gamma_rewards.get(lr, {}):
                matrix[i, j] = lr_gamma_rewards[lr][gamma]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', origin='lower')
    
    # Set ticks
    ax.set_xticks(range(len(gammas)))
    ax.set_yticks(range(len(learning_rates)))
    ax.set_xticklabels([f'{g:.3f}' for g in gammas])
    ax.set_yticklabels([f'{lr:.3f}' for lr in learning_rates])
    
    # Labels
    ax.set_xlabel('Gamma (γ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate (α)', fontsize=12, fontweight='bold')
    ax.set_title('Grid Search: Mean Final Reward (Last 10k Episodes)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Reward', fontsize=11)
    
    # Add text annotations
    for i in range(len(learning_rates)):
        for j in range(len(gammas)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/11_heatmap_grid_search.png', dpi=300, bbox_inches='tight')
    plt.show()


# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":

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
    
    # Generate advanced analysis plots for report
    print("\nGenerating advanced analysis plots...")
    plot_advanced_analysis(results)
    
    # Generate heatmap if multiple learning rates and gammas
    print("\nGenerating grid search heatmap...")
    plot_heatmap_grid_search(results)

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
    
    

    print(f"\nTraining complete! All configs saved to saved_agents/")
    print(f"All plots saved as individual files (1-11) in saved_agents/")
