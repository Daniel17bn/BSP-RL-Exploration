"""
Generate comparison plots from saved training data.
Run this AFTER all SLURM jobs complete to visualize results.

"""

import pickle
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from configs import configs

# ===========================================================
# PLOTTING FUNCTIONS
# ===========================================================

def moving_average(x, window=100):
    """Calculate moving average for smoothing plots."""
    if len(x) < window:
        window = len(x)
    if window <= 1:
        return x
    return np.convolve(x, np.ones(window) / window, mode='valid')

def plot_training_results(results_dict, save_dir='saved_agents/plots'):
    """Plot comprehensive training results for deep RL research."""
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    # 1. Episode rewards with confidence intervals
    plt.figure(figsize=(12, 6))
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        if len(rewards) > 100:
            smoothed = moving_average(rewards, window=100)
            episodes = np.arange(len(smoothed))
            
            # Calculate rolling std for confidence band
            rewards_padded = np.pad(rewards, (50, 50), mode='edge')
            rolling_std = np.array([np.std(rewards_padded[i:i+100]) for i in range(len(rewards))])
            rolling_std = moving_average(rolling_std, window=100)
            
            plt.plot(episodes, smoothed, label=name, color=colors[idx], linewidth=2.5, alpha=0.9)
            plt.fill_between(episodes, smoothed - rolling_std[:len(smoothed)], 
                           smoothed + rolling_std[:len(smoothed)],
                           color=colors[idx], alpha=0.15)
    plt.title("Episode Rewards with Confidence Intervals", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/1_rewards_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Learning curves (raw + smoothed)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Learning Curves Analysis', fontsize=16, fontweight='bold')
    
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        
        # Raw rewards
        axes[0, 0].plot(rewards, label=name, color=colors[idx], alpha=0.4, linewidth=0.8)
        # Smoothed rewards
        if len(rewards) > 100:
            smoothed = moving_average(rewards, window=100)
            axes[0, 1].plot(smoothed, label=name, color=colors[idx], linewidth=2)
        # Cumulative average
        axes[1, 0].plot(np.cumsum(rewards) / (np.arange(len(rewards)) + 1), 
                       label=name, color=colors[idx], linewidth=2)
        # Rolling max
        rolling_max = np.maximum.accumulate(rewards)
        axes[1, 1].plot(rolling_max, label=name, color=colors[idx], linewidth=2)
    
    axes[0, 0].set_title('Raw Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Smoothed Rewards (window=100)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Cumulative Average Reward')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Avg Reward')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Best Reward Achieved (Rolling Max)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Max Reward')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/2_learning_curves_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Episode lengths and survival time
    plt.figure(figsize=(12, 6))
    for idx, (name, data) in enumerate(results_dict.items()):
        lengths = data['episode_lengths']
        if len(lengths) > 100:
            smoothed = moving_average(lengths, window=100)
            episodes = np.arange(len(smoothed))
            plt.plot(episodes, smoothed, label=name, color=colors[idx], linewidth=2, alpha=0.8)
    plt.title("Episode Survival Time (Steps per Episode)", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Steps", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/3_episode_lengths.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Training loss convergence
    plt.figure(figsize=(12, 6))
    for idx, (name, data) in enumerate(results_dict.items()):
        losses = data['losses']
        if len(losses) > 100:
            smoothed = moving_average(losses, window=100)
            steps = np.arange(len(smoothed))
            plt.plot(steps, smoothed, label=name, color=colors[idx], linewidth=2, alpha=0.8)
    plt.title("TD Loss Convergence", fontsize=14, fontweight='bold')
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Loss (Smooth L1)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/4_loss_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Exploration rate (epsilon) decay
    plt.figure(figsize=(12, 6))
    for idx, (name, data) in enumerate(results_dict.items()):
        epsilons = data['episode_epsilons']
        plt.plot(epsilons, label=name, color=colors[idx], linewidth=2, alpha=0.8)
    plt.title("Exploration Rate (ε) Decay Over Training", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Epsilon (ε)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/5_epsilon_decay.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Final performance comparison (boxplot)
    plt.figure(figsize=(10, 6))
    final_rewards = []
    labels = []
    for name, data in results_dict.items():
        # Last 20% of episodes
        final_window = data['episode_rewards'][-int(len(data['episode_rewards']) * 0.2):]
        final_rewards.append(final_window)
        labels.append(name)
    
    bp = plt.boxplot(final_rewards, labels=labels, patch_artist=True,
                     medianprops=dict(color='red', linewidth=2),
                     showfliers=True, notch=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    plt.title('Final Performance Distribution (Last 20% Episodes)', fontsize=14, fontweight='bold')
    plt.ylabel('Reward', fontsize=12)
    plt.xlabel('Configuration', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/6_final_performance_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Sample efficiency comparison
    plt.figure(figsize=(12, 6))
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        # Cumulative sum of rewards
        cumulative_rewards = np.cumsum(rewards)
        plt.plot(cumulative_rewards, label=name, color=colors[idx], linewidth=2, alpha=0.8)
    plt.title("Sample Efficiency (Cumulative Reward)", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Cumulative Reward", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/7_sample_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Performance metrics table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    metrics_data = []
    for name, data in results_dict.items():
        rewards = data['episode_rewards']
        final_window = rewards[-int(len(rewards) * 0.2):]
        
        metrics_data.append([
            name,
            f"{np.mean(final_window):.2f} ± {np.std(final_window):.2f}",
            f"{np.max(rewards):.1f}",
            f"{np.mean(rewards):.2f}",
            f"{len([r for r in final_window if r > 0]) / len(final_window) * 100:.1f}%",
            f"{len(rewards)}"
        ])
    
    table = ax.table(cellText=metrics_data,
                    colLabels=['Config', 'Final Avg±Std', 'Best', 'Overall Avg', 'Success Rate', 'Episodes'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.15, 0.1, 0.12, 0.12, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(metrics_data) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    plt.title('Performance Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f'{save_dir}/8_metrics_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Training stability (variance over time)
    plt.figure(figsize=(12, 6))
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        if len(rewards) > 100:
            # Calculate rolling variance
            rewards_padded = np.pad(rewards, (50, 50), mode='edge')
            rolling_var = np.array([np.var(rewards_padded[i:i+100]) for i in range(len(rewards))])
            smoothed_var = moving_average(rolling_var, window=50)
            plt.plot(smoothed_var, label=name, color=colors[idx], linewidth=2, alpha=0.8)
    plt.title("Training Stability (Rolling Variance)", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Variance", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/9_training_stability.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 10. Learning progress comparison
    plt.figure(figsize=(12, 6))
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        if len(rewards) > 200:
            # Calculate learning progress (improvement rate)
            window = 100
            early = np.mean(rewards[window:window*2])
            late = np.mean(rewards[-window:])
            improvement = [(np.mean(rewards[i:i+window]) - early) / (early + 1e-8) * 100 
                          for i in range(0, len(rewards) - window, 10)]
            plt.plot(np.arange(0, len(rewards) - window, 10), improvement, 
                    label=name, color=colors[idx], linewidth=2, alpha=0.8)
    plt.title("Learning Progress (% Improvement from Baseline)", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Improvement (%)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/10_learning_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SAVED] 10 comprehensive plots saved to {save_dir}/ (no display)")

# ===========================================================
# MAIN PLOTTING SCRIPT
# ===========================================================

def main():
    """Load all saved training data and generate comparison plots."""
    
    print("="*60)
    print("Loading training results for plotting...")
    print("="*60)
    
    results = {}
    missing_configs = []
    
    # Load data for each config
    for config in configs:
        config_name = config['name']
        data_path = f"saved_agents/{config_name.replace(' ', '_')}/data/training_data.pkl"
        
        if os.path.exists(data_path):
            print(f"Loading: {config_name}")
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                results[config_name] = data['results']
        else:
            print(f"✗ Missing: {config_name} ({data_path})")
            missing_configs.append(config_name)
    
    print("="*60)
    
    # Check if we have any results
    if not results:
        print("\n ERROR: No training data found!")
        print("Make sure .pkl files exist in saved_agents/ directory")
        print("Expected structure:")
        for config in configs:
            config_folder = config['name'].replace(' ', '_')
            print(f"  saved_agents/{config_folder}/")
            print(f"    ├── data/training_data.pkl")
            print(f"    ├── models/final.pt")
            print(f"    └── checkpoints/")
        return 1
    
    # Warn about missing configs
    if missing_configs:
        print(f"\n  WARNING: {len(missing_configs)} config(s) missing:")
        for name in missing_configs:
            print(f"  - {name}")
        print(f"\nGenerating plots for {len(results)} config(s) only...")
    else:
        print(f"\n✓ All {len(results)} configs loaded successfully!")
    
    # Generate plots
    print("\n" + "="*60)
    print("Generating comparison plots...")
    print("="*60)
    
    os.makedirs("saved_agents/plots", exist_ok=True)
    plot_training_results(results, save_dir='saved_agents/plots')
    
    print("\n" + "="*60)
    print("✓ COMPLETE!")
    print("="*60)
    print(f"Plots saved to: saved_agents/plots/")
    print("Files generated:")
    print("  1. 1_rewards_confidence.png")
    print("  2. 2_learning_curves_analysis.png")
    print("  3. 3_episode_lengths.png")
    print("  4. 4_loss_convergence.png")
    print("  5. 5_epsilon_decay.png")
    print("  6. 6_final_performance_boxplot.png")
    print("  7. 7_sample_efficiency.png")
    print("  8. 8_metrics_table.png")
    print("  9. 9_training_stability.png")
    print("  10. 10_learning_progress.png")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
