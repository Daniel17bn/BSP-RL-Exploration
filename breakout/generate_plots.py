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
    plt.figure(figsize=(16, 8))
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        if len(rewards) > 200:
            smoothed = moving_average(rewards, window=200)
            episodes = np.arange(len(smoothed))
            
            # Calculate rolling std for confidence band
            rewards_padded = np.pad(rewards, (100, 100), mode='edge')
            rolling_std = np.array([np.std(rewards_padded[i:i+200]) for i in range(len(rewards))])
            rolling_std = moving_average(rolling_std, window=200)
            
            plt.plot(episodes, smoothed, label=name, color=colors[idx], linewidth=4, alpha=0.95)
            plt.fill_between(episodes, smoothed - rolling_std[:len(smoothed)], 
                           smoothed + rolling_std[:len(smoothed)],
                           color=colors[idx], alpha=0.2)
    plt.title("Episode Rewards with Confidence Intervals", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Episode", fontsize=16, fontweight='bold')
    plt.ylabel("Reward", fontsize=16, fontweight='bold')
    plt.legend(fontsize=14, loc='best', framealpha=0.95, edgecolor='black', fancybox=True)
    plt.grid(True, alpha=0.4, linewidth=1)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/1_rewards_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Smoothed learning curves (clean comparison)
    plt.figure(figsize=(16, 8))
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        if len(rewards) > 200:
            smoothed = moving_average(rewards, window=200)
            episodes = np.arange(len(smoothed))
            plt.plot(episodes, smoothed, label=name, color=colors[idx], linewidth=4, alpha=0.95)
    plt.title("Smoothed Learning Curves (200-Episode Moving Average)", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Episode", fontsize=16, fontweight='bold')
    plt.ylabel("Average Reward", fontsize=16, fontweight='bold')
    plt.legend(fontsize=14, loc='best', framealpha=0.95, edgecolor='black', fancybox=True)
    plt.grid(True, alpha=0.4, linewidth=1)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/2_smoothed_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Episode lengths and survival time
    plt.figure(figsize=(16, 8))
    for idx, (name, data) in enumerate(results_dict.items()):
        lengths = data['episode_lengths']
        if len(lengths) > 200:
            smoothed = moving_average(lengths, window=200)
            episodes = np.arange(len(smoothed))
            plt.plot(episodes, smoothed, label=name, color=colors[idx], linewidth=4, alpha=0.95)
    plt.title("Episode Survival Time (Steps per Episode)", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Episode", fontsize=16, fontweight='bold')
    plt.ylabel("Steps", fontsize=16, fontweight='bold')
    plt.legend(fontsize=14, loc='best', framealpha=0.95, edgecolor='black', fancybox=True)
    plt.grid(True, alpha=0.4, linewidth=1)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/3_episode_lengths.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Training loss convergence
    plt.figure(figsize=(16, 8))
    for idx, (name, data) in enumerate(results_dict.items()):
        losses = data['losses']
        if len(losses) > 500:
            smoothed = moving_average(losses, window=500)
            steps = np.arange(len(smoothed))
            plt.plot(steps, smoothed, label=name, color=colors[idx], linewidth=4, alpha=0.95)
    plt.title("TD Loss Convergence", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Training Step", fontsize=16, fontweight='bold')
    plt.ylabel("Loss (Smooth L1)", fontsize=16, fontweight='bold')
    plt.legend(fontsize=14, loc='best', framealpha=0.95, edgecolor='black', fancybox=True)
    plt.grid(True, alpha=0.4, linewidth=1)
    plt.tick_params(labelsize=14)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/4_loss_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Exploration rate (epsilon) decay
    plt.figure(figsize=(16, 8))
    for idx, (name, data) in enumerate(results_dict.items()):
        epsilons = data['episode_epsilons']
        plt.plot(epsilons, label=name, color=colors[idx], linewidth=4, alpha=0.95)
    plt.title("Exploration Rate (e) Decay Over Training", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Episode", fontsize=16, fontweight='bold')
    plt.ylabel("Epsilon (e)", fontsize=16, fontweight='bold')
    plt.legend(fontsize=14, loc='best', framealpha=0.95, edgecolor='black', fancybox=True)
    plt.grid(True, alpha=0.4, linewidth=1)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/5_epsilon_decay.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Final performance comparison (boxplot)
    plt.figure(figsize=(14, 8))
    final_rewards = []
    labels = []
    for name, data in results_dict.items():
        # Last 20% of episodes
        final_window = data['episode_rewards'][-int(len(data['episode_rewards']) * 0.2):]
        final_rewards.append(final_window)
        labels.append(name)
    
    bp = plt.boxplot(final_rewards, labels=labels, patch_artist=True,
                     medianprops=dict(color='red', linewidth=3),
                     showfliers=True, notch=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(2)
    
    plt.title('Final Performance Distribution (Last 20% Episodes)', fontsize=20, fontweight='bold', pad=20)
    plt.ylabel('Reward', fontsize=16, fontweight='bold')
    plt.xlabel('Configuration', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.4, axis='y', linewidth=1)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/6_final_performance_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Reward rate over time (reward per 1000 timesteps)
    plt.figure(figsize=(12, 6))
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        lengths = data['episode_lengths']
        
        # Calculate cumulative timesteps and rewards
        cumulative_steps = np.cumsum(lengths)
        cumulative_rewards = np.cumsum(rewards)
        
        # Calculate reward rate per 1000 steps
        if len(cumulative_steps) > 10:
            # Sample every 10 episodes for clarity
            sample_idx = np.arange(0, len(cumulative_steps), 10)
            reward_rate = []
            steps_sampled = []
            
            for i in sample_idx[1:]:
                if i > 0:
                    rate = (cumulative_rewards[i] - cumulative_rewards[max(0, i-10)]) / \
                           (cumulative_steps[i] - cumulative_steps[max(0, i-10)]) * 1000
                    reward_rate.append(rate)
                    steps_sampled.append(cumulative_steps[i])
            
            plt.plot(steps_sampled, moving_average(reward_rate, window=min(10, len(reward_rate))), 
                    label=name, color=colors[idx], linewidth=2.5, alpha=0.9)
    
    plt.title("Learning Efficiency (Reward per 1000 Timesteps)", fontsize=14, fontweight='bold')
    plt.xlabel("Total Timesteps", fontsize=12)
    plt.ylabel("Reward Rate", fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/7_learning_efficiency.png', dpi=300, bbox_inches='tight')
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
            f"{np.mean(final_window):.2f} +/- {np.std(final_window):.2f}",
            f"{np.max(rewards):.1f}",
            f"{np.mean(rewards):.2f}",
            f"{len([r for r in final_window if r > 0]) / len(final_window) * 100:.1f}%",
            f"{len(rewards)}"
        ])
    
    table = ax.table(cellText=metrics_data,
                    colLabels=['Config', 'Final Avg+/-Std', 'Best', 'Overall Avg', 'Success Rate', 'Episodes'],
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
    
    # 9. Episode length distribution comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Episode Length Distributions', fontsize=16, fontweight='bold')
    
    for idx, (name, data) in enumerate(results_dict.items()):
        lengths = data['episode_lengths']
        
        # Histogram
        axes[idx].hist(lengths, bins=50, color=colors[idx], alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[idx].axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lengths):.0f}')
        axes[idx].axvline(np.median(lengths), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(lengths):.0f}')
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Episode Length (steps)', fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/9_episode_length_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 10. Performance phases (Early/Mid/Late game comparison)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    phase_data = {'Early (0-33%)': [], 'Mid (33-66%)': [], 'Late (66-100%)': []}
    config_names = []
    
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        n_episodes = len(rewards)
        
        # Split into three phases
        early_phase = rewards[:n_episodes//3]
        mid_phase = rewards[n_episodes//3:2*n_episodes//3]
        late_phase = rewards[2*n_episodes//3:]
        
        phase_data['Early (0-33%)'].append(np.mean(early_phase))
        phase_data['Mid (33-66%)'].append(np.mean(mid_phase))
        phase_data['Late (66-100%)'].append(np.mean(late_phase))
        config_names.append(name)
    
    x = np.arange(len(config_names))
    width = 0.25
    
    for i, (phase, values) in enumerate(phase_data.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=phase, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax.set_title('Performance Across Training Phases', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=15, ha='right')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/10_performance_phases.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SAVED] 10 comprehensive plots saved to {save_dir}/ (no display)")
    
    # 11. Convergence analysis (time to reach reward milestones)
    plt.figure(figsize=(12, 6))
    
    milestones = [0, 5, 10, 15, 20]  # Reward thresholds
    config_convergence = {}
    
    for name, data in results_dict.items():
        rewards = data['episode_rewards']
        smoothed = moving_average(rewards, window=50) if len(rewards) > 50 else rewards
        
        episodes_to_milestone = []
        for milestone in milestones:
            # Find first episode where smoothed reward exceeds milestone
            idx = np.where(smoothed >= milestone)[0]
            if len(idx) > 0:
                episodes_to_milestone.append(idx[0])
            else:
                episodes_to_milestone.append(len(rewards))  # Never reached
        
        config_convergence[name] = episodes_to_milestone
    
    # Plot as lines
    for idx, (name, episodes) in enumerate(config_convergence.items()):
        plt.plot(milestones, episodes, marker='o', markersize=8, linewidth=2.5, 
                label=name, color=colors[idx], alpha=0.9)
    
    plt.title("Convergence Speed (Episodes to Reach Reward Thresholds)", fontsize=14, fontweight='bold')
    plt.xlabel("Reward Threshold", fontsize=12)
    plt.ylabel("Episodes Required", fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/11_convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 12. Loss vs Reward correlation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('TD Loss vs Episode Reward Correlation', fontsize=16, fontweight='bold')
    
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        losses = data['losses']
        
        # Downsample losses to match episode count (average loss per episode)
        if len(losses) > len(rewards):
            losses_per_episode = []
            steps_per_episode = len(losses) // len(rewards)
            for i in range(len(rewards)):
                start = i * steps_per_episode
                end = min((i + 1) * steps_per_episode, len(losses))
                losses_per_episode.append(np.mean(losses[start:end]))
            loss_data = np.array(losses_per_episode)
        else:
            loss_data = losses[:len(rewards)]
        
        # Scatter plot with transparency
        axes[idx].scatter(loss_data, rewards, alpha=0.3, s=10, color=colors[idx])
        
        # Add trend line
        if len(loss_data) > 10:
            z = np.polyfit(loss_data, rewards, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(loss_data), max(loss_data), 100)
            axes[idx].plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8, 
                          label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Average TD Loss', fontsize=10)
        axes[idx].set_ylabel('Episode Reward', fontsize=10)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/12_loss_reward_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] 12 comprehensive plots saved to {save_dir}/ (no display)")

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
    print("  1. 1_rewards_confidence.png - Episode rewards with statistical confidence bands")
    print("  2. 2_smoothed_learning_curves.png - Clean learning curve comparison")
    print("  3. 3_episode_lengths.png - Survival time over training")
    print("  4. 4_loss_convergence.png - TD loss convergence (log scale)")
    print("  5. 5_epsilon_decay.png - Exploration strategy visualization")
    print("  6. 6_final_performance_boxplot.png - Statistical performance comparison")
    print("  7. 7_learning_efficiency.png - Reward rate per 1000 timesteps")
    print("  8. 8_metrics_table.png - Comprehensive statistics summary")
    print("  9. 9_episode_length_distributions.png - Episode length histograms")
    print("  10. 10_performance_phases.png - Early/Mid/Late game comparison")
    print("  11. 11_convergence_analysis.png - Time to reach reward milestones")
    print("  12. 12_loss_reward_correlation.png - TD loss vs reward relationship")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
