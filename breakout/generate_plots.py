"""
Comprehensive Visualization Suite for Deep Reinforcement Learning Analysis

This script generates 20 publication-quality plots for analyzing DQN training results
on Atari Breakout. The visualizations cover multiple dimensions of learning:
- Performance metrics (rewards, success rates)
- Sample efficiency (learning speed)
- Training stability (variance, convergence)
- Behavioral analysis (exploration, best episodes)

Designed for scientific reporting and comparative analysis of different
hyperparameter configurations.

Outputs:
- 20 high-resolution PNG plots (300 DPI) suitable for academic papers
- Adaptive smoothing for 10M+ timestep training runs
- Memory-efficient processing for large-scale experiments

Author: [Your Name]
Date: December 2025
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
# SMOOTHING UTILITIES
# ===========================================================

def moving_average(x, window=100):
    """
    Calculate moving average for smoothing noisy training curves.
    
    Why needed: Raw episode rewards are extremely noisy in RL due to:
    - Stochastic environment dynamics
    - Epsilon-greedy exploration causing suboptimal actions
    - Inherent variance in game outcomes
    
    Moving average reveals underlying learning trends by averaging
    over neighboring data points.
    
    Method: Convolution with uniform kernel (simple and efficient)
    Returns: Smoothed array (length reduced by window-1 due to mode='valid')
    """
    if len(x) < window:
        window = max(1, len(x))
    if window <= 1:
        return x
    return np.convolve(x, np.ones(window) / window, mode='valid')

def adaptive_window(data_length, base_window=100, target_points=1000):
    """
    Calculate adaptive smoothing window based on dataset size.
    
    Problem: Fixed window size works poorly across different training lengths:
    - Too small for long runs (10M steps): still noisy
    - Too large for short runs (1M steps): over-smoothed, lose detail
    
    Solution: Scale window with data length while targeting ~1000 points
    for visualization (good balance of detail and smoothness).
    
    This ensures consistent visual quality across different experiment scales.
    """
    # For 10M timesteps, we want smoother curves
    if data_length > 50000:  # Very long training
        return max(base_window, data_length // target_points)
    elif data_length > 10000:  # Long training
        return max(base_window, data_length // (target_points * 2))
    else:
        return base_window

def plot_training_results(results_dict, save_dir='saved_agents/plots2'):
    """
    Generate comprehensive training analysis plots.
    
    This function creates 20 different visualizations covering:
    - Learning curves and convergence
    - Statistical performance metrics
    - Sample efficiency comparisons
    - Training stability analysis
    - Behavioral patterns
    
    All plots use adaptive smoothing for clarity and are optimized
    for large-scale experiments (10M+ timesteps).
    """
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    # ===========================================================
    # PLOT 1: Episode Rewards with Confidence Intervals
    # ===========================================================
    # Purpose: Show learning progress with uncertainty quantification
    # Key insight: Width of confidence band indicates training stability
    # - Narrow band = stable, consistent learning
    # - Wide band = high variance, unstable training
    plt.figure(figsize=(16, 8))
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        window = adaptive_window(len(rewards), base_window=500)
        if len(rewards) > window:
            smoothed = moving_average(rewards, window=window)
            episodes = np.arange(len(smoothed))
            
            # Calculate rolling standard deviation for confidence bands
            # Why: Shows how much variability exists around the mean trend
            # High std = unpredictable performance, low std = reliable agent
            # Downsampled for computational efficiency on large datasets
            half_window = window // 2
            rewards_padded = np.pad(rewards, (half_window, half_window), mode='edge')
            stride = max(1, len(rewards) // 5000)  # Limit to 5000 points for efficiency
            rolling_std = np.array([np.std(rewards_padded[i:i+window]) for i in range(0, len(rewards), stride)])
            # Interpolate back to full length
            x_sparse = np.arange(0, len(rewards), stride)
            rolling_std = np.interp(np.arange(len(rewards)), x_sparse, rolling_std)
            rolling_std = moving_average(rolling_std, window=window)
            
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
    
    # ===========================================================
    # PLOT 2: Smoothed Learning Curves
    # ===========================================================
    # Purpose: Clean comparison of learning speed across configurations
    # Key questions answered:
    # - Which hyperparameters lead to fastest learning?
    # - Does any config achieve higher final performance?
    # - Are there learning plateaus or continuous improvement?
    plt.figure(figsize=(16, 8))
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        window = adaptive_window(len(rewards), base_window=500)
        if len(rewards) > window:
            smoothed = moving_average(rewards, window=window)
            episodes = np.arange(len(smoothed))
            plt.plot(episodes, smoothed, label=name, color=colors[idx], linewidth=4, alpha=0.95)
    plt.title(f"Smoothed Learning Curves ({window}-Episode Moving Average)", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Episode", fontsize=16, fontweight='bold')
    plt.ylabel("Average Reward", fontsize=16, fontweight='bold')
    plt.legend(fontsize=14, loc='best', framealpha=0.95, edgecolor='black', fancybox=True)
    plt.grid(True, alpha=0.4, linewidth=1)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/2_smoothed_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===========================================================
    # PLOT 3: Episode Survival Time
    # ===========================================================
    # Purpose: Track how long agent survives (proxy for skill)
    # In Breakout: Longer episodes = agent keeps ball alive longer
    # Interpretation:
    # - Increasing trend = agent learning to play defensively
    # - High values = skilled at paddle control
    # - Correlation with rewards shows if survival → scoring
    plt.figure(figsize=(16, 8))
    for idx, (name, data) in enumerate(results_dict.items()):
        lengths = data['episode_lengths']
        window = adaptive_window(len(lengths), base_window=500)
        if len(lengths) > window:
            smoothed = moving_average(lengths, window=window)
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
    
    # ===========================================================
    # PLOT 4: TD Loss Convergence
    # ===========================================================
    # Purpose: Monitor training objective (temporal difference error)
    # TD Loss = (Q_predicted - Q_target)²
    # 
    # Expected pattern: Decrease then stabilize
    # - Decreasing = network learning to predict values accurately
    # - Log scale reveals convergence even when loss is small
    # - Plateau indicates learning has converged
    # 
    # Note: Loss can increase if agent explores new strategies!
    plt.figure(figsize=(16, 8))
    for idx, (name, data) in enumerate(results_dict.items()):
        losses = data['losses']
        window = adaptive_window(len(losses), base_window=2000, target_points=2000)
        if len(losses) > window:
            smoothed = moving_average(losses, window=window)
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
    
    # ===========================================================
    # PLOT 6: Final Performance Distribution (Boxplot)
    # ===========================================================
    # Purpose: Statistical comparison of fully-trained agents
    # Uses last 10% of training (converged policy) for fair comparison
    # 
    # Boxplot components:
    # - Box = interquartile range (25th-75th percentile)
    # - Red line = median performance (robust to outliers)
    # - Whiskers = data range (excluding outliers)
    # - Notches = confidence intervals (non-overlapping = significant difference)
    # 
    # This answers: "Which config is reliably best?"
    plt.figure(figsize=(14, 8))
    final_rewards = []
    labels = []
    for name, data in results_dict.items():
        # Last 10% of episodes (more stable for long training)
        final_window = data['episode_rewards'][-int(len(data['episode_rewards']) * 0.1):]
        final_rewards.append(final_window)
        labels.append(name)
    
    bp = plt.boxplot(final_rewards, tick_labels=labels, patch_artist=True,
                     medianprops=dict(color='red', linewidth=3),
                     showfliers=True, notch=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(2)
    
    plt.title('Final Performance Distribution (Last 10% Episodes)', fontsize=20, fontweight='bold', pad=20)
    plt.ylabel('Reward', fontsize=16, fontweight='bold')
    plt.xlabel('Configuration', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.4, axis='y', linewidth=1)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/6_final_performance_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===========================================================
    # PLOT 7: Learning Efficiency (Reward Rate)
    # ===========================================================
    # Purpose: Measure reward accumulation speed
    # Metric: Reward per 1000 timesteps (controls for episode length)
    # 
    # Why important:
    # - Two agents might reach same final performance but at different speeds
    # - Higher efficiency = better sample efficiency
    # - Useful for comparing wall-clock training time
    # 
    # Interpretation: Higher line = faster learning
    plt.figure(figsize=(12, 6))
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        lengths = data['episode_lengths']
        
        # Calculate cumulative timesteps and rewards
        cumulative_steps = np.cumsum(lengths)
        cumulative_rewards = np.cumsum(rewards)
        
        # Calculate reward rate per 1000 steps
        if len(cumulative_steps) > 10:
            # Adaptive sampling based on total episodes
            sample_interval = max(10, len(cumulative_steps) // 1000)
            sample_idx = np.arange(0, len(cumulative_steps), sample_interval)
            reward_rate = []
            steps_sampled = []
            
            for i in sample_idx[1:]:
                if i > 0:
                    lookback = min(sample_interval, i)
                    rate = (cumulative_rewards[i] - cumulative_rewards[i-lookback]) / \
                           (cumulative_steps[i] - cumulative_steps[i-lookback]) * 1000
                    reward_rate.append(rate)
                    steps_sampled.append(cumulative_steps[i])
            
            smooth_window = adaptive_window(len(reward_rate), base_window=50)
            smoothed_rate = moving_average(reward_rate, window=smooth_window)
            # Adjust steps_sampled to match smoothed array length
            steps_smoothed = steps_sampled[:len(smoothed_rate)]
            plt.plot(steps_smoothed, smoothed_rate, 
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
        final_window = rewards[-int(len(rewards) * 0.1):]  # Last 10% for 10M training
        
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
        
        # Adaptive binning for large datasets
        n_bins = min(100, max(50, len(lengths) // 500))
        
        # Histogram
        axes[idx].hist(lengths, bins=n_bins, color=colors[idx], alpha=0.7, edgecolor='black', linewidth=0.5)
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
    
    milestones = [0, 5, 10, 20, 30, 50]  # Reward thresholds for Breakout
    config_convergence = {}
    
    for name, data in results_dict.items():
        rewards = data['episode_rewards']
        window = adaptive_window(len(rewards), base_window=100)
        smoothed = moving_average(rewards, window=window) if len(rewards) > window else rewards
        
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
    
    # ===========================================================
    # PLOT 13: Sample Efficiency (Cumulative Reward)
    # ===========================================================
    # Purpose: Compare total learning progress vs environment interactions
    # 
    # Key metric in RL: Sample efficiency
    # - How many timesteps needed to achieve good performance?
    # - Steeper slope = more sample efficient
    # - Earlier convergence = faster learner
    # 
    # Critical for real-world applications where data collection is expensive
    plt.figure(figsize=(16, 8))
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        lengths = data['episode_lengths']
        
        cumulative_steps = np.cumsum(lengths)
        cumulative_rewards = np.cumsum(rewards)
        
        # Downsample for plotting efficiency
        if len(cumulative_steps) > 5000:
            indices = np.linspace(0, len(cumulative_steps)-1, 5000, dtype=int)
            cumulative_steps = cumulative_steps[indices]
            cumulative_rewards = cumulative_rewards[indices]
        
        plt.plot(cumulative_steps, cumulative_rewards, label=name, 
                color=colors[idx], linewidth=3, alpha=0.9)
    
    plt.title("Sample Efficiency (Cumulative Reward vs Timesteps)", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Timesteps", fontsize=16, fontweight='bold')
    plt.ylabel("Cumulative Reward", fontsize=16, fontweight='bold')
    plt.legend(fontsize=14, loc='best', framealpha=0.95, edgecolor='black', fancybox=True)
    plt.grid(True, alpha=0.4, linewidth=1)
    plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/13_sample_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===========================================================
    # PLOT 14: Training Stability Analysis
    # ===========================================================
    # Purpose: Diagnose training dynamics and stability
    # 
    # Two panels:
    # 1. Rolling Mean: Shows learning trajectory (similar to learning curves)
    # 2. Rolling Variance: Shows training stability over time
    # 
    # Ideal pattern for variance:
    # - High initially (exploring, unstable)
    # - Decreases over time (policy stabilizing)
    # - Low at end (converged, consistent behavior)
    # 
    # High variance throughout = unstable/noisy training (potential hyperparameter issue)
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('Training Stability Analysis', fontsize=20, fontweight='bold')
    
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        window = adaptive_window(len(rewards), base_window=500)
        
        if len(rewards) > window:
            # Rolling mean
            rolling_mean = moving_average(rewards, window=window)
            
            # Rolling variance
            rolling_var = np.array([np.var(rewards[max(0, i-window):i+1]) 
                                   for i in range(window, len(rewards))])
            
            episodes_mean = np.arange(len(rolling_mean))
            episodes_var = np.arange(window, len(rewards))
            
            axes[0].plot(episodes_mean, rolling_mean, label=name, 
                        color=colors[idx], linewidth=3, alpha=0.9)
            axes[1].plot(episodes_var, rolling_var, label=name, 
                        color=colors[idx], linewidth=3, alpha=0.9)
    
    axes[0].set_ylabel('Rolling Mean Reward', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=12, loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=12)
    
    axes[1].set_xlabel('Episode', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Rolling Variance', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=12, loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/14_training_stability.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===========================================================
    # PLOT 15: Sample Efficiency to Milestones
    # ===========================================================
    # Purpose: Quantify learning speed to specific performance thresholds
    # 
    # Key question: How many timesteps to reach reward X?
    # 
    # Comparison method:
    # - Lower bars = fewer timesteps needed = more sample efficient
    # - Missing bars = milestone never reached during training
    # 
    # Useful for:
    # - Identifying which configs learn fastest
    # - Estimating training budget for desired performance
    # - Comparing early vs late-game learning rates
    plt.figure(figsize=(12, 8))
    
    milestones = [5, 10, 20, 30, 40, 50]
    milestone_data = {}
    
    for name, data in results_dict.items():
        rewards = data['episode_rewards']
        lengths = data['episode_lengths']
        cumulative_steps = np.cumsum(lengths)
        
        window = adaptive_window(len(rewards), base_window=100)
        smoothed = moving_average(rewards, window=window) if len(rewards) > window else rewards
        
        timesteps_to_milestone = []
        for milestone in milestones:
            idx = np.where(smoothed >= milestone)[0]
            if len(idx) > 0:
                episode_idx = min(idx[0] + window//2, len(cumulative_steps)-1)
                timesteps_to_milestone.append(cumulative_steps[episode_idx] / 1e6)  # In millions
            else:
                timesteps_to_milestone.append(np.nan)
        
        milestone_data[name] = timesteps_to_milestone
    
    x = np.arange(len(milestones))
    width = 0.8 / len(milestone_data)
    
    for idx, (name, timesteps) in enumerate(milestone_data.items()):
        offset = (idx - len(milestone_data)/2 + 0.5) * width
        bars = plt.bar(x + offset, timesteps, width, label=name, 
                      color=colors[idx], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, timesteps):
            if not np.isnan(val):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.xlabel('Reward Milestone', fontsize=14, fontweight='bold')
    plt.ylabel('Timesteps to Reach (Millions)', fontsize=14, fontweight='bold')
    plt.title('Sample Efficiency: Timesteps to Reach Reward Milestones', fontsize=16, fontweight='bold', pad=15)
    plt.xticks(x, milestones, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/15_milestone_timesteps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===========================================================
    # PLOT 16: Reward Distribution Heatmap
    # ===========================================================
    # Purpose: Visualize how reward distribution evolves during training
    # 
    # Reading the heatmap:
    # - X-axis: Training progress (0% → 100%)
    # - Y-axis: Reward bins (from negative to high scores)
    # - Color intensity: Frequency of rewards in that range
    # 
    # Learning pattern visualization:
    # - Early: Concentration at low rewards (unskilled play)
    # - Middle: Distribution shifts upward (improving)
    # - Late: Peak at higher rewards (skilled, consistent)
    # 
    # Red regions = common reward ranges at that training stage
    fig, axes = plt.subplots(1, min(3, len(results_dict)), figsize=(18, 5))
    if len(results_dict) == 1:
        axes = [axes]
    fig.suptitle('Reward Distribution Heatmap Over Training', fontsize=18, fontweight='bold')
    
    for idx, (name, data) in enumerate(results_dict.items()):
        if idx >= 3:  # Limit to 3 configs for space
            break
            
        rewards = data['episode_rewards']
        
        # Divide training into 50 segments
        n_segments = 50
        segment_size = len(rewards) // n_segments
        
        # Create heatmap data
        heatmap_data = []
        for i in range(n_segments):
            segment = rewards[i*segment_size:(i+1)*segment_size]
            if len(segment) > 0:
                hist, _ = np.histogram(segment, bins=20, range=(-5, 60))
                heatmap_data.append(hist)
        
        heatmap_data = np.array(heatmap_data).T
        
        im = axes[idx].imshow(heatmap_data, aspect='auto', cmap='YlOrRd', 
                             interpolation='bilinear', origin='lower')
        axes[idx].set_title(name, fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Training Progress (%)', fontsize=11)
        axes[idx].set_ylabel('Reward Bins', fontsize=11)
        axes[idx].set_xticks(np.linspace(0, n_segments-1, 5))
        axes[idx].set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        axes[idx].set_yticks(np.linspace(0, 19, 5))
        axes[idx].set_yticklabels(['-5', '10', '25', '40', '60'])
        plt.colorbar(im, ax=axes[idx], label='Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/16_score_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===========================================================
    # PLOT 17: Success Rate Over Time
    # ===========================================================
    # Purpose: Track proportion of successful episodes
    # Success defined: Episodes with positive reward (scored points)
    # 
    # Why important:
    # - In early training, agent might score 0 points (100% failure)
    # - As learning progresses, success rate should increase
    # - Final success rate indicates policy reliability
    # 
    # Interpretation:
    # - Increasing trend = agent learning to score consistently
    # - 100% = agent always scores (never gets shutout)
    # - Plateaus below 100% = some inherent difficulty/stochasticity
    plt.figure(figsize=(16, 8))
    
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        window = adaptive_window(len(rewards), base_window=500)
        
        # Calculate rolling success rate (positive reward episodes)
        success_rate = []
        episodes = []
        for i in range(window, len(rewards)):
            segment = rewards[i-window:i]
            rate = np.sum(np.array(segment) > 0) / len(segment) * 100
            success_rate.append(rate)
            episodes.append(i)
        
        plt.plot(episodes, success_rate, label=name, 
                color=colors[idx], linewidth=3, alpha=0.9)
    
    plt.title("Success Rate Over Training (% Positive Reward Episodes)", 
             fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Episode", fontsize=16, fontweight='bold')
    plt.ylabel("Success Rate (%)", fontsize=16, fontweight='bold')
    plt.ylim([0, 105])
    plt.legend(fontsize=14, loc='best', framealpha=0.95, edgecolor='black', fancybox=True)
    plt.grid(True, alpha=0.4, linewidth=1)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/17_success_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===========================================================
    # PLOT 18: Learning Rate (Improvement Speed)
    # ===========================================================
    # Purpose: Measure instantaneous rate of improvement
    # Metric: Derivative of smoothed reward curve
    # 
    # Positive values = improving
    # Negative values = performance degrading (rare, but possible during exploration)
    # Zero = plateau (learning stalled)
    # 
    # Useful for:
    # - Identifying learning phases (rapid improvement vs plateau)
    # - Detecting when to stop training (prolonged plateau)
    # - Comparing learning dynamics across configs
    plt.figure(figsize=(16, 8))
    
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        window = adaptive_window(len(rewards), base_window=1000)
        
        if len(rewards) > window:
            # Calculate improvement rate
            smoothed = moving_average(rewards, window=window)
            improvement = np.diff(smoothed)
            episodes = np.arange(1, len(improvement)+1)
            
            # Further smooth the improvement
            improvement_smoothed = moving_average(improvement, window=window//4)
            episodes_smoothed = episodes[:len(improvement_smoothed)]
            
            plt.plot(episodes_smoothed, improvement_smoothed, label=name, 
                    color=colors[idx], linewidth=3, alpha=0.9)
    
    plt.title("Learning Rate (Reward Improvement Per Episode)", 
             fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Episode", fontsize=16, fontweight='bold')
    plt.ylabel("Reward Improvement Rate", fontsize=16, fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    plt.legend(fontsize=14, loc='best', framealpha=0.95, edgecolor='black', fancybox=True)
    plt.grid(True, alpha=0.4, linewidth=1)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/18_learning_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===========================================================
    # PLOT 19: Best Reward Progression
    # ===========================================================
    # Purpose: Track peak performance achieved over time
    # 
    # This plot shows: At any point, what's the best the agent has EVER done?
    # 
    # Why track this:
    # - Shows exploration success (finding high-reward strategies)
    # - Monotonically increasing (never decreases)
    # - Rapid initial climb = discovering better strategies
    # - Plateau = reaching skill ceiling
    # 
    # Different from average reward: Shows potential vs typical performance
    plt.figure(figsize=(16, 8))
    
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        
        # Track best reward achieved so far
        best_so_far = []
        current_best = float('-inf')
        for r in rewards:
            if r > current_best:
                current_best = r
            best_so_far.append(current_best)
        
        episodes = np.arange(len(best_so_far))
        plt.plot(episodes, best_so_far, label=name, 
                color=colors[idx], linewidth=3, alpha=0.9)
    
    plt.title("Best Reward Achieved Over Training", 
             fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Episode", fontsize=16, fontweight='bold')
    plt.ylabel("Best Reward So Far", fontsize=16, fontweight='bold')
    plt.legend(fontsize=14, loc='best', framealpha=0.95, edgecolor='black', fancybox=True)
    plt.grid(True, alpha=0.4, linewidth=1)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/19_best_episodes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===========================================================
    # PLOT 20: Comprehensive Summary Dashboard
    # ===========================================================
    # Purpose: Single-page overview of all key metrics
    # 
    # Perfect for:
    # - Presentations (one slide with complete picture)
    # - Quick comparison of multiple configs
    # - Report executive summary
    # 
    # Includes:
    # - Learning curves (main result)
    # - Final performance distribution (statistical summary)
    # - Sample efficiency (cost analysis)
    # - Success rate (reliability)
    # - Detailed statistics table (quantitative comparison)
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Mini plot 1: Learning curves
    ax1 = fig.add_subplot(gs[0, :])
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        window = adaptive_window(len(rewards), base_window=500)
        if len(rewards) > window:
            smoothed = moving_average(rewards, window=window)
            episodes = np.arange(len(smoothed))
            ax1.plot(episodes, smoothed, label=name, color=colors[idx], linewidth=2.5)
    ax1.set_title('Learning Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Reward', fontsize=11)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Mini plot 2: Final performance boxplot
    ax2 = fig.add_subplot(gs[1, 0])
    final_rewards_mini = []
    labels_mini = []
    for name, data in results_dict.items():
        final_window = data['episode_rewards'][-int(len(data['episode_rewards']) * 0.1):]
        final_rewards_mini.append(final_window)
        labels_mini.append(name.replace('_', '\n'))
    bp = ax2.boxplot(final_rewards_mini, tick_labels=labels_mini, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_title('Final Performance', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Reward', fontsize=10)
    ax2.tick_params(labelsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Mini plot 3: Sample efficiency
    ax3 = fig.add_subplot(gs[1, 1])
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        lengths = data['episode_lengths']
        cumulative_steps = np.cumsum(lengths)
        cumulative_rewards = np.cumsum(rewards)
        if len(cumulative_steps) > 1000:
            indices = np.linspace(0, len(cumulative_steps)-1, 1000, dtype=int)
            ax3.plot(cumulative_steps[indices]/1e6, cumulative_rewards[indices], 
                    color=colors[idx], linewidth=2, label=name)
    ax3.set_title('Sample Efficiency', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Timesteps (M)', fontsize=10)
    ax3.set_ylabel('Cumulative Reward', fontsize=10)
    ax3.tick_params(labelsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Mini plot 4: Success rate
    ax4 = fig.add_subplot(gs[1, 2])
    for idx, (name, data) in enumerate(results_dict.items()):
        rewards = data['episode_rewards']
        window = min(500, len(rewards) // 10)
        if window > 10:
            success_rate = []
            episodes = []
            for i in range(window, len(rewards), window//10):
                segment = rewards[max(0, i-window):i]
                rate = np.sum(np.array(segment) > 0) / len(segment) * 100
                success_rate.append(rate)
                episodes.append(i)
            ax4.plot(episodes, success_rate, color=colors[idx], linewidth=2)
    ax4.set_title('Success Rate', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Episode', fontsize=10)
    ax4.set_ylabel('% Positive', fontsize=10)
    ax4.tick_params(labelsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Mini plot 5: Statistics table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('tight')
    ax5.axis('off')
    
    table_data = []
    for name, data in results_dict.items():
        rewards = data['episode_rewards']
        lengths = data['episode_lengths']
        final_window = rewards[-int(len(rewards) * 0.1):]
        
        table_data.append([
            name,
            f"{np.mean(final_window):.2f}",
            f"{np.std(final_window):.2f}",
            f"{np.max(rewards):.1f}",
            f"{np.mean(rewards):.2f}",
            f"{np.sum(lengths)/1e6:.2f}M",
            f"{len(rewards)}"
        ])
    
    table = ax5.table(cellText=table_data,
                     colLabels=['Config', 'Final Mean', 'Final Std', 'Best', 'Overall Mean', 'Total Steps', 'Episodes'],
                     cellLoc='center', loc='center',
                     colWidths=[0.20, 0.12, 0.12, 0.10, 0.13, 0.13, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(7):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(7):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    fig.suptitle('Training Summary Dashboard', fontsize=22, fontweight='bold', y=0.98)
    plt.savefig(f'{save_dir}/20_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] 20 comprehensive plots saved to {save_dir}/ (no display)")

# ===========================================================
# MAIN PLOTTING SCRIPT
# ===========================================================

def main():
    """
    Load saved training data and generate all visualization plots.
    
    Workflow:
    1. Load training_data.pkl files from each config's saved_agents/ folder
    2. Validate data availability (warn about missing configs)
    3. Generate 20 comprehensive plots comparing all configs
    4. Save as publication-quality PNG files (300 DPI)
    
    Expected data structure:
    saved_agents/
      ConfigName/
        data/
          training_data.pkl  <- Contains: episode_rewards, episode_lengths,
                                          episode_epsilons, losses
        models/
          final.pt
        checkpoints/
          ...
    
    Output: saved_agents/plots/ directory with 20 PNG files
    """
    
    print("="*60)
    print("Loading training results for plotting...")
    print("="*60)
    
    results = {}
    missing_configs = []
    
    # Load training data from each configuration
    # Each config should have saved its results during training
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
    print("20 comprehensive plots generated:")
    print("  1. 1_rewards_confidence.png - Episode rewards with statistical confidence bands (adaptive smoothing)")
    print("  2. 2_smoothed_learning_curves.png - Clean learning curve comparison (10M timesteps optimized)")
    print("  3. 3_episode_lengths.png - Survival time over training (adaptive smoothing)")
    print("  4. 4_loss_convergence.png - TD loss convergence (log scale, high-resolution)")
    print("  5. 5_epsilon_decay.png - Exploration strategy visualization")
    print("  6. 6_final_performance_boxplot.png - Statistical performance comparison (last 10%)")
    print("  7. 7_learning_efficiency.png - Reward rate per 1000 timesteps")
    print("  8. 8_metrics_table.png - Comprehensive statistics summary")
    print("  9. 9_episode_length_distributions.png - Episode length histograms")
    print("  10. 10_performance_phases.png - Early/Mid/Late game comparison")
    print("  11. 11_convergence_analysis.png - Time to reach reward milestones")
    print("  12. 12_loss_reward_correlation.png - TD loss vs reward relationship")
    print("  13. 13_sample_efficiency.png - Cumulative reward vs timesteps")
    print("  14. 14_training_stability.png - Rolling mean and variance analysis")
    print("  15. 15_milestone_timesteps.png - Sample efficiency to reward milestones")
    print("  16. 16_score_heatmap.png - Reward distribution heatmap over training")
    print("  17. 17_success_rate.png - Success rate (positive rewards) over time")
    print("  18. 18_learning_rate.png - Reward improvement per episode")
    print("  19. 19_best_episodes.png - Best reward achieved timeline")
    print("  20. 20_summary_dashboard.png - Comprehensive multi-panel summary")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
