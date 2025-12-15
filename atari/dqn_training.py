import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC (no display)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import gymnasium as gym
import ale_py
from tqdm import tqdm
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
import os
import sys
import time
from gymnasium.vector import AsyncVectorEnv

from params import Params
from configs import configs

sns.set_theme()
gym.register_envs(ale_py)

# ===========================================================
# DQN NETWORK
# ===========================================================

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_dim=512):
        super(DQN, self).__init__()
        
        # Convolutional layers for image processing
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate conv output size
        conv_out_size = self._get_conv_out(input_shape)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# ===========================================================
# REPLAY BUFFER
# ===========================================================

class ReplayBuffer:
    """Optimized replay buffer with pre-allocated arrays for HPC."""
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.state_shape = state_shape
        self.position = 0
        self.size = 0
        
        # Pre-allocate arrays for faster access
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size

# ===========================================================
# DQN AGENT
# ===========================================================

class DQNAgent:
    def __init__(self, state_shape, n_actions, params, device, use_amp=True):
        self.n_actions = n_actions
        self.device = device
        self.gamma = params.gamma
        self.batch_size = params.batch_size
        self.learning_starts = params.learning_starts
        self.target_update_freq = params.target_update_frequency
        self.train_freq = params.train_frequency
        self.use_amp = use_amp and device.type == 'cuda'
        
        # Epsilon schedule
        self.epsilon = params.epsilon_start
        self.epsilon_start = params.epsilon_start
        self.epsilon_end = params.epsilon_end
        self.epsilon_decay_steps = params.epsilon_decay_steps
        
        # Networks
        self.policy_net = DQN(state_shape, n_actions, params.hidden_dim).to(device)
        self.target_net = DQN(state_shape, n_actions, params.hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and gradient scaler for mixed precision
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=params.learning_rate)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Replay buffer with pre-allocated arrays
        self.replay_buffer = ReplayBuffer(params.buffer_size, state_shape)
        
        # Training stats
        self.steps_done = 0
        self.episodes_done = 0
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def update_epsilon(self):
        """Linear decay of epsilon."""
        if self.steps_done < self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * \
                          (self.steps_done / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_end
    
    def train_step(self):
        """Perform one training step with mixed precision support."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors (non_blocking for async GPU transfer)
        states = torch.from_numpy(states).to(self.device, non_blocking=True)
        actions = torch.from_numpy(actions).to(self.device, non_blocking=True)
        rewards = torch.from_numpy(rewards).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(next_states).to(self.device, non_blocking=True)
        dones = torch.from_numpy(dones).to(self.device, non_blocking=True)
        
        if self.use_amp:
            # Mixed precision training for ~2-3x speedup
            with autocast():
                # Current Q values
                current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Target Q values
                with torch.no_grad():
                    next_q = self.target_net(next_states).max(1)[0]
                    target_q = rewards + self.gamma * next_q * (1 - dones)
                
                # Compute loss
                loss = F.smooth_l1_loss(current_q, target_q)
            
            # Optimize with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
                target_q = rewards + self.gamma * next_q * (1 - dones)
            
            loss = F.smooth_l1_loss(current_q, target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
            self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from policy net to target net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filepath):
        """Save agent state."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
        }, filepath)
    
    def load(self, filepath):
        """Load agent state."""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']

# ===========================================================
# PREPROCESSING
# ===========================================================

def preprocess_frame(frame):
    """Convert frame to grayscale and resize."""
    # Convert to grayscale
    gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
    # Crop and resize (84x84 is standard for Atari)
    resized = gray[34:194, :]  # Crop to remove score
    resized = resized[::2, ::2]  # Downsample by 2
    return resized.astype(np.float32) / 255.0

class FrameStack:
    """Stack last N frames for input to network."""
    def __init__(self, n_frames=4):
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
    
    def reset(self, frame):
        """Reset with initial frame."""
        processed = preprocess_frame(frame)
        for _ in range(self.n_frames):
            self.frames.append(processed)
        return self.get_state()
    
    def push(self, frame):
        """Add new frame and return stacked state."""
        processed = preprocess_frame(frame)
        self.frames.append(processed)
        return self.get_state()
    
    def get_state(self):
        """Return stacked frames as numpy array."""
        return np.array(self.frames)

# ===========================================================
# TRAINING FUNCTION
# ===========================================================

def train_dqn(env, agent, params, config_name, checkpoint_path=None):
    """Train DQN agent with checkpointing support for HPC."""
    
    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    episode_epsilons = []
    losses = []
    
    # Current episode stats
    current_reward = 0
    current_length = 0
    start_step = 0
    
    # Load checkpoint if resuming
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        episode_rewards = checkpoint_data['episode_rewards']
        episode_lengths = checkpoint_data['episode_lengths']
        episode_epsilons = checkpoint_data['episode_epsilons']
        losses = checkpoint_data['losses']
        start_step = checkpoint_data['step']
    
    # Frame stacker
    frame_stack = FrameStack(n_frames=4)
    
    # Reset environment
    obs, _ = env.reset(seed=params.seed)
    state = frame_stack.reset(obs)
    
    pbar = tqdm(total=params.total_timesteps, initial=start_step, desc=f"Training {config_name}")
    
    for step in range(params.total_timesteps):
        # Select and perform action
        action = agent.select_action(state, training=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Process next state
        next_state = frame_stack.push(next_obs)
        
        # Store transition
        agent.replay_buffer.push(state, action, reward, next_state, float(done))
        
        # Update stats
        current_reward += reward
        current_length += 1
        agent.steps_done += 1
        
        # Train agent
        if step >= params.learning_starts and step % params.train_frequency == 0:
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
        
        # Update target network
        if step % params.target_update_frequency == 0:
            agent.update_target_network()
        
        # Update epsilon
        agent.update_epsilon()
        
        # Handle episode end
        if done:
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)
            episode_epsilons.append(agent.epsilon)
            agent.episodes_done += 1
            
            # Log progress
            if agent.episodes_done % params.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-params.log_interval:])
                avg_loss = np.mean(losses[-100:]) if losses else 0
                pbar.set_postfix({
                    'episode': agent.episodes_done,
                    'avg_reward': f'{avg_reward:.2f}',
                    'epsilon': f'{agent.epsilon:.3f}',
                    'loss': f'{avg_loss:.4f}'
                })
            
            # Reset for next episode
            obs, _ = env.reset()
            state = frame_stack.reset(obs)
            current_reward = 0
            current_length = 0
        else:
            state = next_state
        
        # Save checkpoint (model + training state)
        if (step + 1) % params.save_interval == 0:
            save_path = f"saved_agents/{config_name.replace(' ', '_')}_step_{step+1}.pt"
            agent.save(save_path)
            
            # Save full checkpoint for resumption
            checkpoint_path = f"saved_agents/{config_name.replace(' ', '_')}_checkpoint.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'episode_rewards': episode_rewards,
                    'episode_lengths': episode_lengths,
                    'episode_epsilons': episode_epsilons,
                    'losses': losses,
                    'step': step + 1
                }, f)
        
        pbar.update(1)
    
    pbar.close()
    
    return {
        'episode_rewards': np.array(episode_rewards),
        'episode_lengths': np.array(episode_lengths),
        'episode_epsilons': np.array(episode_epsilons),
        'losses': np.array(losses),
    }

# ===========================================================
# PLOTTING
# ===========================================================

def moving_average(x, window=100):
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
    
    # 10. Learning rate comparison (if applicable)
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
# MAIN
# ===========================================================

def train_single_config(config_idx, gpu_id=0, num_gpus=1):
    """Train a single configuration (for parallel execution)."""
    config = configs[config_idx]
    
    # Set device
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    
    print(f"\n{'='*60}")
    print(f"Training {config['name']} on {device}")
    print(f"{'='*60}")
    
    params = config['params']
    
    # Set random seeds
    torch.manual_seed(params.seed + config_idx)
    np.random.seed(params.seed + config_idx)
    random.seed(params.seed + config_idx)
    
    # Create environment
    env = gym.make('ALE/Breakout-v5', render_mode=params.render_mode)
    
    # Get state and action dimensions
    obs, _ = env.reset()
    n_actions = env.action_space.n
    state_shape = (4, 80, 80)  # 4 stacked 80x80 frames
    
    # Create agent with mixed precision training
    agent = DQNAgent(state_shape, n_actions, params, device, use_amp=True)
    
    # Check for checkpoint
    checkpoint_path = f"saved_agents/{config['name'].replace(' ', '_')}_checkpoint.pkl"
    
    # Train
    training_results = train_dqn(env, agent, params, config['name'], checkpoint_path)
    
    # Save final model
    final_path = f"saved_agents/{config['name'].replace(' ', '_')}_final.pt"
    agent.save(final_path)
    print(f"Saved final model to {final_path}")
    
    # Save training data
    data_path = f"saved_agents/{config['name'].replace(' ', '_')}_data.pkl"
    with open(data_path, 'wb') as f:
        pickle.dump({
            'results': training_results,
            'params': params,
            'config_name': config['name']
        }, f)
    
    env.close()
    return config['name'], training_results


if __name__ == "__main__":
    # Create save directory
    os.makedirs("saved_agents", exist_ok=True)
    os.makedirs("saved_agents/plots", exist_ok=True)
    
    # Parse command line arguments for HPC
    if len(sys.argv) > 1:
        # Single config mode (for SLURM array jobs)
        config_idx = int(sys.argv[1])
        gpu_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        
        print(f"\n[HPC MODE] Training config {config_idx} on GPU {gpu_id}")
        train_single_config(config_idx, gpu_id)
        
    else:
        # Multi-GPU parallel mode (local)
        num_gpus = torch.cuda.device_count()
        num_configs = len(configs)
        
        print(f"\n[PARALLEL MODE] {num_gpus} GPUs available, {num_configs} configs to train")
        
        if num_gpus > 1 and num_configs > 1:
            # Parallel training using multiprocessing
            print("Starting parallel training across multiple GPUs...")
            
            mp.set_start_method('spawn', force=True)
            processes = []
            
            for i in range(min(num_configs, num_gpus)):
                p = mp.Process(target=train_single_config, args=(i, i))
                p.start()
                processes.append(p)
            
            for p in processes:
                p.join()
            
            print("\n[DONE] Parallel training complete!")
            
        else:
            # Sequential training (fallback)
            print("Sequential training (not enough GPUs for parallel)...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            
            results = {}
            for idx, config in enumerate(configs):
                name, result = train_single_config(idx, 0)
                results[name] = result
        
        # Load all results and plot
        print(f"\n{'='*60}")
        print("Loading results and generating comparison plots...")
        print(f"{'='*60}")
        
        results = {}
        for config in configs:
            data_path = f"saved_agents/{config['name'].replace(' ', '_')}_data.pkl"
            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                    results[config['name']] = data['results']
        
        if results:
            plot_training_results(results)
        
        print("\nTraining complete!")
