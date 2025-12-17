import numpy as np
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

from configs import configs

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
            config_folder = f"saved_agents/{config_name.replace(' ', '_')}"
            os.makedirs(f"{config_folder}/checkpoints", exist_ok=True)
            
            # Save model checkpoint
            save_path = f"{config_folder}/checkpoints/step_{step+1}.pt"
            agent.save(save_path)
            
            # Save full checkpoint for resumption
            checkpoint_path = f"{config_folder}/checkpoints/checkpoint.pkl"
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
    
    # Create organized folder structure for this config
    config_folder = f"saved_agents/{config['name'].replace(' ', '_')}"
    os.makedirs(f"{config_folder}/models", exist_ok=True)
    os.makedirs(f"{config_folder}/data", exist_ok=True)
    os.makedirs(f"{config_folder}/checkpoints", exist_ok=True)
    
    # Check for checkpoint
    checkpoint_path = f"{config_folder}/checkpoints/checkpoint.pkl"
    
    # Train
    training_results = train_dqn(env, agent, params, config['name'], checkpoint_path)
    
    # Save final model
    final_path = f"{config_folder}/models/final.pt"
    agent.save(final_path)
    print(f"Saved final model to {final_path}")
    
    # Save training data
    data_path = f"{config_folder}/data/training_data.pkl"
    with open(data_path, 'wb') as f:
        pickle.dump({
            'results': training_results,
            'params': params,
            'config_name': config['name']
        }, f)
    print(f"Saved training data to {data_path}")
    
    env.close()
    return config['name'], training_results


if __name__ == "__main__":
    # Create save directory
    os.makedirs("saved_agents", exist_ok=True)
    
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
        
        print("\nTraining complete! Use generate_plots.py to create visualizations.")
