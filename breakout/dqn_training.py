"""
Deep Q-Network (DQN) Implementation for Atari Breakout

This implementation follows the seminal DQN Nature paper:
"Human-level control through deep reinforcement learning" (Mnih et al., 2015)

Key Features:
- Convolutional neural network for visual input processing
- Experience replay buffer for breaking correlation in observation sequences
- Fixed Q-targets (separate target network) for stable learning
- Epsilon-greedy exploration with linear annealing
- Reward clipping and frame stacking as per the original paper
- Mixed precision training (AMP) for efficient GPU utilization
- HPC-optimized with checkpoint/resume capability

Author: [Your Name]
Date: December 2025
"""

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
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import os
import sys
import cv2  # For efficient frame preprocessing

from configs import configs

gym.register_envs(ale_py)  # Register Atari Learning Environment

# ===========================================================
# DQN NETWORK ARCHITECTURE
# ===========================================================

class DQN(nn.Module):
    """
    Deep Q-Network architecture as described in Mnih et al. (2015).
    
    Architecture:
    - 3 convolutional layers for spatial feature extraction from raw pixels
    - 2 fully connected layers for action-value estimation
    - Input: 4 stacked grayscale frames (84x84x4)
    - Output: Q-values for each possible action
    
    The convolutional layers learn to detect game-relevant features like
    ball position, paddle location, and brick patterns.
    """
    def __init__(self, input_shape, n_actions, hidden_dim=512):
        super(DQN, self).__init__()
        
        # Convolutional layers: Extract spatial hierarchies from frames
        # Layer 1: 8x8 kernels with stride 4 - detects basic features (edges, corners)
        # Layer 2: 4x4 kernels with stride 2 - combines features into patterns
        # Layer 3: 3x3 kernels with stride 1 - refines high-level features
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
# EXPERIENCE REPLAY BUFFER
# ===========================================================

class ReplayBuffer:
    """
    Experience Replay Buffer: A key innovation from the DQN paper.
    
    Purpose: Store and randomly sample past experiences (s, a, r, s', done)
    to break temporal correlations in the training data.
    
    Why it's important:
    1. Sequential states are highly correlated - training on them directly
       causes instability and overfitting
    2. Random sampling from replay buffer creates i.i.d. training batches
    3. Each experience can be reused multiple times (sample efficiency)
    4. Smooths out learning by averaging over many past behaviors
    
    Memory optimization: Uses uint8 (0-255) instead of float32 (0-1) to
    reduce memory usage by 4x, crucial for storing 1M transitions.
    """
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.state_shape = state_shape
        self.position = 0  # Circular buffer position
        self.size = 0      # Current buffer size (up to capacity)
        
        # Pre-allocate arrays for HPC efficiency
        # uint8 storage: saves ~16GB of RAM for 1M capacity buffer
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        # Store states as uint8 (0-255), converting from float32 (0-1)
        self.states[self.position] = (state * 255).astype(np.uint8)
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = (next_state * 255).astype(np.uint8)
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        # Convert uint8 back to float32 (0-1) when sampling
        return (
            self.states[indices].astype(np.float32) / 255.0,
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices].astype(np.float32) / 255.0,
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
        """
        Epsilon-Greedy Action Selection: Balances exploration vs exploitation.
        
        With probability epsilon: take random action (EXPLORE)
        With probability (1-epsilon): take best action per Q-network (EXPLOIT)
        
        Epsilon decays linearly from 1.0 to 0.05 over training:
        - Early training: mostly random (learn about environment)
        - Late training: mostly greedy (exploit learned policy)
        
        This strategy is crucial for discovering optimal behaviors while
        avoiding getting stuck in local optima.
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)  # Random exploration
        else:
            # Greedy action: choose action with highest Q-value
            with torch.no_grad():  # No gradient needed for inference
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()  # Argmax over actions
    
    def update_epsilon(self):
        """Linear decay of epsilon."""
        if self.steps_done < self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * \
                          (self.steps_done / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_end
    
    def train_step(self):
        """
        Perform one DQN training step using the Bellman equation.
        
        Training algorithm:
        1. Sample random minibatch from replay buffer (breaks correlation)
        2. Compute Q(s,a) using policy network
        3. Compute target: r + γ * max_a' Q_target(s',a') using target network
        4. Minimize loss: L = (Q - target)²
        5. Update policy network parameters via gradient descent
        
        Key innovation: Use separate target network (updated periodically)
        to compute Q-targets. This prevents the "chasing a moving target"
        problem that destabilizes standard Q-learning.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample random minibatch: breaks temporal correlation
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors (non_blocking for async GPU transfer)
        states = torch.from_numpy(states).to(self.device, non_blocking=True)
        actions = torch.from_numpy(actions).to(self.device, non_blocking=True)
        rewards = torch.from_numpy(rewards).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(next_states).to(self.device, non_blocking=True)
        dones = torch.from_numpy(dones).to(self.device, non_blocking=True)
        
        # Clear CUDA cache periodically to prevent fragmentation
        if self.device.type == 'cuda' and self.steps_done % 1000 == 0:
            torch.cuda.empty_cache()
        
        if self.use_amp:
            # Mixed precision training: Use FP16 for ~2-3x speedup on modern GPUs
            # Automatically maintains FP32 master weights for numerical stability
            with autocast(device_type='cuda', dtype=torch.float16):
                # Q(s,a): Extract Q-values for actions actually taken
                # gather() selects Q-value for the specific action index
                current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Bellman target: r + γ * max_a' Q_target(s', a')
                # Use target network (not policy net) for stability
                with torch.no_grad():  # No gradients for target computation
                    next_q = self.target_net(next_states).max(1)[0]  # max over actions
                    target_q = rewards + self.gamma * next_q * (1 - dones)
                    # Note: (1 - dones) zeros out future rewards for terminal states
                
                # Temporal Difference (TD) Loss: Huber loss is less sensitive to outliers
                # than MSE, providing more stable training
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
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']

# ===========================================================
# FRAME PREPROCESSING (Following DQN Nature Paper)
# ===========================================================

def preprocess_frame(frame):
    """
    Preprocess Atari frames to reduce dimensionality and computational cost.
    
    Steps (as per Mnih et al., 2015):
    1. Convert RGB to grayscale (210x160x3 → 210x160x1)
       - Color is not essential for Breakout gameplay
       - Reduces input size by 3x
    2. Crop score area (210x160 → 160x160)
       - Score/lives at top don't help agent learn gameplay
       - Focuses network attention on playfield
    3. Resize to 84x84 for computational efficiency
       - Standard size used across all Atari games
    4. Normalize to [0,1] for neural network input stability
    """
    # Luminance conversion: weighted sum matching human perception
    gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
    # Remove score display at top of screen
    cropped = gray[34:194, :]  # Keep only game playfield
    # Downsample to standard 84x84 (reduces computation)
    resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0  # Normalize to [0,1]

class FrameStack:
    """
    Frame Stacking: Stack last 4 frames to provide motion information.
    
    Why needed: A single frame doesn't show velocity/direction.
    Example: Looking at one frame, you can't tell if ball is moving up or down.
    With 4 frames stacked, the network can infer motion from position changes.
    
    This gives the network temporal context without explicit motion detection.
    """
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
# MAIN TRAINING LOOP
# ===========================================================

def train_dqn(env, agent, params, config_name, checkpoint_path=None):
    """
    Main DQN training loop following the algorithm from Mnih et al. (2015).
    
    Training procedure:
    1. Initialize replay buffer D
    2. For each timestep:
       a) Select action using epsilon-greedy
       b) Execute action, observe reward and next state
       c) Store transition in replay buffer
       d) Sample random minibatch from D
       e) Perform gradient descent step on (y - Q)²
       f) Every C steps, update target network Q_target = Q
    
    Enhancements:
    - Checkpoint/resume for long HPC training runs
    - Life-loss handling specific to Breakout
    - Progress tracking and logging
    - Mixed precision training for efficiency
    """
    
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
        
        # Load agent model state
        model_checkpoint_path = os.path.join(
            os.path.dirname(checkpoint_path), 
            f'step_{start_step}.pt'
        )
        if os.path.exists(model_checkpoint_path):
            agent.load(model_checkpoint_path)
            print(f"Loaded agent state from {model_checkpoint_path}")
    
    # Frame stacker
    frame_stack = FrameStack(n_frames=4)
    
    # Reset environment with random no-op starts (standard DQN practice)
    obs, info = env.reset(seed=params.seed)
    # Perform random number of no-op actions (0-30) for stochasticity
    max_no_ops = 30
    no_ops = np.random.randint(0, max_no_ops + 1)
    for _ in range(no_ops):
        obs, _, _, _, _ = env.step(0)  # 0 = NOOP action
    
    state = frame_stack.reset(obs)
    lives = info.get('lives', 5)  # Breakout starts with 5 lives
    # Breakout typically requires a FIRE action to launch the ball.
    # Force FIRE once at the start, and after each life loss.
    need_fire = True
    
    pbar = tqdm(total=params.total_timesteps, initial=start_step, desc=f"Training {config_name}")
    
    for step in range(start_step, params.total_timesteps):
        # Select and perform action
        if need_fire:
            action = 1  # FIRE
            need_fire = False
        else:
            action = agent.select_action(state, training=True)

        next_obs, reward, terminated, truncated, info = env.step(action)

        # CRITICAL: Reward Clipping (DQN Nature paper, Section 4)
        # Clip all rewards to {-1, 0, +1} to:
        # 1. Prevent large reward magnitudes from dominating gradient updates
        # 2. Enable same learning rate across all Atari games
        # 3. Limit scale of error derivatives for more stable learning
        # In Breakout: breaking brick (1-7 points) all become +1
        reward = np.sign(reward)  # Maps any positive reward to +1

        # Life-Loss as Episode Boundary (Important for Breakout!)
        # Problem: Breakout has 5 lives - if we only end episode when all lives
        # are lost, the agent gets delayed feedback about mistakes.
        # 
        # Solution: Treat each life loss as a "mini-episode" for Q-learning:
        # - Store done=True in replay buffer when life is lost
        # - This teaches agent that life loss = bad (immediate penalty signal)
        # - But continue same episode for statistics tracking
        # 
        # This technique significantly speeds up learning in multi-life games.
        new_lives = info.get('lives', lives)
        life_lost = (new_lives < lives)
        if life_lost:
            lives = new_lives
            need_fire = True  # Need to press FIRE to launch ball after life loss

        # Q-learning treats life loss as terminal (better credit assignment)
        done_for_replay = life_lost or terminated or truncated
        
        # Episode only truly ends when all lives exhausted
        real_done = (lives == 0) or truncated
        
        # Process next state
        next_state = frame_stack.push(next_obs)
        
        # Store transition with done flag indicating life loss or episode end
        agent.replay_buffer.push(state, action, reward, next_state, float(done_for_replay))
        
        # Update stats
        current_reward += reward
        current_length += 1
        agent.steps_done += 1
        
        # Train agent
        if step >= params.learning_starts and step % params.train_frequency == 0:
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
        
        # Periodic Target Network Update (every C=1000 steps)
        # Why separate networks?
        # Without: Q-values and targets both change → "chasing moving target" → divergence
        # With: Targets stable for C steps → more stable gradient updates → convergence
        # This is one of the key innovations that made DQN work!
        if step % params.target_update_frequency == 0:
            agent.update_target_network()
        
        # Update epsilon
        agent.update_epsilon()
        
        # Handle episode end (only when all lives exhausted or truncated)
        if real_done:
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
            
            # Reset for next episode with random no-op starts
            obs, info = env.reset()
            # Random no-ops for stochasticity
            max_no_ops = 30
            no_ops = np.random.randint(0, max_no_ops + 1)
            for _ in range(no_ops):
                obs, _, _, _, _ = env.step(0)  # NOOP
            
            state = frame_stack.reset(obs)
            lives = info.get('lives', 5)
            need_fire = True
            current_reward = 0
            current_length = 0
        else:
            state = next_state
        
        # Save checkpoint (model + training state)
        if (step + 1) % params.save_interval == 0:
            config_folder = os.path.join("saved_agents", config_name.replace(' ', '_'))
            os.makedirs(os.path.join(config_folder, "checkpoints"), exist_ok=True)
            
            # Save model checkpoint
            save_path = os.path.join(config_folder, "checkpoints", f"step_{step+1}.pt")
            agent.save(save_path)
            
            # Save full checkpoint for resumption
            checkpoint_path = os.path.join(config_folder, "checkpoints", "checkpoint.pkl")
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
    try:
        config = configs[config_idx]
        
        # Set device
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            torch.cuda.set_device(gpu_id)  # Critical for HPC multi-GPU
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
        
        print(f"\n{'='*60}")
        print(f"Training {config['name']} on {device}")
        print(f"{'='*60}")
        
        params = config['params']
        
        # Set all random seeds for reproducible experiments
        # Different configs get different seeds (params.seed + config_idx)
        # to ensure independent runs while maintaining reproducibility
        torch.manual_seed(params.seed + config_idx)        # PyTorch CPU
        np.random.seed(params.seed + config_idx)           # NumPy
        random.seed(params.seed + config_idx)              # Python random
        if torch.cuda.is_available():
            torch.cuda.manual_seed(params.seed + config_idx)      # PyTorch CUDA
            torch.cuda.manual_seed_all(params.seed + config_idx)  # All GPUs
            # Deterministic mode: same inputs → same outputs (at cost of speed)
            # Essential for scientific reproducibility and debugging
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Create environment
        env = gym.make('ALE/Breakout-v5', render_mode=params.render_mode)
        
        # Get state and action dimensions
        obs, _ = env.reset()
        n_actions = env.action_space.n
        state_shape = (4, 84, 84)  # 4 stacked 84x84 frames (DQN Nature standard)
        
        # Create agent with mixed precision training
        agent = DQNAgent(state_shape, n_actions, params, device, use_amp=True)
        
        # Create organized folder structure for this config
        config_folder = os.path.join("saved_agents", config['name'].replace(' ', '_'))
        os.makedirs(os.path.join(config_folder, "models"), exist_ok=True)
        os.makedirs(os.path.join(config_folder, "data"), exist_ok=True)
        os.makedirs(os.path.join(config_folder, "checkpoints"), exist_ok=True)
        
        # Check for checkpoint
        checkpoint_path = os.path.join(config_folder, "checkpoints", "checkpoint.pkl")
        
        # Train
        training_results = train_dqn(env, agent, params, config['name'], checkpoint_path)
        
        # Save final model
        final_path = os.path.join(config_folder, "models", "final.pt")
        agent.save(final_path)
        print(f"Saved final model to {final_path}")
        
        # Save training data
        data_path = os.path.join(config_folder, "data", "training_data.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump({
                'results': training_results,
                'params': params,
                'config_name': config['name']
            }, f)
        print(f"Saved training data to {data_path}")
        
        env.close()
        return config['name'], training_results
        
    except Exception as e:
        print(f"\n[ERROR] Training failed for config {config_idx}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to ensure proper exit code


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
            
            # Set start method only if not already set
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass  # Already set
            
            processes = []
            
            for i in range(min(num_configs, num_gpus)):
                p = mp.Process(target=train_single_config, args=(i, i))
                p.start()
                processes.append(p)
            
            # Wait for all processes and check for errors
            for i, p in enumerate(processes):
                p.join()
                if p.exitcode != 0:
                    print(f"WARNING: Process {i} exited with code {p.exitcode}")
            
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
