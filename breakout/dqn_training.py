"""
Deep Q-Network (DQN) Implementation for Atari Breakout

This implementation is based on the DQN paper by Mnih et al. (2015),
which demonstrated that deep reinforcement learning could achieve
human-level performance on Atari games.

Implemented Features:
- Convolutional neural network for processing game frames
- Experience replay buffer for storing and sampling transitions
- Target network for stable Q-value estimation
- Epsilon-greedy exploration strategy with decay
- Reward clipping and frame stacking as described in the paper
- Mixed precision training for improved GPU efficiency
- Checkpoint system for training resumption

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
    Deep Q-Network architecture for action-value function approximation.
    
    Network Architecture:
    - 3 convolutional layers for spatial feature extraction
    - 2 fully connected layers for Q-value estimation
    - Input: 4 stacked grayscale frames (84x84 pixels each)
    - Output: Q-values for each possible action
    
    The convolutional layers learn to detect relevant game features such as
    ball position, paddle location, and brick patterns.
    """
    def __init__(self, input_shape, n_actions, hidden_dim=512):
        super(DQN, self).__init__()
        
        # Convolutional layers: Extract hierarchical spatial features
        # Layer 1: 8x8 kernels with stride 4 - detects low-level features (edges, corners)
        # Layer 2: 4x4 kernels with stride 2 - combines features into patterns
        # Layer 3: 3x3 kernels with stride 1 - refines high-level representations
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
    Experience Replay Buffer - a key component from the DQN algorithm.
    
    Stores past experiences (state, action, reward, next_state, done) and
    randomly samples them during training.
    
    Purpose:
    1. Consecutive frames are highly correlated, which can lead to
       overfitting and unstable training
    2. Random sampling breaks temporal correlations, creating more
       independent and identically distributed (i.i.d.) training batches
    3. Each experience can be reused multiple times, improving sample efficiency
    4. Smooths out learning by averaging over diverse past behaviors
    
    Memory Optimization: Uses uint8 (0-255) instead of float32 (0-1),
    reducing memory usage by approximately 4x - essential for storing
    1 million transitions.
    """
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.state_shape = state_shape
        self.position = 0  # Current position in circular buffer
        self.size = 0      # Number of experiences currently stored
        
        # Pre-allocate arrays for computational efficiency
        # uint8 storage reduces memory footprint by ~16GB for 1M capacity
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        # Store states as uint8 (0-255) for memory efficiency
        self.states[self.position] = (state * 255).astype(np.uint8)
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = (next_state * 255).astype(np.uint8)
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        # Convert uint8 back to float32 (0-1) for neural network processing
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
        
        # Epsilon-greedy exploration schedule
        self.epsilon = params.epsilon_start
        self.epsilon_start = params.epsilon_start
        self.epsilon_end = params.epsilon_end
        self.epsilon_decay_steps = params.epsilon_decay_steps
        
        # Networks
        self.policy_net = DQN(state_shape, n_actions, params.hidden_dim).to(device)
        self.target_net = DQN(state_shape, n_actions, params.hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and gradient scaler for mixed precision training
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=params.learning_rate)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(params.buffer_size, state_shape)
        
        # Training stats
        self.steps_done = 0
        self.episodes_done = 0
    
    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection balancing exploration and exploitation.
        
        With probability epsilon: select random action (explore)
        With probability (1-epsilon): select action with highest Q-value (exploit)
        
        Epsilon decays linearly from 1.0 to 0.05 during training:
        - Early training: primarily random actions to discover environment dynamics
        - Later training: primarily greedy actions based on learned policy
        
        This balance is crucial - pure exploitation risks local optima,
        while pure exploration prevents convergence to optimal policy.
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)  # Random exploration
        else:
            # Select action with maximum Q-value
            with torch.no_grad():  # No gradient computation needed for inference
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()  # Argmax over actions
    
    def update_epsilon(self):
        """Linearly decay epsilon from start to end value over specified steps."""
        if self.steps_done < self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * \
                          (self.steps_done / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_end
    
    def train_step(self):
        """
        Performs one DQN training step using sampled experiences.
        
        Algorithm:
        1. Sample random minibatch from replay buffer
        2. Compute current Q(s,a) using policy network
        3. Compute target: r + γ * max Q(s',a') using target network
        4. Calculate temporal difference (TD) loss
        5. Perform gradient descent to update policy network
        
        Key technique: Separate target network (updated periodically) stabilizes
        training by preventing the "moving target" problem where both Q-values
        and targets change simultaneously.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample random minibatch to break temporal correlation
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert numpy arrays to PyTorch tensors and transfer to device
        states = torch.from_numpy(states).to(self.device, non_blocking=True)
        actions = torch.from_numpy(actions).to(self.device, non_blocking=True)
        rewards = torch.from_numpy(rewards).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(next_states).to(self.device, non_blocking=True)
        dones = torch.from_numpy(dones).to(self.device, non_blocking=True)
        
        # Periodically clear CUDA cache to prevent memory fragmentation
        if self.device.type == 'cuda' and self.steps_done % 1000 == 0:
            torch.cuda.empty_cache()
        
        if self.use_amp:
            # Mixed precision training: FP16 computation for ~2-3x speedup
            # while maintaining FP32 master weights for numerical stability
            with autocast(device_type='cuda', dtype=torch.float16):
                # Extract Q-values for actions taken in each transition
                # gather() selects Q-value corresponding to action index
                current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Compute Bellman target: r + γ * max_a' Q_target(s', a')
                # Using target network (not policy network) for stability
                with torch.no_grad():  # No gradients needed for target computation
                    next_q = self.target_net(next_states).max(1)[0]  # Maximum Q-value in next state
                    target_q = rewards + self.gamma * next_q * (1 - dones)
                    # (1 - dones) masks out future rewards for terminal states
                
                # Compute temporal difference loss using Huber loss (smooth L1)
                # More robust to outliers than mean squared error
                loss = F.smooth_l1_loss(current_q, target_q)
            
            # Perform gradient descent with gradient scaling for mixed precision
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard FP32 training
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
        """Synchronize target network weights with policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filepath):
        """Save agent state including networks and optimizer."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
        }, filepath)
    
    def load(self, filepath):
        """Load agent state from checkpoint file."""
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
    Preprocess Atari frames following the methodology from Mnih et al. (2015).
    
    Preprocessing steps:
    1. Convert RGB to grayscale (210x160x3 → 210x160x1)
       - Color information is not essential for Breakout
       - Reduces input dimensionality by factor of 3
    2. Crop score region (210x160 → 160x160)
       - Removes score display which does not aid gameplay decisions
       - Focuses network on playfield area
    3. Resize to 84x84 pixels
       - Standard input size used across all Atari environments
       - Reduces computational requirements
    4. Normalize pixel values to [0,1]
       - Improves neural network training stability
    """
    # Grayscale conversion using luminance weights
    gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
    # Remove score display from top of frame
    cropped = gray[34:194, :]  # Retain playfield only
    # Downsample to standard dimensions
    resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0  # Normalize to [0,1]

class FrameStack:
    """
    Frame stacking utility to provide temporal information to the network.
    
    Motivation: A single frame lacks velocity and direction information.
    For example, ball trajectory cannot be determined from a single snapshot.
    By stacking 4 consecutive frames, the network can infer motion from
    position changes across frames.
    
    This provides temporal context without explicit motion detection or
    recurrent architecture.
    """
    def __init__(self, n_frames=4):
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
    
    def reset(self, frame):
        """Initialize frame stack by replicating the first frame."""
        processed = preprocess_frame(frame)
        for _ in range(self.n_frames):
            self.frames.append(processed)
        return self.get_state()
    
    def push(self, frame):
        """Add new frame to stack and return updated state."""
        processed = preprocess_frame(frame)
        self.frames.append(processed)
        return self.get_state()
    
    def get_state(self):
        """Return current frame stack as numpy array."""
        return np.array(self.frames)

# ===========================================================
# MAIN TRAINING LOOP
# ===========================================================

def train_dqn(env, agent, params, config_name, checkpoint_path=None):
    """
    Main DQN training loop implementing the algorithm from Mnih et al. (2015).
    
    Training procedure:
    1. Initialize replay buffer
    2. For each timestep:
       a) Select action using epsilon-greedy policy
       b) Execute action and observe transition
       c) Store transition in replay buffer
       d) Sample minibatch and perform gradient descent
       e) Periodically update target network
    
    Additional features:
    - Checkpoint/resume capability for long training runs
    - Life-loss handling specific to multi-life games (Breakout)
    - Progress tracking and logging
    - Mixed precision training for computational efficiency
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
    
    # Attempt to resume from checkpoint if available
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
    
    # Reset environment with random no-op start for stochasticity
    obs, info = env.reset(seed=params.seed)
    # Apply 0-30 random no-op actions to vary initial state
    max_no_ops = 30
    no_ops = np.random.randint(0, max_no_ops + 1)
    for _ in range(no_ops):
        obs, _, _, _, _ = env.step(0)  # No-operation action
    
    state = frame_stack.reset(obs)
    lives = info.get('lives', 5)  # Breakout provides 5 lives
    # Breakout requires FIRE action to launch ball at episode start
    # and after each life loss
    need_fire = True
    
    pbar = tqdm(total=params.total_timesteps, initial=start_step, desc=f"Training {config_name}")
    
    for step in range(start_step, params.total_timesteps):
        # Select and execute action
        if need_fire:
            action = 1  # FIRE action
            need_fire = False
        else:
            action = agent.select_action(state, training=True)

        next_obs, reward, terminated, truncated, info = env.step(action)

        # Reward clipping as specified in Mnih et al. (2015)
        # Clip all rewards to {-1, 0, +1} to:
        # 1. Enable same hyperparameters across different games
        # 2. Prevent large rewards from dominating gradient updates
        # 3. Limit scale of error derivatives for stable learning
        # In Breakout: all positive rewards (1-7 points) become +1
        reward = np.sign(reward)

        # Life-loss handling for improved credit assignment in multi-life games
        # Problem: With 5 lives, treating only full game-over as terminal
        # delays learning signal for poor decisions.
        # 
        # Solution: Treat life loss as terminal state for Q-learning:
        # - Store done=True in replay buffer when life is lost
        # - Provides immediate penalty signal for life-losing actions
        # - Continue episode for statistics tracking
        # 
        # This significantly accelerates learning in games with multiple lives.
        new_lives = info.get('lives', lives)
        life_lost = (new_lives < lives)
        if life_lost:
            lives = new_lives
            need_fire = True  # Requires FIRE to launch ball after life loss

        # Treat life loss as terminal for Q-learning updates
        done_for_replay = life_lost or terminated or truncated
        
        # Episode truly ends only when all lives are exhausted
        real_done = (lives == 0) or truncated
        
        # Process next observation
        next_state = frame_stack.push(next_obs)
        
        # Store transition in replay buffer
        agent.replay_buffer.push(state, action, reward, next_state, float(done_for_replay))
        
        # Update statistics
        current_reward += reward
        current_length += 1
        agent.steps_done += 1
        
        # Perform training step once replay buffer has sufficient data
        if step >= params.learning_starts and step % params.train_frequency == 0:
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
        
        # Periodically update target network (typically every 1000 steps)
        # Key insight: Separate target network prevents "moving target" problem
        # where both Q-values and TD targets change simultaneously, causing
        # training instability. Periodic updates keep targets fixed for multiple
        # gradient steps, significantly improving convergence stability.
        # This is one of the critical innovations that made DQN successful.
        if step % params.target_update_frequency == 0:
            agent.update_target_network()
        
        # Update epsilon for exploration-exploitation trade-off
        agent.update_epsilon()
        
        # Handle episode termination (all lives exhausted or truncated)
        if real_done:
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)
            episode_epsilons.append(agent.epsilon)
            agent.episodes_done += 1
            
            # Log progress at regular intervals
            if agent.episodes_done % params.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-params.log_interval:])
                avg_loss = np.mean(losses[-100:]) if losses else 0
                pbar.set_postfix({
                    'episode': agent.episodes_done,
                    'avg_reward': f'{avg_reward:.2f}',
                    'epsilon': f'{agent.epsilon:.3f}',
                    'loss': f'{avg_loss:.4f}'
                })
            
            # Reset environment for new episode
            obs, info = env.reset()
            # Apply random no-ops for initial state variation
            max_no_ops = 30
            no_ops = np.random.randint(0, max_no_ops + 1)
            for _ in range(no_ops):
                obs, _, _, _, _ = env.step(0)  # No-operation
            
            state = frame_stack.reset(obs)
            lives = info.get('lives', 5)
            need_fire = True
            current_reward = 0
            current_length = 0
        else:
            state = next_state
        
        # Save checkpoint at regular intervals for training resumption
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
        
        # Set random seeds for reproducibility
        # Each configuration uses offset seed for independent runs
        torch.manual_seed(params.seed + config_idx)        # PyTorch CPU
        np.random.seed(params.seed + config_idx)           # NumPy
        random.seed(params.seed + config_idx)              # Python random
        if torch.cuda.is_available():
            torch.cuda.manual_seed(params.seed + config_idx)      # PyTorch CUDA
            torch.cuda.manual_seed_all(params.seed + config_idx)  # All GPUs
            # Enable deterministic mode for reproducible results
            # Note: May reduce performance but ensures experimental consistency
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Initialize Atari environment
        env = gym.make('ALE/Breakout-v5', render_mode=params.render_mode)
        
        # Determine state and action space dimensions
        obs, _ = env.reset()
        n_actions = env.action_space.n
        state_shape = (4, 84, 84)  # 4 stacked 84x84 frames
        
        # Initialize DQN agent with mixed precision training
        agent = DQNAgent(state_shape, n_actions, params, device, use_amp=True)
        
        # Create directory structure for saving checkpoints and results
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
    
    # Parse command-line arguments for HPC cluster execution
    if len(sys.argv) > 1:
        # Single config mode (for SLURM array jobs)
        config_idx = int(sys.argv[1])
        gpu_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        
        print(f"\n[HPC MODE] Training config {config_idx} on GPU {gpu_id}")
        train_single_config(config_idx, gpu_id)
        
    else:
        # Local execution mode with multi-GPU support
        num_gpus = torch.cuda.device_count()
        num_configs = len(configs)
        
        print(f"\n[PARALLEL MODE] {num_gpus} GPUs available, {num_configs} configs to train")
        
        if num_gpus > 1 and num_configs > 1:
            # Parallel training across multiple GPUs
            print("Starting parallel training across multiple GPUs...")
            
            # Set multiprocessing start method
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass  # Start method already configured
            
            processes = []
            
            for i in range(min(num_configs, num_gpus)):
                p = mp.Process(target=train_single_config, args=(i, i))
                p.start()
                processes.append(p)
            
            # Wait for all processes to complete
            for i, p in enumerate(processes):
                p.join()
                if p.exitcode != 0:
                    print(f"WARNING: Process {i} exited with code {p.exitcode}")
            
            print("\n[DONE] Parallel training complete!")
            
        else:
            # Sequential training mode (insufficient GPUs for parallelization)
            print("Sequential training (insufficient GPUs for parallel execution)...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            
            results = {}
            for idx, config in enumerate(configs):
                name, result = train_single_config(idx, 0)
                results[name] = result
        
        print("\nTraining complete! Use generate_plots.py to create visualizations.")
