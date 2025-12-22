from typing import NamedTuple

class Params(NamedTuple):
    # Training parameters
    total_timesteps: int
    learning_rate: float
    gamma: float  # discount factor
    
    # Exploration parameters
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    
    # Experience replay
    buffer_size: int
    batch_size: int
    learning_starts: int  # timesteps before training starts
    
    # Network updates
    target_update_frequency: int  # update target network every N steps
    train_frequency: int  # train every N steps
    
    # Network architecture
    hidden_dim: int
    
    # Environment
    seed: int
    render_mode: str  # None, "human", or "rgb_array"
    
    # Logging
    log_interval: int  # log stats every N episodes
    save_interval: int  # save model every N timesteps
