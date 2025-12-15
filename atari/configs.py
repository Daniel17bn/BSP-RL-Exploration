from params import Params

# HPC-Optimized configurations for Breakout DQN experiments
# - buffer_size reduced to 50K (saves 10GB RAM, fits in 25-30GB total)
# - save_interval increased to 100K (less disk I/O on HPC)
# - Maintains research quality hyperparameters

configs = [
    # Baseline Configuration
    {
        "name": "Baseline",
        "params": Params(
            total_timesteps=1_000_000,
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay_steps=100_000,
            buffer_size=50_000,          
            batch_size=32,
            learning_starts=10_000,
            target_update_frequency=1_000,
            train_frequency=4,
            hidden_dim=512,
            seed=42,
            render_mode=None,
            log_interval=10,
            save_interval=100_000,       
        )
    },
    
    # High Learning Rate
    {
        "name": "High Learning Rate",
        "params": Params(
            total_timesteps=1_000_000,
            learning_rate=5e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay_steps=100_000,
            buffer_size=50_000,          
            batch_size=32,
            learning_starts=10_000,
            target_update_frequency=1_000,
            train_frequency=4,
            hidden_dim=512,
            seed=42,
            render_mode=None,
            log_interval=10,
            save_interval=100_000,       
        )
    },
    
    # Larger Batch Size
    {
        "name": "Larger Batch",
        "params": Params(
            total_timesteps=1_000_000,
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay_steps=100_000,
            buffer_size=50_000,          
            batch_size=64,               
            learning_starts=10_000,
            target_update_frequency=1_000,
            train_frequency=4,
            hidden_dim=512,
            seed=42,
            render_mode=None,
            log_interval=10,
            save_interval=100_000,       
        )
    },
    
]
