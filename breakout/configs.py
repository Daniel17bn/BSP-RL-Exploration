from params import Params

TOTAL_TIMESTEPS = 10_000_000

# Experiment 1: Breakout DQN - Three Contrasting Strategies
# Compare different learning philosophies to identify what matters most for Breakout
configs = [
    # Config 1: Baseline - Standard DQN (Nature paper settings adapted for Breakout)
    {
        "name": "Baseline",
        "params": Params(
            total_timesteps=TOTAL_TIMESTEPS,
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=5_000_000,   # 50% of training
            buffer_size=1_000_000,
            batch_size=32,
            learning_starts=100_000,
            target_update_frequency=1_000,
            train_frequency=4,
            hidden_dim=512,
            seed=42,
            render_mode=None,
            log_interval=10,
            save_interval=500_000,
        )
    },

    # Config 2: Fast Learner - Aggressive learning with higher LR and frequent updates
    # Hypothesis: Faster convergence may help in simple environments like Breakout
    {
        "name": "Fast_Learner",
        "params": Params(
            total_timesteps=TOTAL_TIMESTEPS,
            learning_rate=2.5e-4,            # 2.5x higher learning rate
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=2_000_000,   # Fast epsilon decay (20% of training)
            buffer_size=1_000_000,
            batch_size=64,                   # Larger batch for stability with high LR
            learning_starts=100_000,
            target_update_frequency=500,     # 2x more frequent target updates
            train_frequency=4,
            hidden_dim=512,
            seed=42,
            render_mode=None,
            log_interval=10,
            save_interval=500_000,
        )
    },

    # Config 3: Deep Explorer - Extended exploration with higher final epsilon
    # Hypothesis: Thorough exploration may discover better long-term strategies
    {
        "name": "Deep_Explorer",
        "params": Params(
            total_timesteps=TOTAL_TIMESTEPS,
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.10,                # Keep 10% exploration indefinitely
            epsilon_decay_steps=8_000_000,   # Explore for 80% of training
            buffer_size=1_000_000,
            batch_size=32,
            learning_starts=100_000,
            target_update_frequency=2_000,   # Slower, more stable target updates
            train_frequency=4,
            hidden_dim=512,
            seed=42,
            render_mode=None,
            log_interval=10,
            save_interval=500_000,
        )
    },
]
