from params import Params

TOTAL_TIMESTEPS = 10_000_000

configs = [
    # Config 1: Baseline - Standard DQN (Nature paper settings adapted for Breakout)
    # Expected: Steady, reliable learning. Should reach 30-50 points by 10M steps.
    {
        "name": "Baseline",
        "params": Params(
            total_timesteps=TOTAL_TIMESTEPS,
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=5_000_000,   # 50% of training
            buffer_size=1_000_000,            # 1M buffer with uint8
            batch_size=32,
            learning_starts=100_000,          # Start training after 100k steps
            target_update_frequency=1_000,
            train_frequency=4,
            hidden_dim=512,
            seed=42,
            render_mode=None,
            log_interval=10,
            save_interval=500_000,
        )
    },

    # Config 2: Fast Learner - Aggressive but stable learning
    # Hypothesis: Faster initial learning with frequent updates achieves good results sooner
    # Expected: Quick early gains, potential for 40-60+ points with stable convergence
    {
        "name": "Fast_Learner",
        "params": Params(
            total_timesteps=TOTAL_TIMESTEPS,
            learning_rate=1.5e-4,            # 1.5x higher learning rate (stable)
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=3_000_000,   # 30% - Quick but not rushed
            buffer_size=1_000_000,
            batch_size=64,                   # Larger batch for stability
            learning_starts=80_000,          # Start slightly earlier
            target_update_frequency=500,     # 2x more frequent target updates
            train_frequency=4,
            hidden_dim=512,
            seed=42,
            render_mode=None,
            log_interval=10,
            save_interval=500_000,
        )
    },

    # Config 3: Deep Explorer - Extended exploration with conservative updates
    # Hypothesis: Thorough exploration discovers diverse strategies and better long-term play
    # Expected: Slower initial learning, potentially higher final performance (50-80+ points)
    {
        "name": "Deep_Explorer",
        "params": Params(
            total_timesteps=TOTAL_TIMESTEPS,
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.07,                # Mild continued exploration (7%)
            epsilon_decay_steps=7_000_000,   # Explore for 70% of training
            buffer_size=1_000_000,
            batch_size=32,
            learning_starts=120_000,         # More data before training
            target_update_frequency=1_500,   # More conservative target updates
            train_frequency=4,
            hidden_dim=512,
            seed=42,
            render_mode=None,
            log_interval=10,
            save_interval=500_000,
        )
    },
]
