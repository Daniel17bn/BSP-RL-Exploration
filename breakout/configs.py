from params import Params

TOTAL_TIMESTEPS = 10_000_000

configs = [
    # Baseline - Standard DQN settings
    {
        "name": "Baseline",
        "params": Params(
            total_timesteps=TOTAL_TIMESTEPS,
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=5_000_000,
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

    # Fast Learner - Higher learning rate and faster exploration decay
    {
        "name": "Fast_Learner",
        "params": Params(
            total_timesteps=TOTAL_TIMESTEPS,
            learning_rate=1.5e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=3_000_000,
            buffer_size=1_000_000,
            batch_size=64,
            learning_starts=80_000,
            target_update_frequency=500,
            train_frequency=4,
            hidden_dim=512,
            seed=42,
            render_mode=None,
            log_interval=10,
            save_interval=500_000,
        )
    },

    # Deep Explorer - Extended exploration period
    {
        "name": "Deep_Explorer",
        "params": Params(
            total_timesteps=TOTAL_TIMESTEPS,
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.07,
            epsilon_decay_steps=7_000_000,
            buffer_size=1_000_000,
            batch_size=32,
            learning_starts=120_000,
            target_update_frequency=1_500,
            train_frequency=4,
            hidden_dim=512,
            seed=42,
            render_mode=None,
            log_interval=10,
            save_interval=500_000,
        )
    },
]
