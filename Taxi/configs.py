from params import Params

configs = [
        {
            "name": "Config-1 (baseline)",
            "params": Params(
                total_episodes=50_000,
                learning_rate=0.1,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_decay=0.99995,
                epsilon_min=0.05,
                seed=42,
                is_rainy=False,
                fickle_passenger=False,
                n_runs=1,
                action_size=6,
                state_size=500,
            )
        },
        {
            "name": "Config-2 (faster decay)",
            "params": Params(
                total_episodes=50_000,
                learning_rate=0.1,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_decay=0.9999,
                epsilon_min=0.05,
                seed=42,
                is_rainy=False,
                fickle_passenger=False,
                n_runs=1,
                action_size=6,
                state_size=500,
            )
        },
        {
            "name": "Config-3 (higher LR)",
            "params": Params(
                total_episodes=50_000,
                learning_rate=0.3,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_decay=0.99995,
                epsilon_min=0.05,
                seed=42,
                is_rainy=False,
                fickle_passenger=False,
                n_runs=1,
                action_size=6,
                state_size=500,
            )
        },
    ]