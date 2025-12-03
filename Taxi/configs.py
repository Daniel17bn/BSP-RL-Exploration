from params import Params

configs = [
        {
            "name": "Example 1",
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
                action_size=6,
                state_size=500,
            )
        },
        {
            "name": "Example 2",
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
                action_size=6,
                state_size=500,
            )
        },
        {
            "name": "Example 3",
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
                action_size=6,
                state_size=500,
            )
        },
    ]