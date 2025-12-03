from params import Params

configs = [
        {
            "name": "Discount factor 0.9",
            "params": Params(
                total_episodes=10000,
                learning_rate=0.1,
                gamma=0.9,
                epsilon_start=1.0,
                epsilon_decay=0.9995,
                epsilon_min=0.01,
                seed=42,
                is_rainy=False,
                fickle_passenger=False,
                action_size=6,
                state_size=500,
            )
        },
        {
            "name": "Discount factor 0.95",
            "params": Params(
                total_episodes=10000,
                learning_rate=0.1,
                gamma=0.95,
                epsilon_start=1.0,
                epsilon_decay=0.9995,
                epsilon_min=0.01,
                seed=42,
                is_rainy=False,
                fickle_passenger=False,
                action_size=6,
                state_size=500,
            )
        },
        {
            "name": "Discount factor 0.98",
            "params": Params(
                total_episodes=10000,
                learning_rate=0.1,
                gamma=0.98,
                epsilon_start=1.0,
                epsilon_decay=0.9995,
                epsilon_min=0.01,
                seed=42,
                is_rainy=False,
                fickle_passenger=False,
                action_size=6,
                state_size=500,
            )
        },
    ]