from params import Params

# Example configurations for Taxi Q-Learning experiments

configs = [
    # Baseline Configuration
    {
        "name": "Baseline",
        "params": Params(
            total_episodes=10000,
            learning_rate=0.1,
            gamma=0.99,
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
    
    # High Learning Rate
    {
        "name": "High Learning Rate",
        "params": Params(
            total_episodes=10000,
            learning_rate=0.5,
            gamma=0.99,
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
    
    # Fast Epsilon Decay
    {
        "name": "Fast Decay",
        "params": Params(
            total_episodes=10000,
            learning_rate=0.1,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            seed=42,
            is_rainy=False,
            fickle_passenger=False,
            action_size=6,
            state_size=500,
        )
    },
    
    # Complex Environment
    {
        "name": "Complex Environment",
        "params": Params(
            total_episodes=10000,
            learning_rate=0.1,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_decay=0.9995,
            epsilon_min=0.01,
            seed=42,
            is_rainy=True,
            fickle_passenger=True,
            action_size=6,
            state_size=500,
        )
    },
]