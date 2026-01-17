from params import Params


configs = [
    {
        "name": "High LR + Fast Decay",
        "params": Params(
            total_episodes=10000,
            learning_rate=0.5,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_decay=0.9990,
            epsilon_min=0.01,
            seed=42,
            is_rainy=False,
            fickle_passenger=False,
            action_size=6,
            state_size=500,
        )
    },
    
   
    {
        "name": "Optimal Baseline",
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
    
    {
        "name": "Low LR + Slow Decay",
        "params": Params(
            total_episodes=10000,
            learning_rate=0.01,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_decay=0.9999,
            epsilon_min=0.01,
            seed=42,
            is_rainy=False,
            fickle_passenger=False,
            action_size=6,
            state_size=500,
        )
    },
    
    {
        "name": "Low Discount Factor",
        "params": Params(
            total_episodes=10000,
            learning_rate=0.1,
            gamma=0.8,
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
