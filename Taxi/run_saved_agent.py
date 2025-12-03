# run_saved_agent.py

import pickle
import numpy as np
import gymnasium as gym
from params import Params  # Import the real Params class so pickle can reconstruct it

SAVE_PATH = "saved_agents/taxi_qtable.pkl"

# Load saved Q-table + parameters
with open(SAVE_PATH, "rb") as f:
    data = pickle.load(f)

qtable = data["qtable"]
params = data["params"]

# Create environment with saved parameters
env = gym.make(
    "Taxi-v3",
    is_rainy=getattr(params, "is_rainy", False),
    fickle_passenger=getattr(params, "fickle_passenger", False),
    render_mode="human",
)

n_test_episodes = 50

for ep in range(n_test_episodes):
    state, info = env.reset() # no seed to have random starts each episode
    done = False
    total_reward = 0
    steps = 0

    while not done:
        # Pick best learned action
        action = int(np.argmax(qtable[state]))

        # Step environment
        new_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1
        done = terminated or truncated
        state = new_state

    print(f"Episode {ep+1}: reward={total_reward}, steps={steps}")

env.close()
