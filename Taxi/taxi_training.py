import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import gymnasium as gym
from tqdm import tqdm
from typing import NamedTuple

from params import Params

sns.set_theme()


# ===========================================================
# Q-LEARNING
# ===========================================================

class Qlearning:
    def __init__(self, lr, gamma, state_size, action_size):
        self.lr = lr
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.reset_qtable()

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))

    def update(self, state, action, reward, new_state):
        best_next = np.max(self.qtable[new_state])
        td_error = reward + self.gamma * best_next - self.qtable[state, action]
        self.qtable[state, action] += self.lr * td_error

# ===========================================================
# EPSILON-GREEDY POLICY
# ===========================================================

class EpsilonGreedy:
    def __init__(self, epsilon, epsilon_min, epsilon_decay, rng=None):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = rng if rng is not None else np.random.default_rng()

    def choose_action(self, action_space, state, qtable):
        if self.rng.random() < self.epsilon:
            return action_space.sample()
        max_ids = np.where(qtable[state] == np.max(qtable[state]))[0]
        return int(self.rng.choice(max_ids))

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ===========================================================
# TRAINING FUNCTION
# ===========================================================

def run_env(env, learner, explorer, params):
    rewards_log = np.zeros((params.total_episodes, params.n_runs))
    lengths_log = np.zeros((params.total_episodes, params.n_runs))
    eps_log = np.zeros((params.total_episodes, params.n_runs))
    q_change_log = np.zeros((params.total_episodes, params.n_runs))
    qtables_final = np.zeros((params.n_runs, params.state_size, params.action_size))

    # Seed the action space for reproducibility
    env.action_space.seed(params.seed)
    
    episodes = np.arange(params.total_episodes)

    for run in range(params.n_runs):
        learner.reset_qtable()
        explorer.epsilon = params.epsilon_start

        for ep in tqdm(episodes, desc=f"Run {run+1}/{params.n_runs}", leave=False):
            eps_log[ep, run] = explorer.epsilon
            q_prev = learner.qtable.copy()

            state, _ = env.reset() #no seed here to have random starts each run
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action = explorer.choose_action(env.action_space, state, learner.qtable)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                learner.update(state, action, reward, next_state)
                total_reward += reward
                steps += 1
                state = next_state

            rewards_log[ep, run] = total_reward
            lengths_log[ep, run] = steps
            q_change_log[ep, run] = np.max(np.abs(learner.qtable - q_prev))

            explorer.decay()

        qtables_final[run] = learner.qtable

    return rewards_log, lengths_log, eps_log, q_change_log, qtables_final

# ===========================================================
# MOVING AVERAGE
# ===========================================================

def moving_average(x, window=200):
    if window <= 1:
        return x
    return np.convolve(x, np.ones(window) / window, mode="same")

# ===========================================================
# PLOTTING
# ===========================================================

def plot_results(rewards, lengths, epsilons):
    mean_rewards = rewards.mean(axis=1)
    mean_lengths = lengths.mean(axis=1)
    mean_eps = epsilons.mean(axis=1)

    # 1. Rewards raw
    plt.figure(figsize=(12, 5))
    plt.plot(mean_rewards, alpha=0.7)
    plt.title("Reward per Episode (mean across runs)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()

    # 2. Moving average
    plt.figure(figsize=(12, 5))
    plt.plot(moving_average(mean_rewards), linewidth=2)
    plt.title("Moving Average Reward (window=200)")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.grid()
    plt.show()

    # 3. Episode lengths
    plt.figure(figsize=(12, 5))
    plt.plot(mean_lengths)
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid()
    plt.show()

    # 4. Epsilon curve
    plt.figure(figsize=(12, 5))
    plt.plot(mean_eps)
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid()
    plt.show()

# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":
    params = Params(
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

    # Set numpy random seed for reproducibility
    np.random.seed(params.seed)
    rng = np.random.default_rng(params.seed)

    env = gym.make(
        "Taxi-v3",
        is_rainy=params.is_rainy,
        fickle_passenger=params.fickle_passenger
    )

    learner = Qlearning(params.learning_rate, params.gamma, params.state_size, params.action_size)
    explorer = EpsilonGreedy(params.epsilon_start, params.epsilon_min, params.epsilon_decay, rng=rng)

    rewards, lengths, epsilons, q_changes, qtables = run_env(env, learner, explorer, params)

    # PLOTS
    plot_results(rewards, lengths, epsilons)

    # SAVE RESULTS
    save_data = {
        "qtable": qtables[0],
        "params": params,
        "rewards": rewards,
        "lengths": lengths,
        "epsilons": epsilons,
        "q_changes": q_changes,
    }

    with open("saved_agents/taxi_qtable.pkl", "wb") as f:
        pickle.dump(save_data, f)

    print("Training complete. Model saved.")
