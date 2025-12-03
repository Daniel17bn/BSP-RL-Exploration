from typing import NamedTuple

class Params(NamedTuple):
    total_episodes: int
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_decay: float
    epsilon_min: float
    seed: int
    is_rainy: bool
    fickle_passenger: bool
    n_runs: int
    action_size: int
    state_size: int
