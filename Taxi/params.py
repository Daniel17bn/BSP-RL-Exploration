from typing import NamedTuple

class Params(NamedTuple):
    total_episodes: int
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_decay: float
    epsilon_min: float
    seed: int # not used for this env since the passenger starting location should be random
    is_rainy: bool
    fickle_passenger: bool
    action_size: int
    state_size: int
