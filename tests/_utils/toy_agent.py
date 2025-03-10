import numpy as np
from typing import Any
from rlglue.agent import BaseAgent

class ToyAgent(BaseAgent):
    def __init__(self, actions: int):
        super().__init__()
        self.actions = actions
        self.rng = np.random.RandomState(0)

    def start(self, observation: Any) -> int:
        return self.rng.randint(self.actions)

    def step(self, reward: float, observation: Any, extra={}) -> int:
        return self.rng.randint(self.actions)

    def end(self, reward: float, extra={}) -> None:
        ...
