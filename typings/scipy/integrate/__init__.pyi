import numpy as np
from typing import Callable, Sequence

class SolverOutput:
    y: np.ndarray

def solve_ivp(dsdt: Callable[[float, np.ndarray], np.ndarray], dt: Sequence[float], s0: np.ndarray) -> SolverOutput:
    pass
