import numpy as np
from typing import Any, Sequence, Tuple

Coords = Tuple[int, int]

def getState(coords: Coords, shape: Coords) -> int:
    # use clip mode to handle out-of-bounds
    # if agent bumps into wall, will just stay put instead
    return np.ravel_multi_index(coords, shape, mode='clip')

def getCoords(state: int, shape: Coords) -> Coords:
    return np.unravel_index(state, shape)

def findFirstTrigger(arr: Sequence[Any], s: int, a: int, sp: int):
    for element in arr:
        if element.trigger(s, a, sp) and element.name:
            return element

    return None
