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

def predecessor(sp: int, action: int, shape: Coords):
    x, y = getCoords(sp, shape)

    # UP
    if action == 0:
        if y == 0:
            return None

        s = getState((x, y - 1), shape)

        if y == shape[1] - 1:
            return sp, s

        return s

    # RIGHT
    elif action == 1:
        if x == 0:
            return None

        s = getState((x - 1, y), shape)

        if x == shape[0] - 1:
            return sp, s

        return s

    # DOWN
    elif action == 2:
        if y == shape[1] - 1:
            return None

        s = getState((x, y + 1), shape)

        if y == 0:
            return sp, s

        return s

    # LEFT
    else:
        if x == shape[0] - 1:
            return None

        s = getState((x + 1, y), shape)

        if x == 0:
            return sp, s

        return s

def predecessors(sp: int, shape: Coords):
    for a in range(4):
        s = predecessor(sp, a, shape)
        if s is None:
            continue

        elif isinstance(s, tuple):
            for _s in s:
                yield _s, a

        else:
            yield s, a
