from typing import Any, Sequence, Tuple

from PyRlEnvs.utils.math import try2FastJit, clipInt

Coords = Tuple[int, int]

@try2FastJit
def getState(coords: Coords, shape: Coords) -> int:
    x, y = coords
    mx, my = shape

    s = clipInt(y, 0, my) + clipInt(x, 0, mx) * my
    return int(s)

@try2FastJit
def getCoords(state: int, shape: Coords) -> Coords:
    x = int(state // shape[1])
    y = int(state % shape[1])

    return (x, y)

def findFirstTrigger(arr: Sequence[Any], s: int, a: int, sp: int):
    for element in arr:
        if element.trigger(s, a, sp) and element.name:
            return element

    return None

@try2FastJit
def predecessor(sp: int, action: int, shape: Coords):
    x, y = getCoords(sp, shape)

    # need to trick numba into knowing the type of this list
    empty = [i for i in range(0)]

    # UP
    if action == 0:
        if y == 0:
            return empty

        s = getState((x, y - 1), shape)

        if y == shape[1] - 1:
            return [sp, s]

        return [s]

    # RIGHT
    elif action == 1:
        if x == 0:
            return empty

        s = getState((x - 1, y), shape)

        if x == shape[0] - 1:
            return [sp, s]

        return [s]

    # DOWN
    elif action == 2:
        if y == shape[1] - 1:
            return empty

        s = getState((x, y + 1), shape)

        if y == 0:
            return [sp, s]

        return [s]

    # LEFT
    else:
        if x == shape[0] - 1:
            return empty

        s = getState((x + 1, y), shape)

        if x == 0:
            return [sp, s]

        return [s]

def predecessors(sp: int, shape: Coords):
    for a in range(4):
        s = predecessor(sp, a, shape)
        for _s in s:
            yield _s, a
