import numpy as np
from typing import Callable

def rungeKutta(derivs: Callable[[np.ndarray, float], np.ndarray], y0: np.ndarray, ts: np.ndarray):
    yout = np.zeros((len(ts), len(y0)), dtype=y0.dtype)

    yout[0] = y0

    for i in range(len(ts) - 1):
        t = ts[i]
        dt = ts[i + 1] - t
        y0 = yout[i]

        k1 = derivs(y0, t)
        k2 = derivs(y0 + (dt / 2.0) * k1, t + (dt / 2.0))
        k3 = derivs(y0 + (dt / 2.0) * k2, t + (dt / 2.0))
        k4 = derivs(y0 + dt * k3, t + dt)

        yout[i + 1] = y0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return yout

def euler(derivs: Callable[[np.ndarray, float], np.ndarray], y0: np.ndarray, ts: np.ndarray):
    yout = np.zeros((len(ts), len(y0)), dtype=y0.dtype)

    yout[0] = y0

    for i in range(len(ts) - 1):
        t = ts[i]
        dt = ts[i + 1] - t

        y0 = yout[i]
        ydot = derivs(y0, t)

        yout[i + 1] = y0 + ydot * dt

    return yout
