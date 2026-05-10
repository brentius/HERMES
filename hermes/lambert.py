"""Lambert's problem solver.

Thin wrapper over lamberthub.izzo2015 returning heliocentric departure/arrival
velocities in km/s for a given pair of position vectors and time-of-flight.
"""
from __future__ import annotations

import numpy as np
from lamberthub import izzo2015

from .bodies import MU_SUN


def solve_lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    tof_seconds: float,
    mu: float = MU_SUN,
    prograde: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve Lambert's problem.

    Parameters
    ----------
    r1, r2 : (3,) arrays in km
    tof_seconds : time of flight in seconds
    mu : gravitational parameter in km^3/s^2
    prograde : True for prograde transfer (standard for solar-system missions)

    Returns
    -------
    v1, v2 : departure and arrival velocities in km/s
    """
    v1, v2 = izzo2015(mu, np.asarray(r1), np.asarray(r2), tof_seconds, prograde=prograde)
    return np.asarray(v1), np.asarray(v2)
