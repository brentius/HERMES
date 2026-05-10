"""Planet constants and ephemeris lookup.

Units: km, s, kg. Ephemeris frame: ICRS / J2000 ecliptic-aligned heliocentric.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    GCRS,
    HeliocentricTrueEcliptic,
    get_body_barycentric_posvel,
    solar_system_ephemeris,
)
from astropy.time import Time

# Use the high-accuracy JPL DE440 kernel; astropy will download on first use.
solar_system_ephemeris.set("de440")

MU_SUN = 1.32712440018e11  # km^3/s^2


@dataclass(frozen=True)
class Body:
    name: str
    mu: float           # km^3/s^2
    radius: float       # km (mean equatorial)
    min_flyby_alt: float  # km — minimum allowed altitude above surface for a flyby
    soi: float          # km — sphere of influence (approximate)


BODIES: dict[str, Body] = {
    "Sun":     Body("Sun",     MU_SUN,           695_700.0,        0.0,        np.inf),
    "Mercury": Body("Mercury", 2.2032e4,          2_439.7,        200.0,    1.12e5),
    "Venus":   Body("Venus",   3.24859e5,         6_051.8,        300.0,    6.16e5),
    "Earth":   Body("Earth",   3.986004418e5,     6_378.137,      300.0,    9.24e5),
    "Mars":    Body("Mars",    4.282837e4,        3_389.5,        200.0,    5.77e5),
    "Jupiter": Body("Jupiter", 1.26686534e8,     69_911.0,      2_000.0,    4.82e7),
    "Saturn":  Body("Saturn",  3.7931187e7,      58_232.0,      1_000.0,    5.46e7),
    "Uranus":  Body("Uranus",  5.793939e6,       25_362.0,        500.0,    5.18e7),
    "Neptune": Body("Neptune", 6.836529e6,       24_622.0,        500.0,    8.66e7),
}


def get_body(name: str) -> Body:
    try:
        return BODIES[name]
    except KeyError as e:
        raise ValueError(f"Unknown body: {name}. Known: {list(BODIES)}") from e


@lru_cache(maxsize=4096)
def _state_cached(body_name: str, mjd: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    t = Time(mjd, format="mjd", scale="tdb")
    pos, vel = get_body_barycentric_posvel(body_name.lower(), t)
    # Convert from ICRS barycentric to heliocentric ecliptic.
    sun_pos, sun_vel = get_body_barycentric_posvel("sun", t)
    r_icrs = (pos - sun_pos).xyz.to(u.km).value
    v_icrs = (vel - sun_vel).xyz.to(u.km / u.s).value
    # Rotate ICRS -> ecliptic of J2000 (obliquity ~23.4393°).
    eps = np.deg2rad(23.43928)
    R = np.array([
        [1, 0, 0],
        [0, np.cos(eps), np.sin(eps)],
        [0, -np.sin(eps), np.cos(eps)],
    ])
    r = R @ r_icrs
    v = R @ v_icrs
    return tuple(r), tuple(v)


def state(body_name: str, mjd: float) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric ecliptic-J2000 position (km) and velocity (km/s) at MJD (TDB)."""
    if body_name == "Sun":
        return np.zeros(3), np.zeros(3)
    r, v = _state_cached(body_name, float(mjd))
    return np.array(r), np.array(v)
