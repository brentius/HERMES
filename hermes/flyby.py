"""Patched-conic gravity-assist (flyby) math.

Reference frame: planet-centered, but only magnitudes/angles matter here.
A flyby instantaneously rotates the v-infinity vector by an angle 2*delta,
where sin(delta) = 1 / (1 + r_p * v_inf^2 / mu). Powered flybys add an
impulsive Δv at periapsis.
"""
from __future__ import annotations

import numpy as np

from .bodies import Body


def max_turn_angle(v_inf: float, r_p: float, mu: float) -> float:
    """Maximum achievable v_inf turn angle (radians) at periapsis radius r_p."""
    e = 1.0 + r_p * v_inf * v_inf / mu
    return 2.0 * np.arcsin(1.0 / e)


def turn_angle_between(v_inf_in: np.ndarray, v_inf_out: np.ndarray) -> float:
    """Angle (radians) between incoming and outgoing v_inf vectors."""
    cos_a = np.clip(
        np.dot(v_inf_in, v_inf_out) / (np.linalg.norm(v_inf_in) * np.linalg.norm(v_inf_out)),
        -1.0, 1.0,
    )
    return float(np.arccos(cos_a))


def rotate_v_inf(v_inf_in: np.ndarray, delta: float, b_angle: float) -> np.ndarray:
    """Rotate v_inf_in by turn angle `delta` around an axis defined by `b_angle`.

    The axis lies in the B-plane (perpendicular to v_inf_in). `b_angle` selects
    which B-plane direction we rotate about (0 = ecliptic-north-ish reference).
    """
    v = np.asarray(v_inf_in, dtype=float)
    v_hat = v / np.linalg.norm(v)
    # Build an orthonormal basis (v_hat, e1, e2) in the B-plane.
    ref = np.array([0.0, 0.0, 1.0]) if abs(v_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(v_hat, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(v_hat, e1)
    axis = np.cos(b_angle) * e1 + np.sin(b_angle) * e2
    # Rodrigues rotation of v about `axis` by `delta`.
    c, s = np.cos(delta), np.sin(delta)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def powered_flyby_dv(
    v_inf_in: np.ndarray,
    v_inf_out: np.ndarray,
    body: Body,
    r_p: float,
) -> tuple[float, float]:
    """ΔV (km/s) and turn-angle deficit (rad) for a powered flyby.

    Strategy: turn the v_inf vector as far as the geometry allows at radius r_p
    using planet's gravity (free), then apply impulsive Δv at periapsis to make
    up the speed and remaining direction. We approximate the powered cost as
    the magnitude difference of the periapsis-velocity vectors of the two
    hyperbolas (incoming and outgoing), assuming r_p is shared.
    """
    v_in_mag = float(np.linalg.norm(v_inf_in))
    v_out_mag = float(np.linalg.norm(v_inf_out))
    required_turn = turn_angle_between(v_inf_in, v_inf_out)
    # Use the smaller v_inf to determine the bounding turn capability.
    delta_max_in = max_turn_angle(v_in_mag, r_p, body.mu)
    delta_max_out = max_turn_angle(v_out_mag, r_p, body.mu)
    deficit = max(0.0, required_turn - min(delta_max_in, delta_max_out))
    # Periapsis speeds of the two hyperbolas (vis-viva on hyperbola at r_p):
    vp_in = np.sqrt(v_in_mag ** 2 + 2.0 * body.mu / r_p)
    vp_out = np.sqrt(v_out_mag ** 2 + 2.0 * body.mu / r_p)
    dv = abs(vp_out - vp_in)
    return dv, deficit
