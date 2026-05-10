"""Flyby math sanity checks."""
import numpy as np

from hermes.bodies import get_body
from hermes.flyby import max_turn_angle, powered_flyby_dv


def test_max_turn_angle_jupiter_low_v_inf():
    # Low v_inf at Jupiter periapsis = strong turn angle
    jup = get_body("Jupiter")
    delta = max_turn_angle(v_inf=5.0, r_p=jup.radius + 200_000.0, mu=jup.mu)
    assert np.rad2deg(delta) > 60.0  # Voyager-class flybys turn many tens of degrees


def test_unpowered_flyby_zero_dv():
    # Same |v_inf| in/out, fully achievable turn angle => Δv ~ 0
    jup = get_body("Jupiter")
    v_in = np.array([5.0, 0.0, 0.0])
    # Rotate by a small angle in-plane
    theta = np.deg2rad(20.0)
    v_out = np.array([5.0 * np.cos(theta), 5.0 * np.sin(theta), 0.0])
    dv, deficit = powered_flyby_dv(v_in, v_out, jup, r_p=jup.radius + 200_000.0)
    assert dv < 1e-6
    assert deficit == 0.0
