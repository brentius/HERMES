"""Lambert solver sanity check — Curtis 'Orbital Mechanics' Example 5.2.

r1 = (5000, 10000, 2100) km, r2 = (-14600, 2500, 7000) km, tof = 3600 s,
mu_earth. Expected v1 ≈ (-5.992, 1.925, 3.246) km/s.
"""
import numpy as np
import pytest

from hermes.lambert import solve_lambert


def test_curtis_example_5_2():
    mu_earth = 398_600.4418
    r1 = np.array([5000.0, 10000.0, 2100.0])
    r2 = np.array([-14600.0, 2500.0, 7000.0])
    tof = 3600.0
    v1, v2 = solve_lambert(r1, r2, tof, mu=mu_earth, prograde=True)
    np.testing.assert_allclose(v1, [-5.9925, 1.9254, 3.2456], atol=5e-3)
    np.testing.assert_allclose(v2, [-3.3125, -4.1966, -0.38529], atol=5e-3)
