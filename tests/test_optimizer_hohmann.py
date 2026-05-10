"""Smoke test: Earth->Mars optimizer should converge near analytical Hohmann ΔV.

Marked slow because it runs differential_evolution with real ephemeris.
"""
import pytest

from hermes.io import load_mission
from hermes.optimizer import optimize


@pytest.mark.slow
def test_earth_mars_hohmann_convergence(tmp_path):
    spec = load_mission("examples/earth_mars_hohmann.yaml")
    spec.optimizer = {"popsize": 15, "maxiter": 50, "seed": 0, "disp": False}
    sol = optimize(spec)
    # v_inf at Earth for a Hohmann to Mars is ~2.94 km/s; loose bound.
    assert sol.launch_dv_kms < 5.0
    assert 200 < (sol.legs[0].t_arrive_mjd - sol.legs[0].t_depart_mjd) < 320
