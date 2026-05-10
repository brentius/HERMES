"""REBOUND N-body verification of a patched-conic solution.

Builds a heliocentric simulation with the Sun and the bodies involved in the
mission, seeds the spacecraft as a massless test particle with the optimizer's
post-launch state, applies impulsive Δv at each flyby epoch (matching the
optimizer's outgoing v_inf), and reports the closest-approach distance to each
flyby body and the final position miss to the target.
"""
from __future__ import annotations

import numpy as np
import rebound

from .bodies import MU_SUN, get_body, state
from .mission import Solution

DAY = 86400.0
G_SI = 6.67430e-20  # km^3/(kg s^2) — REBOUND uses G=1 by default; we set units instead.


def _setup_sim(epoch_mjd: float, body_names: list[str]) -> rebound.Simulation:
    sim = rebound.Simulation()
    sim.units = ("km", "s", "kg")
    # Sun
    sun = get_body("Sun")
    M_sun = sun.mu / sim.G
    sim.add(m=M_sun, x=0, y=0, z=0, vx=0, vy=0, vz=0, hash="Sun")
    # Planets (heliocentric ecliptic state at epoch)
    for name in body_names:
        if name == "Sun":
            continue
        body = get_body(name)
        M = body.mu / sim.G
        r, v = state(name, epoch_mjd)
        sim.add(m=M, x=r[0], y=r[1], z=r[2], vx=v[0], vy=v[1], vz=v[2], hash=name)
    sim.integrator = "ias15"
    sim.move_to_com()
    return sim


def verify(solution: Solution, dt_record: float = 1.0) -> dict:
    """Integrate the planned trajectory and compare to design.

    Parameters
    ----------
    solution : optimized patched-conic solution
    dt_record : sample interval in days for trajectory output

    Returns
    -------
    report dict with closest-approach distances, miss distance at target, and
    a sampled trajectory list.
    """
    spec = solution.mission
    body_names = list({*spec.sequence, "Sun"})
    t0_mjd = solution.legs[0].t_depart_mjd
    sim = _setup_sim(t0_mjd, body_names)

    # Spacecraft post-launch state: at start body's position with v_dep
    sc_r = np.array(solution.legs[0].r_depart)
    sc_v = np.array(solution.legs[0].v_depart)
    sim.add(m=0.0, x=sc_r[0], y=sc_r[1], z=sc_r[2],
            vx=sc_v[0], vy=sc_v[1], vz=sc_v[2], hash="SC")
    sim.move_to_com()
    sc = sim.particles["SC"]

    # Schedule flyby kicks: at each flyby epoch, set spacecraft velocity to
    # planet_v + v_inf_out (matches the patched-conic plan).
    flyby_kicks = []
    for fb, leg in zip(solution.flybys, solution.legs[1:]):
        # The kick happens at flyby epoch; new velocity is leg.v_depart
        flyby_kicks.append((fb.epoch_mjd, fb.body, np.array(leg.v_depart)))

    final_mjd = solution.legs[-1].t_arrive_mjd
    target_body = solution.legs[-1].to_body

    # Time tracking: REBOUND uses sim.t in seconds since t0_mjd.
    t_total = (final_mjd - t0_mjd) * DAY
    sample_dt = dt_record * DAY

    closest_approach = {i: (np.inf, fb.epoch_mjd) for i, fb in enumerate(solution.flybys)}
    trajectory = []  # (mjd, x, y, z)

    next_kick_idx = 0
    t = 0.0
    while t < t_total:
        next_t = min(t + sample_dt, t_total)
        while next_kick_idx < len(flyby_kicks):
            kick_mjd, kick_body, kick_v = flyby_kicks[next_kick_idx]
            kick_t = (kick_mjd - t0_mjd) * DAY
            if kick_t <= next_t:
                sim.integrate(kick_t)
                planet = sim.particles[kick_body]
                d = float(np.linalg.norm([sc.x - planet.x, sc.y - planet.y, sc.z - planet.z]))
                if d < closest_approach[next_kick_idx][0]:
                    closest_approach[next_kick_idx] = (d, kick_mjd)
                sc.vx, sc.vy, sc.vz = float(kick_v[0]), float(kick_v[1]), float(kick_v[2])
                next_kick_idx += 1
            else:
                break
        sim.integrate(next_t)
        mjd_now = t0_mjd + sim.t / DAY
        trajectory.append((mjd_now, sc.x, sc.y, sc.z))
        for i, fb in enumerate(solution.flybys):
            if abs(mjd_now - fb.epoch_mjd) < 5.0:
                planet = sim.particles[fb.body]
                d = float(np.linalg.norm([sc.x - planet.x, sc.y - planet.y, sc.z - planet.z]))
                if d < closest_approach[i][0]:
                    closest_approach[i] = (d, mjd_now)
        t = next_t

    # Final miss distance from target
    target = sim.particles[target_body]
    miss = float(np.linalg.norm([sc.x - target.x, sc.y - target.y, sc.z - target.z]))

    report = {
        "t0_mjd": t0_mjd,
        "final_mjd": final_mjd,
        "target": target_body,
        "final_miss_km": miss,
        "closest_approaches": [
            {
                "index": i,
                "body": solution.flybys[i].body,
                "distance_km": d,
                "altitude_km": d - get_body(solution.flybys[i].body).radius,
                "epoch_mjd": e,
                "planned_altitude_km": solution.flybys[i].altitude_km,
                "planned_epoch_mjd": solution.flybys[i].epoch_mjd,
            }
            for i, (d, e) in closest_approach.items()
        ],
        "trajectory": trajectory,
    }
    return report
