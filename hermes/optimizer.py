"""Decision-vector encoding and global optimizer for gravity-assist chains.

Decision vector for sequence [B0, B1, ..., BN, BT] (N flybys):
    [t0, T_1, ..., T_{N+1}, r_p_1, ..., r_p_N, b_1, ..., b_N]

t0 in MJD; T_i in days; r_p_i in km (constrained ≥ R_body + min_alt);
b_i is a B-plane angle in [-pi, pi], reserved for future trajectory shaping
(currently unused — flybys are auto-targeted by Lambert-arc velocities).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from .bodies import get_body, state, MU_SUN
from .flyby import powered_flyby_dv, turn_angle_between
from .lambert import solve_lambert
from .mission import MissionSpec, Solution, LegSolution, FlybySolution

DAY = 86400.0


def _bounds(spec: MissionSpec) -> list[tuple[float, float]]:
    bounds = [spec.launch_window_mjd]
    bounds.extend(spec.tof_bounds_days)
    for i in range(spec.n_flybys):
        body = get_body(spec.sequence[i + 1])
        r_min = body.radius + spec.min_flyby_altitude_km
        r_max = body.radius + 50.0 * body.radius   # allow up to 50 R_body
        bounds.append((r_min, r_max))
    for _ in range(spec.n_flybys):
        bounds.append((-np.pi, np.pi))
    return bounds


def _decode(x: np.ndarray, spec: MissionSpec) -> dict:
    n_legs = spec.n_legs
    n_fb = spec.n_flybys
    t0 = float(x[0])
    tofs = [float(v) for v in x[1:1 + n_legs]]
    rps = [float(v) for v in x[1 + n_legs:1 + n_legs + n_fb]]
    bs = [float(v) for v in x[1 + n_legs + n_fb:1 + n_legs + 2 * n_fb]]
    return {"t0": t0, "tofs": tofs, "rps": rps, "bs": bs}


def _evaluate(x: np.ndarray, spec: MissionSpec, return_solution: bool = False):
    decoded = _decode(x, spec)
    t0 = decoded["t0"]
    tofs = decoded["tofs"]
    rps = decoded["rps"]

    # Epochs at each body in the sequence
    epochs = [t0]
    for tof in tofs:
        epochs.append(epochs[-1] + tof)

    # Body states at each epoch
    body_states = []
    for body_name, mjd in zip(spec.sequence, epochs):
        r, v = state(body_name, mjd)
        body_states.append((r, v))

    # Solve Lambert per leg
    leg_v = []   # list of (v_dep, v_arr)
    PENALTY = 1e6
    for i in range(spec.n_legs):
        r1 = body_states[i][0]
        r2 = body_states[i + 1][0]
        tof_s = tofs[i] * DAY
        if tof_s <= 0:
            return PENALTY if not return_solution else None
        try:
            v1, v2 = solve_lambert(r1, r2, tof_s, MU_SUN, prograde=True)
        except Exception:
            return PENALTY if not return_solution else None
        leg_v.append((v1, v2))

    # Launch ΔV: from Earth (or starting body) heliocentric velocity to v_dep_0
    v_planet_launch = body_states[0][1]
    v_inf_launch = leg_v[0][0] - v_planet_launch
    c3 = float(np.dot(v_inf_launch, v_inf_launch))
    launch_dv = float(np.linalg.norm(v_inf_launch))   # cost from Earth's SOI; user typically reports C3

    # Flyby ΔVs
    flyby_dvs = []
    flyby_records = []
    for i in range(spec.n_flybys):
        body_name = spec.sequence[i + 1]
        body = get_body(body_name)
        v_planet = body_states[i + 1][1]
        v_arr = leg_v[i][1]
        v_dep_next = leg_v[i + 1][0]
        v_inf_in = v_arr - v_planet
        v_inf_out = v_dep_next - v_planet
        r_p = rps[i]
        dv, deficit = powered_flyby_dv(v_inf_in, v_inf_out, body, r_p)
        # Penalty for turn-angle deficit (cannot bend v_inf enough at r_p)
        dv_total = dv + 5.0 * deficit * np.linalg.norm(v_inf_in)
        flyby_dvs.append(dv_total)
        flyby_records.append({
            "body": body_name,
            "epoch_mjd": epochs[i + 1],
            "v_inf_in": v_inf_in,
            "v_inf_out": v_inf_out,
            "r_p": r_p,
            "altitude": r_p - body.radius,
            "turn_angle_deg": np.rad2deg(turn_angle_between(v_inf_in, v_inf_out)),
            "powered_dv": dv,
        })

    # Arrival v_inf cost (rendezvous: penalize; flyby-only mission: ignore)
    v_planet_target = body_states[-1][1]
    v_inf_arrive = leg_v[-1][1] - v_planet_target
    arrival_v_inf = float(np.linalg.norm(v_inf_arrive))

    total = launch_dv + sum(flyby_dvs)

    if return_solution:
        legs = []
        for i in range(spec.n_legs):
            legs.append(LegSolution(
                from_body=spec.sequence[i],
                to_body=spec.sequence[i + 1],
                t_depart_mjd=epochs[i],
                t_arrive_mjd=epochs[i + 1],
                r_depart=body_states[i][0].tolist(),
                v_depart=leg_v[i][0].tolist(),
                r_arrive=body_states[i + 1][0].tolist(),
                v_arrive=leg_v[i][1].tolist(),
            ))
        flybys = [
            FlybySolution(
                body=r["body"],
                epoch_mjd=r["epoch_mjd"],
                v_inf_in=r["v_inf_in"].tolist(),
                v_inf_out=r["v_inf_out"].tolist(),
                r_periapsis_km=r["r_p"],
                altitude_km=r["altitude"],
                turn_angle_deg=r["turn_angle_deg"],
                powered_dv_kms=r["powered_dv"],
            )
            for r in flyby_records
        ]
        return Solution(
            mission=spec,
            legs=legs,
            flybys=flybys,
            launch_c3_km2s2=c3,
            launch_dv_kms=launch_dv,
            arrival_v_inf_kms=arrival_v_inf,
            total_dv_kms=total,
            decision_vector=x.tolist(),
        )

    return float(total)


def optimize(spec: MissionSpec) -> Solution:
    bounds = _bounds(spec)
    opts = spec.optimizer or {}
    result = differential_evolution(
        _evaluate,
        bounds=bounds,
        args=(spec,),
        popsize=opts.get("popsize", 30),
        maxiter=opts.get("maxiter", 200),
        seed=opts.get("seed", None),
        tol=opts.get("tol", 1e-4),
        mutation=opts.get("mutation", (0.5, 1.0)),
        recombination=opts.get("recombination", 0.7),
        polish=True,
        workers=opts.get("workers", 1),
        updating="deferred" if opts.get("workers", 1) != 1 else "immediate",
        disp=opts.get("disp", True),
    )
    return _evaluate(result.x, spec, return_solution=True)
