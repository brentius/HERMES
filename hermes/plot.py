"""Plot generation: Plotly 3D HTML, matplotlib 2D ecliptic, ΔV bar chart."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from .bodies import MU_SUN, get_body, state
from .lambert import solve_lambert
from .mission import Solution

DAY = 86400.0


def _propagate_kepler(r0: np.ndarray, v0: np.ndarray, tof_seconds: float, n: int = 200) -> np.ndarray:
    """Sample n+1 points along the Keplerian arc by re-solving Lambert."""
    pts = [r0.copy()]
    # Numerical propagation via small RK steps (avoids depending on a Kepler solver).
    r = r0.astype(float).copy()
    v = v0.astype(float).copy()
    dt = tof_seconds / n
    for _ in range(n):
        # RK4 in two-body field
        def acc(rr):
            mag = np.linalg.norm(rr)
            return -MU_SUN * rr / mag ** 3
        k1v = acc(r)
        k1r = v
        k2v = acc(r + 0.5 * dt * k1r)
        k2r = v + 0.5 * dt * k1v
        k3v = acc(r + 0.5 * dt * k2r)
        k3r = v + 0.5 * dt * k2v
        k4v = acc(r + dt * k3r)
        k4r = v + dt * k3v
        r = r + (dt / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        pts.append(r.copy())
    return np.array(pts)


def _planet_orbit_samples(body_name: str, t0_mjd: float, period_days: float, n: int = 360) -> np.ndarray:
    return np.array([state(body_name, t0_mjd + period_days * k / n)[0] for k in range(n + 1)])


_APPROX_PERIOD_DAYS = {
    "Mercury": 88, "Venus": 225, "Earth": 365.25, "Mars": 687,
    "Jupiter": 4_333, "Saturn": 10_759, "Uranus": 30_687, "Neptune": 60_190,
}


def plot_3d_html(solution: Solution, out_path: Path) -> None:
    fig = go.Figure()
    # Sun
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                               marker=dict(size=6, color="gold"), name="Sun"))
    # Planet orbits
    bodies_to_plot = list(dict.fromkeys(solution.mission.sequence))
    t0 = solution.legs[0].t_depart_mjd
    for name in bodies_to_plot:
        if name == "Sun":
            continue
        orbit = _planet_orbit_samples(name, t0, _APPROX_PERIOD_DAYS.get(name, 365.25))
        fig.add_trace(go.Scatter3d(
            x=orbit[:, 0], y=orbit[:, 1], z=orbit[:, 2],
            mode="lines", line=dict(width=1), name=f"{name} orbit",
        ))
    # Trajectory legs
    for leg in solution.legs:
        r0 = np.array(leg.r_depart)
        v0 = np.array(leg.v_depart)
        tof = (leg.t_arrive_mjd - leg.t_depart_mjd) * DAY
        pts = _propagate_kepler(r0, v0, tof)
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="lines", line=dict(width=4),
            name=f"{leg.from_body}→{leg.to_body}",
        ))
    # Flyby markers
    for fb in solution.flybys:
        r, _ = state(fb.body, fb.epoch_mjd)
        fig.add_trace(go.Scatter3d(
            x=[r[0]], y=[r[1]], z=[r[2]], mode="markers+text",
            marker=dict(size=4, color="red"),
            text=[f"{fb.body} ({fb.altitude_km:.0f} km)"],
            textposition="top center",
            name=f"Flyby {fb.body}",
        ))
    fig.update_layout(
        title=f"{solution.mission.name} — total ΔV {solution.total_dv_kms:.2f} km/s",
        scene=dict(aspectmode="data", xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)"),
    )
    fig.write_html(str(out_path))


def plot_ecliptic_png(solution: Solution, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(0, 0, marker="*", markersize=15, color="gold", label="Sun")
    bodies_to_plot = list(dict.fromkeys(solution.mission.sequence))
    t0 = solution.legs[0].t_depart_mjd
    for name in bodies_to_plot:
        if name == "Sun":
            continue
        orbit = _planet_orbit_samples(name, t0, _APPROX_PERIOD_DAYS.get(name, 365.25))
        ax.plot(orbit[:, 0], orbit[:, 1], lw=0.7, label=f"{name} orbit")
    for leg in solution.legs:
        r0 = np.array(leg.r_depart)
        v0 = np.array(leg.v_depart)
        tof = (leg.t_arrive_mjd - leg.t_depart_mjd) * DAY
        pts = _propagate_kepler(r0, v0, tof)
        ax.plot(pts[:, 0], pts[:, 1], lw=2, label=f"{leg.from_body}→{leg.to_body}")
    for fb in solution.flybys:
        r, _ = state(fb.body, fb.epoch_mjd)
        ax.plot(r[0], r[1], "rx", markersize=10)
        ax.annotate(fb.body, (r[0], r[1]), fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("X (km, ecliptic)")
    ax.set_ylabel("Y (km, ecliptic)")
    ax.set_title(f"{solution.mission.name} (ecliptic projection)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_dv_breakdown_png(solution: Solution, out_path: Path) -> None:
    labels = ["Launch (v∞)"] + [f"FB {fb.body}" for fb in solution.flybys]
    values = [solution.launch_dv_kms] + [fb.powered_dv_kms for fb in solution.flybys]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color="steelblue")
    ax.set_ylabel("ΔV (km/s)")
    ax.set_title(
        f"{solution.mission.name} ΔV breakdown — total {solution.total_dv_kms:.2f} km/s, "
        f"C3 {solution.launch_c3_km2s2:.1f} km²/s²"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
