"""Mission specification and solution dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass
class MissionSpec:
    name: str
    sequence: list[str]                  # e.g. ["Earth", "Venus", "Venus", "Earth", "Jupiter"]
    launch_window_mjd: tuple[float, float]
    tof_bounds_days: list[tuple[float, float]]   # one per leg; len == len(sequence) - 1
    min_flyby_altitude_km: float = 300.0
    optimizer: dict[str, Any] = field(default_factory=dict)
    verify: bool = True

    @property
    def n_flybys(self) -> int:
        return len(self.sequence) - 2

    @property
    def n_legs(self) -> int:
        return len(self.sequence) - 1


@dataclass
class LegSolution:
    from_body: str
    to_body: str
    t_depart_mjd: float
    t_arrive_mjd: float
    r_depart: list[float]   # km
    v_depart: list[float]   # km/s heliocentric
    r_arrive: list[float]
    v_arrive: list[float]


@dataclass
class FlybySolution:
    body: str
    epoch_mjd: float
    v_inf_in: list[float]
    v_inf_out: list[float]
    r_periapsis_km: float
    altitude_km: float
    turn_angle_deg: float
    powered_dv_kms: float


@dataclass
class Solution:
    mission: MissionSpec
    legs: list[LegSolution]
    flybys: list[FlybySolution]
    launch_c3_km2s2: float
    launch_dv_kms: float
    arrival_v_inf_kms: float
    total_dv_kms: float
    decision_vector: list[float]

    def to_dict(self) -> dict:
        return {
            "mission": asdict(self.mission),
            "legs": [asdict(l) for l in self.legs],
            "flybys": [asdict(f) for f in self.flybys],
            "launch_c3_km2s2": self.launch_c3_km2s2,
            "launch_dv_kms": self.launch_dv_kms,
            "arrival_v_inf_kms": self.arrival_v_inf_kms,
            "total_dv_kms": self.total_dv_kms,
            "decision_vector": list(self.decision_vector),
        }
