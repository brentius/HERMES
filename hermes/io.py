"""YAML mission loading and CSV/JSON solution writing."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml
from astropy.time import Time

from .mission import MissionSpec, Solution


def _to_mjd(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(Time(value, format="iso", scale="utc").mjd)


def load_mission(path: Path) -> MissionSpec:
    raw = yaml.safe_load(Path(path).read_text())
    lw = raw["launch_window"]
    return MissionSpec(
        name=raw.get("mission", Path(path).stem),
        sequence=raw["sequence"],
        launch_window_mjd=(_to_mjd(lw[0]), _to_mjd(lw[1])),
        tof_bounds_days=[tuple(b) for b in raw["tof_bounds_days"]],
        min_flyby_altitude_km=raw.get("min_flyby_altitude_km", 300.0),
        optimizer=raw.get("optimizer", {}),
        verify=raw.get("verify", True),
    )


def save_solution_json(solution: Solution, path: Path) -> None:
    Path(path).write_text(json.dumps(solution.to_dict(), indent=2, default=str))


def save_trajectory_csv(trajectory: list[tuple[float, float, float, float]], path: Path) -> None:
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mjd", "x_km", "y_km", "z_km"])
        for row in trajectory:
            w.writerow(row)


def save_verify_report(report: dict, path: Path) -> None:
    # Strip trajectory from JSON; it's saved separately as CSV.
    summary = {k: v for k, v in report.items() if k != "trajectory"}
    Path(path).write_text(json.dumps(summary, indent=2, default=str))
