"""Command-line interface for HERMES."""
from __future__ import annotations

import json
from pathlib import Path

import click

from .io import load_mission, save_solution_json, save_trajectory_csv, save_verify_report
from .mission import MissionSpec, Solution, LegSolution, FlybySolution
from .optimizer import optimize
from .plot import plot_3d_html, plot_dv_breakdown_png, plot_ecliptic_png
from .verify import verify as verify_solution


@click.group()
def main() -> None:
    """HERMES — interplanetary trajectory designer with gravity-assist optimization."""


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path),
              default=None, help="Output directory (default: runs/<mission>/)")
@click.option("--no-verify", is_flag=True, help="Skip REBOUND verification")
@click.option("--no-plot", is_flag=True, help="Skip plot generation")
def design(config: Path, out_dir: Path | None, no_verify: bool, no_plot: bool) -> None:
    """Design a trajectory from a YAML mission spec."""
    spec = load_mission(config)
    out_dir = out_dir or Path("runs") / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Optimizing {spec.name} ({len(spec.sequence) - 1} legs, {spec.n_flybys} flybys)...")
    sol = optimize(spec)
    save_solution_json(sol, out_dir / "solution.json")
    click.echo(f"  total dV: {sol.total_dv_kms:.3f} km/s")
    click.echo(f"  launch C3: {sol.launch_c3_km2s2:.2f} km^2/s^2")
    click.echo(f"  arrival v_inf: {sol.arrival_v_inf_kms:.3f} km/s")
    click.echo(f"  → {out_dir / 'solution.json'}")
    if not no_plot:
        plot_3d_html(sol, out_dir / "trajectory_3d.html")
        plot_ecliptic_png(sol, out_dir / "ecliptic.png")
        plot_dv_breakdown_png(sol, out_dir / "dv_breakdown.png")
        click.echo(f"  plots → {out_dir}")
    if spec.verify and not no_verify:
        click.echo("Verifying with REBOUND IAS15...")
        report = verify_solution(sol)
        save_trajectory_csv(report["trajectory"], out_dir / "trajectory.csv")
        save_verify_report(report, out_dir / "verify_report.json")
        click.echo(f"  final miss: {report['final_miss_km']:.0f} km from {report['target']}")
        for ca in report["closest_approaches"]:
            click.echo(
                f"  flyby {ca['index']} ({ca['body']}): "
                f"actual alt {ca['altitude_km']:.0f} km vs planned {ca['planned_altitude_km']:.0f} km"
            )


def _solution_from_json(path: Path) -> Solution:
    raw = json.loads(Path(path).read_text())
    spec = MissionSpec(
        name=raw["mission"]["name"],
        sequence=raw["mission"]["sequence"],
        launch_window_mjd=tuple(raw["mission"]["launch_window_mjd"]),
        tof_bounds_days=[tuple(b) for b in raw["mission"]["tof_bounds_days"]],
        min_flyby_altitude_km=raw["mission"]["min_flyby_altitude_km"],
        optimizer=raw["mission"]["optimizer"],
        verify=raw["mission"]["verify"],
    )
    legs = [LegSolution(**l) for l in raw["legs"]]
    flybys = [FlybySolution(**f) for f in raw["flybys"]]
    return Solution(
        mission=spec, legs=legs, flybys=flybys,
        launch_c3_km2s2=raw["launch_c3_km2s2"],
        launch_dv_kms=raw["launch_dv_kms"],
        arrival_v_inf_kms=raw["arrival_v_inf_kms"],
        total_dv_kms=raw["total_dv_kms"],
        decision_vector=raw["decision_vector"],
    )


@main.command()
@click.argument("solution_json", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=None)
def verify(solution_json: Path, out_dir: Path | None) -> None:
    """Re-run REBOUND verification on a saved solution."""
    sol = _solution_from_json(solution_json)
    out_dir = out_dir or solution_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report = verify_solution(sol)
    save_trajectory_csv(report["trajectory"], out_dir / "trajectory.csv")
    save_verify_report(report, out_dir / "verify_report.json")
    click.echo(f"Final miss: {report['final_miss_km']:.0f} km from {report['target']}")


@main.command()
@click.argument("solution_json", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=None)
def plot(solution_json: Path, out_dir: Path | None) -> None:
    """Generate plots from a saved solution."""
    sol = _solution_from_json(solution_json)
    out_dir = out_dir or solution_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_3d_html(sol, out_dir / "trajectory_3d.html")
    plot_ecliptic_png(sol, out_dir / "ecliptic.png")
    plot_dv_breakdown_png(sol, out_dir / "dv_breakdown.png")
    click.echo(f"Plots → {out_dir}")


if __name__ == "__main__":
    main()
