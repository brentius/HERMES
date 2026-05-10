# HERMES

Spacecraft trajectory designer with gravity-assist chain optimization.

- **Patched-conic** Lambert solver for fast multi-body trajectory design
- **Differential-evolution** global optimizer over launch + flyby dates
- **REBOUND IAS15** N-body verification of the optimized solution
- CLI driven by YAML mission configs; outputs HTML 3D / PNG / CSV / JSON

## Install

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

First run will download the JPL DE440 ephemeris kernel (~120 MB) via astropy.

## Usage

```powershell
hermes design examples/earth_mars_hohmann.yaml
hermes design examples/cassini_vveje.yaml --out runs/cassini
hermes plot runs/cassini/solution.json
hermes verify runs/cassini/solution.json
```

Outputs land in `runs/<mission>/`:
- `solution.json` — optimized trajectory + ΔV breakdown
- `trajectory_3d.html` — interactive Plotly view
- `ecliptic.png` — top-down ecliptic projection
- `dv_breakdown.png` — bar chart of launch + flyby ΔVs
- `trajectory.csv` — REBOUND-integrated state samples
- `verify_report.json` — closest-approach distances vs planned

## Mission YAML

```yaml
mission: cassini_vveje
sequence: [Earth, Venus, Venus, Earth, Jupiter]
launch_window: ["1997-10-01", "1997-11-30"]
tof_bounds_days:
  - [80, 200]
  - [350, 500]
  - [50, 150]
  - [400, 800]
min_flyby_altitude_km: 300
optimizer: { popsize: 40, maxiter: 300, seed: 42 }
verify: true
```

## Tests

```powershell
pytest                # fast unit tests
pytest -m slow        # includes optimizer convergence test
```
