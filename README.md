# Veriscope — finite-window early warning and gating

## Overview

Veriscope is a small, typed core plus a runnable reference pipeline for finite-window model monitoring. It implements:

- A fixed-partition, product-TV stability test over a declared window of metrics.
- A prequential gain check (bits/sample) versus a baseline.
- Finite-sample slack ε_stat aggregated over metrics to avoid overreacting at low evidence.
- A common transport that calibrates each metric into [0,1] so distances are comparable and composable.
- Deterministic, resource-bounded optional heavy metrics and a complete, reproducible training/evaluation sweep.

## Core concepts

**WindowDecl:** A declaration of what you will measure.

- `epsilon` (ε): stability resolution in product-TV.
- `metrics`: names of metric channels (e.g. `["var_out_k", "eff_dim"]`).
- `weights`: per-metric weights (normalized internally).
- `bins`: histogram bins for fixed-partition TV.
- `cal_ranges`: per-metric calibration ranges `(lo, hi)` to map raw values into [0,1].
- `interventions`: optional post-processors `T(·)` (e.g. identity, small shifts/scales).

**DeclTransport:** A common adapter `G_T` that applies the declared calibration to any metric series, ensuring values land in [0,1]. It includes a naturality probe to catch gauge slippage early.

**GateEngine:** The finite-window gate combining:

- Stability: worst `D_W(past, recent) ≤ ε − ε_stat`.
- Gain: prequential gain ≥ `gain_thresh` (bits/sample).
- Sensitivity budget: `κ_sens ≤ ε_sens` (optional).
- Evidence floor: minimum finite pairs across metrics.

**ε_stat aggregation:** Conservative finite-sample slack per metric using a Ben-Hamou–Hoppen–Kohayakawa style bound for k-bin histograms,
\[
\varepsilon_\text{stat}(n, k, \alpha) = \sqrt{\frac{\log k + \log (1/\alpha)}{2n}},
\]
then aggregated by the declared weights.

**`tv_hist_fixed`:** Total-variation distance between two fixed-range [0,1] histograms (robust to empties).

## Installation

Requirements: Python 3.9+, NumPy, pandas, matplotlib, filelock, pyarrow, PyTorch (>= 2.1), torchvision (>= 0.16).

Optional dependencies:

- Topological metric (H0 persistence) via `ripser`: install extras `"topo"`.

### From PyPI (**when published**)

```bash
pip install veriscope
pip install "veriscope[topo]"  # to enable persistent homology metric
```

### From source

```bash
git clone https://github.com/your-org/veriscope.git
cd veriscope
pip install -e .
pip install -e ".[topo]"  # optional
```

## Quick start (core API)

Minimal, self-contained gating example on two metrics:

```python
import numpy as np
from veriscope.core.window import FRWindow
from veriscope import (
    WindowDecl, DeclTransport, GateEngine,
    aggregate_epsilon_stat, tv_hist_fixed, assert_naturality
)

# 1) Declare the window Φ_W
decl = WindowDecl(
    epsilon=0.08,
    metrics=["var_out_k", "eff_dim"],
    weights={"var_out_k": 0.5, "eff_dim": 0.5},
    bins=16,
    cal_ranges={"var_out_k": (0.0, 1.0), "eff_dim": (0.0, 64.0)},  # transport → [0,1]
    interventions=(lambda x: x,),
)
decl.normalize_weights()

# 2) Attach the common transport and (optionally) probe naturality
tp = DeclTransport(decl)
decl.attach_transport(tp)
assert_naturality(
    tp,
    [lambda z: z, lambda z: z[1:-1] if getattr(z, "ndim", 1) == 1 and z.size > 2 else z],
)

# 3) Instantiate the FR window and gate
frwin = FRWindow(decl=decl, transport=tp, tests=())
gate = GateEngine(
    frwin=frwin,
    gain_thresh=0.10,          # bits/sample
    eps_stat_alpha=0.05,       # α for ε_stat
    eps_stat_max_frac=0.25,    # cap ε_stat ≤ frac * ε
    eps_sens=0.04,             # κ_sens budget
    min_evidence=16,           # require at least this many finite pairs
)

# 4) One finite window: past vs recent slices for each metric
rng = np.random.default_rng(0)
past = {
    "var_out_k": rng.uniform(0.20, 0.40, size=128),
    "eff_dim":   rng.uniform(10.0, 20.0, size=128),
}
recent = {
    "var_out_k": rng.uniform(0.22, 0.42, size=128),
    "eff_dim":   rng.uniform(11.0, 19.0, size=128),
}

# 5) Evidence and ε_stat under the declared transport
apply = tp.apply
tpast   = {m: apply(m, v) for m, v in past.items()}
trecent = {m: apply(m, v) for m, v in recent.items()}
counts = {
    m: int(
        min(
            np.isfinite(tpast[m]).sum(),
            np.isfinite(trecent[m]).sum(),
        )
    )
    for m in decl.metrics
}

eps_stat = aggregate_epsilon_stat(decl, counts, alpha=0.05)

# 6) Prequential gain and κ_sens (na → not enforced)
gain_bits = 0.12
kappa_sens = np.nan  # no check

# 7) Gate check
result = gate.check(
    past=past,              # raw is fine; GateEngine applies the declared transport
    recent=recent,
    counts_by_metric=counts,
    gain_bits=gain_bits,
    kappa_sens=kappa_sens,
    eps_stat_value=eps_stat,
)

print("OK:", result.ok)
print("Audit:", result.audit)
```

## A note on `tv_hist_fixed`

If you only need a fixed-range TV:

```python
from veriscope import tv_hist_fixed

d = tv_hist_fixed([0.1, 0.2, 0.3], [0.1, 0.4], bins=16)
```

## Command-line runner (reference experiment)

The package ships a runnable reference pipeline to reproduce a controlled early-warning study on CIFAR-10 with an external STL-10 monitor stream.

- Entry point: `veriscope` (console script)
- Help: `veriscope --help`
- Fast smoke run (downloads data on first use):

  ```bash
  SCAR_SMOKE=1 SCAR_OUTDIR=./out_smoke SCAR_DATA=./data veriscope
  ```

- Full run (longer):

  ```bash
  SCAR_OUTDIR=./out_full SCAR_DATA=./data veriscope
  ```

### Key environment variables

- `SCAR_OUTDIR`: output directory (default `./scar_bundle_phase4`)
- `SCAR_DATA`: data root; CIFAR-10 and STL-10 will be downloaded if absent
- `SCAR_SMOKE=1`: tiny sweep for quick end-to-end
- `SCAR_CALIB=1`: calibration mode for deployment knobs
- `SCAR_GATE_MIN_EVIDENCE`: minimum finite transported pairs across metrics required by the gate
- `SCAR_GATE_GAIN_THRESH`: override prequential gain threshold (bits/sample)
- `SCAR_GATE_EPSILON`: override ε
- `SCAR_FAMILY_Z_THR`: z-gate threshold used by the learned detector family gate
- `CUDA_VISIBLE_DEVICES`: sets device; defaults to CPU when empty

## What the runner produces

- `bundle_runs_<tag>.parquet` (or `.csv` fallback): per-epoch logs for all runs
- `bundle_runs_eval_with_overlays_{soft,hard}.parquet`: unified overlays with warn/collapse epochs
- `window_decl_calibrated.json`, `window_provenance_decl.json(.sha256)`: the deployed Φ_W capsule
- `calibration_provenance.json`, `precedence_summary.json`: where knobs came from
- `gate_gain_thresh_calibration.json`: gain threshold learned from controls
- `repro.json`, `env.json`, `pip_freeze.txt`: reproducibility capsule
- `figs/*.png`: per-metric spaghetti plots and warn/collapse rasters
- Artifacts with checksums (`.md5`, `.sha256`)

## Determinism and budgets

- Deterministic by default; the runner configures `CUBLAS_WORKSPACE_CONFIG` and disables TF32 fast paths.
- Heavy metrics (sliced W2, topological persistence) are guarded by per-call and total budgets; see `BudgetLedger` in the runner. These budgets can be tuned via the runner’s configuration knobs (e.g. `sw2_budget_ms`, `ripser_budget_ms`).

## API surface (import paths)

Top-level re-exports:

```python
from veriscope import (
    WindowDecl,
    DeclTransport,
    assert_naturality,
    GateEngine,
    GateResult,
    epsilon_statistic_bhc,
    aggregate_epsilon_stat,
    tv_hist_fixed,
)
```

FR window container:

```python
from veriscope.core.window import FRWindow
```

### Signatures (abridged)

- `WindowDecl(epsilon: float, metrics: Sequence[str], weights: Dict[str, float], bins: int, interventions=(), cal_ranges: Dict[str, (float, float)])`
  - `normalize_weights() -> None`
  - `attach_transport(transport) -> None`

- `DeclTransport(decl, ..., atol=1e-6)`
  - `apply(ctx: str, x: ndarray) -> ndarray` in `[0,1]`
  - `natural_with(restrict: Callable) -> bool`

- `assert_naturality(transport, restrictors: Iterable[Callable], msg: str) -> None`

- `GateEngine(frwin: FRWindow, gain_thresh: float, eps_stat_alpha: float, eps_stat_max_frac: float, eps_sens: float, min_evidence=0)`
  - `check(past: Dict[str, ndarray], recent: Dict[str, ndarray], counts_by_metric: Dict[str,int], gain_bits: float, kappa_sens: float, eps_stat_value: float) -> GateResult`

- `epsilon_statistic_bhc(n: int, k: int, alpha=0.05) -> float` in `[0,1]`
- `aggregate_epsilon_stat(window: WindowDecl, counts_by_metric: Dict[str,int], alpha=0.05) -> float`
- `tv_hist_fixed(z0, z1, bins: int) -> float` in `[0,1]`

## Configuration defaults (selected)

The package exports minimal, window-agnostic defaults in `veriscope.config.CFG`. Important keys you’ll often tune:

- `gate_window`: 16
- `gate_bins`: 16
- `gate_min_evidence`: 16
- `gate_gain_thresh`: 0.10 (bits/sample; calibrated in the runner)
- `gate_epsilon`: 0.08
- `gate_eps_stat_max_frac`: 0.25 (cap fraction for ε_stat)
- `eps_sens` or `gate_epsilon_sens`: 0.04 (κ_sens)
- `warn_consec`: 3
- `device`: `"cuda"` when `CUDA_VISIBLE_DEVICES` is set, else `"cpu"`

## Development

- Style and linting: `ruff` (line length 120). See `pyproject.toml` `[tool.ruff]`.
- Static typing: `mypy` is configured in strict mode for package sources; third-party imports are ignored to keep the signal clean until hot spots are typed.
- Tests and smoke runs: `SCAR_SMOKE=1 veriscope` executes a tiny, end-to-end sweep and writes artifacts under `SCAR_OUTDIR`.
- Optional features: install extras `[topo]` to enable persistent homology via `ripser`.

## FAQ

**Do I need a GPU?**  
No. The runner will use CUDA if available; otherwise it falls back to CPU. Heavy metrics are budgeted to keep runs bounded.

**What if `ripser` is not installed?**  
`pers_H0` will be NaN and any topology-dependent paths are skipped; everything else works.

**Where do “bits/sample” come from?**  
Prequential gain is computed as `(log p_baseline − log p_model) / ln(2)`, averaged over the finite window.

## License

See the `LICENSE` file in the repository. The package metadata currently declares “Other/Proprietary License”.

## Citation

If you use this code or the fixed-partition finite-window gate in your work, please cite the repository. A formal citation will be added in a later release.

## Changelog

**0.1.0**

- First refactor exposing the finite-window core (`WindowDecl`, `DeclTransport`, `GateEngine`, ε_stat aggregation, TV) with a runnable CIFAR-10/STL-10 reference pipeline and provenance capsules.
