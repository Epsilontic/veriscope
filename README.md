<div align="center">

# 🔬 Veriscope

**Early-warning detection of neural network training pathologies**

  
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: Dual](https://img.shields.io/badge/License-Dual-brightgreen.svg)](./COMMERCIAL_LICENSE.md)
[![CI](https://github.com/Epsilontic/veriscope/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Epsilontic/veriscope/actions/workflows/ci.yml)

*Detect representation collapse before it's too late.*

[Quick Start](#quick-start) • [Pilot kit overview](#pilot-kit-overview) • [Gate Semantics](#gate-semantics) • [Core API](#core-api) • [Output Artifacts](#output-artifacts) • [Contributing](#contributing) • [Governance](#governance--community) • [Citation](#citation)

</div>

---

## Overview

Veriscope is an open research project developing tools to detect early signs of collapse in internal model diversity before they lead to brittle or unsafe behavior. Standard metrics can look healthy while representation structure quietly degrades. Veriscope provides **auditable, reproducible monitoring signals** to surface these risks.

At a high level, Veriscope turns training-time monitoring into a reproducible decision contract:

- Declare an observable window (Φ_W) via `WindowDecl`.
- Normalize metrics under a declared transport (`DeclTransport`).
- Evaluate stability via a windowed divergence with finite-evidence uncertainty (`ε_stat`).
- Emit structured audit fields and artifacts for post-hoc review.

---

## Key capabilities

- **Representation drift detection**
  - Feature-geometry signals (e.g., effective rank, variance-outside-k)
  - Dynamics signals (FTLE-like entropy gradients; experimental)
  - Cadenced heavy metrics (sliced-W2; H0 persistence with `ripser` when available)
- **Finite Realism gate**
  - Windowed divergence with prequential gain, ε-stat, and κ-sensitivity probes (optional)
  - Resource budgets for heavy metrics with deterministic JL projections
- **Calibratable ground truth (unsupervised)**
  - Rank-only soft-collapse threshold calibrated on control runs
  - Per-factor robust gradient cutoff for hard collapse
- **Learned detector with deployable gate**
  - Grouped CV, FP-calibrated τ→τ′ mapping under the same gate used at inference
- **Audit-ready artifacts**
  - Provenance capsules (resolved config, environment, window declaration) with content digests (where enabled)
  - Decisions and intermediate quantities logged with structured audit fields for post-hoc review
- **Reproducibility and determinism**
  - Deterministic seeds, cuBLAS workspace config, and safe data-loading policies

---

## Gate Semantics

`GateEngine` evaluates stability by computing `worst_DW` (a windowed divergence across tracked metrics, reduced as “worst over interventions”) and comparing it to an effective threshold `eps_eff`.

### Windowed divergence

- `w_sum = Σ_m |w[m]|` (defaults to 1.0 if empty)
- For each intervention `T` (often identity only): compute `TV_m(T(past), T(recent))`
- Combine per intervention: `DW_T = Σ_m (|w[m]| / w_sum) * TV_m`
- Reduce: `worst_DW = max_T DW_T` (ignoring non-finite per-metric TVs)

### Effective threshold (as implemented)

- `eps_stat_cap = min(max(eps_stat_value, 0), cap_frac * ε)`
- `eps_eff = max(0, ε + eps_sens - eps_stat_cap)`
- Stability exceeds when `worst_DW > eps_eff`

### Policy and optional checks

- **Policy** controls how stability and gain combine (`either`, `conjunction`, `persistence`, `persistence_stability`).

- **Gain**: non-finite gain is treated as “not checked” (audited but does not auto-fail).

- **κ sensitivity**: non-finite κ is treated as “not checked”.

- Multi-metric consensus (`min_metrics_exceeding`): can suppress stability exceedance even when `worst_DW > eps_eff` if too few per-metric TVs exceed `eps_eff`.

For exact decision logic, rely on `GateEngine.check(...).audit`, the calibration CSV fields, and the CLI validators (`veriscope validate`, `veriscope diff`) described in `docs/productization.md`.

---

## Status

- ✅ CIFAR-10 reproduction suite (PyTorch)
- ✅ GPT runner with gate presets and spike/corruption injection
- 🔜 Extension to additional language-model fine-tuning runs
- 🔜 Public Python package with verifier + benchmark tools

---

## Requirements

- Python 3.9+
- PyTorch ≥ 2.1
- Optional GPU with CUDA 11/12 (CPU supported; heavy metrics slower)
- `numpy`, `pandas`, `matplotlib`
- Optional:
  - `filelock` (safer dataset downloads)
  - `ripser` (H0 persistence; falls back to NaN if unavailable)
  - `pyarrow` or `fastparquet` (for parquet output; CSV fallback automatic)

Example installation (editable install):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

Optional extras (if configured in `pyproject.toml`):

```bash
pip install -e ".[topo]"
```

Note: dependencies are declared in `pyproject.toml`, so you normally do not need a `requirements.txt`.

---

## Datasets

- **CIFAR-10** (train/test) — auto-downloaded into `SCAR_DATA` if missing
- **STL-10** (test split) — external, label-free monitor stream (with resize); falls back to a clean validation split if unavailable
- **Shakespeare (char)** — used by the GPT runner; prepared via nanoGPT scripts

---

## Quick Start

After installation, run Veriscope via explicit subcommands (recommended).
Use `veriscope --help` to see the available subcommands.

### CIFAR smoke run (fast end-to-end)

```bash
export SCAR_DATA=./data
veriscope run cifar --smoke --outdir ./out/cifar_smoke_$(date +%Y%m%d_%H%M%S)
# If the console script is not available:
# python -m veriscope run cifar --smoke --outdir ./out/cifar_smoke_$(date +%Y%m%d_%H%M%S)
```

Runs a small CIFAR sweep (few seeds, short epochs) into the specified outdir.

### Full CIFAR sweep (more seeds/epochs)

```bash
export SCAR_DATA=./data
veriscope run cifar --outdir ./out/cifar_full_$(date +%Y%m%d_%H%M%S)
```

> Note: a full sweep can take hours depending on hardware.

### HF quick start (transformers runner)

```bash
# Requires transformers + datasets
veriscope run hf --gate-preset tuned_v0 --outdir ./out/hf_smoke_$(date +%Y%m%d_%H%M%S) -- \
  --model gpt2 \
  --dataset wikitext:wikitext-2-raw-v1 \
  --dataset_split train \
  --max_steps 50 \
  --device cpu
```

Runs a short HF training loop and writes a capsule in the specified outdir.
For a scripted version, see `scripts/run_hf_smoke.sh`.

---

## Pilot kit overview

If you are piloting Veriscope with a design partner, the pilot kit provides:
- A control + injected-pathology workflow with a shareable report and calibration summary.
- Success criteria grounded in the v0 contract.
- A stable artifact bundle to share with Veriscope.

For operational semantics (exit codes, schemas, artifact contracts, and validation), see `docs/productization.md`.
For narrative guidance, success criteria, and sharing guidance, see `docs/pilot/README.md`.
For script usage, outputs, and troubleshooting, see `scripts/pilot/README.md`.

Pilot harness commands (copy/paste):

```bash
# Prereq: clone and prepare nanoGPT data (see GPT runner section below).
# Control run (baseline)
bash scripts/pilot/run.sh ./out/pilot_control -- --dataset shakespeare_char --nanogpt_dir ./nanoGPT

# Injected run (pathology via data corruption flags)
bash scripts/pilot/run.sh ./out/pilot_injected -- --dataset shakespeare_char --nanogpt_dir ./nanoGPT \
  --data_corrupt_at 2500 --data_corrupt_len 400 --data_corrupt_frac 0.15 --data_corrupt_mode permute
# (See `python -m veriscope.runners.gpt.train_nanogpt --help` for corruption flag details.)

# Score calibration
python scripts/pilot/score.py \
  --control-dir ./out/pilot_control \
  --injected-dir ./out/pilot_injected \
  --out calibration.json \
  --out-md calibration.md

# Negative controls on a completed capsule
bash scripts/pilot/negative_controls.sh ./out/pilot_control
```

---

## GPT runner (nanoGPT)

The GPT runner wraps nanoGPT training with FR gating and emits a capsule directory (results summary + provenance + optional calibration CSV). For the authoritative contract, defer to `docs/productization.md`.

```bash
# Clone nanoGPT alongside veriscope
git clone https://github.com/karpathy/nanoGPT.git

# Prepare data (e.g., shakespeare_char)
cd nanoGPT
python data/shakespeare_char/prepare.py
cd ..

veriscope run gpt --outdir ./out/gpt_run --force -- \
  --dataset shakespeare_char \
  --nanogpt_dir ./nanoGPT \
  --gate_preset tuned_v0
```

Advanced: run the module directly (bypasses the CLI wrapper):

```bash
python -m veriscope.runners.gpt.train_nanogpt \
  --dataset shakespeare_char \
  --nanogpt_dir ./nanoGPT \
  --gate_preset tuned_v0
```
---

## Core API

### Window Declaration (Φ_W)

```python
from veriscope.core.window import WindowDecl
from veriscope.core.transport import DeclTransport

decl = WindowDecl(
    epsilon=0.12,
    metrics=["var_out_k", "eff_dim"],
    weights={"var_out_k": 0.5, "eff_dim": 0.5},
    bins=16,
    cal_ranges={"var_out_k": (0, 1), "eff_dim": (0, 64)},
)

transport = DeclTransport(decl)
decl.attach_transport(transport)
```

### Gate Check

```python
import numpy as np
from veriscope.core.window import FRWindow
from veriscope.core.gate import GateEngine

fr_win = FRWindow(decl=decl, transport=transport, tests=())

gate = GateEngine(
    frwin=fr_win,
    gain_thresh=0.05,
    eps_stat_alpha=0.05,
    eps_stat_max_frac=0.25,
    eps_sens=0.04,
    min_evidence=16,
)

result = gate.check(
    past={"var_out_k": np.array([...]), "eff_dim": np.array([...])},
    recent={"var_out_k": np.array([...]), "eff_dim": np.array([...])},
    counts_by_metric={"var_out_k": 100, "eff_dim": 100},
    gain_bits=0.08,
    kappa_sens=np.nan,
    eps_stat_value=0.01,
)

print("OK:", result.ok)
print("worst_DW:", result.audit.get("worst_DW"))
print("eps_eff:", result.audit.get("eps_eff"))
```

---

## Output artifacts

### CIFAR sweep artifacts (run outdir)

- **Run logs**
  - `bundle_runs_{tag}.parquet` (CSV fallback if parquet unavailable)
  - `errors_{tag}.log` (per-run failures, if any)
- **Learned detector and calibration**
  - `learned_detector_{tag}.json`
  - `tau_mapping_{tag}.json` (τ→τ′ mapping under deployed gate)
  - `ph_directions_{tag}.json` (CUSUM directions per metric)
  - `rp_adequacy_{tag}.json` (JL/native agreement pre-warm)
- **Ground truth and baselines**
  - `grad_cutoff_by_factor_{tag}.json`
  - `baseline_events_{ewma|sma|vote|seq|newma}_{tag}.csv`
  - `detector_events_{tag}.csv`
- **Overlays and summaries**
  - `bundle_runs_eval_with_overlays_{tag}.parquet`
  - `bundle_runs_eval_with_overlays_soft.parquet`
  - `bundle_runs_eval_with_overlays_hard.parquet`
  - `summary_learned_{tag}.csv`
  - `summary_baseline_{gate|ewma|sma|vote|seq|newma}_{tag}.csv`
- **Invariants and provenance**
  - `artifact_invariants_provenance.json`
  - `window_provenance.json`
  - `env.json`, `repro.json`, `pip_freeze.txt`
- **Figures**
  - `figs/*.png`

Writes are atomic where possible; parquet falls back to CSV automatically. Legacy tooling may also respect `SCAR_OUTDIR` if set.

### Pilot capsule outputs (`OUTDIR`)

- **Required capsule files**
  - `results_summary.json`
  - `window_signature.json`
  - `run_config_resolved.json`
- **CLI outputs captured by the pilot harness**
  - `validate.txt`
  - `inspect.txt`
  - `report.md`
  - `report_stderr.txt`
  - `git_sha.txt`
  - `version.txt`
- **Calibration outputs (from score.py)**
  - `calibration.json`
  - `calibration.md`

---

## Reproducibility and determinism

- Seeds set for Python, NumPy, and PyTorch (including CUDA)
- cuBLAS deterministic workspace configured and probed; falls back to `warn_only` if unavailable
- DataLoader persistence disabled for per-epoch loaders to avoid file descriptor leaks
- JL projections use bounded LRU cache with deterministic seeding

---

## Roadmap

- Broaden modality coverage beyond CIFAR-10
- Package release with verifiers, scoring scripts, and stable APIs
- Alternative topology backends (MST-based) for `ripser`-free environments

---

---

## Contributing

Contributions are welcome. For non-trivial changes, please open an issue first to discuss scope and design.

**CLA required:** All contributions require signing the Contributor License Agreement. See `INDIVIDUAL_CLA.md`.

---

## Governance & Community

Veriscope’s governance and community expectations are documented here:

- [Governance](./GOVERNANCE.md)
- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Security Policy](./SECURITY.md)
- [Release Process](./RELEASE_PROCESS.md)

---

## License

This project is dual-licensed:

- **GNU Affero General Public License v3.0 (AGPL-3.0-only)**
- **Commercial license** (for organizations preferring not to comply with AGPL terms)

The dual-license model ensures Veriscope remains open and auditable for research and open-source use,
while also enabling sustainable commercial support and long-term maintenance.

Use AGPL-3.0-only for open-source compliance; obtain a commercial license if you prefer not to comply with AGPL obligations for proprietary or hosted deployments.

See `LICENSE` and `COMMERCIAL_LICENSE.md` for details.

---

## Citation

If you publish results using this repository, please cite both the software and, where relevant, the underlying theory.

Use the repository’s `CITATION.cff` (GitHub: “Cite this repository”) and include the exact version/tag (or commit hash) used.

- Holmander, C. (2025). *Finite Realism: Epistemology, Metaphysics, and Physics Under Finite Resources*. Zenodo. https://doi.org/10.5281/zenodo.17226485

---

## Contact

**Maintainer:** Craig Holmander  
📧 craig.holm@protonmail.com  
🌐 ORCID: https://orcid.org/0009-0002-3145-8498
