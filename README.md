<div align="center">

# Veriscope

**Early-warning detection of neural network training pathologies**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Extras: torch | hf](https://img.shields.io/badge/extras-torch%20%7C%20hf-orange.svg)](#optional-extras)
[![License: Dual](https://img.shields.io/badge/License-Dual-brightgreen.svg)](./COMMERCIAL_LICENSE.md)
[![CI](https://github.com/Epsilontic/veriscope/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Epsilontic/veriscope/actions/workflows/ci.yml)

Runners: **CIFAR (PyTorch)** • **GPT (nanoGPT)** • **HF (transformers)**

*Detect representation collapse before it's too late.*

[Quick Start](#quick-start) • [Docs](./docs/) • [Pilot kit](#pilot-kit-overview) • [CLI](#capsule-operations) • [Core API](#core-api) • [Artifacts](#output-artifacts) • [License](#license)

</div>

---

## 5-minute AI2 reviewer path

Start with the synthetic reviewer packet (fast, no long training run required):

```bash
source ~/venv/bin/activate
veriscope --help
python -m veriscope.cli.main --help
veriscope validate docs/examples/reviewer_packet/run_a
veriscope report docs/examples/reviewer_packet/run_a --format text
veriscope diff docs/examples/reviewer_packet/run_a docs/examples/reviewer_packet/run_b
```

Optional smoke run:

```bash
bash scripts/run_gpt_smoke.sh /tmp/veriscope_smoke_test -- --max_iters 1
capdir="$(cat /tmp/veriscope_smoke_test/capdir.txt)"
veriscope validate "$capdir"
veriscope report "$capdir" --format text
```

Notes:
- CPU smoke can be slow on some hosts.
- Tiny smoke runs may end with `final_decision=skip` and zero evaluated gates; this is expected.

---

## Overview

Veriscope is a CLI-first tool for detecting early signs of training-time failure (representation collapse / drift) *before* loss curves or eval metrics make the issue obvious. It produces **auditable, reproducible capsule artifacts** you can validate, diff, and share.

In ~30 seconds, what you get:

- A repeatable way to run training with monitoring and emit a **capsule** (run directory) with stable contracts.
- A finite-window gate that turns monitored signals into an explicit status enum (`pass | warn | fail | skip`) plus structured audit fields.
- Validators (`veriscope validate`, `veriscope diff`) and reporting (`veriscope report`) for post-hoc review and sharing.

What we enforce (scoped to the declared window in `window_signature.json`):
- Capsule integrity checks are contract-enforced and auditable (`veriscope validate`).
- Validation is deterministic and read-only (no artifact mutation during validate/report).
- Run comparability is explicit and hash-gated by `window_signature_ref.hash` (+ gate preset policy).

Pipeline overview:

```
training loop
  -> metric snapshots (geometry / dynamics / optional heavy metrics)
  -> window normalization (declared transport)
  -> finite-window gate (decision + audit)
  -> capsule artifacts (results + signature + provenance)
```

This is formalized as a declared window + transport + gate → capsule artifacts:

- Declare an observable window via `WindowDecl`.
- Normalize metrics under a declared transport (`DeclTransport`).
- Evaluate stability via a windowed divergence with finite-evidence uncertainty (`ε_stat`).
- Emit structured audit fields and artifacts for post-hoc review.

---

## What it detects (operationally)

In this project, “collapse” is operational: it refers to internal representation structure becoming unusually *low-diversity* or *homogeneous* relative to a declared window and calibration, often preceding brittle behavior. Typical modes include:

- Rank/effective-dimension collapse
- Activation homogenization / dead features
- Mode dropping / distributional drift in internal features

What Veriscope outputs:
- A status enum (`pass | warn | fail | skip`) plus audit fields (e.g., `worst_DW`, `eps_eff`, evidence counts).
- A capsule directory with machine-readable artifacts (see “Output artifacts”).
- Canonical status fields live in artifacts as per-gate `decision` and summary `final_decision` (optional legacy booleans like `ok`/`warn` may appear).

What it does **not** guarantee:
- It does not prove safety or correctness of the trained model.
- It does not replace evals; it is an early-warning and audit surface for training dynamics within a declared window.

---

## Documentation map

| Topic | Where |
| --- | --- |
| v0 product/ops operational guidance (CLI semantics, exit codes, invariants) | `docs/productization.md` |
| Frozen artifact contract (hashing, canonicalization, governance chaining) | `docs/contract_v1.md` |
| Incubation readiness scope + claims/non-claims | `docs/incubation_readiness.md` |
| Calibration protocol (window-scoped) + report shape | `docs/calibration_protocol_v0.md` |
| Distributed mode contract and fail-loud rules | `docs/distributed_mode.md` |
| Pilot kit overview (what to run + what to share) | `docs/pilot/README.md` |
| Pilot harness scripts (exact commands + scoring) | `scripts/pilot/README.md` |

Normative precedence: `docs/contract_v1.md` is the authoritative contract. All other docs are derived companion guidance.

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

## Gate semantics (short version)

`GateEngine` evaluates stability by computing `worst_DW` (a windowed divergence across tracked metrics) and comparing it to an effective threshold `eps_eff`. The emitted status is an explicit enum (`pass | warn | fail | skip`); `skip` means “not evaluated” and is neutral.

For decision and exit-code behavior, defer to:
- `docs/contract_v1.md` (single normative contract)
- `docs/productization.md` (derived operational guidance)

---

## Status & scope

- ✅ CIFAR-10 reproduction suite (PyTorch)
- ✅ GPT runner with gate presets and spike/corruption injection
- ✅ HF runner (transformers) smoke workflows
- 🔜 Extension to additional language-model fine-tuning runs

Stability statement:
- **Artifact contract (v1)** is intended to be stable and is documented in `docs/contract_v1.md`.
- The **Python API** may evolve; the CLI + artifacts are the primary interface for pilots.
- v0 scope is **single-node** by default; distributed workflows are v1+.

---

## Requirements & Installation

### Core install

The base package targets Python 3.9+ and keeps dependencies minimal:
- `numpy`
- `pydantic`
- `typing_extensions`
- `filelock`

Install from source (editable):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

Or install from PyPI (when published):

```bash
pip install veriscope
```

### Optional extras

Install only the extras you need:

- **GPT/CIFAR runners (PyTorch)**  
  ```bash
  pip install -e ".[torch]"
  ```
- **HF runner (transformers/datasets/accelerate)**  
  ```bash
  pip install -e ".[hf]"
  ```
- **Analysis / parquet / plotting helpers (optional)**  
  ```bash
  pip install -e ".[report]"
  ```
  Installs `pandas`, `matplotlib`, and `pyarrow`.
- **Topological metrics (`ripser`)**  
  ```bash
  pip install -e ".[topo]"
  ```

If you’re installing from PyPI (once published), you can request extras directly:

```bash
pip install "veriscope[torch]"
pip install "veriscope[hf]"
pip install "veriscope[report]"
pip install "veriscope[topo]"
```

Note: all dependencies are declared in `pyproject.toml`, so you typically do not need a `requirements.txt`.

---

## Datasets

- **CIFAR-10** (train/test) — auto-downloaded into `SCAR_DATA` if missing
- **STL-10** (test split) — external, label-free monitor stream (with resize); falls back to a clean validation split if unavailable
- **Shakespeare (char)** — used by the GPT runner; prepared via nanoGPT scripts

Note: `SCAR_*` environment variables are legacy names used across the repository. They remain supported for now; see `docs/productization.md` for the boundary contract.

---

## Quick Start

After installation, run Veriscope via explicit subcommands (recommended).
Use `veriscope --help` to see the available subcommands.

### Golden Path (one command)

```bash
bash scripts/run_gpt_smoke.sh
```

This command emits a smoke capsule under `./out/`. Then verify it:

```bash
veriscope validate ./out/gpt_smoke_YYYYMMDD_HHMMSS
veriscope report ./out/gpt_smoke_YYYYMMDD_HHMMSS --format text
```

For `scripts/run_gpt_smoke.sh`, smoke defaults are reviewer-safe:
- default device is `cpu` (override with `VERISCOPE_GPT_SMOKE_DEVICE` or forwarded `--device ...`)
- default gate preset is `tuned_v0` unless you explicitly pass `--gate_preset/--gate-preset`

For pilot-scoped guarantees and boundaries, see:
- `docs/incubation_readiness.md`
- `docs/calibration_protocol_v0.md`
- `docs/distributed_mode.md`

### CIFAR smoke run (fast end-to-end)

```bash
export SCAR_DATA=./data
veriscope run cifar --smoke --outdir ./out/cifar_smoke_$(date +%Y%m%d_%H%M%S)
# If the console script is not available:
# python -m veriscope.cli.main run cifar --smoke --outdir ./out/cifar_smoke_$(date +%Y%m%d_%H%M%S)
```

Runs a small CIFAR sweep (few seeds, short epochs) into the specified outdir.

Expected output (minimum):

```bash
ls OUTDIR
# ... results.json results_summary.json window_signature.json run_config_resolved.json ...

cat OUTDIR/results_summary.json | head -n 20
# {
#   "schema_version": 1,
#   "run_id": "...",
#   "window_signature_ref": {"hash": "...", "path": "window_signature.json"},
#   "run_status": "success",
#   "counts": {"evaluated": 1, "skip": 0, "pass": 1, "warn": 0, "fail": 0},
#   "final_decision": "pass"
# }

veriscope validate OUTDIR
echo $?
# 0 means "valid capsule"
```

### Full CIFAR sweep (more seeds/epochs)

```bash
export SCAR_DATA=./data
veriscope run cifar --outdir ./out/cifar_full_$(date +%Y%m%d_%H%M%S)
```

> Note: a full sweep can take hours depending on hardware.

### HF quick start (transformers runner)

```bash
# Requires pip install -e ".[hf]"
veriscope run hf --gate-preset tuned_v0 --outdir "./out/hf_smoke_$(date +%Y%m%d_%H%M%S)" -- \
  --model gpt2 \
  --dataset wikitext:wikitext-2-raw-v1 \
  --dataset_split train \
  --max_steps 50 \
  --device cpu
```

Runs a short HF training loop and writes a capsule in the specified outdir.
For a scripted version, see `scripts/run_hf_smoke.sh`.

---

## Capsule operations

Common capsule commands:

```bash
veriscope validate OUTDIR
veriscope report OUTDIR --format md
veriscope report OUTDIR --format text
veriscope diff OUTDIR_A OUTDIR_B
veriscope override OUTDIR --status pass --reason "Known infrastructure noise"
```

Compare multiple capsules:

```bash
veriscope report --compare OUTDIR_A OUTDIR_B --format md
```

Comparison notes:
- `veriscope diff` and `veriscope report --compare` require a valid `governance_log.jsonl` in every compared OUTDIR; missing or invalid governance fails the command.
- Capsules marked `partial=true` are not comparable and are rejected by both `diff` and `report --compare`.
- Single-capsule `veriscope report OUTDIR` is less strict than compare mode and can still render while surfacing governance warnings.

---

## Pilot kit overview

If you are piloting Veriscope with a design partner, the pilot kit provides:
- A control + injected-pathology workflow with a shareable report and calibration summary.
- Success criteria grounded in the v0 contract.
- A stable artifact bundle to share with Veriscope.

Minimum share set (contract-aligned):
- non-partial capsules: `window_signature.json`, `results.json`, `results_summary.json`
- partial capsules: `window_signature.json` and `results_summary.json` with `partial=true`
- `run_config_resolved.json` is recommended for reproducibility context

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

The GPT runner wraps nanoGPT training with FR gating and emits a capsule directory (results summary + provenance + optional calibration CSV). For the normative artifact contract, defer to `docs/contract_v1.md`; for operational runner guidance, see `docs/productization.md`.

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

# GateResult exposes legacy booleans; the canonical status is `decision` in artifacts.
# Contract rule: if audit.evaluated is false, decision == "skip" (neutral).
evaluated = bool(result.audit.get("evaluated", True))
decision = (
    "skip"
    if not evaluated
    else (
        "warn"
        if bool(getattr(result, "warn", False))
        else ("pass" if bool(getattr(result, "ok", False)) else "fail")
    )
)

print("decision:", decision)
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

### Capsule outputs (`OUTDIR`)

Terminology:
- **Capsule**: a single immutable run directory with signature + results (+ optional governance overlays).
- **Sweep**: a collection of runs plus aggregated tables/figures and learned-detector artifacts.

**Artifact contract (v1):**
- **Non-partial capsules (required):**
  - `results.json`
  - `results_summary.json`
  - `window_signature.json`
- **Partial capsules (required):**
  - `window_signature.json`
  - `results_summary.json` with `partial=true`
- **Fail-marker invariant (enforced):**
  - `results_summary.first_fail_iter` is required when `counts.fail > 0`, and must be `null` when `counts.fail == 0`.
  - `first_fail_iter.txt` must exist iff `results_summary.first_fail_iter` is non-null.
- **Recommended:**
  - `run_config_resolved.json`
- **Optional governance / overlays (append-only; do not mutate raw artifacts):**
  - `manual_judgement.json`
  - `manual_judgement.jsonl`
  - `governance_log.jsonl`

**Pilot harness outputs (captured in `OUTDIR`):**
- `validate.txt`
- `inspect.txt`
- `report.md`
- `report_stderr.txt`
- `git_sha.txt`
- `version.txt`

**Calibration outputs (from `scripts/pilot/score.py`):**
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

Veriscope is dual-licensed. The default license is **AGPL-3.0-only**; **commercial licenses are available**
for organizations that cannot or do not want to comply with AGPL obligations (including network use).

- Open-source license: `LICENSE`
- Commercial terms: `COMMERCIAL_LICENSE.md`
- Notices: `NOTICE`
- Trademark policy: `TRADEMARK.md`

Package metadata declares SPDX `AGPL-3.0-only`; commercial terms are defined in `COMMERCIAL_LICENSE.md`.

Commercial licensing contact: see the “Contact” section below.

---

## Citation

If you publish results using this repository, please cite both the software and, where relevant, the underlying theory.

Use the repository’s `CITATION.cff` (GitHub: “Cite this repository”) and include the exact version/tag (or commit hash) used.

- Holmander, C. (2025). *Finite Realism: Epistemology, Metaphysics, and Physics Under Finite Resources*. Zenodo. https://doi.org/10.5281/zenodo.17226485

---

## Contact

**Maintainer:** Craig Holmander  
craig.holm@protonmail.com  
ORCID: https://orcid.org/0009-0002-3145-8498
