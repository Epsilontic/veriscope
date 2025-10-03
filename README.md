# Veriscope

Early-warning system for representation drift in machine learning models.

Veriscope is an open research project developing tools to detect early signs of collapse in internal model diversity before they lead to brittle or unsafe behavior. Standard metrics can look healthy while representation structure quietly degrades. Veriscope provides **auditable, reproducible monitoring signals** to surface these hidden risks.

---

## Key capabilities

- **Representation drift detection**  
  Feature-geometry signals (e.g., effective rank, variance‑outside‑k)  
  Dynamics signals (FTLE‑like entropy gradients)  
  Cadenced heavy metrics (sliced‑W2; H0 persistence with `ripser` when available)
- **Finite Realism gate**  
  Fixed‑partition product–TV with prequential gain, ε_stat, and κ_sens probes  
  Resource budgets for heavy metrics with deterministic JL projections
- **Calibratable ground truth (unsupervised)**  
  Rank‑only soft‑collapse threshold calibrated on control runs  
  Per‑factor robust gradient cutoff for hard collapse
- **Learned detector with deployable gate**  
  Grouped CV, FP‑calibrated τ→τ′ mapping under the same gate used at inference
- **Audit‑ready artifacts**  
  Tamper‑evident provenance capsules (environment, window declaration, budgets)  
  Overlay files for scoring and independent verification
- **Reproducibility and determinism**  
  Deterministic seeds, cuBLAS workspace config, and safe data‑loading policies

---

## Status

This is an early‑stage research prototype. Interfaces and outputs may change. Expect a reproduction suite for CIFAR‑10 first; extension to other modalities will follow.

- ✅ CIFAR‑10 reproduction suite (PyTorch)
- 🔜 Extension to language‑model fine‑tuning runs
- 🔜 Public Python package with verifier + benchmark tools

---

## Requirements

- Python 3.9+
- PyTorch ≥ 1.13 (2.x recommended), `torchvision`
- Optional GPU with CUDA 11/12 (CPU is supported; heavy metrics will be slower)
- `numpy`, `pandas`, `matplotlib`
- Optional:
  - `filelock` (safer dataset downloads)
  - `ripser` (for H0 persistence; otherwise falls back to NaN rather than crash)
  - `pyarrow` or `fastparquet` (for writing parquet; CSV fallback is automatic)

Example install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# If you don’t have a requirements file yet, minimally:
# pip install torch torchvision numpy pandas matplotlib filelock ripser pyarrow
```

---

## Datasets

- **CIFAR‑10** (train/test) — downloaded automatically into `SCAR_DATA` if absent  
- **STL‑10** (test split) — used as an external, label‑free monitor stream (with resize); falls back to a clean validation split if unavailable

---

## Quick start

**Smoke run (fast end‑to‑end):**

```bash
export SCAR_SMOKE=1
export SCAR_OUTDIR=./out_scar_smoke
export SCAR_DATA=./data
python veriscope_oct.py
```

This runs a tiny sweep (few seeds, short epochs) and writes artifacts to `SCAR_OUTDIR`. Expect a few minutes on GPU; longer on CPU.

**Full sweep (more seeds/epochs):**

```bash
unset SCAR_SMOKE
export SCAR_OUTDIR=./out_scar_full
export SCAR_DATA=./data
python veriscope_oct.py
```

> Note: a full sweep can take hours depending on hardware.

---

## Configuration and environment

Main knobs (env vars override defaults):

- `SCAR_OUTDIR`: output directory for all artifacts
- `SCAR_DATA`: dataset root (CIFAR‑10 / STL‑10)
- `SCAR_SMOKE=1`: small budget, at least one heavy pass
- `SCAR_NUM_WORKERS`: DataLoader workers (`0` recommended for reproducibility)
- `SCAR_EVAL_SPLIT`: `eval` (default) | `calib` | `both`

Gate / detector:

- `SCAR_FAMILY_Z_THR`: z‑threshold for family‑gate confirmation (default ≈ 2.903)
- `SCAR_WARN_CONSEC`, `SCAR_FAMILY_WINDOW`: warn persistence and local window

Soft GT calibration:

- `SCAR_FIXED_GT_RANK_MIN`: override effective‑rank threshold (disables quantile calibration)
- `SCAR_SOFT_Q`: quantile for soft GT (when not using a fixed threshold)

Determinism:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set automatically if unset)

You can also edit `CFG` in the script to adjust seeds, epochs, heavy‑metrics cadence, and budgets.

---

## Outputs and artifacts

Written under `SCAR_OUTDIR`:

- **Run logs**  
  `bundle_runs_{tag}.parquet` (CSV fallback if parquet unavailable)  
  `errors_{tag}.log` (per‑run failures, if any)
- **Learned detector and calibration**  
  `learned_detector_{tag}.json`  
  `tau_mapping_{tag}.json` (τ→τ′ mapping under the deployed gate)  
  `ph_directions_{tag}.json` (CUSUM directions per metric)  
  `rp_adequacy_{tag}.json` (JL/native agreement pre‑warm)
- **Ground truth and baselines**  
  `grad_cutoff_by_factor_{tag}.json`  
  `baseline_events_{ewma|sma|vote|seq|newma}_{tag}.csv`  
  `detector_events_{tag}.csv`
- **Overlays and summaries**  
  `bundle_runs_eval_with_overlays_{tag}.parquet`  
  `bundle_runs_eval_with_overlays_soft.parquet`  
  `bundle_runs_eval_with_overlays_hard.parquet`  
  `summary_learned_{tag}.csv`  
  `summary_baseline_{gate|ewma|sma|vote|seq|newma}_{tag}.csv`
- **Invariants and provenance**  
  `artifact_invariants_provenance.json`  
  `window_provenance.json`  
  `env.json`, `repro.json`, `pip_freeze.txt`
- **Figures**  
  `figs/*.png` (per‑metric spaghetti and event raster plots)

Each write is atomic where possible; parquet writes fall back to CSV automatically with a format marker. Overlays are always emitted for both soft and hard tags to simplify downstream scoring.

---

## Reproducibility and determinism

- Seeds are set for Python, NumPy, and PyTorch (including CUDA).  
- cuBLAS deterministic workspace is configured and probed; if unavailable, the run falls back to `warn_only` to avoid device asserts.  
- DataLoader persistence is disabled for per‑epoch loaders to avoid file‑descriptor leaks.  
- JL projections use a bounded LRU cache with deterministic seeding.

---

## Evaluation protocol (CIFAR‑10)

- **Unified unsupervised GT**  
  Soft: low effective rank (`eff_dim` or `eff_dim_gt`), calibrated from control runs  
  Hard: NaN/inf or gradient explosion beyond robust per‑factor cutoffs
- **Learned detector**  
  Trained on calibration seeds; FP‑mapped under the deployed gate to meet an FP cap  
  Evaluated on held‑out seeds; run‑level FP and lead‑time reported
- **Baselines**  
  EWMA/SMA on loss, vote over geometry/dynamics, sequential PH + rank, NEWMA

In smoke mode, some heavy metrics are skipped or reduced in cadence to keep runtime short.

---

## Roadmap

- Broaden modality coverage beyond CIFAR‑10  
- Package release with verifiers, scoring scripts, and stable APIs  
- Alternative topology backends (MST‑based) for `ripser`‑free environments

---

## Documentation

- Conceptual framework: **Finite Realism: Epistemology, Metaphysics, and Physics Under Finite Resources** — <https://doi.org/10.5281/zenodo.17226485>

Technical notes and experiment logs will be added as the project matures.

---

## Contributing

Feedback, replication attempts, and methodological critiques are welcome. Please open an issue or contact the maintainer. As the code stabilizes, we will add contribution guidelines and a lightweight governance model.

---

## License

This project is dual‑licensed:

- **GNU Affero General Public License v3.0 (AGPL‑3.0‑only)**  
- **Commercial license** (for organizations who prefer not to comply with AGPL terms)

See `LICENSE` for full details. Use under AGPL‑3.0 is free. For commercial licensing inquiries, contact the maintainer.

---

## Citation

If you publish results using this repository, please cite both the software and, where relevant, the underlying theory.

**Citing the software (Veriscope).**  
Use the repository’s `CITATION.cff` (GitHub: “Cite this repository”) and include the exact version/tag (or commit hash) you used. If a DOI is minted for a release (e.g., via Zenodo), include that DOI in your citation.

**Citing the theory (Finite Realism).**  
When discussing the conceptual framework that motivates Veriscope (e.g., window/gate formalism and finite‑resource constraints), cite:

- Holmander, C. (2025). *Finite Realism: Epistemology, Metaphysics, and Physics Under Finite Resources*. Zenodo. <https://doi.org/10.5281/zenodo.17226485>

---

## Contact

**Maintainer:** Craig Holmander  
📧 craig.holm@protonmail.com  
🌐 ORCID: <https://orcid.org/0009-0002-3145-8498>
