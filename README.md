# README.md — veriscope  
  
<div align="center">  
  
# 🔬 Veriscope  
  
**Early-warning detection of neural network training pathologies**  
  
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
  
*Detect representation collapse before it's too late.*  
  
[Quick Start](#quick-start) • [Documentation](#core-concepts) • [Examples](#usage-examples) • [Citation](#citation)  
  
</div>  
  
---  
  
## Overview  
  
Veriscope is a research framework for detecting **soft collapse**, **mode collapse**, and **gradient instabilities** in neural network training — often *before* standard metrics show any warning signs. While loss curves may look healthy, internal representation structure can quietly degrade, leading to brittle or unsafe models.  
  
### The Problem  
  
Standard training metrics (loss, accuracy) can remain stable while:  
- Effective dimensionality collapses (representations become low-rank)  
- Feature diversity degrades (mode collapse)  
- Sensitivity to perturbations increases (fragile representations)  
  
### The Solution  
  
Veriscope provides **mathematically rigorous gating decisions** based on:  
- **Geometric metrics**: Effective dimension, variance outside top-k eigenspace  
- **Dynamical metrics**: FTLE-like entropy gradients, gradient noise scale  
- **Distributional metrics**: Sliced Wasserstein-2, total variation distance  
- **Topological metrics**: H0 persistence (optional, via ripser)  
  
All wrapped in a **Finite Realism (FR) framework** that turns monitoring into an auditable decision contract: you declare an observable window (Φ_W) via `WindowDecl`, calibrate normalization ranges on healthy controls, account for finite-evidence uncertainty via ε_stat, and enforce a sensitivity budget κ_sens. The output is a pass/fail gate with explicit reasons and artifacts suitable for post-hoc review.
  
---  
  
## Key Features  
  
### 🎯 Multi-Modal Support  
- **Vision models**: ResNet-18 on CIFAR-10 with STL-10 external monitoring  
- **Language models**: nanoGPT integration with transformer hidden-state probing  
  
### 🔒 Rigorous Gating Framework  
```  
┌─────────────────────────────────────────────────────────────┐  
│                     GateEngine Decision                      │  
├─────────────────────────────────────────────────────────────┤  
│  ✓ Gain check:      prequential_gain ≥ threshold           │  
│  ✓ Stability check: D_W(past, recent) ≤ ε - ε_stat         │  
│  ✓ Sensitivity:     κ_sens ≤ ε_sens                         │  
├─────────────────────────────────────────────────────────────┤  
│  PASS: All checks satisfied → Continue training             │  
│  FAIL: Any check violated  → Alert / halt / intervene       │  
└─────────────────────────────────────────────────────────────┘  
```  
  
### 📊 Regime-Anchored Detection  
Detect not just *changes* but *sustained deviations* from healthy baselines:  
- **Change detection**: D_W(past, recent) — catches transitions  
- **Regime detection**: D_W(reference, recent) — catches drift from healthy state  
  
### 🔬 Audit-Ready Artifacts  
- Provenance capsules with content digests (tamper-evident audit artifacts)  
- Reproducible with deterministic seeding  
- All decisions logged with full audit trails  
  
---  
  
## Architecture  
  
```  
veriscope/  
├── core/                        # FR mathematical framework  
│   ├── window.py               # WindowDecl, FRWindow  
│   ├── gate.py                 # GateEngine (stability checking)  
│   ├── regime.py               # RegimeAnchoredGateEngine (drift)  
│   ├── transport.py            # Metric normalization  
│   ├── calibration.py          # ε-statistics  
│   └── ipm.py                  # Divergence measures (TV, SW2)  
│  
├── runners/  
│   ├── legacy/                 # CIFAR-10 runner  
│   │   ├── legacy_cli_refactor.py  # Main entry point  
│   │   ├── features.py         # Feature extraction  
│   │   ├── metrics_heavy.py    # Budgeted heavy metrics  
│   │   └── ...  
│   │  
│   └── gpt/                    # GPT/Transformer runner  
│       ├── adapter.py          # nanoGPT → veriscope bridge  
│       ├── train_nanogpt.py    # Training with FR gating  
│       └── analyze_gates.py    # Gate analysis tools  
│  
├── detectors/                  # Detection algorithms  
│   ├── baselines.py            # PH, CUSUM, NEWMA  
│   └── learned.py              # Learned detector  
│  
├── config.py                   # Configuration management  
├── fr_integration.py           # FR wiring utilities  
└── pipeline.py                 # High-level API  
```  
  
---  
  
## Installation  
  
### Prerequisites  
- Python 3.9+  
- PyTorch ≥ 1.13 (2.x recommended)  
- CUDA 11/12 (optional, CPU supported)  
  
### Standard Installation  
  
```bash  
# Clone the repository  
git clone https://github.com/Epsilontic/veriscope.git  
cd veriscope  
  
# Create virtual environment  
python -m venv .venv  
source .venv/bin/activate  # On Windows: .venv\Scripts\activate  
  
# Install in editable mode  
pip install -e .  
```  
  
### With Optional Dependencies  
  
```bash  
# For topology metrics (H0 persistence)  
pip install -e ".[topo]"  
  
# For development  
pip install -e ".[dev]"  
```  
  
### For GPT Experiments  
  
```bash  
# Clone nanoGPT alongside veriscope  
git clone https://github.com/karpathy/nanoGPT.git  
  
# Prepare data (e.g., shakespeare_char)  
cd nanoGPT  
python data/shakespeare_char/prepare.py  
cd ..  
```  
  
---  
  
## Quick Start  
  
### 🚀 Smoke Test (5-10 minutes)  
  
Verify installation with a minimal run:  
  
```bash
# Smoke mode accepts truthy values: 1, true, yes, on
export SCAR_SMOKE=true
export SCAR_OUTDIR=./out_smoke
export SCAR_DATA=./data

veriscope
```

This command runs the default legacy CIFAR runner via the `veriscope` console script and writes a minimal artifact bundle to `SCAR_OUTDIR`.

Expected outputs include:
- `window_decl_calibrated.json`
- `summary_*.csv`
- `bundle_runs_*.parquet`
  
### 🔬 Full CIFAR-10 Sweep  
  
Run the complete evaluation suite:  
  
```bash  
export SCAR_OUTDIR=./out_full  
export SCAR_DATA=./data  
  
veriscope  
```  
  
> **Note**: A full sweep takes 2-8 hours depending on hardware.  
  
### 🤖 GPT Training with Gating

Train a small GPT with veriscope monitoring:  

```bash  
cd veriscope  

python -m veriscope.runners.gpt.train_nanogpt \  
    --dataset shakespeare_char \  
    --nanogpt_dir ../nanoGPT \  
    --gate_preset tuned  
```  

#### With Pathology Injection (for testing detection)  

```bash  
# Inject data corruption at iteration 2500  
python -m veriscope.runners.gpt.train_nanogpt \  
    --dataset shakespeare_char \  
    --nanogpt_dir ../nanoGPT \  
    --data_corrupt_at 2500 \  
    --data_corrupt_len 400 \  
    --data_corrupt_frac 0.15 \  
    --data_corrupt_mode permute  
```  

#### GPT Spike Experiment (recommended reproducible protocol)

This experiment validates **change-only corruption detection** on a controlled, localized pathology window and uses **regime** as a separate long-horizon drift alarm.

**Canonical config (Spike v1)**  
See the header comment block in `veriscope/runners/gpt/train_nanogpt.py` for the empirically validated settings. The key parameters are:

- `--metric_interval 2`
- `--gate_window 75`
- `--gate_warmup 1500`
- `--gate_epsilon 0.25`
- `--gate_eps_stat_max_frac 0.15`
- `--gate_min_evidence 75`
- `--gate_gain_thresh -0.002`

**A) Corruption run (15% token permutation, 2500–2900)**

```bash
python -m veriscope.runners.gpt.train_nanogpt \
    --dataset shakespeare_char \
    --nanogpt_dir ../nanoGPT \
    --device cuda \
    --metric_interval 2 \
    --gate_window 75 \
    --gate_warmup 1500 \
    --gate_epsilon 0.25 \
    --gate_eps_stat_max_frac 0.15 \
    --gate_min_evidence 75 \
    --gate_gain_thresh -0.002 \
    --data_corrupt_at 2500 \
    --data_corrupt_len 400 \
    --data_corrupt_frac 0.15 \
    --data_corrupt_mode permute
```

**B) Clean control run (same settings, no corruption flags)**

```bash
python -m veriscope.runners.gpt.train_nanogpt \
    --dataset shakespeare_char \
    --nanogpt_dir ../nanoGPT \
    --device cuda \
    --metric_interval 2 \
    --gate_window 75 \
    --gate_warmup 1500 \
    --gate_epsilon 0.25 \
    --gate_eps_stat_max_frac 0.15 \
    --gate_min_evidence 75 \
    --gate_gain_thresh -0.002
```

**C) Analyze both runs (spike overlap at 2500–2900)**

```bash
python -m veriscope.runners.gpt.analyze_gates \
    --results ./out/veriscope_gpt_run.json \
    --spike_start 2500 \
    --spike_len 400 \
    --gate_window 75 \
    --metric_interval 2
```

**How to read the output**

- For spike experiments, use the analyzer’s **EXECUTIVE SUMMARY → “CORRUPTION DETECTION (change-only)”** block.
- Treat **“BASELINE DRIFT (regime)”** as a separate signal (long-horizon deviation from a known-good reference).
- Do **not** score spike experiments using the union gate (change ∨ regime) when regime is active; it is intentionally conservative.
  
---  
  
## Usage Examples  
  
### Programmatic API  
  
```python  
from veriscope.pipeline import run_one, run_sweep, evaluate  
  
# Run a single seed with a specific pathology  
df = run_one(  
    seed=42,  
    tag="experiment",  
    monitor_ds=None,  # Uses default  
    factor={"name": "uniform_label_noise", "p": 0.3}  
)  
  
# Or run the full sweep  
df_all = run_sweep(tag="full")  
if df_all is not None:  
    evaluate(df_all, tag="full")  
```  
  
### Custom Gate Integration  
  
```python  
from veriscope.core.window import WindowDecl  
from veriscope.core.transport import DeclTransport  
from veriscope.core.gate import GateEngine, FRWindow  
  
# Define your observable metric space  
window_decl = WindowDecl(  
    epsilon=0.12,  
    metrics=["var_out_k", "eff_dim", "cos_disp"],  
    weights={"var_out_k": 0.4, "eff_dim": 0.4, "cos_disp": 0.2},  
    bins=16,  
    cal_ranges={  
        "var_out_k": (0.0, 1.0),  
        "eff_dim": (0.0, 64.0),  
        "cos_disp": (0.0, 0.5),  
    },  
)  
  
# Create gate engine  
transport = DeclTransport(window_decl)  
window_decl.attach_transport(transport)  
fr_win = FRWindow(decl=window_decl, transport=transport, tests=())  
  
gate = GateEngine(  
    frwin=fr_win,  
    gain_thresh=0.05,        # bits/sample  
    eps_stat_alpha=0.05,  
    eps_stat_max_frac=0.25,  
    eps_sens=0.04,  
    min_evidence=16,  
)  
  
# In your training loop:  
result = gate.check(  
    past=past_metrics,       # Dict[str, np.ndarray]  
    recent=recent_metrics,   # Dict[str, np.ndarray]  
    counts_by_metric=counts, # Dict[str, int]  
    gain_bits=prequential_gain,  
    kappa_sens=sensitivity,  
    eps_stat_value=eps_stat,  
)  
  
if not result.ok:  
    print(f"⚠️  Gate FAIL: {result.audit['reason']}")  
    # Take action: log, alert, reduce LR, checkpoint, etc.  
```  
  
### Regime-Anchored Detection  
  
```python  
from veriscope.core.regime import RegimeAnchoredGateEngine, RegimeConfig  
  
# Wrap base gate with regime detection  
regime_config = RegimeConfig(  
    enabled=True,  
    epsilon_mult=1.5,          # Regime ε = 1.5 × base ε  
    reference_build_span=1500,  # Build window iterations  
    min_evidence_per_metric=50,  
)  
  
regime_gate = RegimeAnchoredGateEngine(  
    base_engine=gate,  
    fr_win=fr_win,  
    config=regime_config,  
    gate_warmup=1000,  
    gate_window=100,  
)  
  
# Check returns combined decision  
result = regime_gate.check(  
    past=past_metrics,  
    recent=recent_metrics,  
    counts_by_metric=counts,  
    gain_bits=gain,  
    kappa_sens=kappa,  
    eps_stat_value=eps_stat,  
    iter_num=current_iteration,  
)  
  
# Access detailed audit  
print(f"Change OK: {result.audit['change_ok']}")  
print(f"Regime OK: {result.audit['regime_ok']}")  
print(f"Reference established: {result.audit.get('ref_established_at')}")  
```  
  
---  
  
## Core Concepts  
  
### Window Declaration (Φ_W)  
  
The `WindowDecl` defines the **observable metric space** — which aspects of training you're monitoring:  
  
| Field | Purpose |  
|-------|---------|  
| `epsilon` | Divergence tolerance (how much drift is acceptable) |  
| `metrics` | Which metrics to track |  
| `weights` | Relative importance of each metric |  
| `bins` | Histogram resolution for TV distance |  
| `cal_ranges` | Normalization ranges (calibrated from healthy runs) |  
  
### Gate Checks  
  
| Check | Formula | Interpretation |  
|-------|---------|----------------|  
| **Gain** | `gain_bits ≥ threshold` | Model is still learning |  
| **Stability** | `D_W(past, recent) ≤ ε - ε_stat` | Distribution is stable |  
| **Sensitivity** | `κ_sens ≤ ε_sens` | Not overly sensitive to perturbations |  
  
### Ground Truth Labels  

Veriscope generates **operational collapse event labels** (heuristic, audit-logged) from training signals:

| Type | Detection Criteria |  
|------|---------------------|  
| **Hard** | NaN/Inf in loss, gradient explosion (2+ consecutive) |  
| **Soft** | Effective dimension below calibrated threshold (with patience) |  
  
---  
  
## Configuration  
  
### Environment Variables  
  
| Variable | Default | Description |  
|----------|---------|-------------|  
| `SCAR_OUTDIR` | `./scar_bundle_phase4` | Output directory |  
| `SCAR_DATA` | `./data` | Dataset root |  
| `SCAR_SMOKE` | `0` | Enable smoke mode (quick E2E). Truthy values: `1`, `true`, `yes`, `on`. |  
| `SCAR_FR` | `0` | Enable FR gating |  
| `SCAR_GATE_EPSILON` | `0.08` | Gate divergence tolerance |  
| `SCAR_GATE_MIN_EVIDENCE` | `16` | Minimum samples before evaluating |  
| `SCAR_WARN_CONSEC` | `3` | Consecutive failures to trigger |  
| `SCAR_NUM_WORKERS` | `0` | DataLoader workers |

Note: Veriscope parses boolean environment variables using a truthy convention. For example, `SCAR_SMOKE=true` and `SCAR_SMOKE=1` behave identically.
  
### Gate Tuning  
  
```bash  
# Stricter detection (more sensitive)  
export SCAR_GATE_EPSILON=0.08  
export SCAR_GATE_GAIN_THRESH=0.05  
  
# More tolerant (fewer false alarms)  
export SCAR_GATE_EPSILON=0.15  
export SCAR_GATE_GAIN_THRESH=0.02  
```  
  
### GPT-Specific Presets  
  
```bash  
# Use tuned defaults for GPT  
python -m veriscope.runners.gpt.train_nanogpt --gate_preset tuned  
  
# Tuned preset applies:  
# --gate_window 100  
# --gate_warmup 1000  
# --gate_epsilon 0.15  
# --gate_gain_thresh -0.003  
# --gate_min_evidence 20  
```  
  
---  
  
## Output Artifacts  
  
After a run, `SCAR_OUTDIR` contains:  
  
```  
out/  
├── bundle_runs_{tag}.parquet      # All metrics per epoch  
├── detector_events_{tag}.csv      # Learned detector warnings  
├── baseline_events_*.csv          # Baseline detector warnings  
├── summary_*.csv                  # Detection rate, FP rate, lead time  
│  
├── learned_detector_{tag}.json    # Trained detector weights  
├── window_decl_calibrated.json    # Calibrated WindowDecl  
├── tau_mapping_{tag}.json         # Threshold calibration  
│  
├── artifact_invariants_provenance.json  # Audit trail  
├── env.json                       # Environment snapshot  
├── pip_freeze.txt                 # Dependencies  
│  
└── figs/                          # Visualization  
    ├── var_out_k_{tag}.png  
    ├── eff_dim_{tag}.png  
    ├── gate_warn_{tag}.png  
    └── event_raster_{tag}.png  
```  
  
---  
  
## Analyzing Results  
  
### Gate Analysis (GPT)  
  
```bash  
python -m veriscope.runners.gpt.analyze_gates \  
    --results ./out/veriscope_gpt_run.json \  
    --spike_start 2500 \  
    --spike_len 400 \  
    --gate_window 100 \  
    --metric_interval 2  
```  
  
Output:  
```  
═══════════════════════════════════════════════════════════════════  
 EXECUTIVE SUMMARY  
═══════════════════════════════════════════════════════════════════  
  
┌─ CORRUPTION DETECTION ────────────────────────────────────────┐  
│  Precision:    0.923  (when alarmed, was it real?)            │  
│  Recall:       0.857  (did we catch the corruption?)          │  
│  Specificity:  0.981  (quiet when clean?)                     │  
└───────────────────────────────────────────────────────────────┘  
```  
  
### Metrics Tracked  
  
| Metric | Description | Schedule |  
|--------|-------------|----------|  
| `var_out_k` | Variance outside top-k eigenspace | Every epoch |  
| `eff_dim` | Effective dimension | Every epoch |  
| `cos_disp` | Cosine dispersion | Every epoch |  
| `ftle` | FTLE proxy (entropy gradient) | Every epoch |  
| `sw2` | Sliced Wasserstein-2 | Cadenced (heavy) |  
| `pers_H0` | H0 persistence | Cadenced (heavy) |  
| `gate_warn` | Gate decision (1=pass, 0=fail) | Per window |  
| `gate_worst_tv` | Maximum TV distance | Per window |  
  
---  
  
## Troubleshooting  
  
### Gate Never Evaluates  
  
**Symptom**: `gate_reason` is always `insufficient_history` or `insufficient_evidence`  
  
**Solutions**:  
```bash  
# Reduce minimum evidence requirement  
export SCAR_GATE_MIN_EVIDENCE=8  
  
# Or reduce window size  
export SCAR_GATE_WINDOW=8  
```  
  
### Too Many False Alarms  
  
**Symptom**: Gate fails on healthy runs  
  
**Solutions**:  
```bash  
# Increase epsilon (more tolerance)  
export SCAR_GATE_EPSILON=0.15  
  
# Or increase eps_stat cap  
export SCAR_GATE_EPS_STAT_MAX_FRAC=0.30  
```  
  
### ripser Unavailable  
  
**Symptom**: `pers_H0` is always NaN  
  
**Solution**:  
```bash  
pip install ripser  
# Or accept that H0 metrics won't be available  
```  
  
### CUDA Determinism Errors  
  
**Symptom**: cuBLAS workspace errors  
  
**Solution**:  
```bash  
export CUBLAS_WORKSPACE_CONFIG=:4096:8  
```  
  
---  
  
## Performance Considerations  
  
| Hardware | Smoke Run | Full Sweep |  
|----------|-----------|------------|  
| RTX 3090 | ~5 min | ~2 hours |  
| RTX 2080 | ~10 min | ~4 hours |  
| CPU only | ~30 min | ~12+ hours |  
  
Heavy metrics (SW2, H0) have **budgets** to prevent runaway compute:  
- `sw2_budget_ms`: Per-call ceiling (default 200ms)  
- `ripser_budget_ms`: Per-call ceiling (default 250ms)  
- `total_heavy_budget_ms`: Per-run ceiling (default 180s)  
  
---  
  
## Reproducibility  
  
Veriscope is designed for **strict reproducibility**:  
  
- Deterministic seeding for Python, NumPy, PyTorch  
- cuBLAS workspace configuration for GPU determinism  
- JL projections use bounded LRU cache with fixed seeds  
- DataLoader workers use deterministic initialization  
- Environment snapshots saved with every run  
  
To verify reproducibility:  
```bash  
# Run twice  
SCAR_OUTDIR=./run1 veriscope  
SCAR_OUTDIR=./run2 veriscope  
  
# Compare checksums  
md5sum run1/bundle_runs_*.parquet run2/bundle_runs_*.parquet  
```  
  
---  
  
## Roadmap  
  
- [x] CIFAR-10 reproduction suite  
- [x] nanoGPT integration  
- [ ] Hugging Face Transformers integration  
- [ ] Diffusion model monitoring  
- [ ] Real-time dashboard  
- [ ] Alternative topology backends (MST-based)  
- [ ] Public Python package on PyPI  
  
---  
  
## Documentation  
  
### Theory  
- **Finite Realism framework**: [Finite Realism: Epistemology, Metaphysics, and Physics Under Finite Resources](https://doi.org/10.5281/zenodo.17226485)  
  
### Code Documentation  
- **Agents.md**: Comprehensive guide for AI coding assistants  
- **Inline docstrings**: All core modules are documented  
  
---  
  
## Contributing  
  
We welcome:  
- 🐛 Bug reports and fixes  
- 📊 Replication attempts on new datasets/models  
- 🔬 Methodological critiques and improvements  
- 📝 Documentation improvements  
  
Please open an issue to discuss significant changes before submitting PRs.  
  
---  
  
## License  
  
Dual-licensed:  
- **AGPL-3.0** for open-source use  
- **Commercial license** available for organizations  
  
See [LICENSE](LICENSE) for details.  
  
---  
  
## Citation  
  
If you use Veriscope in your research, please cite:  
  
```bibtex  
@software{veriscope2025,  
  author = {Holmander, Craig},  
  title = {Veriscope: Early-Warning Detection of Neural Network Training Pathologies},  
  year = {2025},  
  url = {https://github.com/Epsilontic/veriscope}  
}  
  
@article{holmander2025finiterealism,  
  author = {Holmander, Craig},  
  title = {Finite Realism: Epistemology, Metaphysics, and Physics Under Finite Resources},  
  year = {2025},  
  doi = {10.5281/zenodo.17226485}  
}  
```  
  
---  
  
## Contact  
  
**Maintainer**: Craig Holmander  
  
📧 craig.holm@protonmail.com    
🌐 ORCID: [0009-0002-3145-8498](https://orcid.org/0009-0002-3145-8498)  
  
---  
  
<div align="center">  
  
*Catch collapse before it catches you.*  
  
</div>
