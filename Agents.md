# Agents.md — veriscope

> **Purpose of this document**
>
> This file defines the *conceptual and architectural contract* of veriscope. It is not a tutorial and not a full API reference. Read this document before modifying core behavior, adding metrics, or changing gating semantics.

## Project Overview

**veriscope** is a research framework for **early warning detection of neural network training pathologies** (soft/hard collapse, mode collapse, gradient explosion). It implements a mathematically rigorous "Finite Realism" (FR) gating system based on information-theoretic divergence bounds, with applications to both vision models (ResNet/CIFAR-10) and language models (nanoGPT).

### Core Mission  
Detect training instabilities **before** catastrophic collapse by monitoring geometric, topological, and dynamical properties of the learned representation space, then making pass/fail gating decisions based on calibrated divergence thresholds.

### Why Gates (Pass/Fail) Instead of Scores
veriscope frames early warning as a *decision problem*, not a ranking or alerting problem. Binary gates are intentional: they support calibrated intervention (pause, rollback, adjust) under finite evidence budgets. Continuous anomaly scores are useful diagnostically, but gating decisions are made only when divergence bounds, statistical uncertainty (ε_stat), and sensitivity budgets jointly certify unacceptable risk.

---

## Architecture Overview

```  
veriscope/  
├── config.py                    # Central configuration with env overrides  
├── core/                        # FR mathematical framework  
│   ├── window.py               # WindowDecl, FRWindow definitions  
│   ├── transport.py            # DeclTransport (metric normalization)  
│   ├── gate.py                 # GateEngine (stability checking)  
│   ├── regime.py               # RegimeAnchoredGateEngine (drift detection)  
│   ├── calibration.py          # ε-statistics and epsilon resolution  
│   └── ipm.py                  # IPM distances (TV, product-TV)  
├── fr_integration.py           # FR wiring, window installation  
├── pipeline.py                 # High-level run_one/run_sweep/evaluate  
├── detectors/  
│   ├── baselines.py            # PH/CUSUM baseline facades  
│   └── learned.py              # Learned detector facade  
└── runners/  
    ├── legacy/                 # CIFAR-10/STL-10 runner (full implementation)  
    │   ├── legacy_cli_refactor.py  # Main CLI entry point (~4000 lines)  
    │   ├── data.py             # Data loading, splits, corruption  
    │   ├── model.py            # ResNet-18 for CIFAR  
    │   ├── features.py         # Feature extraction, JL projection  
    │   ├── metrics_heavy.py    # SW2, H0 persistence (budgeted)  
    │   ├── monitors.py         # Entropy, confidence monitors  
    │   ├── probes.py           # κ_sens sensitivity probes  
    │   ├── budget.py           # Heavy metric budget ledger  
    │   ├── runtime.py          # Global CFG/OUTDIR/BUDGET state  
    │   ├── determinism.py      # Seeding, strict determinism  
    │   ├── eval/core.py        # Evaluation, events, summaries  
    │   └── detectors/          # Baseline and learned implementations  
    └── gpt/  
        ├── adapter.py          # nanoGPT → veriscope adapter  
        ├── train_nanogpt.py    # GPT training with FR gating  
        └── analyze_gates.py    # Gate recall/precision analysis  
        # GPT runners are used for hypothesis validation and transfer tests, not as the primary development regime
```

## Lifecycle Overview

1. **Calibration** — Run control experiments (factor='none') to establish normalization ranges and ε-statistics.
2. **Window Installation** — Declare `WindowDecl` and install it via FR integration.
3. **Build Phase** — Accumulate healthy reference statistics (no gating failures allowed).
4. **Live Gating** — Evaluate gain, stability, and sensitivity during training.
5. **Post-hoc Evaluation** — Compute detection rates, lead times, and false positives.

---

## Key Concepts

### 1. WindowDecl (Window Declaration)  
Declares the observable metric space Φ_W:  
```python  
WindowDecl(  
    epsilon=0.12,                    # Divergence tolerance  
    metrics=["var_out_k", "eff_dim"], # Tracked metrics  
    weights={"var_out_k": 0.5, "eff_dim": 0.5},  
    bins=16,                          # Histogram bins for TV  
    cal_ranges={"var_out_k": (0, 1), "eff_dim": (0, 64)},  # Normalization ranges  
    interventions=(lambda x: x,),     # Test functions  
)  
```  

### 2. GateEngine  
Performs the core stability check:  
- **Gain check**: prequential gain ≥ threshold (learning is progressing)  
- **Stability check**: D_W(past, recent) ≤ ε - ε_stat (distribution stable)  
- **Sensitivity check**: κ_sens ≤ ε_sens (not overly sensitive to perturbations)  

### 3. RegimeAnchoredGateEngine  
Extended gate with reference-based drift detection:  
- **Change detection**: D_W(past, recent) — detects transitions  
- **Regime detection**: D_W(ref, recent) — detects sustained deviation from healthy baseline  
- Reference established only during a configurable "build phase" when model is healthy  

### 4. Metrics Tracked  
| Metric | Description | Schedule |  
|--------|-------------|----------|  
| `var_out_k` | Variance outside top-k eigenspace | Every epoch |  
| `eff_dim` | Effective dimension (participation ratio) | Every epoch |  
| `cos_disp` | Cosine similarity dispersion | Every epoch |  
| `ftle` | Finite-time Lyapunov exponent proxy | Every epoch |  
| `sw2` | Sliced Wasserstein-2 distance | Heavy (cadenced) |  
| `pers_H0` | H0 persistence (ripser) | Heavy (cadenced) |  
| `mon_entropy` | Monitor set entropy | Cadenced |  

### 5. Ground Truth (GT) Detection  
Unified GT labels collapse events:  
- **Hard collapse**: NaN/Inf in loss/gradients, gradient explosion  
- **Soft collapse**: Sustained low effective dimension (rank collapse)  

---

## Important Files & Entry Points

### Main Entry Points  
1. **`veriscope/runners/legacy_cli_refactor.py::main()`** — CIFAR sweep CLI  
2. **`veriscope/runners/gpt/train_nanogpt.py`** — GPT training with gating  
3. **`veriscope/pipeline.py`** — Programmatic API (`run_one`, `run_sweep`, `evaluate`)  

### Configuration  
- **`veriscope/config.py`** — Defaults and env loading (`CFG` dict)  
- **`veriscope/runners/legacy/runtime.py`** — Runtime state (`install_runtime()`)  
- Environment variables: `SCAR_SMOKE`, `SCAR_FR`, `SCAR_GATE_*`, `SCAR_OUTDIR`, etc.  

### Core FR Framework  
- **`veriscope/core/window.py`** — `WindowDecl`, `FRWindow`, `window_decl_identity_hash`  
- **`veriscope/core/gate.py`** — `GateEngine`, `GateResult`  
- **`veriscope/core/regime.py`** — `RegimeAnchoredGateEngine`, `RegimeConfig`  
- **`veriscope/core/transport.py`** — `DeclTransport`, `assert_naturality`  
- **`veriscope/core/ipm.py`** — `tv_hist_fixed`, `d_Pi`, `dPi_product_tv`, `D_W`  
- **`veriscope/core/calibration.py`** — `epsilon_statistic_bhc`, `aggregate_epsilon_stat`  

---

## Common Tasks

### Adding a New Metric  
1. Compute the metric in the training loop (e.g., `legacy_cli_refactor.py::run_one`)  
2. Add to `WindowDecl.metrics` and `WindowDecl.weights`  
3. Calibrate `cal_ranges` from control runs (factor='none')  
4. Update `det_features` in detector training if used for learned detection  

### Tuning Gate Parameters  
```python  
CFG["gate_window"] = 16        # Epochs per half-window  
CFG["gate_epsilon"] = 0.12     # Base tolerance  
CFG["gate_gain_thresh"] = 0.05 # bits/sample learning threshold  
CFG["gate_min_evidence"] = 16  # Min samples before evaluating  
CFG["warn_consec"] = 3         # Consecutive failures to trigger  
```  

### Running Smoke Tests  
```bash  
SCAR_SMOKE=1 python -m veriscope.runners.legacy_cli_refactor  
```  
Smoke mode reduces epochs, seeds, and relaxes thresholds for quick E2E validation.  

### Enabling FR Gating  
```bash  
SCAR_FR=1 python -m veriscope.runners.legacy_cli_refactor  
```  

---

## Code Patterns

### Environment-Driven Configuration  
```python  
if env_truthy("SCAR_SMOKE"):  
    cfg["epochs"] = 24  
    cfg["gate_window"] = 3  

# Env overrides have highest precedence  
v = os.environ.get("SCAR_GATE_EPSILON")  
if v is not None:  
    CFG["gate_epsilon"] = float(v)  
```  

### Robust Numeric Handling  
```python  
from veriscope.runners.legacy.utils import as_int, as_float, to_numeric_series  

x = as_int(cfg.get("epochs"), default=72)  
y = as_float(audit.get("worst_DW", np.nan), default=float("nan"))  
```  

### Gate Check Pattern  
```python  
result = gate_engine.check(  
    past=past_dict,           # {metric: np.ndarray}  
    recent=recent_dict,       # {metric: np.ndarray}  
    counts_by_metric=counts,  # {metric: int}  
    gain_bits=gain,           # float  
    kappa_sens=kappa,         # float  
    eps_stat_value=eps_stat,  # float  
)  
if not result.ok:  
    # Gate failed — potential instability  
    print(f"Gate FAIL: {result.audit}")  
```  

### WindowDecl Cloning  
```python  
# Preferred: use copy_with() or dataclasses.replace()  
regime_decl = base_decl.copy_with(epsilon=0.18)  
```  

---

## Testing & Validation

### Key Invariants  
1. **ε_stat ∈ [0, ε * max_frac]** — Statistical uncertainty is bounded  
2. **gate_warn=1 means PASS, gate_warn=0 means FAIL** — Important for metrics  
3. **Non-evaluated epochs are neutral (PASS)** — Insufficient history doesn't fail  
4. **Reference only established during build phase** — Prevents "bad but stationary" bug  

### Common Failure Modes  
- **Gate never evaluates**: Check `gate_min_evidence`, `gate_window`, and warmup  
- **False alarms in smoke**: Increase `gate_eps_inflation_max` or `gate_epsilon`  
- **Missing metrics**: Verify `cal_ranges` exists for all tracked metrics  

---

## Dependencies

### Required  
- Python ≥3.9  
- PyTorch, torchvision  
- NumPy, pandas, matplotlib  

### Optional  
- `ripser` — For H0 persistence (topology)  
- `filelock` — For dataset download coordination  
- `nanoGPT` — For GPT experiments (external checkout)  

---

## Important Warnings

1. **DO NOT import from `legacy_cli_refactor.py` in core modules** — Phase-1 boundary  
2. **CFG is a live mutable dict** — Always mutate in place, never rebind  
3. **Determinism requires `CUBLAS_WORKSPACE_CONFIG=:4096:8`** — Set early  
4. **Heavy metrics have budgets** — SW2 and ripser calls are rate-limited  
5. **Smoke mode changes many defaults** — Check `CFG_SMOKE` and `SMOKE_CRITICAL_KEYS`  

---

## Glossary

| Term | Definition |  
|------|------------|  
| **FR** | Finite Realism — the mathematical framework |  
| **D_W** | Operational distinguishability on Φ_W |  
| **ε** | Epsilon — divergence tolerance |  
| **ε_stat** | Statistical uncertainty from finite samples |  
| **TV** | Total Variation distance |  
| **SW2** | Sliced Wasserstein-2 distance |  
| **PH** | Persistent Homology / Page-Hinkley test |  
| **GT** | Ground Truth (collapse labels) |  
| **warm_idx** | warmup + ph_burn epochs |  
| **κ_sens** | Sensitivity budget (response to perturbations) |  

---

## Scope Note on GPT Experiments
Transformer experiments (nanoGPT) are treated as *validation domains* to test transfer of FR gating logic beyond vision models. Success or failure here informs metric choice and window design; it does not redefine core FR semantics.

---

## Contact & Contributing

This is a research codebase. Key design decisions are documented in docstrings and the `regime.py` module header contains detailed rationale for reference-anchored detection.
