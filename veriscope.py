# Veriscope Bundle (Phase-4; invariants + FP-calibratable GT + robust learner)
#
# Delta from previous Phase-3 draft based on P0/P1 critique:
#   P0:
#     • Aggregation fallback: read CSV if parquet read fails when collating runs.
#     • Ripser import guard: sweep runs without ripser (pers_H0=NaN instead of crashing).
#     • Non-persistent per-epoch DataLoader workers: avoid “too many open files”.
#   P1:
#     • Deterministic FTLE-lowent quantile via manual sort.
#     • Overlay assert → warn (don’t kill evaluation on single mismatch).
#     • STL10 preflight with fallback to clean_val monitor.
#     • Per-tag error logs; SW2 equal-N uses a shared index when shapes match.
#
# Env:
#   SCAR_OUTDIR=...      # output dir
#   SCAR_DATA=...        # data root (CIFAR-10 / STL10 will be downloaded if absent)
#   SCAR_SMOKE=1         # optional: tiny sweep for quick end-to-end

# flake8: noqa: E122,E128

import hashlib
import json
import math
import os
import sys
import random
import subprocess
import time
import shutil
import inspect
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd

try:
    from filelock import FileLock
except Exception:
    FileLock = None  # graceful fallback if filelock isn't installed

# Ensure deterministic cuBLAS workspace is configured before torch/cuBLAS init (CUDA 11.x/12.x)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torchvision

# Strict determinism hygiene: disable TF32 matmul/cudnn paths
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    pass

# Optional stability: pin CPU thread counts to reduce nondeterminism and contention
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
# AMP compatibility shim: prefer torch.amp, fall back to torch.cuda.amp, otherwise no-op
try:
    from torch import amp as _amp_mod
    _AMP_SRC = "torch.amp"
except Exception:
    try:
        from torch.cuda import amp as _amp_mod  # type: ignore
        _AMP_SRC = "torch.cuda.amp"
    except Exception:
        _AMP_SRC = "none"
        class _NoAutocast:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        class _NoGradScaler:
            def __init__(self, enabled=False): self.enabled = bool(enabled)
            def scale(self, loss): return loss
            def unscale_(self, opt): pass
            def step(self, opt): opt.step()
            def update(self): pass
        class _NoAmp:
            autocast = _NoAutocast
            GradScaler = _NoGradScaler
        amp = _NoAmp()
    else:
        import inspect as _inspect
        _sig = _inspect.signature(_amp_mod.autocast)
        if "device_type" in _sig.parameters:
            amp = _amp_mod
        else:
            # Wrap autocast to ignore device_type on older torch.cuda.amp
            class _AutocastWrapper:
                def __init__(self, device_type=None, enabled=True):
                    self._ctx = _amp_mod.autocast(enabled=enabled)
                def __enter__(self): return self._ctx.__enter__()
                def __exit__(self, exc_type, exc, tb): return self._ctx.__exit__(exc_type, exc, tb)
            class _AmpShim:
                autocast = _AutocastWrapper
                GradScaler = _amp_mod.GradScaler
            amp = _AmpShim()
else:
    amp = _amp_mod
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import transforms

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


import traceback

_SW2_CPU_WARNED = False

# ---- P0: ripser import guard ----
try:
    from ripser import ripser as _ripser

    def _ripser_safe(X):
        return _ripser(X, maxdim=0)

except Exception as e:
    try:
        if mp.current_process().name == "MainProcess":
            print(f"[WARN] ripser unavailable ({e!r}) — pers_H0 will be NaN.")
    except Exception:
        pass

    def _ripser_safe(X):
        raise RuntimeError("ripser_unavailable")


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("PYTHONHASHSEED", "0")

try:
    if mp.current_process().name == "MainProcess":
        print("Torch:", torch.__version__, "TV:", torchvision.__version__, "CUDA:", torch.version.cuda)
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
except Exception:
    pass
    
# Numerically-stable sigmoid used across train/OOF/eval

def _sigmoid_stable(z, cap: float = 60.0):
    """Return sigmoid(z) with clipping to avoid overflow in exp()."""
    zc = np.clip(z, -float(cap), float(cap))
    return 1.0 / (1.0 + np.exp(-zc))

# === Finite‑window guard primitives (product–TV + prequential gain) ===
from dataclasses import dataclass  # already imported above; safe if duplicate

@dataclass
class Window:
    epsilon: float                  # resolution threshold in product–TV
    weights: dict                   # per‑context weights, e.g. {"raw":0.4,"detrend":0.3,"zscore":0.3}
    bins: int = 16                  # histogram bins for product–TV
    interventions: tuple = ()       # tuple of callables T(x) taking and returning arrays

def _tv_hist(a: np.ndarray, b: np.ndarray, bins: int) -> float:
    # assumes values ~[0,1]; clip if unsure
    a = np.clip(np.asarray(a, dtype=float), 0.0, 1.0)
    b = np.clip(np.asarray(b, dtype=float), 0.0, 1.0)
    ha, _ = np.histogram(a, bins=bins, range=(0, 1), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(0, 1), density=True)
    # avoid div‑by‑zero: if empty, TV=0
    if ha.sum() == 0 and hb.sum() == 0:
        return 0.0
    if ha.sum() == 0: ha = np.ones_like(ha)
    if hb.sum() == 0: hb = np.ones_like(hb)
    ha = ha / ha.sum()
    hb = hb / hb.sum()
    return 0.5 * np.abs(ha - hb).sum()

def product_tv(obs_by_ctx: dict, pred_by_ctx: dict, weights: dict, bins: int) -> float:
    tv = 0.0
    for c, w in weights.items():
        if c in obs_by_ctx and c in pred_by_ctx:
            tv += float(w) * _tv_hist(obs_by_ctx[c], pred_by_ctx[c], bins)
    return float(tv)

def prequential_gain(logp_model: np.ndarray, logp_baseline: np.ndarray) -> float:
    # positive means the model compresses better than the baseline
    # both arrays should be aligned to the same observation order
    return float((np.asarray(logp_baseline) - np.asarray(logp_model)).sum())

def earned_warning(
    obs_ctx: dict,          # {"raw": np.array, "detrend": np.array, "zscore": np.array, ...}
    pred_ctx: dict,         # same keys as obs_ctx; also pred_ctx["logp"] = per-record model log-probs
    base_ctx: dict,         # baseline; base_ctx["logp"] = per-record baseline log-probs
    window: Window,
    gain_thresh: float,
    transport=lambda x: x,  # alignment map applied to both sides after T
):
    """Return (ok, audit) where ok==True means the window passes stability + gain."""
    gain = prequential_gain(pred_ctx.get("logp", []), base_ctx.get("logp", []))
    if not np.isfinite(gain) or gain < gain_thresh:
        return False, {"why": "no_prequential_gain", "gain": float(gain)}

    # strip logp channel for TV computation
    pred_no_log = {k: v for k, v in pred_ctx.items() if k != "logp"}
    obs_no_log  = {k: v for k, v in obs_ctx.items() if k != "logp"}

    worst = 0.0
    intervs = window.interventions or (lambda x: x,)
    for T in intervs:
        tv = product_tv(
            {k: transport(T(v)) for k, v in pred_no_log.items()},
            {k: transport(T(v)) for k, v in obs_no_log.items()},
            window.weights,
            window.bins,
        )
        if tv > worst:
            worst = tv

    ok = worst <= window.epsilon
    return ok, {"gain": float(gain), "worst_tv": float(worst)}

# Default window (non-invasive; unused unless you call earned_warning)
DEFAULT_WINDOW = Window(
    epsilon=0.08,
    weights={"raw": 0.4, "detrend": 0.3, "zscore": 0.3},
    bins=16,
    interventions=(
        lambda x: x,                           # identity
        lambda x: x * 1.05,                    # scale up
        lambda x: x * 0.95,                    # scale down
        lambda x: np.clip(x + 0.05, 0, 1),     # shift up
    ),
)

# --- Finite Realism • Window Budgets, Fixed-Partition Gate, and Provenance ---
from dataclasses import dataclass as _fr_dataclass  # local alias to avoid shadowing

@_fr_dataclass
class WindowBudget:
    """Finite resource budgets for heavy metrics and the overall run."""
    sw2_ms: int = 200                 # per-call budget for sliced W2
    ripser_ms: int = 250              # per-call budget for H0 persistence
    total_heavy_ms: int = 180_000     # total wall-clock budget for heavy metrics in a run
    sw2_calls: int = 1_000_000        # defensive ceiling; tuned by CFG
    ripser_calls: int = 1_000_000     # defensive ceiling; tuned by CFG

class BudgetLedger:
    """Per-process budget ledger to enforce finite-window resource use."""
    def __init__(self, limits: WindowBudget):
        self.lim = limits
        self.spent_ms = {"sw2": 0.0, "ripser": 0.0, "heavy_total": 0.0}
        self.calls = {"sw2": 0, "ripser": 0}

    def allow(self, what: str) -> bool:
        if what == "sw2":
            return (
                self.calls["sw2"] < int(self.lim.sw2_calls)
                and self.spent_ms["heavy_total"] < float(self.lim.total_heavy_ms)
            )
        if what == "ripser":
            return (
                self.calls["ripser"] < int(self.lim.ripser_calls)
                and self.spent_ms["heavy_total"] < float(self.lim.total_heavy_ms)
            )
        return True

    def charge(self, what: str, ms: float):
        v = float(max(0.0, ms))
        if what in self.spent_ms:
            self.spent_ms[what] += v
            self.calls[what] += 1
        self.spent_ms["heavy_total"] += v

# Global ledger is instantiated after CFG is defined.
BUDGET: Optional[BudgetLedger] = None


def _window_hash(window: Window) -> str:
    """Deterministic id for the admissible window; used in provenance."""
    try:
        payload = {
            "epsilon": float(window.epsilon),
            "weights": {str(k): float(v) for k, v in (window.weights or {}).items()},
            "bins": int(window.bins),
            "interventions": len(tuple(window.interventions or ())),
        }
        s = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()
    except Exception:
        return ""


def write_window_provenance(outdir: Path, window: Window, cfg: dict, budget: WindowBudget):
    """Persist a tamper-evident window provenance capsule."""
    try:
        capsule = {
            "window_hash": _window_hash(window),
            "window": {
                "epsilon": float(window.epsilon),
                "weights": {str(k): float(v) for k, v in (window.weights or {}).items()},
                "bins": int(window.bins),
                "n_interventions": len(tuple(window.interventions or ())),
            },
            "budgets": {
                "sw2_ms": int(budget.sw2_ms),
                "ripser_ms": int(budget.ripser_ms),
                "total_heavy_ms": int(budget.total_heavy_ms),
                "sw2_calls": int(budget.sw2_calls),
                "ripser_calls": int(budget.ripser_calls),
            },
            "cfg_pointers": {
                "heavy_every": int(cfg.get("heavy_every", 0)),
                "sw2_n_proj": int(cfg.get("sw2_n_proj", 0)),
                "rp_repeats": int(cfg.get("rp_repeats", 0)),
                "rp_dim_topo": int(cfg.get("rp_dim_topo", 0)),
            },
        }
        s = json.dumps(capsule, sort_keys=True)
        capsule["sha256"] = hashlib.sha256(s.encode()).hexdigest()
        save_json(capsule, outdir / "window_provenance.json")
    except Exception:
        pass

# --- Fixed-partition Finite Realism declarations (Φ_W, G_T, ε_stat, κ_sens scaffolding) ---

class WindowDecl:
    def __init__(self, epsilon: float, metrics: List[str], weights: Dict[str, float],
                 bins: int, interventions: tuple, cal_ranges: Dict[str, Tuple[float, float]]):
        self.epsilon = float(epsilon)
        self.metrics = list(metrics)
        self.weights = dict(weights)
        self.bins = int(bins)
        self.interventions = tuple(interventions)
        self.cal_ranges = dict(cal_ranges)

    def transport(self, name: str, x: np.ndarray) -> np.ndarray:
        lo, hi = self.cal_ranges.get(name, (None, None))
        if lo is None or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(np.asarray(x, float))
        z = (np.asarray(x, float) - lo) / max(1e-12, (hi - lo))
        return np.clip(z, 0.0, 1.0)

# Global predeclared fixed-partition window (Φ_W); set this once per run
WINDOW_DECL: Optional[WindowDecl] = None

def install_window_decl(win: WindowDecl) -> None:
    """Setter to install Φ_W for gating."""
    global WINDOW_DECL
    WINDOW_DECL = win


def calibrate_window_from_controls(df_control: pd.DataFrame, metrics: List[str], weights: Dict[str, float],
                                   bins: int, epsilon: float, interventions: tuple) -> WindowDecl:
    """Predeclare Φ_W from factor=='none' controls after warm; freeze transport ranges per metric."""
    cal_ranges: Dict[str, Tuple[float, float]] = {}
    for m in metrics:
        if m in df_control.columns:
            col = pd.to_numeric(df_control[m], errors="coerce").to_numpy(dtype=float)
            col = col[np.isfinite(col)]
            if col.size >= 16:
                lo = float(np.nanpercentile(col, 1.0))
                hi = float(np.nanpercentile(col, 99.0))
            else:
                lo, hi = (0.0, 1.0)
        else:
            lo, hi = (0.0, 1.0)
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            lo, hi = (0.0, 1.0)
        cal_ranges[m] = (lo, hi)
    return WindowDecl(epsilon=epsilon, metrics=metrics, weights=weights, bins=bins,
                      interventions=interventions, cal_ranges=cal_ranges)


def tv_hist_fixed(z0: np.ndarray, z1: np.ndarray, bins: int) -> float:
    ha, _ = np.histogram(z0, bins=bins, range=(0.0, 1.0), density=True)
    hb, _ = np.histogram(z1, bins=bins, range=(0.0, 1.0), density=True)
    if ha.sum() == 0 and hb.sum() == 0:
        return 0.0
    if ha.sum() == 0: ha = np.ones_like(ha)
    if hb.sum() == 0: hb = np.ones_like(hb)
    ha = ha / ha.sum()
    hb = hb / hb.sum()
    return 0.5 * float(np.abs(ha - hb).sum())


def dPi_product_tv(window: WindowDecl, past_by_metric: Dict[str, np.ndarray],
                   recent_by_metric: Dict[str, np.ndarray]) -> float:
    tv = 0.0
    for m, w in window.weights.items():
        if m not in past_by_metric or m not in recent_by_metric:
            continue
        a = window.transport(m, past_by_metric[m])
        b = window.transport(m, recent_by_metric[m])
        tv += float(w) * tv_hist_fixed(a, b, window.bins)
    return float(tv)


def epsilon_statistic_bhc(n: int, k: int, alpha: float = 0.05) -> float:
    """Bretagnolle–Huber–Carol-type bound in TV for multinomial with k bins."""
    if n <= 0:
        return float("inf")
    x = float(np.log(2.0 / max(1e-12, alpha)))
    return math.sqrt(((max(1, k) - 1) * x) / max(1, n)) + (x / max(1, n))


def aggregate_epsilon_stat(window: WindowDecl, counts_by_metric: Dict[str, int],
                           k_override: Optional[int] = None, alpha: float = 0.05) -> float:
    total = 0.0
    for m, w in window.weights.items():
        n = int(counts_by_metric.get(m, 0))
        k = int(k_override) if k_override is not None else int(window.bins)
        eps_m = epsilon_statistic_bhc(n=n, k=k, alpha=alpha)
        total += float(w) * eps_m
    return float(total)


@torch.no_grad()
def collect_feature_snapshot(model: nn.Module, pool_loader, device, metrics: List[str], ref_mu_sig):
    """Collect a light snapshot of features/metrics over a small pool."""
    Z_geom, Z_geom_native = _features_for_loader(
        model, pool_loader, device,
        n_batches=max(1, CFG.get("metric_batches", 1)),
        cap=int(CFG.get("metric_total_cap", 256)),
        ref_mu_sig=ref_mu_sig, run_key=0, epoch=0,
    )
    out: Dict[str, np.ndarray] = {}
    try:
        v_ok, eff, *_ = variance_outside_k(Z_geom)
        out["var_out_k"] = np.array([float(v_ok)], dtype=float)
        out["eff_dim"]   = np.array([float(eff)], dtype=float)
    except Exception:
        out["var_out_k"] = np.array([float("nan")], dtype=float)
        out["eff_dim"]   = np.array([float("nan")], dtype=float)
    out.setdefault("ftle", np.array([float("nan")], dtype=float))
    return out


def kappa_sens_probe(model: nn.Module, opt: torch.optim.Optimizer, pool_loader, device,
                     window: WindowDecl, ref_mu_sig, probe_cfg: Dict) -> float:
    """Compute κ_sens using predeclared micro-probes. Always restores model params."""
    base = collect_feature_snapshot(model, pool_loader, device, window.metrics, ref_mu_sig)
    kappa = 0.0
    if probe_cfg.get("aug_probe", True):
        try:
            temp_loader = probe_cfg.get("aug_loader_factory", lambda: pool_loader)()
            interv = collect_feature_snapshot(model, temp_loader, device, window.metrics, ref_mu_sig)
            tv = dPi_product_tv(window, base, interv)
            if np.isfinite(tv):
                kappa = max(kappa, float(tv))
        except Exception:
            pass
    if probe_cfg.get("lr_probe", False):
        state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        try:
            lr0 = float(opt.param_groups[0].get("lr", CFG.get("base_lr", 0.1)))
            micro_lr = float(probe_cfg.get("lr_factor", 1.1)) * lr0
            micro_opt = torch.optim.SGD(model.parameters(), lr=micro_lr, momentum=0.0, weight_decay=0.0)
            it = iter(pool_loader)
            for _ in range(2):
                try:
                    xb, yb = next(it)
                except StopIteration:
                    break
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                micro_opt.zero_grad(set_to_none=True)
                loss.backward()
                micro_opt.step()
            interv2 = collect_feature_snapshot(model, pool_loader, device, window.metrics, ref_mu_sig)
            tv2 = dPi_product_tv(window, base, interv2)
            if np.isfinite(tv2):
                kappa = max(kappa, float(tv2))
        except Exception:
            pass
        finally:
            model.load_state_dict({k: v.to(device) for k, v in state.items()})
    return float(kappa)


def gate_check(window: WindowDecl,
               past: Dict[str, np.ndarray],
               recent: Dict[str, np.ndarray],
               counts: Dict[str, int],
               gain: float,
               gain_thresh: float,
               kappa_sens: float,
               eps_stat_alpha: float = 0.05) -> Tuple[int, Dict[str, float]]:
    """Joint gate: prequential gain + fixed-partition product–TV with ε_stat and κ_sens."""
    worst = 0.0
    intervs = window.interventions or (lambda x: x,)
    for T in intervs:
        tv_sum = 0.0
        for m, w in window.weights.items():
            a = T(window.transport(m, past.get(m, np.array([], dtype=float))))
            b = T(window.transport(m, recent.get(m, np.array([], dtype=float))))
            tv_sum += float(w) * tv_hist_fixed(a, b, window.bins)
        worst = max(worst, tv_sum)
    eps_stat = aggregate_epsilon_stat(window, counts_by_metric=counts, alpha=eps_stat_alpha)
    ok_gain = np.isfinite(gain) and (gain >= gain_thresh)
    ok_kappa = np.isfinite(kappa_sens) and (kappa_sens <= window.epsilon)
    ok_stability = np.isfinite(worst) and ((worst + eps_stat) <= window.epsilon)
    return int(bool(ok_gain and ok_stability and ok_kappa)), {
        "gain": float(gain),
        "worst_tv": float(worst),
        "eps_stat": float(eps_stat),
        "kappa_sens": float(kappa_sens if np.isfinite(kappa_sens) else 0.0),
    }


def write_window_audit(outdir: Path, window_decl: WindowDecl, note: str = ""):
    """Persist the predeclared Φ_W (fixed partitions, transports, bins, epsilon)."""
    try:
        payload = {
            "epsilon": float(window_decl.epsilon),
            "metrics": list(window_decl.metrics),
            "weights": {str(k): float(v) for k, v in window_decl.weights.items()},
            "bins": int(window_decl.bins),
            "cal_ranges": {k: [float(v[0]), float(v[1])] for k, v in window_decl.cal_ranges.items()},
            "interventions": [f"predeclared_T_{i}" for i, _ in enumerate(window_decl.interventions)],
            "notes": note or "Fixed partitions and transport calibrated from factor=='none' after warm",
        }
        save_json(payload, OUTDIR / "window_audit.json")
    except Exception:
        pass

# ---------------------------
# Output & config
# ---------------------------
OUTDIR = Path(os.environ.get("SCAR_OUTDIR", "./scar_bundle_phase4"))
try:
    if mp.current_process().name == "MainProcess":
        OUTDIR.mkdir(parents=True, exist_ok=True)
        sentinel = OUTDIR / "PHASE_TAG.txt"
        if not sentinel.exists():
            sentinel.write_text("Scar–Collapse Bundle Phase-4")
        else:
            sentinel_tag = sentinel.read_text().strip()
            if "Phase-4" not in sentinel_tag:
                raise RuntimeError(f"OUTDIR {OUTDIR} appears to be from a different phase: {sentinel_tag}")
except Exception:
    pass
DATA_ROOT = os.environ.get("SCAR_DATA", "./data")

C = 10  # CIFAR-10 classes

CFG = dict(
# device & determinism
device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
deterministic=True,  # strict determinism ON by default
amp=True,  # ignored when deterministic=True
# optimizer / schedule
base_lr=0.3,
momentum=0.9,
weight_decay=5e-4,
warmup=2,
cosine=True,
# grad safety
grad_clip_norm=5.0,  # default clip to stabilize early steps
# sweep budget
epochs=72,
batch=256,
# seeds
seeds_calib=list(range(401, 411)),  # 10 seeds for calibration
seeds_eval=list(range(511, 541)),  # 30 seeds for evaluation
# factorial pathologies (exactly one active per run)
factors=[
{"name": "none"},
{
"name": "none_safe",
"base_lr_scale": 0.33,
"override_grad_clip": 6.0,
},  # healthier run for FP denom
{"name": "uniform_label_noise", "p": 0.6},
{"name": "class_skew", "base_p": 0.4, "hot_scale": 1.7, "hot_k": 4},
{"name": "long_tail", "base_p": 0.4, "pareto_a": 3.0},
{"name": "input_corruption", "blur_p": 0.3, "noise_std": 0.03},
{
"name": "class_dropout_window",
"drop_classes": 1,
"drop_frac": 0.8,
"start": 10,
"end": 30,
},
{"name": "lr_spike", "epoch": 5, "factor": 6.0},
{"name": "mom_pulse", "start": 12, "duration": 3, "momentum": 0.0},
],
# monitor stream
monitor_source="external",  # {"external","clean_val","noisy_train"}
monitor_labels=False,  # optional: compute monitor_acc if True
metric_batches=3,  # per-epoch monitor mini-batches for mean/std estimates
metric_total_cap=512,  # per-epoch cap per metric (mean over <= cap samples)
# external monitor config (STL-10 resized, CIFAR-normalised)
external_monitor=dict(
dataset="STL10",
split="test",
resize_to=32,
pool_size=4000,  # per-run pool
ent_subset=1200,  # for entropy/confidence (independent stream)
ent_every=2,
),
# splits / normalisation frame
monitor_val_per_class=80,  # used only if monitor_source=="clean_val"
norm_ref_per_class=100,  # μ_ref, σ_ref frozen once at init per run (from CIFAR train)
# feature geometry
geom_std="ref",  # freeze σ to reference across epochs
geom_rp_dim=48,  # JL for covariance space in metrics
# variance-outside-k
var_k_energy=0.90,
var_k_max=32,
# heavy metrics cadence & budgets
heavy_every=6,
rp_dim_topo=16,
rp_repeats=8,
rp_agg="median",
rp_fixed=True,
ripser_budget_ms=250,
topo_sample_n=192,
sw2_budget_ms=200,
sw2_n_proj=96,  # lower to 64 for 8GB GPUs
# PH calibration (directions learned on SOFT collapses only)
ph_win=20,
ph_burn=6,
ph_two_sided=False,
ph_lambda=3.0,
ph_min_points=10,
# coherence/slope early-warning (observability only)
slope_w=7,
slope_tol=5e-4,
slope_persist_min=13,
corr_w=13,
coh_thresh=0.80,
coh_minlen=3,
# sequential PH / rank baseline params
ph_win_short=8,
seq_cusum_lambda=3.0,
rank_win=8,
# learned detector
detector_horizon=5,  # epochs before soft collapse considered positive
detector_L2_grid=[1e-3, 1e-2, 1e-1],
detector_steps_grid=[200, 400, 800],
detector_lr=1e-2,
detector_cv_folds=5,  # grouped CV over seeds
warn_vote=2,  # quorum k for vote baseline
warn_consec=3,  # consecutive-epoch hits required for a warn (env SCAR_WARN_CONSEC overrides)
det_use_missing=False,  # do NOT feed missingness indicators to learner
# baselines over train loss (reported; not used by learner)
ewma_alpha=0.2,
sma_k=5,
# NEWMA baseline (fast/slow rates)
newma_fast=0.3,
newma_slow=0.03,
# unified ground truth calibration (unsupervised)
gt_rank_min=8.0,  # effective rank threshold (native eff_dim)
gt_rank_q=0.075,  # quantile for eff_dim(_gt) when calibrating gt_rank_min
gt_patience=2,  # soft patience
# workers & I/O
num_workers=int(os.environ.get("SCAR_NUM_WORKERS", "0")),
# ops controls
max_failures=6,  # abort sweep if this many runs fail
skip_cos_disp_in_smoke=True,  # short-circuit cosine_dispersion when SCAR_SMOKE=1
skip_gns_in_smoke=True,       # short-circuit gradient_noise_scale when SCAR_SMOKE=1
# compatibility / backend flags
compat_mode=True,  # keep PH baselines on ripser during this calibration cycle
lock_sw2_backend=True,  # reserved: keep a stable SW2 backend per run
)

# Default family z-gate; overridden by SCAR_FAMILY_Z_THR if set
CFG.setdefault("family_z_thr", 2.903)

def _env_float_in_range(name: str, default: float, lo: float, hi: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        x = float(v)
        if not (lo <= x <= hi):
            raise ValueError
        return x
    except Exception:
        print(f"[WARN] ignoring {name}={v!r}; using {default}")
        return default

# --- env override for family z-gate (used by deployed detector gate) ---
CFG["family_z_thr"] = _env_float_in_range("SCAR_FAMILY_Z_THR", CFG.get("family_z_thr", 2.903), 0.25, 20.0)
try:
    if mp.current_process().name == "MainProcess":
        print(f"[env] family_z_thr={CFG['family_z_thr']:.3f}")
except Exception:
    pass

# Warn if determinism is requested but the cuBLAS workspace value is unexpected
try:
    if torch.cuda.is_available() and CFG.get("deterministic", True):
        val = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
        if val not in (":4096:8", ":16:8"):
            print(
            "[WARN] Deterministic=True but CUBLAS_WORKSPACE_CONFIG is not a standard deterministic value; using ':4096:8'."
            )
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
except Exception:
    pass

# Derive topo backend from compat flag (keep ripser during this cycle)
try:
    if CFG.get("compat_mode", True):
        CFG["topo_backend"] = "ripser"
    else:
        CFG["topo_backend"] = "mst"  # placeholder for next cycle
except Exception:
    pass

SUCCESS_TARGET = dict(min_detect_rate=0.90, min_lead=3, max_early_fp_rate=0.05)

# Vote baseline metrics — gradient/loss removed to avoid GT leakage / cadence bias.
VOTE_METRICS = ["cos_disp", "var_out_k", "ftle", "mon_entropy"]

# Scheduled metrics (cadenced/missing by design) — never fed to the learner.
SCHEDULED_METRICS = ["sw2", "pers_H0", "mon_entropy", "avg_max_prob"]

# TTL for scheduled metrics propagation to avoid stale ffill artifacts
CFG.setdefault("scheduled_ttl", 2 * CFG.get("heavy_every", 6))
# Family gate neighborhood size (epochs) for local confirmation
CFG.setdefault("gate_early_exit", True)

# Finite-window heavy-metric budgets (env-overridable)
CFG.setdefault("total_heavy_budget_ms", 180_000)
CFG.setdefault("ripser_calls_cap", 1_000_000)
CFG.setdefault("sw2_calls_cap", 1_000_000)

# Gate parameters for fixed-partition finite realism (predeclared Φ_W)
CFG.setdefault("gate_window", 16)
CFG.setdefault("gate_bins", 16)
CFG.setdefault("gate_epsilon", 0.08)
CFG.setdefault("gate_gain_thresh", 5.0)

# Instantiate the global budget ledger using current CFG limits
try:
    _wb = WindowBudget(
        sw2_ms=int(CFG.get("sw2_budget_ms", 200)),
        ripser_ms=int(CFG.get("ripser_budget_ms", 250)),
        total_heavy_ms=int(CFG.get("total_heavy_budget_ms", 180_000)),
        sw2_calls=int(CFG.get("sw2_calls_cap", 1_000_000)),
        ripser_calls=int(CFG.get("ripser_calls_cap", 1_000_000)),
    )
    BUDGET = BudgetLedger(_wb)
except Exception:
    BUDGET = BudgetLedger(WindowBudget())

# Smoke-mode overrides (fast E2E)
CFG_SMOKE = dict(
    seeds_calib=[401, 402],
    seeds_eval=[511, 512],
    epochs=16,
    heavy_every=16,      # will be overwritten by the update below
    rp_repeats=1,
    sw2_n_proj=64,
    gate_window=6,       # 2*W=12 <= 16 so the gate runs
    warmup=4,            # NEW
    ph_burn=0,           # NEW
)

# Ensure smoke mode executes at least one heavy pass
CFG_SMOKE.update({
"heavy_every": max(1, int(CFG_SMOKE.get("epochs", 16)) // 4)
})

def seeds_for_eval_from_env(CFG_dict):
    """Return seeds for evaluation based on SCAR_EVAL_SPLIT.
    SCAR_EVAL_SPLIT: "eval" (default) | "calib" | "both".
    """
    mode = os.environ.get("SCAR_EVAL_SPLIT", "eval").lower().strip()
    if mode == "both":
        return list(CFG_dict.get("seeds_calib", [])) + list(CFG_dict.get("seeds_eval", []))
    if mode == "calib":
        return list(CFG_dict.get("seeds_calib", []))
    return list(CFG_dict.get("seeds_eval", []))


# ---------------------------
# Repro / env utils
# ---------------------------
def _cuda_hash():
    try:
        exe = shutil.which("nvidia-smi")
        if not exe:
            return ""
        s = subprocess.check_output([exe, "-q"], stderr=subprocess.DEVNULL, text=True)[:20000]
        return hashlib.md5(s.encode()).hexdigest()
    except Exception:
        return ""


def _pip_freeze_md5():
    try:
        s = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL, text=True
        )[:50000]
        return hashlib.md5(s.encode()).hexdigest()
    except Exception:
        return ""


def save_json(obj, path: Path):
    def jsonable(o):
        try:
            json.dumps(o)
            return o
        except Exception:
            if isinstance(o, torch.device):
                return str(o)
            return str(o)

    if isinstance(obj, dict):
        obj = {k: jsonable(v) for k, v in obj.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def update_json(path: Path, patch: Dict):
    try:
        cur = json.loads(path.read_text())
        if not isinstance(cur, dict):
            cur = {}
    except Exception:
        cur = {}
    cur.update(patch or {})
    save_json(cur, path)


# Repro capsule: persist environment/cuda/library hashes once per sweep
try:
    repro_path = OUTDIR / "repro.json"
    if mp.current_process().name == "MainProcess":
        if not repro_path.exists():
            save_json({
            "cuda_hash_md5": _cuda_hash(),
            "pip_freeze_md5": _pip_freeze_md5(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
            "deterministic": bool(CFG.get("deterministic", True)),
            }, repro_path)
except Exception:
    pass

# Persist the finite-window provenance alongside repro info
try:
    if (mp.current_process().name == "MainProcess") and (BUDGET is not None):
        write_window_provenance(OUTDIR, DEFAULT_WINDOW, CFG, BUDGET.lim)
except Exception:
    pass


# Helper to enable strict determinism or fallback to warn_only if unavailable
def _enable_strict_det_or_fallback():
    """Enable strict deterministic algorithms; if cuBLAS determinism is unavailable,
    fall back to warn_only=True so the run proceeds without device-side asserts."""
    if not torch.cuda.is_available():
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except Exception:
            pass
        return
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        # Force one cuBLAS GEMM under strict determinism to validate workspace config
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        (a @ b).sum().item()
        print("[det] strict determinism OK (cuBLAS probe passed)")
    except Exception as e:
        print(f"[WARN] strict determinism failed on cuBLAS probe: {e!r}; falling back to warn_only")
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_all(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        _enable_strict_det_or_fallback()


def new_gen(master: int, off: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed((master * 7919 + off * 104729) % (2**63 - 1))
    return g


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def safe_write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    try:
        df.to_parquet(tmp)
        tmp.replace(path)
    except Exception:
        # best-effort cleanup of tmp on failure
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        try:
            path.with_suffix(".format.txt").write_text("csv")
        except Exception:
            pass


# --- Overlay writer: always emit both overlays (soft + hard) for scoring ---
def _write_both_overlays(ov_all: pd.DataFrame, outdir: Path):
    """
    Always emit BOTH overlays (soft + hard) regardless of which GT tags appear
    so downstream scoring can consistently read either file.

    Expects a column 'collapse_tag_gt' containing one of {'none','soft','hard'}.
    Includes 'none' rows in both outputs to keep FP denominators consistent.
    """
    assert "collapse_tag_gt" in ov_all.columns, "overlay missing 'collapse_tag_gt'"

    soft = ov_all[ov_all["collapse_tag_gt"].isin(["none", "soft"])].copy()
    hard = ov_all[ov_all["collapse_tag_gt"].isin(["none", "hard"])].copy()

    soft_path = outdir / "bundle_runs_eval_with_overlays_soft.parquet"
    hard_path = outdir / "bundle_runs_eval_with_overlays_hard.parquet"

    # Write atomically via existing helper
    safe_write_parquet(soft, soft_path)
    safe_write_parquet(hard, hard_path)

# --- Warn persistence (k-consecutive) applied per (seed, factor) group ---
def _apply_warn_persistence(df: pd.DataFrame, k: int):
    if not isinstance(k, int) or k <= 1:
        return df
    warn_cols = [c for c in df.columns if c.startswith("is_warn_epoch_")]
    if not warn_cols:
        return df
    keys = ["seed", "factor"]
    for col in warn_cols:
        df[col] = df.groupby(keys)[col].transform(
        lambda s: s.rolling(k, min_periods=k).sum().ge(k)
        )
    return df


def file_md5(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _u01_from_hash(*xs) -> float:
    h = hashlib.blake2b("::".join(map(str, xs)).encode(), digest_size=8).digest()
    v = int.from_bytes(h, byteorder="big", signed=False)
    return (v & ((1 << 53) - 1)) / float(1 << 53)


def _std0(X: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """dim-0 std with cross-version safety."""
    try:
        # Newer PyTorch prefers 'correction'
        return X.std(dim=0, keepdim=keepdim, correction=0)
    except TypeError:
        # Older PyTorch uses 'unbiased'
        return X.std(0, keepdim=keepdim, unbiased=False)


# ---------------------------
# Data & loaders
# ---------------------------
def _cifar_datasets():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    tfm_tr = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    ])
    tfm_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    ])
    if FileLock is not None:
        lock_path = os.path.join(DATA_ROOT, "cifar.lock")
        with FileLock(lock_path):
            tr_aug  = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True,  download=True,  transform=tfm_tr)
            tr_eval = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True,  download=False, transform=tfm_eval)
            te_eval = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True,  transform=tfm_eval)
    else:
        tr_aug  = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True,  download=True,  transform=tfm_tr)
        tr_eval = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True,  download=False, transform=tfm_eval)
        te_eval = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True,  transform=tfm_eval)
    return tr_aug, tr_eval, te_eval


def _stl10_monitor_dataset(cfg_ext):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    size = cfg_ext["resize_to"]
    tfm = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    ])
    if FileLock is not None:
        lock_path = os.path.join(DATA_ROOT, "stl10.lock")
        with FileLock(lock_path):
            ds = torchvision.datasets.STL10(root=DATA_ROOT, split=cfg_ext["split"], download=True, transform=tfm)
    else:
        ds = torchvision.datasets.STL10(root=DATA_ROOT, split=cfg_ext["split"], download=True, transform=tfm)
    return ds


# ---- P0/P1: DataLoader persistence policy helper ----
def make_loader(
ds,
batch,
shuffle,
workers,
gen,
device,
sampler: Optional[Sampler] = None,
persistent: Optional[bool] = None,
):
    """
    If `persistent` is None:
        pass
    • persist only when we expect to reuse the loader across epochs AND have workers (sampler provided)
    • otherwise False (esp. for per-epoch recreated loaders)
    """
    if persistent is None:
        persistent = (workers > 0) and (sampler is not None)
    else:
        # Clamp: persistent_workers requires num_workers > 0
        persistent = bool(persistent) and (workers > 0)
    use_pin = CFG.get("pin_memory", None)
    if use_pin is None:
        # Prefer pin_memory on CUDA even when num_workers == 0
        pin = (device.type == "cuda")
    else:
        pin = bool(use_pin)
    _kwargs = {}
    try:
        if "persistent_workers" in inspect.signature(DataLoader).parameters:
            _kwargs["persistent_workers"] = persistent
    except Exception:
        pass
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=(shuffle and sampler is None),
        num_workers=workers,
        pin_memory=pin,
        generator=gen,
        sampler=sampler,
        drop_last=False,
        worker_init_fn=seed_worker if workers > 0 else None,
        **_kwargs,
    )


def subset_loader_from_indices(ds, idxs: np.ndarray, batch: int, shuffle: bool, seed: int, device):
    sub = Subset(ds, [int(i) for i in idxs])
    gen = torch.Generator().manual_seed(100000 + seed)
    use_pin = CFG.get("pin_memory", None)
    if use_pin is None:
        pin = (device.type == "cuda")
    else:
        pin = bool(use_pin)
    # This loader is reused across epochs; allow persistence.
    _kwargs = {}
    try:
        if "persistent_workers" in inspect.signature(DataLoader).parameters:
            _kwargs["persistent_workers"] = (CFG["num_workers"] > 0)
    except Exception:
        pass
    return DataLoader(
    sub,
    batch_size=min(256, batch),
    shuffle=shuffle,
    generator=gen,
    num_workers=CFG["num_workers"],
    pin_memory=pin,
    drop_last=False,
    worker_init_fn=seed_worker if CFG["num_workers"] > 0 else None,
    **_kwargs,
    )


def _balanced_indices_by_class(labels, per_class, rng):
    idx_by_c = {c: [] for c in range(C)}
    for i, y in enumerate(labels):
        idx_by_c[int(y)].append(i)
    take = []
    for c in range(C):
        pool = np.array(idx_by_c[c], dtype=np.int64)
        rng.shuffle(pool)
        k = min(per_class, len(pool))
        take += pool[:k].tolist()
        idx_by_c[c] = pool[k:]
    rng.shuffle(take)
    return take, idx_by_c


def _take_from_pools(pools: Dict[int, np.ndarray], per_class: int, rng):
    take = []
    for c in range(C):
        pool = pools[c]
        pool_copy = pool.copy()
        rng.shuffle(pool_copy)
        k = min(per_class, len(pool_copy))
        take += pool_copy[:k].tolist()
        pools[c] = pool_copy[k:]
    rng.shuffle(take)
    return take


def load_splits(seed: int):
    tr_aug, tr_eval, te_eval = _cifar_datasets()
    rng = np.random.default_rng(4242 + seed)

    labels_train = np.array(tr_eval.targets, dtype=np.int64)

    mon_take, pools = _balanced_indices_by_class(labels_train, CFG["monitor_val_per_class"], rng)
    mon_val = Subset(tr_eval, mon_take)

    norm_ref_take = _take_from_pools(pools, CFG["norm_ref_per_class"], rng)
    norm_ref = Subset(tr_eval, norm_ref_take)

    all_idx = np.arange(len(tr_aug), dtype=np.int64)
    mask = np.ones(len(all_idx), dtype=bool)
    if mon_val is not None:
        mask[np.array(mon_take, dtype=np.int64)] = False
    mask[np.array(norm_ref_take, dtype=np.int64)] = False
    tr_take = all_idx[mask].tolist()

    splits_path = OUTDIR / f"splits_seed{seed}.json"
    save_json(
    dict(
    MONITOR_VAL=(mon_val.indices if mon_val is not None else []),
    NORM_REF=norm_ref_take,
    TRAIN_REST=tr_take,
    ),
    splits_path,
    )
    return tr_aug, tr_take, mon_val, norm_ref, splits_path


# ---------------------------
# Factorial pathologies (one active per run)
# ---------------------------
def apply_factor_to_labels(y: np.ndarray, factor: Dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(10_000 + seed)
    out = y.copy()
    name = factor["name"]
    if name == "uniform_label_noise":
        p = float(factor.get("p", 0.0))
        flip = rng.random(len(y)) < p
        k = rng.integers(1, C, size=flip.sum())
        out[flip] = (y[flip] + k) % C
    elif name == "class_skew":
        base_p = float(factor.get("base_p", 0.0))
        hot_k = int(factor.get("hot_k", 4))
        hot = rng.choice(np.arange(C), size=max(1, hot_k), replace=False)
        p_class = np.full(C, base_p, dtype=np.float32)
        p_class[hot] = np.clip(base_p * float(factor.get("hot_scale", 1.5)), 0, 0.98)
        u = rng.random(len(y))
        flip = u < p_class[y]
        k = rng.integers(1, C, size=flip.sum())
        out[flip] = (y[flip] + k) % C
    elif name == "long_tail":
        base_p = float(factor.get("base_p", 0.0))
        a = float(factor.get("pareto_a", 3.0))
        tail = rng.pareto(a=a, size=C)
        tail = tail / tail.max()
        p_class = np.clip(base_p * (0.5 + 0.8 * tail), 0, 0.98)
        flip = rng.random(len(y)) < p_class[y]
        k = rng.integers(1, C, size=flip.sum())
        out[flip] = (y[flip] + k) % C
    else:
        pass
    return out


def maybe_corrupt_input(
x: torch.Tensor, factor: Dict, seed: int, epoch: int, idx: int
) -> torch.Tensor:
    name = factor["name"]
    if name != "input_corruption":
        return x
    # blur (deterministic per (seed,epoch,idx))
    if _u01_from_hash("blur", seed, epoch, idx) < float(factor.get("blur_p", 0.0)):
        k = 3
        pad = (k - 1) // 2
        x = torch.nn.functional.avg_pool2d(x.unsqueeze(0), k, stride=1, padding=pad).squeeze(0)
    # additive noise
    std = float(factor.get("noise_std", 0.0))
    if std > 0:
        g = torch.Generator(device=x.device).manual_seed(
        int(1e6 * _u01_from_hash("noise", seed, epoch, idx))
        )
        noise = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype) * std
        x = (x + noise).clamp_(-3, 3)
    return x


class FactorisedTrainDataset(torch.utils.data.Dataset):
    """CIFAR10 train indices with a single active factor per run. Provides epoch-aware dropout semantics."""

    def __init__(
    self, base: torchvision.datasets.CIFAR10, tr_indices: List[int], factor: Dict, seed: int
    ):
        self.base = base
        self.indices = list(tr_indices)
        orig = np.array(base.targets, dtype=np.int64)[np.array(self.indices, dtype=np.int64)]
        self.factor = factor
        self.seed = seed
        self.labels = apply_factor_to_labels(orig, factor, seed)
        self.drop_cfg = None
        if factor["name"] == "class_dropout_window":
            rng = np.random.default_rng(90_000 + seed)
            self.drop_cfg = dict(
            drop_classes=set(
            rng.choice(
            np.arange(C), size=max(1, int(factor.get("drop_classes", 1))), replace=False
            ).tolist()
            ),
            drop_frac=float(factor.get("drop_frac", 0.7)),
            start=int(factor.get("start", 10)),
            end=int(factor.get("end", 30)),
            )
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def should_drop(self, i: int) -> bool:
        if self.drop_cfg is None:
            return False
        if not (self.drop_cfg["start"] <= self._epoch < self.drop_cfg["end"]):
            return False
        y = int(self.labels[i])
        if y not in self.drop_cfg["drop_classes"]:
            return False
        u = _u01_from_hash("drop", self.seed, self._epoch, int(self.indices[i]))
        return u < self.drop_cfg["drop_frac"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x, _ = self.base[idx]
        x = maybe_corrupt_input(x, self.factor, self.seed, self._epoch, idx)
        return x, int(self.labels[i])


class DropoutAwareSampler(Sampler):
    """Epoch-aware sampler that excludes indices marked for dropout BEFORE batching (deterministic)."""

    def __init__(self, dataset: FactorisedTrainDataset, batch_size: int, seed: int):
        self.ds = dataset
        self.bs = batch_size
        self.seed = seed
        self._cache_epoch = None
        self._cache_valid = None

    def set_epoch(self, epoch: int):
        self.ds.set_epoch(epoch)
        self._cache_epoch = None

    def _refresh(self):
        if self._cache_epoch == self.ds._epoch and self._cache_valid is not None:
            return
        valid = [i for i in range(len(self.ds)) if not self.ds.should_drop(i)]
        rng = np.random.default_rng(7777 + self.seed + self.ds._epoch)
        rng.shuffle(valid)
        if self.bs and len(valid) >= self.bs:
            valid = valid[: (len(valid) // self.bs) * self.bs]
        self._cache_epoch = self.ds._epoch
        self._cache_valid = valid

    def __iter__(self):
        self._refresh()
        return iter(self._cache_valid)

    def __len__(self):
        self._refresh()
        return len(self._cache_valid)


# ---------------------------
# Model & schedule
# ---------------------------
def make_model():
    try:
        m = torchvision.models.resnet18(weights=None, num_classes=C)
    except TypeError:
        m = torchvision.models.resnet18(pretrained=False, num_classes=C)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def make_opt(model):
    return torch.optim.SGD(
    model.parameters(),
    lr=CFG["base_lr"],
    momentum=CFG["momentum"],
    weight_decay=CFG["weight_decay"],
    nesterov=True,
    )


def lr_at(epoch: int, total: int, base: float, warmup: int, cosine: bool):
    if warmup > 0 and epoch < warmup:
        return base * (epoch + 1) / warmup
    if cosine:
        t = (epoch - warmup) / max(1, total - warmup)
        return base * 0.5 * (1 + math.cos(math.pi * t))
    return base


@torch.no_grad()
def penult(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    h = model.relu(model.bn1(model.conv1(x)))
    h = model.layer1(h)
    h = model.layer2(h)
    h = model.layer3(h)
    h = model.layer4(h)
    h = model.avgpool(h)
    return torch.flatten(h, 1)


# ---------------------------
# JL cache (LRU bounded)
# ---------------------------
class JLCache:
    def __init__(self, capacity: int = 64):
        self.cap = capacity
        self.store: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()

    def get(
    self,
    d_in: int,
    q: int,
    run_key: int,
    epoch: int,
    fixed: bool,
    device="cpu",
    dtype=torch.float32,
    ):
        q = int(min(q, d_in))
        # Cache on logical keys only (CPU, float32); move/cast on return to avoid fragmentation
        key = (d_in, q, run_key if fixed else (run_key, epoch))
        if key in self.store:
            self.store.move_to_end(key)
            A_cpu = self.store[key]
        else:
            base_seed = 1234567 + d_in * 13 + q * 29 + (run_key * 7) + (0 if fixed else epoch * 17)
            g = torch.Generator(device="cpu").manual_seed(base_seed)
            A_cpu = torch.randn(
            d_in, q, generator=g, device="cpu", dtype=torch.float32
            ) / math.sqrt(q)
            self.store[key] = A_cpu
            if len(self.store) > self.cap:
                self.store.popitem(last=False)
        return A_cpu.to(device=device, dtype=dtype)


_JL = JLCache(capacity=128)


# ---------------------------
# Metrics (multi-batch mean/std)
# ---------------------------
@torch.no_grad()
def _features_for_loader(
model,
loader,
device,
n_batches: int,
cap: int,
ref_mu_sig: Optional[Tuple[torch.Tensor, torch.Tensor]],
run_key: int,
epoch: int,
):
    model.eval()
    feats = []
    cnt = 0
    it = iter(loader)
    for _ in range(n_batches):
        try:
            xb, _ = next(it)
        except StopIteration:
            it = iter(loader)
            try:
                xb, _ = next(it)
            except Exception as e:
                print(f"[WARN] features fetch failed: {e}")
                break
        except Exception as e:
            print(f"[WARN] features fetch failed: {e}")
            break
        xb = xb.to(device)
        h = penult(model, xb).detach().cpu().float()
        feats.append(h)
        cnt += h.shape[0]
        if cnt >= cap:
            break

    H = (
    torch.cat(feats, 0)
    if feats
    else torch.zeros(
    (0, (ref_mu_sig[0].shape[1] if ref_mu_sig is not None else 1)), dtype=torch.float32
    )
    )
    if H.numel() == 0:
        if ref_mu_sig is not None:
            mu, sig_ref = ref_mu_sig
        else:
            d = H.shape[1]
            mu = torch.zeros((1, d), dtype=torch.float32)
            sig_ref = torch.ones((1, d), dtype=torch.float32)
    else:
        if ref_mu_sig is None:
            mu = H.mean(dim=0, keepdim=True)
            sig_ref = _std0(H, keepdim=True) + 1e-8
        else:
            mu, sig_ref = ref_mu_sig
    std_frame = sig_ref  # freeze to reference across epochs
    Z_geom_native = ((H - mu) / std_frame).to(torch.float32)

    Z_geom = Z_geom_native
    if CFG["geom_rp_dim"]:
        d_in = Z_geom.shape[1]
        q = int(min(CFG["geom_rp_dim"], d_in))
        if q < d_in:
            A = _JL.get(d_in, q, run_key, epoch, CFG["rp_fixed"], device="cpu")
            Z_geom = (Z_geom_native @ A).to(torch.float32)
    return Z_geom, Z_geom_native


def cov_eigs(Z: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, int]:
    n = Z.shape[0]
    COV = (Z.T @ Z) / max(1, (n - 1))
    COV = COV + eps * torch.eye(COV.shape[0], device=Z.device, dtype=Z.dtype)
    try:
        w = torch.linalg.eigvalsh(COV)
    except Exception:
        w = torch.linalg.eigvals(COV).real
    neg = int((w < -1e-8).sum().item())
    return w.clamp_min(0), neg


def choose_k_by_energy(eigs: torch.Tensor, energy: float, kmax: int) -> Tuple[int, float, bool]:
    if not (0.0 < energy <= 0.999999):
        raise AssertionError("var_k_energy must be in (0,1).")
    s = eigs.sum()
    if s <= 0 or eigs.numel() == 0:
        return 1, float("nan"), False
    w_desc = torch.flip(eigs, dims=[0])
    cum_top = torch.cumsum(w_desc, dim=0)
    idxs = (cum_top >= energy * s).nonzero(as_tuple=False)
    if idxs.numel() == 0:
        return 1, float("nan"), False
    k = int(idxs[0].item()) + 1
    k = max(1, min(k, int(min(eigs.shape[0], kmax))))
    tail_frac = float(((s - cum_top[k - 1]) / (s + 1e-12)).cpu().item())
    return k, tail_frac, True


def variance_outside_k(Z: torch.Tensor) -> Tuple[float, float, int, float, int, int]:
    w, neg = cov_eigs(Z)
    ok = 1
    k, tail_mass, good = choose_k_by_energy(w, CFG["var_k_energy"], CFG["var_k_max"])
    if not good:
        ok = 0
        k = 1
        tail_mass = float("nan")
    s = w.sum().clamp_min(1e-12)
    topk = w[-k:].sum()
    var_out_k = float(1.0 - (topk / s))
    eff_dim = float((s**2 / (w.pow(2).sum().clamp_min(1e-12))).cpu())
    return var_out_k, eff_dim, int(k), float(tail_mass), ok, neg


# --- spectral_r2: top-2 energy ratio from covariance eigenvalues
def spectral_r2(Z: torch.Tensor) -> float:
    """Top-2 energy ratio (λ1+λ2)/Σλ using covariance eigenvalues; nan if d<2 or degenerate."""
    try:
        w, _ = cov_eigs(Z)
        if w.numel() < 2:
            return float("nan")
        s = float(w.sum().cpu().item())
        if not np.isfinite(s) or s <= 0:
            return float("nan")
        r2 = float(((w[-1] + w[-2]) / w.sum()).cpu().item())
        return r2
    except Exception:
        return float("nan")


@torch.no_grad()
def cosine_dispersion(Z: torch.Tensor, seed: int, epoch: int, sample: int = 800) -> float:
    """Cosine‑similarity dispersion of JL/geom features (observability only)."""
    if os.environ.get("SCAR_SMOKE", "0") == "1" and CFG.get("skip_cos_disp_in_smoke", False):
        return float("nan")
    n = Z.shape[0]
    if n <= 2:
        return float("nan")
    sample_n = int(min(max(2, sample), n))
    _seed = int(1e6 * _u01_from_hash("cos", seed, epoch))
    # Device-safe generator
    if Z.device.type == "cuda":
        g = torch.Generator(device=Z.device).manual_seed(_seed)
        idx = torch.randperm(n, generator=g, device=Z.device)[:sample_n]
    else:
        g = torch.Generator().manual_seed(_seed)
        idx = torch.randperm(n, generator=g)[:sample_n]
    X = F.normalize(Z[idx], dim=1)
    G = X @ X.T
    off = G - torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    var = off.pow(2).sum() / max(1, off.numel() - G.shape[0])
    return float(var.sqrt().item())


def h0_total_persistence_np(X: np.ndarray) -> float:
    try:
        dgm0 = _ripser_safe(X)["dgms"][0]
    except Exception:
        return float("nan")
    if dgm0.size == 0:
        return 0.0
    lifetimes = dgm0[:, 1] - dgm0[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    return float(lifetimes.sum())


def topo_h0_jl_agg(
Z: torch.Tensor,
q: int,
repeats: int,
run_key: int,
epoch: int,
agg: str = "median",
sample_n: int = 192,
) -> Tuple[float, int, float, int]:
    """
    Returns: (value, n_successful_repeats, elapsed_ms, sampled_n_each_repeat)
    On any exception inside a repeat, that repeat is skipped. If none succeed, value=nan.
    """
    t0 = time.time()
    vals = []
    used_n = 0
    for r in range(repeats):
        # Enforce both per-call and global finite-window budgets
        if (time.time() - t0) * 1000.0 > CFG["ripser_budget_ms"]:
            break
        if (BUDGET is not None) and (not BUDGET.allow("ripser")):
            break
        try:
            t_rep = time.time()
            A = _JL.get(
            Z.shape[1],
            min(q, Z.shape[1]),
            run_key + 1009 * (r + 1),
            epoch,
            CFG["rp_fixed"],
            device=Z.device,
            dtype=Z.dtype,
            )
            Zr = Z @ A
            X = (Zr - Zr.mean(dim=0, keepdim=True)) / (_std0(Zr, keepdim=True) + 1e-8)
            n = X.shape[0]
            if n > sample_n:
                _seed = 17_000_000 + run_key + 31 * epoch + 101 * r
                if X.device.type == "cuda":
                    g = torch.Generator(device=X.device).manual_seed(_seed)
                    idx = torch.randperm(n, generator=g, device=X.device)[:sample_n]
                else:
                    g = torch.Generator().manual_seed(_seed)
                    idx = torch.randperm(n, generator=g)[:sample_n]
                X = X[idx]
                used_n = sample_n
            else:
                used_n = n
            vals.append(h0_total_persistence_np(X.cpu().numpy()))
            ms_rep = (time.time() - t_rep) * 1000.0
            try:
                if BUDGET is not None:
                    BUDGET.charge("ripser", ms_rep)
            except Exception:
                pass
        except Exception:
            continue
    elapsed = (time.time() - t0) * 1000.0
    if not vals:
        return float("nan"), 0, elapsed, used_n
    val = float(np.median(vals) if agg == "median" else np.mean(vals))
    return val, len(vals), elapsed, used_n


@torch.no_grad()
def sliced_w2_gpu(
Zt: torch.Tensor, Zt1: torch.Tensor, n_proj: int, seed: int, device
) -> Tuple[float, float, int]:
    """
    Compute sliced W2 distance with deterministic equal-N downsample.

    Determinism: uses a per-(seed) torch.Generator; when shapes match the same
    permutation index is applied to Zt and Zt1 (shared-index equal-N).
    Numerics: direction vectors are normalized with a small epsilon to avoid
    rare division-by-zero under deterministic seeds. Returns (value, elapsed_ms, n_proj_done).
    """
    t0 = time.time()
    Zt, Zt1 = Zt.to(device), Zt1.to(device)
    if Zt.shape[1] == 0 or Zt1.shape[1] == 0:
        return float("nan"), (time.time() - t0) * 1000.0, 0
    n = min(Zt.shape[0], Zt1.shape[0])
    if n < 2:
        return float("nan"), (time.time() - t0) * 1000.0, 0
    # CPU/GPU-safe generator
    if device.type == "cuda":
        g = torch.Generator(device=device).manual_seed(91001 + seed * 1000)
    else:
        # Warn once per process when running SW2 on CPU
        global _SW2_CPU_WARNED
        if not _SW2_CPU_WARNED:
            print("[WARN] SW2 running on CPU; this may be slow.")
            _SW2_CPU_WARNED = True
        g = torch.Generator().manual_seed(91001 + seed * 1000)

    # Equal‑N shared‑index path: reuse the same random permutation for both epochs (deterministic)
    if Zt.shape[0] == Zt1.shape[0] == n:
        if device.type == "cuda":
            idx = torch.randperm(n, generator=g, device=device)
        else:
            idx = torch.randperm(n, generator=g)
        Zt = Zt[:n][idx]
        Zt1 = Zt1[:n][idx]
    else:
        if device.type == "cuda":
            idx = torch.randperm(Zt.shape[0], generator=g, device=device)[:n]
            idx1 = torch.randperm(Zt1.shape[0], generator=g, device=device)[:n]
        else:
            idx = torch.randperm(Zt.shape[0], generator=g)[:n]
            idx1 = torch.randperm(Zt1.shape[0], generator=g)[:n]
        Zt = Zt[idx]
        Zt1 = Zt1[idx1]

    d = Zt.shape[1]
    dirs = torch.randn(n_proj, d, generator=g, device=device, dtype=Zt.dtype)
    norm = dirs.norm(dim=1, keepdim=True)
    dirs = dirs / (norm + 1e-12)
    pt_sorted, _ = torch.sort(Zt @ dirs.T, dim=0)
    pt1_sorted, _ = torch.sort(Zt1 @ dirs.T, dim=0)
    # --- ensure GPU ops complete before measuring elapsed time ---
    msq = ((pt_sorted - pt1_sorted) ** 2).mean()
    val = float(torch.sqrt(torch.clamp(msq, min=0.0)).item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ms = (time.time() - t0) * 1000.0
    return val, ms, n_proj


# ---------------------------
# PH series preprocessing helper (centralized)
# ---------------------------
def _prep_series_for_ph(g: pd.DataFrame, metric: str) -> List[float]:
    """Prepare a metric series for PH detection: ffill scheduled metrics, apply validity masks,
    and gate pers_H0 by min repeats and time budget. Returns a Python list with NaNs for invalid epochs.
    """
    s = g[metric].copy()
    if metric in SCHEDULED_METRICS:
        s = s.ffill()
        # apply TTL to scheduled metrics to avoid stale carry-over
        arr = s.to_numpy(dtype=float)
        age = np.full_like(arr, np.inf, dtype=float)
        last = -1
        for i, v in enumerate(arr):
            if np.isfinite(v):
                last = i
            age[i] = (i - last) if last >= 0 else np.inf
        ttl = float(CFG.get("scheduled_ttl", 2 * CFG.get("heavy_every", 6)))
        arr = np.where(age <= ttl, arr, np.nan)
    else:
        arr = s.to_numpy(dtype=float)
    vcol = f"{metric}_valid"
    if vcol in g.columns:
        mask = g[vcol].astype(bool).to_numpy()
        arr = np.where(mask, arr, np.nan)
    if metric == "pers_H0":
        if "topo_done" in g.columns:
            min_rep = int(math.ceil(CFG.get("rp_repeats", 8) / 2))
            td = g["topo_done"].astype(float).to_numpy()
            arr = np.where((td >= min_rep) & np.isfinite(arr), arr, np.nan)
        if "topo_ms" in g.columns:
            ms = g["topo_ms"].astype(float).to_numpy()
            budget = float(CFG.get("ripser_budget_ms", 250))
            arr = np.where(np.isfinite(arr) & np.isfinite(ms) & (ms <= 0.9 * budget), arr, np.nan)
        # Require enough samples for the topology estimate (item #3)
        nused = None
        if "n_topo_sampled" in g.columns:
            nused = g["n_topo_sampled"].astype(float).to_numpy()
        elif "topo_n_used" in g.columns:
            # our logging uses this name for per-repeat sample count
            nused = g["topo_n_used"].astype(float).to_numpy()
        if nused is not None:
            arr = np.where(np.isfinite(arr) & (nused >= 64), arr, np.nan)
    return [float(x) if np.isfinite(x) else float("nan") for x in arr]


def sliced_w2_gpu_budget(Zt, Zt1, n_proj, seed, device, budget_ms):
    try:
        if (BUDGET is not None) and (not BUDGET.allow("sw2")):
            return float("nan"), 0.0, 0, False
        val, ms, nproj_done = sliced_w2_gpu(Zt, Zt1, n_proj, seed, device)
    except Exception:
        return float("nan"), 0.0, 0, False
    ok = (ms <= budget_ms) and np.isfinite(val) and (nproj_done > 0)
    try:
        if BUDGET is not None:
            BUDGET.charge("sw2", ms)
    except Exception:
        pass
    return (val if ok else float("nan")), ms, int(nproj_done if ok else 0), bool(ok)


def monitor_entropy(model, loader, device) -> float:
    model.eval()
    ent_sum = 0.0
    total = 0
    with torch.inference_mode():
        it = iter(loader)
        while True:
            try:
                xb, _ = next(it)
            except StopIteration:
                break
            except Exception as e:
                print(f"[WARN] monitor fetch failed: {e}")
                break
            xb = xb.to(device)
            p = F.softmax(model(xb), dim=1).clamp_min(1e-12)
            ent = -(p * p.log()).sum(dim=1)
            ent_sum += float(ent.sum().item())
            total += xb.shape[0]
    return ent_sum / max(1, total)


# --- monitor_margin_median: median logit margin (top-1 minus top-2) over loader (label-free)
def monitor_margin_median(model, loader, device) -> float:
    model.eval()
    margins = []
    with torch.inference_mode():
        it = iter(loader)
        while True:
            try:
                xb, _ = next(it)
            except StopIteration:
                break
            except Exception as e:
                print(f"[WARN] monitor fetch failed: {e}")
                break
            xb = xb.to(device)
            logits = model(xb)
            top2 = torch.topk(logits, k=2, dim=1).values
            m = (top2[:, 0] - top2[:, 1]).detach().cpu().float()
            margins.append(m)
    if not margins:
        return float("nan")
    mm = torch.cat(margins, dim=0)
    return float(torch.median(mm).item())


def monitor_avg_conf(model, loader, device) -> float:
    model.eval()
    s = 0.0
    n = 0
    with torch.inference_mode():
        it = iter(loader)
        while True:
            try:
                xb, _ = next(it)
            except StopIteration:
                break
            except Exception as e:
                print(f"[WARN] monitor fetch failed: {e}")
                break
            xb = xb.to(device)
            p = F.softmax(model(xb), dim=1)
            m = p.max(dim=1)[0]
            s += float(m.sum().item())
            n += xb.shape[0]
    return s / max(1, n)


def monitor_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        it = iter(loader)
        while True:
            try:
                xb, yb = next(it)
            except StopIteration:
                break
            except Exception as e:
                print(f"[WARN] monitor fetch failed: {e}")
                break
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(1, total)


def ftle_entropy_grad(model, xb_small: torch.Tensor) -> float:
    """Median log-norm of entropy gradient (AMP-safe)."""
    model.eval()
    dtype = next(model.parameters()).dtype
    x = xb_small.detach().clone().to(dtype).requires_grad_(True)
    logits = model(x)
    p = F.softmax(logits, dim=1).clamp_min(1e-12)
    H = -(p * p.log()).sum(dim=1).mean()
    grad = torch.autograd.grad(H, x, retain_graph=False, create_graph=False)[0]
    v = grad.view(grad.shape[0], -1).norm(dim=1).add_(1e-9).log_()
    vs, _ = torch.sort(v)
    idx = (vs.numel() - 1) // 2
    return float(vs[idx].float().item())


def ftle_entropy_grad_lowent(model, xb_small: torch.Tensor, q: float = 0.30) -> float:
    """FTLE variant: backprop mean entropy on the low-entropy quantile only (deterministic percentile)."""
    model.eval()
    dtype = next(model.parameters()).dtype
    x = xb_small.detach().clone().to(dtype).requires_grad_(True)
    logits = model(x)
    p = F.softmax(logits, dim=1).clamp_min(1e-12)
    per = -(p * p.log()).sum(dim=1)  # per-sample entropy
    # ---- P1: manual percentile by sort for determinism ----
    vals = per.detach().float()
    k = max(0, min(vals.numel() - 1, int(math.floor(q * (vals.numel() - 1)))))
    thr = torch.sort(vals).values[k]
    mask_bool = per < thr
    if int(mask_bool.sum().item()) < 8:
        idx_lowent = torch.topk(vals, k=min(8, vals.shape[0]), largest=False).indices
        xsub = x[idx_lowent]
    else:
        xsub = x[mask_bool]
    logits_sub = model(xsub)
    psub = F.softmax(logits_sub, dim=1).clamp_min(1e-12)
    Hsub = -(psub * psub.log()).sum(dim=1).mean()
    grad = torch.autograd.grad(Hsub, xsub, retain_graph=False, create_graph=False)[0]
    v = grad.view(grad.shape[0], -1).norm(dim=1).add_(1e-9).log_()
    vs, _ = torch.sort(v)
    idx = (vs.numel() - 1) // 2
    return float(vs[idx].float().item())


#
# ---------------------------
# Coherence & slope helpers (observability only)
# ---------------------------
def _rolling_slope(x_arr: np.ndarray, w: int = 9) -> np.ndarray:
    """Centered linear-fit slope over window w (odd). NaN near edges."""
    x = np.asarray(x_arr, dtype=float)
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=float)
    if w % 2 == 0:
        w += 1
    h = w // 2
    out = np.full(n, np.nan, dtype=float)
    t_full = np.arange(-h, h + 1, dtype=float)
    denom_full = np.sum((t_full - t_full.mean()) ** 2) + 1e-12
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        tt = np.arange(lo - i, hi - i, dtype=float)
        xx = x[lo:hi]
        if np.isfinite(xx).sum() >= 3:
            tm = tt.mean()
            xm = np.nanmean(xx)
            num = np.nansum((tt - tm) * (xx - xm))
            denom = np.nansum((tt - tm) ** 2) if hi - lo < w else denom_full
            out[i] = num / (denom + 1e-12)
    return out


def _slope_regime(slopes: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Map slope to regimes: -1 (decreasing), 0 (flat), +1 (increasing)."""
    s = np.asarray(slopes, dtype=float)
    r = np.zeros_like(s, dtype=float)
    r[s > +tol] = +1.0
    r[s < -tol] = -1.0
    r[~np.isfinite(s)] = np.nan
    return r


def _regime_persist_flags(r: np.ndarray, min_len: int = 3) -> dict:
    """Return dict of boolean arrays for persistent regimes over >= min_len consecutive steps."""
    rr = np.asarray(r)
    n = len(rr)

    def _flags(val: float) -> np.ndarray:
        flags = np.zeros(n, dtype=bool)
        cnt = 0
        for i in range(n):
            if rr[i] == val:
                cnt += 1
                if cnt >= min_len:
                    flags[i] = True
            else:
                cnt = 0
        return flags

    return {"neg": _flags(-1.0), "pos": _flags(+1.0), "flat": _flags(0.0)}


def _rolling_corr(x_arr: np.ndarray, y_arr: np.ndarray, w: int = 21) -> np.ndarray:
    """Centered Pearson correlation over window w."""
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    n = min(len(x), len(y))
    if n == 0:
        return np.zeros(0, dtype=float)
    if w % 2 == 0:
        w += 1
    h = w // 2
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        xs = x[lo:hi]
        ys = y[lo:hi]
        mask = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[mask]
        ys = ys[mask]
        if xs.size >= 3:
            xm, ym = xs.mean(), ys.mean()
            num = np.sum((xs - xm) * (ys - ym))
            den = np.sqrt(np.sum((xs - xm) ** 2)) * np.sqrt(np.sum((ys - ym) ** 2)) + 1e-12
            out[i] = num / den
    return out


def _coh_persist_below(
corr_series: np.ndarray, thresh: float = 0.85, min_len: int = 5
) -> np.ndarray:
    """Boolean array: True where rolling corr < thresh for >= min_len consecutive epochs."""
    c = np.asarray(corr_series, dtype=float)
    n = len(c)
    flags = np.zeros(n, dtype=bool)
    cnt = 0
    for i in range(n):
        if np.isfinite(c[i]) and c[i] < thresh:
            cnt += 1
            if cnt >= min_len:
                flags[i] = True
        else:
            cnt = 0
    return flags


# ---------------------------
# PH & sequential helpers
# ---------------------------
def _ph_on_z(zs: List[float], lam: float, direction: str) -> Tuple[Optional[int], List[float]]:
    s = 0.0
    track = []
    for t, z in enumerate(zs):
        if direction == "up":
            s = max(0.0, s + z)
            track.append(s)
            if s > lam:
                return t, track
        else:
            s = min(0.0, s + z)
            track.append(s)
            if s < -lam:
                return t, track
    return None, track


def robust_z_series(xs: List[float], win: int, burn_in: int) -> List[float]:
    thr = max(burn_in, win, 2)
    zs = []
    for t, x in enumerate(xs):
        if t < thr:
            zs.append(0.0)
            continue
        a = max(0, t - win)
        b = t
        w = [v for v in xs[a:b] if np.isfinite(v)]
        if len(w) < 4:
            zs.append(0.0)
            continue
        med = float(np.median(w))
        mad = float(np.median(np.abs(np.array(w) - med))) + 1e-8
        z = (x - med) / (1.4826 * mad)
        zs.append(z)
    return zs


def ph_window_sparse(
xs: List[float],
win: int,
lam: float,
direction: str,
burn_in: int,
min_points: int,
two_sided: bool,
) -> Tuple[Optional[int], List[float], List[float]]:
    """
    Sparse CUSUM-on-robust-z over a series that may contain NaNs.

    We operate in the compacted index space (``comp``) consisting only of finite
    entries of ``xs``. The robust-z threshold ``thr = max(burn_in, win, 2)`` is
    applied in comp-space. Note: by construction, indices below ``thr`` in comp
    have well-defined windows; no assumption ties comp[:thr] to epoch space.

    We map detections and the running CUSUM track back to epoch indices via
    ``comp_idx_to_time(i) = idxs[i]``. We also enforce a minimum availability
    of finite points after burn-in in comp-space: if fewer than ``min_points``
    finite observations exist in ``comp[thr:]``, the detector does not fire.
    """
    thr = max(burn_in, win, 2)

    # indices of finite observations in original time space
    idxs = [i for i, x in enumerate(xs) if np.isfinite(x)]
    if not idxs:
        return None, [0.0] * len(xs), [0.0] * len(xs)

    # compacted finite-valued series
    comp = [xs[i] for i in idxs]

    # enforce availability after burn-in in comp-space
    comp_after_thr = [v for v in comp[thr:] if np.isfinite(v)]
    if len(comp_after_thr) < int(min_points):
        zs = robust_z_series(comp, win, burn_in)
        return None, zs, [0.0] * len(xs)

    # compute robust z in comp-space
    zs = robust_z_series(comp, win, burn_in)

    def _detect(zs_list, dir_label):
        t_comp, tr_comp = _ph_on_z(zs_list, lam, dir_label)
        if t_comp is None:
            return None, tr_comp
        if t_comp < thr:
            return None, tr_comp
        return t_comp, tr_comp

    if two_sided:
        t_up, tr_up = _detect(zs, "up")
        t_dn, tr_dn = _detect(zs, "down")
        if t_up is None and t_dn is None:
            t_comp = None
            tr_use = [0.0] * len(comp)
        else:
            if t_up is None:
                t_comp, tr_use = t_dn, tr_dn
            elif t_dn is None:
                t_comp, tr_use = t_up, tr_up
            else:
                t_comp = t_up if t_up <= t_dn else t_dn
                tr_use = tr_up if t_up <= t_dn else tr_dn
    else:
        t_comp, tr_use = _detect(zs, direction)

    track_full = [0.0] * len(xs)
    if t_comp is not None:
        # map the comp-space track starting at comp index thr back to time indices
        start = thr
        end = min(len(tr_use), len(comp))
        for k in range(start, end):
            ti = idxs[k]
            if 0 <= ti < len(track_full):
                track_full[ti] = tr_use[k]
        t_time = idxs[t_comp]
        return t_time, zs, track_full
    else:
        return None, zs, track_full


def _delta(xs: List[float]) -> List[float]:
    out = [np.nan] * len(xs)
    for t in range(1, len(xs)):
        a, b = xs[t - 1], xs[t]
        out[t] = (b - a) if (np.isfinite(a) and np.isfinite(b)) else np.nan
    return out


# --- flatten model parameters and compute L2 norm (for weight drift)
def _flatten_params_l2(model: nn.Module) -> Tuple[float, torch.Tensor]:
    vec = []
    with torch.no_grad():
        for p in model.parameters():
            vec.append(p.detach().float().view(-1).cpu())
    if not vec:
        return 0.0, torch.zeros(0, dtype=torch.float32)
    v = torch.cat(vec, dim=0)
    return float(torch.linalg.vector_norm(v).item()), v


# --- gradient noise scale (per-batch, microbatch split)
def gradient_noise_scale(
model: nn.Module, xb: torch.Tensor, yb: torch.Tensor, micro: int = 4
) -> Tuple[float, int]:
    """Compute a cheap GNS proxy on a single batch by comparing microbatch gradients.
    Runs in TRAIN mode (to match BN/dropout behavior), does not step the optimizer, and
    restores the original model.training flag afterward. Returns (gns, n_micro) with gns=nan on failure."""
    if os.environ.get("SCAR_SMOKE", "0") == "1" and CFG.get("skip_gns_in_smoke", False):
        return float("nan"), 0
    try:
        was_training = model.training
        model.train()  # ensure BN/dropout behavior like train
        B = xb.shape[0]
        micro = max(2, min(int(micro), B))
        sizes = [B // micro] * micro
        for i in range(B % micro):
            sizes[i] += 1
        splits_x = torch.split(xb.detach(), sizes)
        splits_y = torch.split(yb.detach(), sizes)

        grads = []
        for xs, ys in zip(splits_x, splits_y):
            # clear grads
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            logits = model(xs)
            loss = F.cross_entropy(logits, ys)
            loss.backward()
            g_chunks = []
            for p in model.parameters():
                if p.grad is not None:
                    g_chunks.append(p.grad.detach().float().view(-1).cpu())
            grads.append(torch.cat(g_chunks) if g_chunks else torch.zeros(1))

        # clear grads again
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        # restore original mode
        if not was_training:
            model.eval()

        if len(grads) < 2:
            return float("nan"), len(grads)
        # Bail out if any empty gradient vectors (shape mismatch risk)
        if any((g is None) or (getattr(g, "numel", lambda: 0)() == 0) for g in grads):
            return float("nan"), 0

        G = torch.stack(
        [g if g.numel() > 0 else torch.zeros_like(grads[0]) for g in grads], dim=0
        )
        gbar = G.mean(dim=0)
        num = torch.sum((G - gbar).pow(2))
        den = torch.sum(gbar.pow(2)) + 1e-12
        # Unbiased factor B/(B-1)
        gns = float((num / den).item() * (len(grads) / max(1, len(grads) - 1)))
        return gns, len(grads)
    except Exception:
        # best-effort fallback
        try:
            # ensure we don't leave grads hanging or the model in the wrong mode
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            model.eval()
        except Exception:
            pass
        return float("nan"), 0


def cusum_one_sided(
zs: List[float], lam: float, direction: str = "down"
) -> Tuple[Optional[int], List[float]]:
    s = 0.0
    track = []
    for t, z in enumerate(zs):
        if not np.isfinite(z):
            track.append(s)
            continue
        if direction == "down":
            s = min(0.0, s + z)
            track.append(s)
            if s < -lam:
                return t, track
        else:
            s = max(0.0, s + z)
            track.append(s)
            if s > lam:
                return t, track
    return None, track


def newma_warn_epoch(
xs: List[float], fast: float, slow: float, lam: float, burn_in: int
) -> Optional[int]:
    mu_f = 0.0
    mu_s = 0.0
    for t, x in enumerate(xs):
        if not np.isfinite(x):
            continue
        a = float(x)
        mu_f = (1 - fast) * mu_f + fast * a
        mu_s = (1 - slow) * mu_s + slow * a
        if t >= burn_in:
            s = mu_f - mu_s
            if abs(s) > lam:
                return t
    return None


# ---------------------------
# GT (unsupervised): robust hard + rank-only soft
# ---------------------------
def gt_collapse_time(run_df: pd.DataFrame, grad_cutoff: float) -> Tuple[Optional[int], str]:
    patience = int(CFG.get("gt_patience", 2))
    g = run_df.sort_values("epoch")
    ep = g["epoch"].to_numpy()
    nan_flag = g["nan_flag"].astype(bool).to_numpy()
    grad_rel = g["grad_norm_rel"].to_numpy()
    warm_idx = CFG["warmup"] + CFG["ph_burn"]

    # HARD: NaNs or gradient explosion — require 2 consecutive after warm
    consec = 0
    for t in range(len(ep)):
        if ep[t] < warm_idx:
            consec = 0
            continue
        hard_now = nan_flag[t] or (np.isfinite(grad_rel[t]) and grad_rel[t] >= grad_cutoff)
        if hard_now:
            consec += 1
            if consec >= 2:
                return int(ep[t - 1]), "hard"  # first epoch of confirmation window
        else:
            consec = 0

    # Terminal early-stop hard tag:
    # If the shard ended early (emax < epochs-1) and the final logged row has nan_flag==1,
    # treat it as a confirmed hard collapse at the last observed epoch.
    try:
        epochs_expected = int(CFG.get("epochs", None))
    except Exception:
        epochs_expected = None
    if epochs_expected is not None and len(ep) > 0:
        e_last = int(ep[-1])  # g is sorted by epoch; last observed epoch
        if (e_last < epochs_expected - 1) and bool(nan_flag[-1]):
            # metadata check removed; keep legacy nan_flag gate
            return e_last, "hard"

    # SOFT: rank-only (native eff_dim below threshold) with patience
    eff = g["eff_dim_gt"].to_numpy() if "eff_dim_gt" in g.columns else g["eff_dim"].to_numpy()
    consec = 0
    t_first = None
    for t in range(len(ep)):
        cond_rank = np.isfinite(eff[t]) and (eff[t] <= CFG["gt_rank_min"])
        if cond_rank:
            consec += 1
            if consec == 1:
                t_first = int(ep[t])
            if consec >= patience:
                return t_first, "soft"
        else:
            consec = 0
            t_first = None
    return None, "none"


# ---------------------------
# Learned detector utilities
# ---------------------------
def _metrics_matrix_with_missing(
g: pd.DataFrame, metric_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    X = g[metric_cols].to_numpy(dtype=np.float32)
    M = np.isnan(X).astype(np.float32)
    return X, M


def _fit_global_robust_norm_precollapse(
df_cal: pd.DataFrame, cols: List[str]
) -> Dict[str, Tuple[float, float]]:
    """Robust med/MAD estimated only on pre-collapse epochs (or all epochs for none-runs)."""
    stats: Dict[str, Tuple[float, float]] = {}
    if df_cal.empty:
        return {m: (0.0, 1.0) for m in cols}

    for m in cols:
        sub = df_cal.copy()
        if {"collapse_tag_gt", "t_collapse_gt"}.issubset(sub.columns):
            pre_parts: List[pd.DataFrame] = []
            for (sd, fc), g in sub.groupby(["seed", "factor"]):
                tag = str(g["collapse_tag_gt"].iloc[0])
                tc = g["t_collapse_gt"].iloc[0]
                if (tag == "soft") and pd.notna(tc):
                    pre_parts.append(g[g["epoch"] < int(tc)])
                elif tag == "hard":
                    # exclude hard-collapse runs from normalization
                    continue
                else:
                    pre_parts.append(g)
            sub = pd.concat(pre_parts, ignore_index=True) if pre_parts else sub.iloc[0:0]

        arr = (
        sub[m].to_numpy(dtype=np.float32)
        if m in sub.columns
        else np.array([], dtype=np.float32)
        )
        fin = np.isfinite(arr)
        if fin.sum() >= 16:
            med = float(np.median(arr[fin]))
            mad = float(np.median(np.abs(arr[fin] - med))) + 1e-8
        else:
            med, mad = 0.0, 1.0
        stats[m] = (med, 1.4826 * mad)
    return stats


def _fit_global_robust_norm(
df_cal: pd.DataFrame, cols: List[str]
) -> Dict[str, Tuple[float, float]]:
    stats = {}
    X = df_cal[cols].to_numpy(dtype=np.float32)
    for j, m in enumerate(cols):
        col = X[:, j]
        finite = np.isfinite(col)
        if finite.sum() >= 16:
            med = float(np.median(col[finite]))
            mad = float(np.median(np.abs(col[finite] - med))) + 1e-8
        else:
            med, mad = 0.0, 1.0
        stats[m] = (med, 1.4826 * mad)
    return stats


def _apply_global_norm_impute(
X: np.ndarray, stats: Dict[str, Tuple[float, float]], cols: List[str]
) -> np.ndarray:
    Xn = X.copy()
    for j, m in enumerate(cols):
        med, scale = stats[m]
        if scale <= 0:
            scale = 1.0
        col = Xn[:, j]
        mask = ~np.isfinite(col)
        col[mask] = med
        Xn[:, j] = (col - med) / scale
    Xn[np.isinf(Xn)] = 0.0
    return Xn


def _train_logistic_ridge_balanced(
X: np.ndarray, y: np.ndarray, groups: np.ndarray, steps: int, lr: float, l2: float
) -> Tuple[np.ndarray, float]:
    # Guard: degenerate labels (all negatives or all positives) -> deterministic "disabled" classifier
    if (y.sum() == 0) or (y.sum() == len(y)):
        w = np.zeros(X.shape[1], dtype=np.float32)
        b = -10.0 if y.sum() == 0 else 10.0
        return w, float(b)

    pos = max(1e-6, y.mean())
    w_pos = 0.5 / pos
    w_neg = 0.5 / (1 - pos)
    w = np.zeros(X.shape[1], dtype=np.float32)
    b = 0.0
    for _ in range(steps):
        # Clip logits for numerical stability
        z = np.clip(X @ w + b, -20.0, 20.0)
        p = _sigmoid_stable(z)
        w_i = np.where(y > 0.5, w_pos, w_neg)
        grad_w = (X.T @ ((p - y) * w_i)) / len(y) + l2 * w
        grad_b = float(np.mean((p - y) * w_i))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def _cv_grouped_fit(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fp_mask: np.ndarray,
    steps_grid,
    l2_grid,
    lr: float,
    folds: int,
    fp_cap: float,
    epochs: np.ndarray,
    tc_soft: np.ndarray,
    warm_idx: int,
) -> Optional[Tuple[int, float, float]]:
    """Pick (steps, l2, threshold) by maximizing expected OOF lead time under an FP cap (tie-break: sensitivity)."""

    # --- safety short-circuit: no positives at all ---
    if y.sum() == 0:
        return None

    uniq = np.unique(groups)
    if len(uniq) < max(2, folds):
        folds = max(2, min(len(uniq), folds))
    folds_idx = _partition_groups_with_positives(uniq.copy(), y, groups, folds)

    best_lead = -np.inf
    best_sens = -np.inf
    best_info: Optional[Tuple[int, float, float]] = None

    for l2 in l2_grid:
        for steps in steps_grid:
            # Out-of-fold predictions
            oof_p = np.full_like(y, np.nan, dtype=np.float32)
            for va_seeds in folds_idx:
                tr = ~np.isin(groups, va_seeds)
                va = ~tr
                if va.sum() == 0 or tr.sum() == 0:
                    continue
                w, b = _train_logistic_ridge_balanced(
                    X[tr], y[tr], groups[tr], steps=steps, lr=lr, l2=l2
                )
                z_va = (X[va] @ w + b)
                oof_p[va] = _sigmoid_stable(z_va)

            mask = np.isfinite(oof_p)
            if mask.sum() == 0:
                continue
            elig = (fp_mask == 1) & (y == 0) & mask

            # candidate thresholds from quantiles over relevant scores
            rel = mask & (elig | (y == 1))
            scores = oof_p[rel]
            try:
                ts = (
                    np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 61), method="linear"))
                    if scores.size > 0
                    else np.linspace(0.05, 0.95, 91)
                )
            except TypeError:
                # numpy & python version compatibility
                ts = (
                    np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 61), interpolation="linear"))
                    if scores.size > 0
                    else np.linspace(0.05, 0.95, 91)
                )

            for t in ts:
                pred = oof_p >= t

                # FP rate on eligible negatives (>=warm)
                fp_rate = pred[elig].mean() if elig.any() else 0.0
                if not np.isfinite(fp_rate) or fp_rate > fp_cap:
                    continue

                # lead time and sensitivity at the run level (soft collapses only)
                leads: List[float] = []
                hits = 0
                total_pos = 0
                for seed in uniq.astype(np.int64):
                    gmask = (groups == seed) & mask
                    if not gmask.any():
                        continue
                    tc = np.nanmean(tc_soft[groups == seed])  # soft only; NaN otherwise
                    if np.isfinite(tc):
                        total_pos += 1
                        idx = np.where(gmask & (epochs >= warm_idx) & pred)[0]
                        if idx.size > 0:
                            t_warn = int(epochs[idx.min()])
                            if t_warn < tc:
                                leads.append(tc - t_warn)
                                hits += 1

                sens = (hits / max(1, total_pos)) if total_pos > 0 else 0.0
                mean_lead = float(np.mean(leads)) if len(leads) > 0 else -np.inf

                if (mean_lead > best_lead + 1e-9) or (
                    abs(mean_lead - best_lead) <= 1e-9 and sens > best_sens
                ):
                    best_lead, best_sens, best_info = mean_lead, sens, (steps, l2, float(t))

    return best_info


# --- τ→τ′ mapping helpers (deterministic, uses deployed gate logic) ---
def _partition_groups_with_positives(
uniq_groups: np.ndarray,
y: np.ndarray,
groups: np.ndarray,
folds: int = 5,
) -> List[np.ndarray]:
    """
    Deterministic stratified partition of group ids into folds.
    Strategy: put all positive groups into k buckets round‑robin, then
    fill remaining buckets with negatives round‑robin. Falls back to
    even split if there are no positive groups.
    """
    uniq = np.array(sorted(uniq_groups.astype(np.int64)))
    k = int(max(2, min(len(uniq), int(folds))))
    # identify groups with at least one positive label
    pos_groups = []
    neg_groups = []
    for g_id in uniq:
        mask = (groups.astype(np.int64) == g_id)
        if mask.any() and (y[mask].sum() > 0):
            pos_groups.append(int(g_id))
        else:
            neg_groups.append(int(g_id))
    # initialize empty buckets
    buckets = [list() for _ in range(k)]
    # round‑robin assign positives first
    for i, g_id in enumerate(pos_groups):
        buckets[i % k].append(g_id)
    # then distribute negatives
    for i, g_id in enumerate(neg_groups):
        buckets[i % k].append(g_id)
    # convert to numpy arrays
    return [np.array(sorted(b), dtype=np.int64) for b in buckets]


def _oof_probs_for_params(
Xn: np.ndarray, y: np.ndarray, groups: np.ndarray, steps: int, l2: float, lr: float, folds: int
) -> np.ndarray:
    """
    Deterministic grouped OOF probabilities for a fixed (steps, l2) setting.
    - Groups are seeds; we partition them deterministically so every train split
    contains positives (when available).
    - Returns a float32 vector of OOF probabilities aligned with `y`.
    """
    # Ensure a sensible number of folds
    uniq = np.unique(groups.astype(np.int64))
    k = int(max(2, min(len(uniq), int(folds))))
    folds_idx = _partition_groups_with_positives(uniq.copy(), y, groups, folds=k)

    p = np.full(y.shape, np.nan, dtype=np.float32)
    for va_seeds in folds_idx:
        # boolean masks for this fold
        va_mask = np.isin(groups, va_seeds)
        tr_mask = ~va_mask
        if tr_mask.sum() == 0 or va_mask.sum() == 0:
            continue

        # train on train groups only
        w, b = _train_logistic_ridge_balanced(
        Xn[tr_mask], y[tr_mask], groups[tr_mask], steps=steps, lr=lr, l2=l2
        )
        # score validation groups
        z = Xn[va_mask] @ w + b
        p[va_mask] = _sigmoid_stable(z)
    return p


def _first_run_end(hit_idx: np.ndarray, L: int) -> int:
    """Return index into hit_idx for the end of the first run of length >= L, or -1."""
    if len(hit_idx) < L:
        return -1
    r = 1
    for j in range(1, len(hit_idx)):
        if hit_idx[j] == hit_idx[j - 1] + 1:
            r += 1
            if r >= L:
                return j
        else:
            r = 1
    return -1


def _fam_alarm_at(
i: int,
Z: np.ndarray,
det_features: List[str],
dir_map: Dict[str, str],
cols: List[str],
z_thr: float,
K: int,
) -> bool:
    if not cols:
        return False
    lo, hi = max(0, i - K), min(Z.shape[0] - 1, i + K)
    for m in cols:
        if m not in det_features:
            continue
        j = det_features.index(m)
        zwin = Z[lo : hi + 1, j]
        d = dir_map.get(m)
        if d is None:
            continue  # skip metrics without calibrated direction
        if d == "up" and np.nanmax(zwin) >= z_thr:
            return True
        if d == "down" and np.nanmin(zwin) <= -z_thr:
            return True
    return False


def _gated_runlevel_fp_for_threshold(
meta_rows: pd.DataFrame,
X_raw: np.ndarray,
p: np.ndarray,
det_features: List[str],
stats: Dict[str, Tuple[float, float]],
dir_map: Dict[str, str],
rp_flags: Dict[Tuple[int, str], int],
warm: int,
z_thr: float,
K: int,
warn_consec: int,
) -> float:
    """Run-level FP on factor=='none' after warm, under deployed gate at fixed τ.
    Threshold is applied by masking p to -inf *before* this is called.
    """
    Xz = _apply_global_norm_impute(X_raw, stats, det_features)
    geom_cols = [c for c in ("cos_disp", "var_out_k") if c in det_features]
    dyn_cols = [c for c in ("ftle", "ftle_lowent") if c in det_features]

    run_warn: Dict[Tuple[int, str], int] = {}
    for (sd, fc), indices in meta_rows.groupby(["seed", "factor"]).groups.items():
        idx = np.asarray(sorted(indices), dtype=np.int64)
        sub = meta_rows.iloc[idx].sort_values("epoch")
        order = np.argsort(sub["epoch"].to_numpy())
        idx = idx[order]
        epochs = meta_rows.iloc[idx]["epoch"].to_numpy().astype(np.int64)
        Zg = Xz[idx]
        pg = p[idx]

        elig = (epochs >= warm) & np.isfinite(pg) & (pg > -np.inf)
        hit_idx = np.where(elig)[0]

        j_end = _first_run_end(hit_idx, warn_consec)
        t_warn = None
        if j_end >= 0:
            i = hit_idx[j_end]
            geom_ok = _fam_alarm_at(i, Zg, det_features, dir_map, geom_cols, z_thr, K)
            dyn_ok = _fam_alarm_at(i, Zg, det_features, dir_map, dyn_cols, z_thr, K)
            rp_under = bool(rp_flags.get((int(sd), str(fc)), 0))
            gate_ok = dyn_ok if rp_under else (geom_ok or dyn_ok)
            if gate_ok:
                t_warn = int(epochs[i])
        run_warn[(int(sd), str(fc))] = t_warn

    none_runs = [
    (int(sd), str(fc))
    for (sd, fc), _ in meta_rows.groupby(["seed", "factor"])
    if str(fc) == "none"
    ]
    flags = []
    for k in none_runs:
        tw = run_warn.get(k, None)
        flags.append(bool(tw is not None and tw >= warm))
    return float(np.mean(flags)) if flags else float("nan")


def map_threshold_to_gated_fp(
meta_rows: pd.DataFrame,
X_raw: np.ndarray,
p: np.ndarray,
det_features: List[str],
stats: Dict[str, Tuple[float, float]],
dir_map: Dict[str, str],
rp_flags: Dict[Tuple[int, str], int],
warm: int,
z_thr: float,
K: int,
warn_consec: int,
fp_cap: float,
) -> Tuple[float, float]:
    """Deterministically map τ→τ′ so gated run-level FP on factor=='none' ≤ fp_cap.
    Returns (tau_prime, measured_fp).
    """
    ep = meta_rows["epoch"].to_numpy().astype(np.int64)
    fac = meta_rows["factor"].astype(str).to_numpy()
    elig = (ep >= warm) & np.isfinite(p) & (fac == "none")
    scores = np.asarray(p[elig], dtype=np.float32)
    if scores.size == 0:
        print("[WARN] τ′ calibration skipped: no factor=='none' rows after warm; failing closed with τ′=1.0")
        return 1.0, float("nan")

    qs = np.linspace(0.05, 0.995, 191, dtype=np.float64)
    try:
        ts = np.unique(np.quantile(scores, qs, method="linear"))
    except TypeError:
        ts = np.unique(np.quantile(scores, qs, interpolation="linear"))

    best_tau, best_fp = ts[-1], float("inf")
    for t in ts:
        p_masked = np.where(p >= t, p, -np.inf).astype(np.float32)
        fp = _gated_runlevel_fp_for_threshold(
        meta_rows,
        X_raw,
        p_masked,
        det_features,
        stats,
        dir_map,
        rp_flags,
        warm,
        z_thr,
        K,
        warn_consec,
        )
        if np.isnan(fp):
            continue
        if fp <= fp_cap:
            best_tau, best_fp = float(t), float(fp)
            break
        best_fp = float(fp)
    return best_tau, best_fp


# ---------------------------
# Factor assignment (even across ALL seeds)
# ---------------------------
def assign_factors_evenly(all_seeds: List[int]) -> Dict[int, Dict]:
    facs = CFG["factors"]
    mapping = {}
    for i, s in enumerate(sorted(all_seeds)):
        mapping[int(s)] = facs[i % len(facs)]
    return mapping


_DET_LOGGED = False


# ---------------------------
# One run
# ---------------------------
def _pick_ftle_batch(loader, seed: int, epoch: int):
    try:
        L = len(loader)
    except Exception:
        L = 1
    if L <= 0:
        L = 1
    offset = int((seed * 1009 + epoch * 9173) % L)
    it = iter(loader)
    for _ in range(offset):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)
    xb, _ = next(it)
    return xb


def run_one(seed: int, tag: str, monitor_ds, factor: Dict) -> pd.DataFrame:
    device = CFG["device"]
    seed_all(seed, CFG["deterministic"])
    # One-time determinism status log per process
    global _DET_LOGGED
    if not _DET_LOGGED:
        try:
            print(
            "[det] status | deterministic=%s | CUBLAS_WORKSPACE_CONFIG=%s | TF32(matmul=%s, cudnn=%s)"
            % (
            str(bool(CFG.get("deterministic", True))),
            os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            str(torch.backends.cuda.matmul.allow_tf32),
            str(torch.backends.cudnn.allow_tf32),
            )
            )
        except Exception:
            pass
        # Emit a determinism probe file once per process
        try:
            if mp.current_process().name == "MainProcess":
                update_json(OUTDIR / "env_probe.json", {
                "cuda_hash": _cuda_hash(),
                "pip_freeze_md5": _pip_freeze_md5(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "cuda": torch.version.cuda,
                "deterministic_flag": bool(CFG.get("deterministic", True)),
                })
        except Exception:
            pass
        _DET_LOGGED = True
    amp_enabled = CFG["amp"] and (not CFG["deterministic"]) and device.type == "cuda"

    tr_aug, tr_take, mon_val, norm_ref, splits_path = load_splits(seed)
    run_id = f"s{seed}-{factor['name']}"
    tr_ds = FactorisedTrainDataset(tr_aug, tr_take, factor=factor, seed=seed)
    sampler = (
    DropoutAwareSampler(tr_ds, CFG["batch"], seed=seed)
    if factor["name"] == "class_dropout_window"
    else None
    )

    # monitor loaders
    ent_every = int(CFG.get("external_monitor", {}).get("ent_every", 2))
    pool_loader = None
    ent_loader = None
    monitor_source = str(CFG.get("monitor_source", "clean_val"))

    if monitor_source == "external":
        try:
            if monitor_ds is None:
                raise RuntimeError("external monitor dataset unavailable")

            rng = np.random.default_rng(777000 + seed)
            N = len(monitor_ds)
            pool_size = min(int(CFG["external_monitor"]["pool_size"]), N)
            if pool_size <= 0:
                raise RuntimeError("external monitor pool_size is 0")

            pool_idxs = rng.choice(np.arange(N, dtype=np.int64), size=pool_size, replace=False)
            pool_loader = subset_loader_from_indices(
                monitor_ds, pool_idxs, CFG["batch"], shuffle=True, seed=seed, device=device
            )

            # disjoint entropy set (fallback to pool if diff is empty)
            ent_pool = np.setdiff1d(np.arange(N, dtype=np.int64), pool_idxs, assume_unique=True)
            if ent_pool.size == 0:
                ent_pool = pool_idxs

            ent_n = min(int(CFG["external_monitor"]["ent_subset"]), ent_pool.size)
            if ent_n <= 0:
                raise RuntimeError("external monitor ent_subset is 0")

            ent_idxs = rng.choice(ent_pool, size=ent_n, replace=False)
            ent_loader = subset_loader_from_indices(
                monitor_ds, ent_idxs, CFG["batch"], shuffle=False, seed=seed + 1, device=device
            )

            update_json(
                splits_path,
                {
                    "STL10_POOL": [int(i) for i in pool_idxs.tolist()],
                    "STL10_ENT":  [int(i) for i in ent_idxs.tolist()],
                },
            )
        except Exception as e:
            print(f"[WARN] external monitor failed ({e}); falling back to clean_val for this run only")
            monitor_source = "clean_val"
            pool_loader = None
            ent_loader = None

    # resolve non-external sources (or fallback) unconditionally
    if monitor_source == "noisy_train":
        pool_loader = make_loader(
            tr_ds, CFG["batch"], True, CFG["num_workers"], new_gen(seed, 2), device, persistent=True
        )
        ent_loader = pool_loader
    elif (monitor_source == "clean_val") or (pool_loader is None) or (ent_loader is None):
        pool_loader = make_loader(
            mon_val, CFG["batch"], True, CFG["num_workers"], new_gen(seed, 2), device, persistent=True
        )
        ent_loader = pool_loader

    try:
        mon_sig = {
            "source": monitor_source,
            "pool_len": len(pool_loader.dataset) if hasattr(pool_loader, "dataset") else -1,
            "ent_len": len(ent_loader.dataset) if hasattr(ent_loader, "dataset") else -1,
            "resize_to": CFG["external_monitor"]["resize_to"] if monitor_source == "external" else None,
        }
        update_json(splits_path, {"MONITOR_SIGNATURE": mon_sig})
    except Exception:
        pass

    # norm_ref loader for frozen μ,σ (model-agnostic; reused => allow persistence)
    norm_ref_loader = make_loader(
        norm_ref,
        CFG["batch"],
        True,
        CFG["num_workers"],
        new_gen(seed, 999),
        device,
        persistent=True,
    )

    def probe_aug_loader_factory():
        base_loader = pool_loader
        try:
            class _ProbeDS(torch.utils.data.Dataset):
                def __init__(self, base): self.base = base
                def __len__(self): return len(self.base)
                def __getitem__(self, i):
                    x, y = self.base[i]
                    try: dev = x.device
                    except Exception: dev = "cpu"
                    g = torch.Generator(device=dev).manual_seed(123 + seed * 997 + int(i))
                    if isinstance(x, torch.Tensor):
                        noise = torch.randn(x.shape, dtype=x.dtype, device=dev, generator=g) * 0.02
                        x2 = (x + noise).clamp(-3, 3)
                        if int(_u01_from_hash("flip_probe", seed, int(i)) * 2) == 1 and x2.ndim == 3 and x2.shape[-1] >= 2:
                            x2 = torch.flip(x2, dims=[2])
                    else:
                        x2 = x
                    return x2, y
            return DataLoader(
                _ProbeDS(base_loader.dataset),
                batch_size=base_loader.batch_size,
                shuffle=False,
                num_workers=getattr(base_loader, "num_workers", 0),
                pin_memory=getattr(base_loader, "pin_memory", False),
                drop_last=False,
                worker_init_fn=seed_worker if getattr(base_loader, "num_workers", 0) > 0 else None,
            )
        except Exception:
            return base_loader

    model = make_model().to(device)
    opt = make_opt(model)
    scaler = amp.GradScaler(enabled=amp_enabled)

    # penultimate dim check (model-agnostic; eager assert)
    with torch.no_grad():
        try:
            xb_chk, _ = next(iter(pool_loader))
        except Exception as e:
            print(f"[WARN] pool fetch failed for dim check: {e}; falling back to norm_ref_loader")
            try:
                xb_chk, _ = next(iter(norm_ref_loader))
            except Exception as e2:
                print(
                f"[WARN] norm_ref fetch failed for dim check: {e2}; defaulting penult_dim=512"
                )
                d_pen = 512
            else:
                d_pen = int(penult(model, xb_chk.to(device)).shape[1])
        else:
            d_pen = int(penult(model, xb_chk.to(device)).shape[1])
        print(f"[dim] seed={seed} penult_dim={d_pen}")

    # --------- FIXED REFERENCE (μ,σ) IN NATIVE PENULTIMATE SPACE via norm_ref ----------
    with torch.no_grad():
        model.eval()
        feats = []
        cnt = 0
        it = iter(norm_ref_loader)
        for _ in range(max(1, CFG["metric_batches"])):
            try:
                xb, _ = next(it)
            except StopIteration:
                it = iter(norm_ref_loader)
                try:
                    xb, _ = next(it)
                except Exception as e:
                    print(f"[WARN] norm_ref fetch failed: {e}")
                    break
            except Exception as e:
                print(f"[WARN] norm_ref fetch failed: {e}")
                break
            xb = xb.to(device)
            h = penult(model, xb).detach().cpu().float()
            feats.append(h)
            cnt += h.shape[0]
            if cnt >= CFG["metric_total_cap"]:
                break

        try:
            H = (
            torch.cat(feats, 0)
            if feats
            else penult(model, next(iter(norm_ref_loader))[0].to(device)).detach().cpu().float()
            )
        except Exception as e:
            print(f"[WARN] norm_ref fallback failed: {e}; using empty feature frame")
            H = torch.zeros((0, d_pen), dtype=torch.float32)

        # ✅ Guard empty H to avoid NaN means and ensure correct feature width
        if H.numel() == 0:
            mu = torch.zeros((1, d_pen), dtype=torch.float32)
            sig_ref = torch.ones((1, d_pen), dtype=torch.float32)
        else:
            mu = H.mean(dim=0, keepdim=True)
            sig_ref = _std0(H, keepdim=True) + 1e-8

        ref_mu_sig = (mu, sig_ref)
        assert ref_mu_sig[0].shape[1] == H.shape[1], "ref stats dimension mismatch"

        # --- Insert fallback WindowDecl if missing ---
        if WINDOW_DECL is None:
            try:
                snap = collect_feature_snapshot(model, pool_loader, device, ["var_out_k", "eff_dim"], ref_mu_sig=None)
                eff_est = float(snap.get("eff_dim", np.array([np.nan], dtype=float))[0])
            except Exception:
                eff_est = float("nan")
            hi_eff = (2.0 * eff_est) if np.isfinite(eff_est) else 2.0
            cal_ranges = {"var_out_k": (0.0, 1.0), "eff_dim": (0.0, max(1.0, float(hi_eff)))}
            window_decl = WindowDecl(
                epsilon=float(CFG.get("gate_epsilon", 0.08)),
                metrics=["var_out_k", "eff_dim"],
                weights={"var_out_k": 0.5, "eff_dim": 0.5},
                bins=int(CFG.get("gate_bins", 16)),
                interventions=(lambda x: x,),
                cal_ranges=cal_ranges,
            )
            install_window_decl(window_decl)
            try:
                write_window_audit(OUTDIR, window_decl, note="Fallback per-run WindowDecl")
            except Exception:
                pass
    # ----------------------------------------------------------------------

    logs = []
    loss_hist = []
    run_key = seed
    last_Z_geom = None
    last_Z_geom_native = None
    ewma_loss = None
    
    # --- gate early-exit state (k-consecutive gate_warn after warm) ---
    gate_hits_consec = 0
    gate_halt_epoch = None
    warn_consec_eff = int(os.environ.get("SCAR_WARN_CONSEC", str(CFG.get("warn_consec", 3))))
    warm_epoch = int(CFG.get("warmup", 0) + CFG.get("ph_burn", 0))
    gate_early_exit = bool(CFG.get("gate_early_exit", True))
    # --- coherence/slope buffers ---
    _eff_dim_series = []
    _var_out_k_series = []
    _r2_series = []
    _margin_series = []
    _sw2_series = []  # capture heavy-only SW2 (nan on light epochs)
    _prev_param_vec = None
    nan_seen = False

    # factor-local overrides
    base_lr_scale = float(factor.get("base_lr_scale", 1.0))
    override_clip = factor.get("override_grad_clip", None)

    for epoch in range(0, CFG["epochs"]):
        did_retry_fetch = False
        did_retry_step = False
        # per-epoch reshuffle (deterministic under seeded generator)
        if sampler is None:
            # ---- P0: disable persistent workers for per-epoch recreated loader ----
            tr_loader = make_loader(
            tr_ds,
            CFG["batch"],
            shuffle=True,
            workers=CFG["num_workers"],
            gen=new_gen(seed, 1 + epoch),
            device=device,
            sampler=None,
            persistent=False,
            )
        else:
            tr_loader = make_loader(
            tr_ds,
            CFG["batch"],
            shuffle=False,
            workers=CFG["num_workers"],
            gen=new_gen(seed, 1),
            device=device,
            sampler=sampler,
            persistent=True,
            )
            sampler.set_epoch(epoch)
        tr_ds.set_epoch(epoch)

        # LR & optimizer pathology tweaks
        lr = (
        lr_at(epoch, CFG["epochs"], CFG["base_lr"], CFG["warmup"], CFG["cosine"])
        * base_lr_scale
        )
        mom = CFG["momentum"]
        if factor["name"] == "lr_spike" and epoch == int(factor.get("epoch", 5)):
            lr *= float(factor.get("factor", 6.0))
        if factor["name"] == "mom_pulse":
            start = int(factor.get("start", 12))
            dur = int(factor.get("duration", 3))
            if start <= epoch < start + dur:
                mom = float(factor.get("momentum", 0.0))
        for pg in opt.param_groups:
            pg["lr"] = lr
            if "momentum" in pg:
                pg["momentum"] = mom

        # effective grad clip
        clip_eff = (
        float(override_clip)
        if (override_clip is not None)
        else float(CFG["grad_clip_norm"] or 0.0)
        )
        if clip_eff <= 0:
            clip_eff = None

        # lightweight feature stats for this epoch (on monitor pool)
        with torch.no_grad():
            Z_geom, Z_geom_native = _features_for_loader(
            model,
            pool_loader,
            device,
            n_batches=max(1, CFG["metric_batches"]),
            cap=CFG["metric_total_cap"],
            ref_mu_sig=ref_mu_sig,
            run_key=run_key,
            epoch=epoch,
            )
        n_feats = int(Z_geom.shape[0])
        vout_k, effd, k_used, tail_mass, v_ok, neg_eigs = variance_outside_k(Z_geom)
        vout_k_nat, effd_nat, _, _, v_ok_nat, neg_eigs_nat = variance_outside_k(Z_geom_native)
        r2 = spectral_r2(Z_geom)
        cosd = cosine_dispersion(Z_geom, seed=seed, epoch=epoch)

        # --- coherence & slope channels (observability only) ---
        _eff_dim_series.append(effd)
        _var_out_k_series.append(vout_k)
        _r2_series.append(r2)

        s_w = int(CFG.get("slope_w", 9))
        tol = float(CFG.get("slope_tol", 1e-3))
        p_min = int(CFG.get("slope_persist_min", 3))

        s_eff = _rolling_slope(np.array(_eff_dim_series, dtype=float), w=s_w)
        s_var = _rolling_slope(np.array(_var_out_k_series, dtype=float), w=s_w)
        s_r2 = _rolling_slope(np.array(_r2_series, dtype=float), w=s_w)
        slope_r2 = float(s_r2[-1]) if s_r2.size and np.isfinite(s_r2[-1]) else float("nan")
        r_eff = _slope_regime(s_eff, tol=tol)
        r_var = _slope_regime(s_var, tol=tol)
        rf_eff = _regime_persist_flags(r_eff, min_len=p_min)
        rf_var = _regime_persist_flags(r_var, min_len=p_min)

        c_w = int(CFG.get("corr_w", 21))
        c_th = float(CFG.get("coh_thresh", 0.85))
        c_min = int(CFG.get("coh_minlen", 5))
        corr_ev = _rolling_corr(
        np.array(_eff_dim_series, dtype=float), np.array(_var_out_k_series, dtype=float), w=c_w
        )
        coh_flags = _coh_persist_below(corr_ev, thresh=c_th, min_len=c_min)

        slope_eff_dim = float(s_eff[-1]) if s_eff.size and np.isfinite(s_eff[-1]) else float("nan")
        slope_var_out_k = (
        float(s_var[-1]) if s_var.size and np.isfinite(s_var[-1]) else float("nan")
        )
        reg_eff_dim = (
        float(r_eff[-1]) if r_eff.size and np.isfinite(r_eff[-1]) else float("nan")
        )  # -1/0/+1
        reg_var_out_k = float(r_var[-1]) if r_var.size and np.isfinite(r_var[-1]) else float("nan")
        reg_eff_neg_persist = bool(rf_eff["neg"][-1]) if rf_eff["neg"].size else False
        reg_var_pos_persist = bool(rf_var["pos"][-1]) if rf_var["pos"].size else False
        corr_effdim_varoutk = (
        float(corr_ev[-1]) if corr_ev.size and np.isfinite(corr_ev[-1]) else float("nan")
        )
        coh_break_persist = bool(coh_flags[-1]) if coh_flags.size else False

        # heavy metrics
        pers_h0, topo_done, topo_ms, topo_n_used = float("nan"), 0, 0.0, 0
        sw, sw_ms, sw_proj_done, sw_valid = float("nan"), 0.0, 0, 0
        sw_nat, sw_nat_ms, sw_nat_proj_done, sw_nat_valid = float("nan"), 0.0, 0, 0
        if (epoch % CFG["heavy_every"]) == 0:
            pers_h0, topo_done, topo_ms, topo_n_used = topo_h0_jl_agg(
            Z_geom,
            q=CFG["rp_dim_topo"],
            repeats=CFG["rp_repeats"],
            run_key=run_key,
            epoch=epoch,
            agg=CFG.get("rp_agg", "median"),
            sample_n=CFG.get("topo_sample_n", 192),
            )
            if last_Z_geom is not None:
                sw, sw_ms, sw_proj_done, ok = sliced_w2_gpu_budget(
                last_Z_geom, Z_geom, CFG["sw2_n_proj"], seed, device, CFG["sw2_budget_ms"]
                )
                sw_valid = int(ok)
            if last_Z_geom_native is not None:
                sw_nat, sw_nat_ms, sw_nat_proj_done, okn = sliced_w2_gpu_budget(
                last_Z_geom_native,
                Z_geom_native,
                CFG["sw2_n_proj"],
                seed + 7,
                device,
                CFG["sw2_budget_ms"],
                )
                sw_nat_valid = int(okn)
            _sw2_series.append(sw if np.isfinite(sw) else np.nan)
        else:
            _sw2_series.append(np.nan)

        # Compute 3-point median over most recent finite heavy SW2 values
        def _last_k_finite(arr, k):
            out = []
            for x in reversed(arr):
                if np.isfinite(x):
                    out.append(x)
                if len(out) >= k:
                    break
            return out

        sw2_recent = _last_k_finite(_sw2_series, 3)
        sw2_med3 = float(np.median(sw2_recent)) if sw2_recent else float("nan")
        s_sw2 = _rolling_slope(
        np.array([v for v in _sw2_series], dtype=float), w=max(3, CFG["heavy_every"])
        )
        sw2_slope = float(s_sw2[-1]) if s_sw2.size and np.isfinite(s_sw2[-1]) else float("nan")

        # Update buffers after metric computation
        last_Z_geom = Z_geom.clone()
        last_Z_geom_native = Z_geom_native.clone()

        # entropy & confidence monitors
        mon_ent = float("nan")
        mon_conf = float("nan")
        mon_acc = float("nan")
        margin_med = float("nan")
        slope_margin = float("nan")
        if (epoch % ent_every) == 0:
            mon_ent = monitor_entropy(model, ent_loader, device)
            mon_conf = monitor_avg_conf(model, ent_loader, device)
            margin_med = monitor_margin_median(model, ent_loader, device)
            _margin_series.append(margin_med)
            s_mar = _rolling_slope(np.array(_margin_series, dtype=float), w=s_w)
            slope_margin = (
            float(s_mar[-1]) if s_mar.size and np.isfinite(s_mar[-1]) else float("nan")
            )
            if CFG["monitor_labels"]:
                try:
                    mon_acc = monitor_accuracy(model, ent_loader, device)
                except Exception:
                    mon_acc = float("nan")

        # FTLE channels
        ftle_val = float("nan")
        ftle_lowent_val = float("nan")
        ftle_valid = 0
        try:
            xb_small = _pick_ftle_batch(pool_loader, seed=seed, epoch=epoch).to(device)
            # Compute FTLE in float32 for stability (disable autocast)
            with amp.autocast(device_type=device.type, enabled=False):
                ftle_val = ftle_entropy_grad(model, xb_small[:128].float())
                ftle_lowent_val = ftle_entropy_grad_lowent(model, xb_small[:256].float())
            ftle_valid = 1
        except Exception:
            ftle_valid = 0

        # TRAIN
        train_loss_sum = 0.0
        train_cnt = 0
        grad_norm_med = float("nan")
        grad_norm_rel = float("nan")
        grad_valid = 0
        model.train()
        norms = []
        it = iter(tr_loader)
        while True:
            try:
                xb, yb = next(it)
            except StopIteration:
                break
            except Exception as e:
                if not did_retry_fetch:
                    print(f"[WARN] batch fetch failed once; retrying… (seed={seed}, epoch={epoch}): {repr(e)}")
                    did_retry_fetch = True
                    continue
                nan_seen = True
                print(f"[WARN] batch fetch failed (seed={seed}, epoch={epoch}): {repr(e)}")
                traceback.print_exc()
                break

            xb, yb = xb.to(device), yb.to(device)
            try:
                if amp_enabled:
                    with amp.autocast(device_type=device.type, enabled=amp_enabled):
                        logits = model(xb)
                        loss = F.cross_entropy(logits, yb)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    if clip_eff:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_eff))
                    scaler.step(opt)
                    scaler.update()
                else:
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if clip_eff:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_eff))
                    opt.step()
            except Exception as e:
                if not did_retry_step:
                    print(f"[WARN] train step failed once; retrying… (seed={seed}, epoch={epoch}): {repr(e)}")
                    did_retry_step = True
                    continue
                nan_seen = True
                print(f"[WARN] train step failed (seed={seed}, epoch={epoch}): {repr(e)}")
                traceback.print_exc()
                break

            g2 = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    if torch.isnan(g).any() or torch.isinf(g).any():
                        nan_seen = True
                    g2 += float((g * g).sum().item())
            if g2 > 0 and np.isfinite(g2):
                norms.append(math.sqrt(g2))
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                nan_seen = True
            train_loss_sum += float(loss.detach().item()) * xb.shape[0]
            train_cnt += xb.shape[0]
        if norms:
            grad_norm_med = float(np.median(norms))
            with torch.no_grad():
                p2 = 0.0
                for p in model.parameters():
                    pp = p.detach().float()
                    p2 += float((pp * pp).sum().item())
            grad_norm_rel = float(grad_norm_med / (math.sqrt(max(p2, 1e-12)) + 1e-12))
            grad_valid = 1
        train_loss = train_loss_sum / max(1, train_cnt)
        ewma_loss = (
        train_loss
        if epoch == 0 or np.isnan(train_loss)
        else (
        CFG["ewma_alpha"] * train_loss
        + (1 - CFG["ewma_alpha"]) * (ewma_loss if np.isfinite(ewma_loss) else train_loss)
        )
        )
        loss_hist.append(train_loss)
        sma_loss = (
        float(np.mean(loss_hist[-CFG["sma_k"] :]))
        if len(loss_hist) >= CFG["sma_k"]
        else float("nan")
        )

        sw_ratio = float("nan")
        if sw_valid and sw_nat_valid and np.isfinite(sw) and np.isfinite(sw_nat):
            sw_ratio = float(sw_nat / (sw + 1e-12))

        # --- weight drift and GNS ---
        drift_abs = float("nan")
        drift_rel = float("nan")
        gns = float("nan")
        gns_n = 0
        try:
            cur_norm, cur_vec = _flatten_params_l2(model)
            if _prev_param_vec is not None and _prev_param_vec.numel() == cur_vec.numel():
                d = torch.linalg.vector_norm(cur_vec - _prev_param_vec).item()
                drift_abs = float(d)
                denom = math.sqrt(max(cur_norm**2, 1e-12))
                drift_rel = float(d / (denom + 1e-12))
            _prev_param_vec = cur_vec
        except Exception:
            pass
        try:
            # GNS on the very first batch of the training loader for this epoch
            it_g = iter(tr_loader)
            xb_g, yb_g = next(it_g)
            xb_g, yb_g = xb_g.to(device), yb_g.to(device)
            gns, gns_n = gradient_noise_scale(
            model, xb_g[: min(128, xb_g.shape[0])], yb_g[: min(128, yb_g.shape[0])], micro=4
            )
        except Exception:
            gns, gns_n = float("nan"), 0

        # --- fixed-partition gate (Φ_W) with ε_stat and κ_sens ---
        try:
            gate_warn = 0
            gate_gain = float("nan")
            gate_worst_tv = float("nan")
            gate_eps_stat = float("nan")
            gate_kappa = float("nan")

            W_gate = int(CFG.get("gate_window", 16))
            if (WINDOW_DECL is not None) and (len(logs) >= 2 * W_gate):
                recent = logs[-(2 * W_gate):]

                def _series(seg, key):
                    v = np.array([float(r.get(key, np.nan)) for r in seg], dtype=float)
                    return v[np.isfinite(v)]

                metrics_for_tv = list(WINDOW_DECL.weights.keys())
                past_dict = {m: _series(recent[:W_gate], m) for m in metrics_for_tv}
                recent_dict = {m: _series(recent[W_gate:], m) for m in metrics_for_tv}
                counts = {m: int(min(past_dict[m].size, recent_dict[m].size)) for m in metrics_for_tv}

                # prequential gain over last W on train vs baseline loss
                model_losses = np.array([float(r.get("train_loss", np.nan)) for r in logs[-W_gate:]], dtype=float)
                base_losses  = np.array([float(r.get("ewma_loss",  np.nan)) for r in logs[-W_gate:]], dtype=float)
                mask = np.isfinite(model_losses) & np.isfinite(base_losses)
                gate_gain = float((base_losses[mask] - model_losses[mask]).sum()) if mask.any() else float("nan")

                # κ_sens probe sparsely (augmentation-only by default)
                gate_kappa = 0.0
                try:
                    if (epoch % max(1, int(CFG.get("heavy_every", 6)))) == 0:
                        gate_kappa = kappa_sens_probe(
                            model, opt, pool_loader, device, WINDOW_DECL, ref_mu_sig,
                            probe_cfg={
                                "aug_probe": True,
                                "aug_loader_factory": probe_aug_loader_factory,
                                "lr_probe": False,
                            },
                        )
                except Exception:
                    gate_kappa = 0.0

                flag, audit = gate_check(
                    WINDOW_DECL,
                    past_dict,
                    recent_dict,
                    counts,
                    gain=gate_gain,
                    gain_thresh=float(CFG.get("gate_gain_thresh", 5.0)),
                    kappa_sens=(gate_kappa if np.isfinite(gate_kappa) else 0.0),
                )
                gate_warn = int(flag)
                gate_worst_tv = float(audit.get("worst_tv", np.nan))
                gate_eps_stat = float(audit.get("eps_stat", np.nan))
                gate_kappa = float(audit.get("kappa_sens", np.nan))
        except Exception:
            gate_warn = 0
            gate_gain = float("nan")
            gate_worst_tv = float("nan")
            gate_eps_stat = float("nan")
            gate_kappa = float("nan")

        logs.append(
            dict(
                run_id=run_id,
                seed=seed,
                factor=factor["name"],
                epoch=epoch,
                lr=lr,
                momentum=mom,
                nan_flag=int(nan_seen),
                train_loss=train_loss,
                ewma_loss=ewma_loss,
                sma_loss=sma_loss,
                grad_norm=grad_norm_med,
                grad_norm_rel=grad_norm_rel,
                grad_valid=grad_valid,
                grad_norm_rel_valid=grad_valid,
                var_out_k=vout_k,
                var_out_k_valid=int(v_ok),
                eff_dim=effd,
                eff_dim_gt=effd_nat,
                k_used=k_used,
                tail_mass=tail_mass,
                neg_eigs=neg_eigs,
                cos_disp=cosd,
                # coherence/slope channels
                slope_eff_dim=slope_eff_dim,
                slope_var_out_k=slope_var_out_k,
                reg_eff_dim=reg_eff_dim,
                reg_var_out_k=reg_var_out_k,
                reg_eff_neg_persist=reg_eff_neg_persist,
                reg_var_pos_persist=reg_var_pos_persist,
                corr_effdim_varoutk=corr_effdim_varoutk,
                coh_break_persist=coh_break_persist,
                pers_H0=pers_h0,
                topo_valid=int(topo_done >= 1),
                topo_done=int(topo_done),
                topo_ms=float(topo_ms),
                n_topo_sampled=int(topo_n_used),
                sw2=sw,
                sw2_ms=float(sw_ms),
                n_sw2_proj_done=int(sw_proj_done),
                sw2_valid=int(sw_valid),
                sw2_native=sw_nat,
                sw2_native_ms=float(sw_nat_ms),
                n_sw2_native_proj_done=int(sw_nat_proj_done),
                sw2_native_valid=int(sw_nat_valid),
                sw2_ratio_native=sw_ratio,
                ftle=ftle_val,
                ftle_lowent=ftle_lowent_val,
                ftle_valid=ftle_valid,
                mon_entropy=mon_ent,
                avg_max_prob=mon_conf,
                monitor_acc=(mon_acc if CFG["monitor_labels"] else float("nan")),
                n_features=n_feats,
                ph_win=CFG["ph_win"],
                ph_lambda=CFG["ph_lambda"],
                ph_two_sided=int(CFG["ph_two_sided"]),
                warn_vote=CFG.get("warn_vote", 2),
                heavy_every=CFG["heavy_every"],
                metric_batches=CFG["metric_batches"],
                var_k_energy=CFG["var_k_energy"],
                var_k_max=CFG["var_k_max"],
                # --- new metrics ---
                var_out_k_native=vout_k_nat,
                var_out_k_native_valid=int(v_ok_nat),
                neg_eigs_native=neg_eigs_nat,
                r2=r2,
                slope_r2=slope_r2,
                sw2_med3=sw2_med3,
                sw2_slope=sw2_slope,
                margin_med=margin_med,
                slope_margin=slope_margin,
                drift_abs=drift_abs,
                drift_rel=drift_rel,
                gns=gns,
                gns_n=gns_n,
                # finite-window gate diagnostics
                gate_warn=int(gate_warn),
                gate_gain=float(gate_gain),
                gate_worst_tv=float(gate_worst_tv),
                gate_window=int(CFG.get("gate_window", 16)),
                gate_bins=int(CFG.get("gate_bins", 16)),
                gate_epsilon=float(CFG.get("gate_epsilon", 0.08)),
                gate_gain_thresh=float(CFG.get("gate_gain_thresh", 5.0)),
                # fixed-partition gate diagnostics
                gate_eps_stat=float(gate_eps_stat),
                gate_kappa=float(gate_kappa),
            )
        )
        # hard-stop training when the finite-window gate warns for k consecutive epochs (after warm)
        if gate_early_exit:
            if epoch >= warm_epoch:
                if int(gate_warn) == 1:
                    gate_hits_consec += 1
                else:
                    gate_hits_consec = 0
                if gate_hits_consec >= warn_consec_eff and gate_halt_epoch is None:
                    gate_halt_epoch = int(epoch)
                    # persist a terminal marker compatible with resume logic
                    try:
                        (OUTDIR / f"runs_seed{seed}_{tag}.done.json").write_text(
                            json.dumps({
                                "terminal": True,
                                "reason": "gate_halt",
                                "epoch": int(epoch),
                                "gate_hits_consec": int(gate_hits_consec)
                            })
                        )
                    except Exception:
                        pass
                    break
#         if nan_seen:
#             # ensure we persist a terminal marker even if the failure was detected without an exception break
#             try:
#                 (OUTDIR / f"runs_seed{seed}_{tag}.done.json").write_text(
#                     json.dumps({"terminal": True, "reason": "nan_flag", "epoch": int(epoch)})
#                 )
#             except Exception:
#                 pass
#             break
        if nan_seen:
            # ensure we persist a terminal marker even if the failure was detected without an exception break
            try:
                (OUTDIR / f"runs_seed{seed}_{tag}.done.json").write_text(
                    json.dumps({"terminal": True, "reason": "nan_flag", "epoch": int(epoch)})
                )
            except Exception:
                pass
            break

    df = pd.DataFrame(logs)
    return df
# ---------------------------
# Direction calibration for PH & grad cutoff
# ---------------------------
def calibrate_ph_directions(df_cal: pd.DataFrame, metrics: List[str]) -> Dict[str, str]:
    dir_map = {}
    warm = CFG["warmup"] + CFG["ph_burn"]
    for m in metrics:
        win_m = CFG["ph_win_short"] if (m in SCHEDULED_METRICS) else CFG["ph_win"]
        best = None
        best_dir = "up"
        for d in ["up", "down"]:
            leads = []
            for (seed, factor), g in df_cal.groupby(["seed", "factor"]):
                g = g.sort_values("epoch")
                t_c = g["t_collapse_gt"].iloc[0] if "t_collapse_gt" in g.columns else None
                ctag = g["collapse_tag_gt"].iloc[0] if "collapse_tag_gt" in g.columns else "none"
                if pd.isna(t_c) or ctag != "soft":
                    continue
                xs_all = _prep_series_for_ph(g, m)
                pre = g["epoch"].to_numpy() < int(t_c)
                xs = np.where(pre, np.array(xs_all, dtype=float), np.nan).tolist()
                t, _, _ = ph_window_sparse(xs, win=win_m, lam=CFG["ph_lambda"], direction=d, burn_in=warm, min_points=CFG["ph_min_points"], two_sided=CFG["ph_two_sided"])
                if t is not None:
                    leads.append(int(t_c) - t)
            if leads:
                avg = float(np.mean(leads))
                if (best is None) or (avg > best):
                    best = avg
                    best_dir = d
        dir_map[m] = best_dir
    return dir_map


def calibrate_grad_cutoff_per_factor(df_cal: pd.DataFrame) -> Dict[str, float]:
    """
    Robust per-factor cutoff:
        pass
    - exclude epochs with NaN/inf flags
    - for runs without collapse, restrict to early stable window to avoid tail inflation
    - use median + 4·MAD (MAD scaled by 1.4826)
    """
    out = {}
    warm = CFG["warmup"] + CFG["ph_burn"]
    guard = warm + CFG["ph_win"]  # early stable window cap
    for factor, gfac in df_cal.groupby("factor"):
        vals = []
        for (_, _), g in gfac.groupby(["seed", "factor"]):
            g = g.sort_values("epoch")
            t_c = g["t_collapse_gt"].iloc[0] if "t_collapse_gt" in g.columns else np.nan
            sub = g[(~g["nan_flag"].astype(bool)) & np.isfinite(g["grad_norm_rel"])]
            if pd.isna(t_c):
                sub = sub[sub["epoch"] < guard]
            else:
                sub = sub[sub["epoch"] < int(t_c)]
            vals.extend(sub["grad_norm_rel"].tolist())
        if vals:
            med = float(np.median(vals))
            mad = float(np.median(np.abs(np.array(vals) - med))) + 1e-9
            out[str(factor)] = med + 4.0 * (1.4826 * mad)
        else:
            out[str(factor)] = np.inf
    return out


# ---------------------------
# Events & evaluation (UNIFIED GT)
# ---------------------------
def compute_events(df: pd.DataFrame, metrics_for_ph, dir_map):
    rows, dbg = [], []
    burn = CFG["warmup"] + CFG["ph_burn"]
    win = CFG["ph_win"]
    lam = CFG["ph_lambda"]
    min_points = CFG["ph_min_points"]
    for (seed, factor), g in df.groupby(["seed", "factor"]):
        g0 = g[g.epoch >= 0].sort_values("epoch").copy()
        t_collapse = (
        int(g0["t_collapse_gt"].iloc[0]) if pd.notna(g0["t_collapse_gt"].iloc[0]) else None
        )
        ctag = str(g0["collapse_tag_gt"].iloc[0])
        t_map = {}
        for m in metrics_for_ph:
            xs = _prep_series_for_ph(g0, m)
            win_m = CFG["ph_win_short"] if (m in SCHEDULED_METRICS) else win
            d = dir_map.get(m, "up")
            t, zs, cs = ph_window_sparse(
            xs,
            win=win_m,
            lam=lam,
            direction=d,
            burn_in=burn,
            min_points=min_points,
            two_sided=CFG["ph_two_sided"],
            )
            thr = max(burn, win_m)
            violation = 0
            if t is not None and t < thr:
                violation = 1
                t = None
            t_map[m] = t
            dbg.append(
            dict(
            seed=int(seed),
            factor=str(factor),
            metric=m,
            z_scores=json.dumps([float(z) for z in zs]),
            cusum=json.dumps([float(s) for s in cs]),
            retro_violation=int(violation),
            )
            )
        rows.append(
        dict(
        run_id=f"s{int(seed)}-{str(factor)}",
        seed=int(seed),
        factor=str(factor),
        collapse_tag=ctag,
        t_collapse=t_collapse,
        ph_win=CFG["ph_win"],
        ph_lambda=CFG["ph_lambda"],
        ph_two_sided=int(CFG["ph_two_sided"]),
        heavy_every=CFG["heavy_every"],
        metric_batches=CFG["metric_batches"],
        var_k_energy=CFG["var_k_energy"],
        var_k_max=CFG["var_k_max"],
        **{f"t_{k}": v for k, v in t_map.items()},
        )
        )
    return pd.DataFrame(rows), pd.DataFrame(dbg)


def _first_t_column(tr_df: pd.DataFrame) -> Optional[str]:
    cols = [c for c in tr_df.columns if c.startswith("t_") and c != "t_collapse"]
    if not cols:
        return None
    if len(cols) == 1:
        return cols[0]
    counts = {c: tr_df[c].notna().sum() for c in cols}
    return max(counts, key=counts.get)


def mark_events_epochwise(df_runs: pd.DataFrame, events: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Add boolean overlays to df_runs for warn/collapse based on `events`:
        pass
    - is_warn_epoch_<prefix>: True on the K-length window ending at t_warn (inclusive)
    - is_collapse_epoch_<prefix>: True exactly at t_collapse

    Auto-picks the warn source:
        pass
    * uses 't_warn' if present; otherwise falls back to the most-populated 't_<metric>' column
    (via _first_t_column), ignoring 't_collapse'.
    Robust if multiple rows exist per (seed, factor): the first row is used.
    No-ops cleanly on empty frames.
    """
    df = df_runs.copy()
    warn_col = f"is_warn_epoch_{prefix}"
    col_col = f"is_collapse_epoch_{prefix}"
    df[warn_col] = False
    df[col_col] = False

    if events is None or events.empty:
        return df

    # pick warn key: prefer explicit 't_warn', else best 't_<metric>' column
    warn_key = "t_warn" if ("t_warn" in events.columns) else _first_t_column(events)

    ev = events.set_index(["seed", "factor"])

    K = int(CFG.get("warn_consec", 3))
    for (seed, factor), g in df.groupby(["seed", "factor"], sort=False):
        key = (int(seed), str(factor))
        if key not in ev.index:
            continue

        row = ev.loc[key]
        # If ev.loc returns a DataFrame (duplicated keys), take the first row
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        # --- warn window ---
        if warn_key is not None and warn_key in row.index:
            tw = row[warn_key]
            if pd.notna(tw) and K > 0:
                twi = int(tw)
                e = g["epoch"].to_numpy(dtype=int)
                win = (e >= twi - (K - 1)) & (e <= twi)
                if win.any():
                    df.loc[g.index[win], warn_col] = True

        # --- collapse point ---
        tc = (row["t_collapse"] if "t_collapse" in row.index else row.get("t_collapse_gt", np.nan))
        if pd.notna(tc):
            tci = int(tc)
            df.loc[g.index, col_col] = g["epoch"].to_numpy(dtype=int) == tci

    return df


def assert_overlay_consistency(df_epoch: pd.DataFrame, events: pd.DataFrame, prefix: str):
    warn_col = f"is_warn_epoch_{prefix}"
    col_col = f"is_collapse_epoch_{prefix}"
    ev = events.set_index(["seed", "factor"])
    mismatches = []
    for (seed, factor), g in df_epoch.groupby(["seed", "factor"]):
        key = (int(seed), str(factor))
        if key not in ev.index:
            continue
        tw_flags = g[g[warn_col]].sort_values("epoch")["epoch"].tolist()
        tc_flags = g[g[col_col]].sort_values("epoch")["epoch"].tolist()
        # Use the last True epoch for warn overlays because mark_events_epochwise marks a K-length window ending at t_warn.
        tw = int(tw_flags[-1]) if tw_flags else None
        tc = int(tc_flags[0]) if tc_flags else None
        e_tw = ev.loc[key, "t_warn"]
        e_tc = ev.loc[key, "t_collapse"]
        bad_warn = (
        (pd.notna(e_tw) and tw is None)
        or (pd.isna(e_tw) and tw is not None)
        or (pd.notna(e_tw) and tw != int(e_tw))
        )
        bad_col = (
        (pd.notna(e_tc) and tc is None)
        or (pd.isna(e_tc) and tc is not None)
        or (pd.notna(e_tc) and tc != int(e_tc))
        )
        if bad_warn or bad_col:
            mismatches.append(
            {
            "seed": int(seed),
            "factor": str(factor),
            "warn_overlay": tw,
            "warn_events": (int(e_tw) if pd.notna(e_tw) else None),
            "collapse_overlay": tc,
            "collapse_events": (int(e_tc) if pd.notna(e_tc) else None),
            }
            )
    if mismatches:
        cap = 50
        path = OUTDIR / f"overlay_mismatches_{prefix}.json"
        save_json({"count": len(mismatches), "items": mismatches[:cap]}, path)
        print(f"[WARN] overlay mismatches: {len(mismatches)} (showing first {cap}; details in {path})")


def bootstrap_stratified(rows: pd.DataFrame, B: int = 200) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(123456)
    factors = sorted(rows["factor"].unique().tolist())
    if not factors:
        return {}
    vals_detect = []
    vals_fp = []
    vals_med = []
    warm = CFG["warmup"] + CFG["ph_burn"]
    for _ in range(B):
        boot_parts = []
        for f in factors:
            seeds_f = sorted(rows[rows["factor"] == f]["seed"].unique().tolist())
            if not seeds_f:
                continue
            draw = rng.choice(seeds_f, size=len(seeds_f), replace=True)
            for s in draw:
                boot_parts.append(rows[(rows["factor"] == f) & (rows["seed"] == s)])
        if not boot_parts:
            continue
        boot = pd.concat(boot_parts, ignore_index=True)

        trig = boot[boot["collapse_tag"] == "soft"]
        ncol = len(trig)
        succ = int(
        (
        (trig["t_warn"].notna())
        & ((trig["t_collapse"] - trig["t_warn"]) >= SUCCESS_TARGET["min_lead"])
        ).sum()
        )
        vals_detect.append(succ / max(1, ncol))

        non_trig = boot[boot["collapse_tag"] == "none"]
        denom = max(1, len(non_trig))
        fp = (
        float(np.mean((non_trig["t_warn"].notna()) & (non_trig["t_warn"] >= warm)))
        if denom > 0
        else 1.0
        )
        vals_fp.append(fp)

        leads = (trig["t_collapse"] - trig["t_warn"]).dropna().to_numpy()
        vals_med.append(float(np.median(leads)) if leads.size > 0 else np.nan)

    def ci(v):
        arr = np.array(v, dtype=np.float32)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return (np.nan, np.nan)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return (float(lo), float(hi))

    return dict(detect_rate_ci=ci(vals_detect), fp_rate_ci=ci(vals_fp), lead_median_ci=ci(vals_med))


def summarize_detection(rows: pd.DataFrame, warm_idx: int) -> pd.DataFrame:
    # tolerate empty/malformed inputs
    if rows is None or len(rows) == 0:
        return pd.DataFrame(columns=["kind", "n", "successes", "value", "lo", "hi"])
    if "collapse_tag" not in rows.columns:
        if "collapse_tag_gt" in rows.columns:
            rows = rows.rename(columns={"collapse_tag_gt": "collapse_tag"})
        else:
            rows = rows.copy()
            rows["collapse_tag"] = "none"
    out = []
    trig = rows[rows["collapse_tag"] == "soft"].copy()
    n_collapse = len(trig)
    successes = int(
    (
    (trig["t_warn"].notna())
    & ((trig["t_collapse"] - trig["t_warn"]) >= SUCCESS_TARGET["min_lead"])
    ).sum()
    )
    detect_rate = successes / max(1, n_collapse)
    if n_collapse > 0:
        z = 1.96
        phat = detect_rate
        denom = 1 + z**2 / n_collapse
        center = (phat + z * z / (2 * n_collapse)) / denom
        half = (
        z
        * math.sqrt((phat * (1 - phat) / n_collapse) + z * z / (4 * n_collapse * n_collapse))
        / denom
        )
        d_lo, d_hi = center - half, center + half
    else:
        d_lo = d_hi = np.nan
    non_trig = rows[rows["collapse_tag"] == "none"]
    fp_nontrig = (
    float(np.mean((non_trig["t_warn"].notna()) & (non_trig["t_warn"] >= warm_idx)))
    if len(non_trig) > 0
    else 1.0
    )
    leads = (trig["t_collapse"] - trig["t_warn"]).dropna().to_numpy()
    med = q1 = q3 = np.nan
    if leads.size > 0:
        med = float(np.median(leads))
        q1 = float(np.percentile(leads, 25))
        q3 = float(np.percentile(leads, 75))
    out.append(
    dict(
    kind="detect_rate",
    n=n_collapse,
    successes=successes,
    value=detect_rate,
    lo=d_lo,
    hi=d_hi,
    )
    )
    out.append(dict(kind="fp_nontriggered_after_warm", n=int(len(non_trig)), value=fp_nontrig))
    out.append(dict(kind="lead_time", n=int(leads.size), med=med, q1=q1, q3=q3))
    boot = bootstrap_stratified(rows)
    out.append(
    dict(
    kind="detect_rate_ci_boot",
    lo=boot.get("detect_rate_ci", (np.nan, np.nan))[0],
    hi=boot.get("detect_rate_ci", (np.nan, np.nan))[1],
    )
    )
    out.append(
    dict(
    kind="fp_rate_ci_boot",
    lo=boot.get("fp_rate_ci", (np.nan, np.nan))[0],
    hi=boot.get("fp_rate_ci", (np.nan, np.nan))[1],
    )
    )
    out.append(
    dict(
    kind="lead_median_ci_boot",
    lo=boot.get("lead_median_ci", (np.nan, np.nan))[0],
    hi=boot.get("lead_median_ci", (np.nan, np.nan))[1],
    )
    )
    return pd.DataFrame(out)


def summarize_runlevel_fp(events: pd.DataFrame, warm_idx: int) -> float:
    """
    Run-level FP = fraction of 'none' runs that have any t_warn at/after warm_idx.
    Robust to empty frames / missing columns.
    """
    if events is None or events.empty:
        return float("nan")
    req = {"seed", "factor", "collapse_tag", "t_warn"}
    if not req.issubset(set(events.columns)):
        return float("nan")
    none_runs = events[events["collapse_tag"] == "none"].copy()
    if none_runs.empty:
        return float("nan")
    flags = []
    for (sd, fc), g in none_runs.groupby(["seed", "factor"]):
        tw = g["t_warn"].dropna()
        hit = (len(tw) > 0) and (int(tw.iloc[0]) >= int(warm_idx))
        flags.append(bool(hit))
    return float(np.mean(flags)) if flags else float("nan")


# ---------------------------------
# Efficacy tightening: AUC & CIs
# ---------------------------------

def _wilcoxon_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Tie-aware rank AUC (equiv. to Mann–Whitney U / (m*n))."""
    x = np.concatenate([pos, neg])
    r = pd.Series(x).rank(method="average").to_numpy()
    m = len(pos)
    n = len(neg)
    if m == 0 or n == 0:
        return np.nan
    r_pos = r[:m].sum()
    U = r_pos - m * (m + 1) / 2.0
    return float(U / (m * n))


def _bootstrap_ci(values: np.ndarray, B: int = 1000, alpha: float = 0.05, seed: int = 1337) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (np.nan, np.nan)
    idx = np.arange(vals.size)
    boots = []
    for _ in range(B):
        samp = rng.choice(idx, size=idx.size, replace=True)
        boots.append(float(np.median(vals[samp])))
    lo, hi = np.quantile(boots, [alpha / 2.0, 1 - alpha / 2.0])
    return (float(lo), float(hi))


def safe_auc_by_factor(df: pd.DataFrame, score_col: str, label_col: str, min_per_side: int = 2,
family_map: Optional[Dict[str, str]] = None, B: int = 1000) -> pd.DataFrame:
    """
    Compute per-factor Δ and AUC with bootstrap CIs. Ensures each factor has at least
    `min_per_side` seeds on both sides of label_col. Optionally pool factors via `family_map`.
    Expects one row per (seed,factor) with last-epoch summary scores.
    Returns rows with: group (factor or family), n_pos, n_neg, delta_med, delta_lo, delta_hi,
    auc, auc_lo, auc_hi.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["group","n_pos","n_neg","delta_med","delta_lo","delta_hi","auc","auc_lo","auc_hi"])

    fam = df.copy()
    if family_map:
        fam["group"] = fam["factor"].map(lambda f: family_map.get(str(f), str(f)))
    else:
        fam["group"] = fam["factor"].astype(str)

    out_rows = []
    rng = np.random.default_rng(4242)
    for grp, g in fam.groupby("group"):
        # aggregate at seed level
        g_seed = g.groupby("seed").agg({score_col: "median", label_col: "first"}).reset_index()
        pos = g_seed[g_seed[label_col] == 1][score_col].to_numpy(dtype=float)
        neg = g_seed[g_seed[label_col] == 0][score_col].to_numpy(dtype=float)
        if len(pos) < min_per_side or len(neg) < min_per_side:
            # under-sampled branch: bootstrap Δ = median(pos) - median(neg) by resampling sides separately
            d = float(np.median(pos) - np.median(neg)) if (pos.size and neg.size) else np.nan
            B = int(B) if isinstance(B, (int, np.integer)) else 1000
            rng_local = np.random.default_rng(11000)
            deltas = []
            for _ in range(B):
                pos_b = rng_local.choice(pos, size=len(pos), replace=True) if pos.size else np.array([])
                neg_b = rng_local.choice(neg, size=len(neg), replace=True) if neg.size else np.array([])
                if pos_b.size and neg_b.size:
                    deltas.append(float(np.median(pos_b) - np.median(neg_b)))
            if deltas:
                dlo, dhi = np.quantile(deltas, [0.025, 0.975])
            else:
                dlo, dhi = (np.nan, np.nan)
            out_rows.append(dict(group=str(grp), n_pos=len(pos), n_neg=len(neg),
            delta_med=float(d), delta_lo=float(dlo), delta_hi=float(dhi),
            auc=np.nan, auc_lo=np.nan, auc_hi=np.nan))
            continue
        # Δ bootstrap
        base = g_seed[[score_col, label_col]].to_numpy()
        idx = np.arange(base.shape[0])
        deltas = []
        aucs = []
        for _ in range(B):
            samp = rng.choice(idx, size=idx.size, replace=True)
            s = base[samp]
            pos_s = s[s[:, 1] == 1, 0]
            neg_s = s[s[:, 1] == 0, 0]
            if pos_s.size == 0 or neg_s.size == 0:
                continue
            deltas.append(float(np.median(pos_s) - np.median(neg_s)))
            aucs.append(_wilcoxon_auc(pos_s, neg_s))
        d_med = float(np.median(deltas)) if deltas else np.nan
        d_lo, d_hi = (np.quantile(deltas, [0.025, 0.975]) if deltas else (np.nan, np.nan))
        auc = _wilcoxon_auc(pos, neg)
        a_lo, a_hi = (np.quantile(aucs, [0.025, 0.975]) if aucs else (np.nan, np.nan))
        out_rows.append(dict(group=str(grp), n_pos=len(pos), n_neg=len(neg),
        delta_med=d_med, delta_lo=float(d_lo), delta_hi=float(d_hi),
        auc=float(auc), auc_lo=float(a_lo), auc_hi=float(a_hi)))
    return pd.DataFrame(out_rows)


# ---------------------------------
# Dynamics checks for outliers
# ---------------------------------

def _theil_sen_slope(y: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    """Robust slope (median of pairwise slopes). Works on small n (e.g., last 8 epochs)."""
    y = np.asarray(y, dtype=float)
    if x is None:
        x = np.arange(y.size, dtype=float)
    else:
        x = np.asarray(x, dtype=float)
    m = []
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if np.isfinite(y[i]) and np.isfinite(y[j]) and x[j] != x[i]:
                m.append((y[j] - y[i]) / (x[j] - x[i]))
    return float(np.median(m)) if m else np.nan


def monotonicity_checks_for_outliers(df_runs: pd.DataFrame, warm_idx: int, tail: int = 8) -> pd.DataFrame:
    """
    For each (seed,factor) flagged as outlier elsewhere, compute:
        pass
    - sign(Theil–Sen slope) over last `tail` epochs for FTLE and drift_abs
    - FTLE <= 0 over [warm_idx, end]
    Returns a tidy table with booleans and slopes for auditing.
    """
    rows = []
    for (sd, fc), g in df_runs.groupby(["seed", "factor"]):
        gg = g.sort_values("epoch")
        tail_g = gg.tail(tail)
        x = tail_g["epoch"].to_numpy(dtype=float)
        ft = tail_g.get("ftle", pd.Series(dtype=float)).to_numpy(dtype=float)
        dr = tail_g.get("drift_abs", pd.Series(dtype=float)).to_numpy(dtype=float)
        s_ft = _theil_sen_slope(ft, x)
        s_dr = _theil_sen_slope(dr, x)
        post = gg[gg["epoch"] >= int(warm_idx)]
        ft_ok = bool(np.all(np.nan_to_num(post.get("ftle", pd.Series(dtype=float)).to_numpy(dtype=float)) <= 0.0)) if len(post) else False
        rows.append(dict(seed=int(sd), factor=str(fc),
        slope_ftle=s_ft, slope_drift_abs=s_dr,
        slope_ftle_pos=bool(np.isfinite(s_ft) and s_ft > 0),
        slope_drift_pos=bool(np.isfinite(s_dr) and s_dr > 0),
        ftle_nonpos_after_warm=ft_ok))
    return pd.DataFrame(rows)


# ---------------------------------
# Artifact invariants & provenance
# ---------------------------------

def compute_invariants_and_provenance(df_runs: pd.DataFrame, artifact_csv: Optional[Path] = None) -> Dict[str, Any]:
    """Persist invariants and provenance. FTLE invariant uses a control-derived bound (q99 * margin)."""
    required = {"factor", "epoch"}
    if not required.issubset(set(df_runs.columns)):
        print("[WARN] invariants skipped: missing columns", sorted(list(required - set(df_runs.columns))))
        return {}
    # Calibrate FTLE bound on controls ('none') after warm
    warm = int(CFG["warmup"]) + int(CFG["ph_burn"])
    ctl = (
    df_runs[(df_runs["factor"] == "none") & (df_runs["epoch"] >= warm)]
    if isinstance(df_runs, pd.DataFrame)
    else pd.DataFrame()
    )
    try:
        if not ctl.empty and ("ftle" in ctl.columns):
            arr = ctl["ftle"].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            ctl_q99 = float(np.nanquantile(arr, 0.99)) if arr.size > 0 else np.nan
        else:
            ctl_q99 = np.nan
    except Exception:
        ctl_q99 = np.nan

    margin = float(CFG.get("ftle_q99_margin", 1.05))
    abs_cap = CFG.get("ftle_q99_cap", None)
    if abs_cap is not None:
        try:
            ftle_cap = float(abs_cap)
        except Exception:
            ftle_cap = np.nan
    else:
        ftle_cap = ctl_q99 * margin if np.isfinite(ctl_q99) else np.nan

    inv = []
    for (sd, fc), g in df_runs.groupby(["seed", "factor"]):
        gg = g.sort_values("epoch")
        post = gg[gg["epoch"] >= warm] if "epoch" in gg.columns else gg

        # FTLE boundedness by q99 of this run vs global cap
        q99_run = np.nan
        if "ftle" in post.columns and len(post) > 0:
            vals = post["ftle"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                try:
                    q99_run = float(np.nanquantile(vals, 0.99))
                except TypeError:
                    q99_run = float(np.nanquantile(vals, 0.99, interpolation="linear"))
        ftle_ok = bool(np.isfinite(ftle_cap) and np.isfinite(q99_run) and (q99_run <= ftle_cap))

        neg_ok = bool(np.nanmax(gg.get("neg_eigs", pd.Series([0])).to_numpy(dtype=float)) == 0)
        k_used_med = float(np.nanmedian(gg.get("k_used", pd.Series([np.nan])).to_numpy(dtype=float)))
        k_ok = bool(np.isfinite(k_used_med) and abs(k_used_med - 4.0) <= 0.5)

        inv.append(
        dict(
        seed=int(sd),
        factor=str(fc),
        ftle_bounded=ftle_ok,
        ftle_q99_run=q99_run,
        neg_eigs_zero=neg_ok,
        k_used_approx4=k_ok,
        )
        )

    invariants = pd.DataFrame(inv)
    out = {
    "invariants": invariants.to_dict(orient="records"),
    "summary": {
    "ftle_bounded_pass_rate": float(np.mean(invariants["ftle_bounded"].astype(bool))) if not invariants.empty else np.nan,
    "neg_eigs_zero_pass_rate": float(np.mean(invariants["neg_eigs_zero"].astype(bool))) if not invariants.empty else np.nan,
    "k_used_approx4_pass_rate": float(np.mean(invariants["k_used_approx4"].astype(bool))) if not invariants.empty else np.nan,
    },
    "provenance": {},
    }

    # provenance
    prov: Dict[str, Any] = {}
    try:
        prov["python"] = sys.version
    except Exception:
        pass
    try:
        import numpy as _np
        prov["numpy"] = _np.__version__
    except Exception:
        pass
    try:
        import pandas as _pd
        prov["pandas"] = _pd.__version__
    except Exception:
        pass
    try:
        import torch as _torch
        prov["torch"] = _torch.__version__
    except Exception:
        pass
    try:
        prov["cuda"] = torch.version.cuda
    except Exception:
        pass
    if artifact_csv is not None:
        try:
            prov["artifact_md5"] = file_md5(artifact_csv)
        except Exception:
            pass

    prov["ftle_cap_source"] = ("abs_cap" if abs_cap is not None else "control_q99_margin")
    prov["ftle_control_q99"] = float(ctl_q99) if np.isfinite(ctl_q99) else None
    prov["ftle_margin"] = float(margin)
    prov["ftle_cap_value"] = float(ftle_cap) if np.isfinite(ftle_cap) else None
    prov["warm_idx"] = int(warm)
    out["provenance"] = prov

    try:
        save_json(out, OUTDIR / "artifact_invariants_provenance.json")
    except Exception:
        pass
    return out


#
# ---------------------------
# RP adequacy: JL vs native agreement pre-warm
# ---------------------------
def rp_adequacy_flags(
df: pd.DataFrame, warm: int, corr_min: float = 0.9, min_pts: int = 8
) -> Dict[Tuple[int, str], int]:
    """Return {(seed,factor): 1/0} flag where geometry appears under-resolved in JL space.
    Criteria: Pearson corr between (eff_dim vs eff_dim_gt) or (var_out_k vs var_out_k_native)
    on pre-warm epochs is below corr_min with at least min_pts finite pairs."""
    flags: Dict[Tuple[int, str], int] = {}
    for (seed, factor), g in df.groupby(["seed", "factor"]):
        gg = g.sort_values("epoch")
        pre = gg[gg["epoch"] < warm]
        # eff_dim vs eff_dim_gt
        x1 = (
        pre["eff_dim"].to_numpy(dtype=float)
        if "eff_dim" in pre.columns
        else np.array([], dtype=float)
        )
        y1 = (
        pre["eff_dim_gt"].to_numpy(dtype=float)
        if "eff_dim_gt" in pre.columns
        else (
        pre["eff_dim"].to_numpy(dtype=float)
        if "eff_dim" in pre.columns
        else np.array([], dtype=float)
        )
        )
        m1 = np.isfinite(x1) & np.isfinite(y1)
        corr1 = np.nan
        if m1.sum() >= min_pts:
            corr1 = float(np.corrcoef(x1[m1], y1[m1])[0, 1])
        # var_out_k vs native
        x2 = (
        pre["var_out_k"].to_numpy(dtype=float)
        if "var_out_k" in pre.columns
        else np.array([], dtype=float)
        )
        y2 = (
        pre["var_out_k_native"].to_numpy(dtype=float)
        if "var_out_k_native" in pre.columns
        else np.full_like(x2, np.nan)
        )
        m2 = np.isfinite(x2) & np.isfinite(y2)
        corr2 = np.nan
        if m2.sum() >= min_pts:
            corr2 = float(np.corrcoef(x2[m2], y2[m2])[0, 1])
        flag = int(
        ((np.isfinite(corr1) and corr1 < corr_min) or (np.isfinite(corr2) and corr2 < corr_min))
        )
        flags[(int(seed), str(factor))] = flag
    return flags


# ---------------------------
# Plotting (reads unified epoch overlays)
# ---------------------------
def _spaghetti(ax, df_noise: pd.DataFrame, metric: str):
    for seed, g in df_noise.groupby("seed"):
        gg = g[g.epoch >= 0].sort_values("epoch")
        ax.plot(gg["epoch"], gg[metric], alpha=0.75, lw=1.1, label=f"s={seed}")


def make_plots(df, tr_like, tag, overlay_prefix="learned"):
    (OUTDIR / "figs").mkdir(exist_ok=True)
    shade0, shade1 = -1, CFG["warmup"]
    metrics = [
        "avg_max_prob",
        "mon_entropy",
        "train_loss",
        "ewma_loss",
        "sma_loss",
        "grad_norm",
        "grad_norm_rel",
        "cos_disp",
        "var_out_k",
        "eff_dim",
        "eff_dim_gt",
        "ftle",
        "ftle_lowent",
        "sw2",
        "sw2_native",
        "sw2_ratio_native",
        "pers_H0",
        # --- gate diagnostics ---
        "gate_gain",
        "gate_worst_tv",
        "gate_warn",
    ]
    for m in metrics:
        if m not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        _spaghetti(ax, df, m)
        ax.axvspan(shade0, shade1, color="#eee", alpha=0.5, label="warm-up")
        ax.set_title(f"{m} | [{tag}]")
        ax.set_xlabel("epoch")
        fig.tight_layout()
        fig.savefig(OUTDIR / f"figs/{m}_{tag}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    warn_col = f"is_warn_epoch_{overlay_prefix}"
    col_col = f"is_collapse_epoch_{overlay_prefix}"
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(tr_like) + 2))
    rng = np.random.default_rng(123)
    y = 0
    for (seed, factor), sub in df.groupby(["seed", "factor"]):
        sub = sub.sort_values("epoch")
        ax.plot(sub.epoch, [y] * len(sub), color="#dddddd", lw=1)
        w = sub[sub[warn_col]].epoch.tolist()
        c = sub[sub[col_col]].epoch.tolist()
        if w:
            jitter = (rng.random() - 0.5) * 0.3
            ax.scatter([w[0]], [y + jitter], s=60, color="tab:red", marker="|", lw=2)
        if c:
            jitter = (rng.random() - 0.5) * 0.3
            ctag = (
            tr_like[(tr_like.seed == seed) & (tr_like.factor == factor)]["collapse_tag"].iloc[0]
            if not tr_like.empty
            else "none"
            )
            color = "black" if ctag == "soft" else ("tab:orange" if ctag == "hard" else "tab:gray")
            ax.scatter([c[0]], [y + jitter], s=60, color=color, marker="|", lw=2)
        tagc = (
        tr_like[(tr_like.seed == seed) & (tr_like.factor == factor)]["collapse_tag"].iloc[0]
        if not tr_like.empty
        else ""
        )
        ax.text(-2, y, f"{factor} s={int(seed)} [{tagc}]", ha="right", va="center", fontsize=8)
        y += 1
    ax.axvspan(shade0, shade1, color="#eee", alpha=0.5)
    warm = CFG["warmup"] + CFG["ph_burn"]
    heavy0 = int(((warm + CFG["heavy_every"] - 1) // CFG["heavy_every"]) * CFG["heavy_every"])
    ax.axvline(heavy0, color="#999", lw=1, alpha=0.6)
    ax.set_ylim(-1, y + 1)
    ax.set_xlabel("epoch")
    ax.set_yticks([])
    ax.set_title(f"Warnings (red) vs collapse (black=soft, orange=hard) [{tag}]")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"figs/event_raster_{tag}.png", dpi=140)
    plt.close()


# ---------------------------
# Scheduled-metric helpers (for baselines & safe transforms)
# ---------------------------
def _ffill_scheduled(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    for (seed, factor), g in df.groupby(["seed", "factor"]):
        idx = g.index
        for m in SCHEDULED_METRICS:
            if m in df.columns:
                df.loc[idx, m] = g[m].ffill()
    return df


def _stationaryize_scheduled(df_in: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df_in.copy()
    for (seed, factor), g in df.groupby(["seed", "factor"]):
        idx = g.index
        for m in cols:
            if m in df.columns:
                vals = g[m].astype(float).to_numpy()
                d = np.full_like(vals, np.nan, dtype=float)
                for t in range(1, len(vals)):
                    a, b = vals[t - 1], vals[t]
                    d[t] = (b - a) if (np.isfinite(a) and np.isfinite(b)) else np.nan
                df.loc[idx, m] = d
    return df


# ---------------------------
# Sweep
# ---------------------------


# --- resume completeness helper ---
def _epoch_bounds_csv(path: Path):
    """Return (emin, emax, nrows) for a CSV shard, or (None,None,0) on failure."""
    try:
        df = pd.read_csv(path, usecols=["epoch"])
        return int(df["epoch"].min()), int(df["epoch"].max()), int(len(df))
    except Exception:
        return None, None, 0


def _epoch_bounds(path: Path):
    """Return (emin, emax, nrows) for a parquet shard, or (None,None,0) on failure."""
    try:
        df = pd.read_parquet(path, columns=["epoch"])
        return int(df["epoch"].min()), int(df["epoch"].max()), int(len(df))
    except Exception:
        return None, None, 0


# --- terminal-done marker helper ---
def _has_terminal_done(seed: int, tag: str):
    """Check for terminal marker file written by run_one() when a shard ends early."""
    path = OUTDIR / f"runs_seed{seed}_{tag}.done.json"
    try:
        j = json.loads(path.read_text())
        return bool(j.get("terminal", False)), j
    except Exception:
        return False, {}


def run_sweep(tag: str):
    monitor_ds = None
    if CFG["monitor_source"] == "external":
        # ---- P1: dataset preflight / fallback ----
        try:
            monitor_ds = _stl10_monitor_dataset(CFG["external_monitor"])
        except Exception as e:
            print(f"[WARN] external monitor unavailable ({e!r}); falling back to clean_val.")
            CFG["monitor_source"] = "clean_val"

    # factor mapping across ALL seeds
    all_seeds = sorted(CFG["seeds_calib"] + CFG["seeds_eval"])
    mapping_all = assign_factors_evenly(all_seeds)
    split_map = {int(s): ("calib" if s in CFG["seeds_calib"] else "eval") for s in all_seeds}
    save_json(
    {
    "mapping": {str(k): mapping_all[k]["name"] for k in mapping_all},
    "split": {str(k): split_map[k] for k in split_map},
    },
    OUTDIR / "seed_factor_map.json",
    )

    paths = []
    fail_count = 0

    for seed in all_seeds:
        parq = OUTDIR / f"runs_seed{seed}_{tag}.parquet"
        csvp = parq.with_suffix(".csv")

        # Resume only if shard appears complete; otherwise redo the seed
        if (parq.exists() and parq.stat().st_size > 0) or (
        csvp.exists() and csvp.stat().st_size > 0
        ):
            probe = parq if parq.exists() else csvp
            emin = emax = nrows = None
            if probe.suffix == ".parquet":
                emin, emax, nrows = _epoch_bounds(probe)
            elif probe.suffix == ".csv":
                emin, emax, nrows = _epoch_bounds_csv(probe)
            else:
                emin = emax = None
                nrows = 0

            # complete iff at least 2 rows and max epoch reached (epochs-1)
            complete = (
            (emax is not None)
            and (nrows is not None)
            and (nrows >= 2)
            and (emax == int(CFG.get("epochs", 1)) - 1)
            )

            # CSV fallback completeness: if parquet probe incomplete but CSV exists and is complete, resume from CSV
            if (not complete) and (probe.suffix != ".csv") and csvp.exists():
                emin_csv, emax_csv, nrows_csv = _epoch_bounds_csv(csvp)
                complete_csv = (
                (emax_csv is not None)
                and (nrows_csv is not None)
                and (nrows_csv >= 2)
                and (emax_csv == int(CFG.get("epochs", 1)) - 1)
                )
                if complete_csv:
                    print(
                    f"[resume] skipping seed={seed} (csv complete: emax={emax_csv}, rows={nrows_csv})"
                    )
                    paths.append(csvp)  # aggregator now handles CSV directly
                    continue

            # Terminal marker: skip if present
            done, info = _has_terminal_done(seed, tag)
            if done:
                print(f"[resume] skipping seed={seed} (terminal: {info.get('reason', 'unknown')})")
                if parq.exists():
                    paths.append(parq)
                elif csvp.exists():
                    paths.append(csvp)
                continue

            if complete:
                print(
                f"[resume] skipping seed={seed} ({probe.suffix[1:]} complete: emax={emax}, rows={nrows})"
                )
                paths.append(parq)  # always append the parquet basename for parquet case
                continue
            else:
                print(
                f"[resume] redoing seed={seed} ({probe.suffix[1:]} incomplete: emin={emin}, emax={emax}, rows={nrows})"
                )
                # fall through to run_one

        f = mapping_all[seed]
        print(f"=== Run[{tag}]: seed={seed} factor={f['name']} ===", flush=True)
        try:
            df = run_one(seed, tag=tag, monitor_ds=monitor_ds, factor=f)
            safe_write_parquet(df, parq)
            paths.append(parq)
        except Exception as e:
            fail_count += 1
            errlog = OUTDIR / f"errors_{tag}.log"
            msg = (
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"seed={seed} factor={f['name']} tag={tag} -> {repr(e)}\n"
            )
            try:
                with open(errlog, "a", encoding="utf-8") as fh:
                    fh.write(msg)
            finally:
                pass
            print(f"[WARN] Run failed (seed={seed}): {e}")
            if fail_count >= int(CFG.get("max_failures", 6)):
                print(f"[ABORT] Reached max_failures={CFG.get('max_failures')} — aborting sweep.")
                break

    if not paths:
        print(f"No runs completed for {tag}.")
        return None

    # ---- P0: parquet aggregation fallback (CSV if parquet read fails) ----
    frames = []
    for p in sorted(paths):
        # Case 1: p already points to a CSV
        if p.suffix == ".csv" and p.exists():
            try:
                df = pd.read_csv(p)
                if len(df) > 0:
                    frames.append(df)
                    continue
            except Exception as e:
                print(f"[WARN] failed reading {p.name}: {e}")

        # Case 2: try parquet at the given path
        if p.exists() and p.suffix != ".csv":
            try:
                df = pd.read_parquet(p)
                if len(df) > 0:
                    frames.append(df)
                    continue
            except Exception:
                pass

        # Case 3: fallback to CSV with same basename
        csvp = p.with_suffix(".csv")
        if csvp.exists():
            try:
                df = pd.read_csv(csvp)
                if len(df) > 0:
                    frames.append(df)
                    continue
            except Exception as e:
                print(f"[WARN] failed reading {csvp.name}: {e}")

        print(f"[WARN] missing artifact for {p.stem}; skipping.")

    if not frames:
        print(f"No readable run artifacts for {tag}.")
        return None

    df_all = pd.concat(frames, ignore_index=True)

    out_path = OUTDIR / f"bundle_runs_{tag}.parquet"
    safe_write_parquet(df_all, out_path)

    # Compute checksum for whichever file actually exists (parquet or CSV fallback)
    target = out_path if out_path.exists() else out_path.with_suffix(".csv")
    md5 = file_md5(target)
    Path(str(target) + ".md5").write_text(md5 + "\n")
    print(f"[checksum] {target.name} md5={md5}")
    return df_all



def evaluate(df_all: pd.DataFrame, tag: str):
    # Early guards for empty/epoch-0 aggregates or missing columns
    if df_all is None or len(df_all) == 0 or not {"epoch", "seed"}.issubset(df_all.columns):
        print("[eval] no usable rows or missing epoch/seed; skipping evaluate()")
        return
    try:
        if int(pd.to_numeric(df_all["epoch"], errors="coerce").max()) <= 0:
            print("[eval] aggregate has only epoch 0; skipping evaluate()")
            return
    except Exception:
        pass
    warm_idx = CFG["warmup"] + CFG["ph_burn"]
    cal_mask = df_all["seed"].isin(CFG["seeds_calib"])
    eval_seeds = seeds_for_eval_from_env(CFG)
    ev_mask = df_all["seed"].isin(eval_seeds)
    df_cal_raw = df_all[cal_mask].copy()
    df_eval_raw = df_all[ev_mask].copy()
    no_eval = (df_eval_raw is None) or getattr(df_eval_raw, "empty", True)
    if no_eval:
        print(
        "[eval] no evaluation seeds present; will skip learned/baseline evaluation but still write calibration aggregates."
        )

    # ---- Initial GT (t_c0/ctag0 with infinite cutoff) ----
    def add_gt(df_in):
        rows = []
        for (seed, factor), g in df_in.groupby(["seed", "factor"]):
            g = g.sort_values("epoch").copy()
            t_c0, ctag0 = gt_collapse_time(g, grad_cutoff=np.inf)
            g["t_collapse_gt"] = t_c0
            g["collapse_tag_gt"] = ctag0
            rows.append(g)
        return pd.concat(rows, ignore_index=True) if rows else df_in

    df_cal_raw = add_gt(df_cal_raw)

    # ---- Calibrate grad explosion cutoff per-factor from healthy epochs (robust) ----
    grad_cut_by_factor = calibrate_grad_cutoff_per_factor(df_cal_raw)
    save_json(grad_cut_by_factor, OUTDIR / f"grad_cutoff_by_factor_{tag}.json")

    # ---- FINAL GT with per-factor cutoff ----
    def add_gt_final(df_in):
        # Ensure GT columns exist even on empty frames
        if df_in is None or len(df_in) == 0:
            df_empty = df_in.copy() if isinstance(df_in, pd.DataFrame) else pd.DataFrame()
            if "collapse_tag_gt" not in df_empty.columns:
                df_empty["collapse_tag_gt"] = []
            if "t_collapse_gt" not in df_empty.columns:
                df_empty["t_collapse_gt"] = []
            return df_empty
        rows = []
        for (seed, factor), g in df_in.groupby(["seed", "factor"]):
            g = g.sort_values("epoch").copy()
            cutoff = grad_cut_by_factor.get(str(factor), np.inf)
            t_c, ctag = gt_collapse_time(g, grad_cutoff=cutoff)
            g["t_collapse_gt"] = t_c
            g["collapse_tag_gt"] = ctag
            rows.append(g)
        return pd.concat(rows, ignore_index=True) if rows else df_in

    # ---- Soft GT calibration (eff_dim_gt): env override priority ----
    # Priority: explicit fixed threshold > quantile from env (SCAR_SOFT_Q) > default calibration
    fixed_thr_env = os.environ.get("SCAR_FIXED_GT_RANK_MIN", "").strip()
    soft_q_env = os.environ.get("SCAR_SOFT_Q", "").strip()
    gtq = float(CFG.get("gt_rank_q", 0.075))

    def _soft_quantile_from_controls(df_src, q, warm_idx):
        """Compute q-quantile of eff_dim_gt (preferred) or eff_dim on control ('none') after warm_idx.
        Robust to missing columns; returns None if not computable.
        """
        # Require control columns to exist
        if df_src is None or not isinstance(df_src, pd.DataFrame):
            return None
        if ("factor" not in df_src.columns) or ("epoch" not in df_src.columns):
            return None

        # Filter to controls after warm_idx (coerce epochs to numeric)
        try:
            ep = pd.to_numeric(df_src["epoch"], errors="coerce")
            base = df_src[(df_src["factor"] == "none") & (ep >= int(warm_idx))]
        except Exception:
            return None

        # Prefer eff_dim_gt; fall back to eff_dim; bail if neither exists
        if "eff_dim_gt" in base.columns:
            series = pd.to_numeric(base["eff_dim_gt"], errors="coerce")
        elif "eff_dim" in base.columns:
            series = pd.to_numeric(base["eff_dim"], errors="coerce")
        else:
            return None

        # Drop NaNs and compute quantile
        series = series.dropna()
        if series.empty:
            return None
        arr = series.to_numpy()
        try:
            return float(np.quantile(arr, float(q)))
        except TypeError:
            # numpy <1.22 compatibility
            return float(np.quantile(arr, float(q), interpolation="linear"))
        except Exception:
            return None

    if fixed_thr_env:
        try:
            CFG["gt_rank_min"] = float(fixed_thr_env)
            save_json(
            {
            "gt_rank_min": CFG["gt_rank_min"],
            "gt_rank_q": None,
            "warm": int(warm_idx),
            "source": "env_fixed",
            },
            OUTDIR / f"gt_rank_calibration_{tag}.json",
            )
            print(f"[calib] gt_rank_min fixed from env: {CFG['gt_rank_min']:.3f}")
        except Exception as e:
            print(f"[calib] bad SCAR_FIXED_GT_RANK_MIN={fixed_thr_env!r}: {e} — falling back to quantile/env/default")
            fixed_thr_env = ""

    if not fixed_thr_env and soft_q_env:
        try:
            q_env = float(soft_q_env)
            assert 0.0 < q_env < 1.0
            qv = _soft_quantile_from_controls(df_cal_raw, q_env, warm_idx)
            if qv is not None and np.isfinite(qv):
                CFG["gt_rank_min"] = float(qv)
                save_json(
                {
                "gt_rank_min": CFG["gt_rank_min"],
                "gt_rank_q": q_env,
                "warm": int(warm_idx),
                "source": "env_quantile",
                },
                OUTDIR / f"gt_rank_calibration_{tag}.json",
                )
                print(f"[calib] gt_rank_min from env quantile q={q_env:.2f}: {CFG['gt_rank_min']:.3f}")
            else:
                print("[calib] no 'none' rows post-warm for SCAR_SOFT_Q; using default soft GT calibration")
                soft_q_env = ""
        except Exception as e:
            print(f"[WARN] SCAR_SOFT_Q invalid ({soft_q_env}): {e}; using default soft GT calibration")
            soft_q_env = ""

    if not fixed_thr_env and not soft_q_env:
        # Default: derive from CFG['gt_rank_q'] on calibration controls after warm
        qv = _soft_quantile_from_controls(df_cal_raw, gtq, warm_idx)
        if qv is None or (not np.isfinite(qv)):
            print("[calib] WARN: gt_rank_min quantile is NaN/None; using prior CFG['gt_rank_min'].")
            qv = float(CFG.get("gt_rank_min", 8.0))
        CFG["gt_rank_min"] = float(qv)
        save_json(
        {
        "gt_rank_min": CFG["gt_rank_min"],
        "gt_rank_q": gtq,
        "warm": int(warm_idx),
        "source": "quantile",
        },
        OUTDIR / f"gt_rank_calibration_{tag}.json",
        )
        print(f"[calib] gt_rank_min set from factor=='none' @q={gtq:.3f} after warm={warm_idx}: {CFG['gt_rank_min']:.3f}")

    # Compute GT exactly once using this finalized threshold
    df_cal_raw = add_gt_final(df_cal_raw)
    df_eval_raw = add_gt_final(df_eval_raw)

    # sanity: fail fast if GT didn't stick
    assert "collapse_tag_gt" in df_cal_raw.columns, "add_gt_final failed to stamp GT on df_cal_raw"
    if "collapse_tag_gt" not in df_eval_raw.columns:
        print(
        "[eval] WARN: add_gt_final did not stamp GT on df_eval_raw; continuing with empty eval."
        )

    # ---- PH direction calibration on SOFT collapses only (pre-onset only) ----
    soft_cal = df_cal_raw[df_cal_raw["collapse_tag_gt"] == "soft"]
    ph_metrics = [
    "cos_disp",
    "var_out_k",
    "eff_dim",
    "ftle",
    "ftle_lowent",
    "mon_entropy",
    "ewma_loss",
    "sma_loss",
    "sw2",
    "pers_H0",
    ]
    dir_map = calibrate_ph_directions(soft_cal, ph_metrics)
    save_json(dir_map, OUTDIR / f"ph_directions_{tag}.json")

    # ---- Run-level collapse tag map for evaluation (needed by summarize_detection) ----
    collapse_tag_map = {}
    for (sd, fc), g in df_eval_raw.groupby(["seed", "factor"]):
        ctag_this = "none"
        if "collapse_tag_gt" in g.columns:
            gx = g[g["epoch"] >= warm_idx]
            vals = gx["collapse_tag_gt"].dropna().astype(str).unique().tolist()
            if not vals:
                vals = g["collapse_tag_gt"].dropna().astype(str).unique().tolist()
            if vals:
                ctag_this = vals[0]
        elif "is_collapse_epoch_gt" in g.columns and bool(g["is_collapse_epoch_gt"].any()):
            ctag_this = "soft"
        collapse_tag_map[(int(sd), str(fc))] = ctag_this

    # ---- RP adequacy on eval: JL/native agreement pre-warm ----
    rp_flags = rp_adequacy_flags(
        df_eval_raw,
        warm_idx,
        corr_min=float(CFG.get("rp_corr_min", 0.9)),
        min_pts=int(CFG.get("rp_min_pts", 8)),
    )
    save_json(
    {
    "corr_min": float(CFG.get("rp_corr_min", 0.9)),
    "min_pts": int(CFG.get("rp_min_pts", 8)),
    "flags": {f"s{seed}-{factor}": int(flag) for (seed, factor), flag in rp_flags.items()},
    },
    OUTDIR / f"rp_adequacy_{tag}.json",
    )

    # ---- Baselines (PH on various pools) using unified GT ----
    tr_vote_ev, _ = compute_events(df_eval_raw, metrics_for_ph=VOTE_METRICS, dir_map=dir_map)

    dir_ewma = calibrate_ph_directions(soft_cal, ["ewma_loss"])
    dir_sma = calibrate_ph_directions(soft_cal, ["sma_loss"])
    tr_ewma, _ = compute_events(df_eval_raw, metrics_for_ph=["ewma_loss"], dir_map=dir_ewma)
    tr_sma, _ = compute_events(df_eval_raw, metrics_for_ph=["sma_loss"], dir_map=dir_sma)

    # ---- Prepare learned detector data (schedule-independent features only) ----
    det_features = [
    "cos_disp",
    "var_out_k",
    "ftle",
    "ftle_lowent",
    # slope/coherence channels
    "slope_eff_dim",
    "slope_var_out_k",
    "reg_eff_dim",
    "reg_var_out_k",
    "corr_effdim_varoutk",
    ]

    df_cal_det = df_cal_raw.copy()
    df_eval_det = df_eval_raw.copy()

    # Build training matrix on calibration runs — negatives restricted to pre-onset for collapsed runs
    X_list = []
    M_list = []
    y_list = []
    g_list = []
    fp_mask_list = []
    ep_list = []
    tc_soft_list = []
    for (seed, factor), g in df_cal_det.groupby(["seed", "factor"]):
        g = g.sort_values("epoch").copy()
        t_c = g["t_collapse_gt"].iloc[0]
        ctag = g["collapse_tag_gt"].iloc[0]
        is_nontrig = ctag == "none"
        if pd.notna(t_c) and ctag in ("soft", "hard"):
            g = g[g["epoch"] < int(t_c)].copy()
        X, M = _metrics_matrix_with_missing(g, det_features)
        y = np.zeros(len(g), dtype=np.float32)
        if pd.notna(t_c) and ctag == "soft":
            for t in range(len(g)):
                ep = int(g.iloc[t]["epoch"])
                if ep >= warm_idx and (int(t_c) - ep <= CFG["detector_horizon"]):
                    y[t] = 1.0
        grp = np.full(len(g), int(seed), dtype=np.int64)
        X_list.append(X)
        M_list.append(M)
        y_list.append(y)
        g_list.append(grp)
        fp_mask = ((g["epoch"].to_numpy() >= warm_idx) & is_nontrig).astype(np.float32)
        fp_mask_list.append(fp_mask)
        ep_list.append(g["epoch"].to_numpy().astype(np.int64))
        tc_soft_list.append(
            np.full(
                len(g), (int(t_c) if (pd.notna(t_c) and ctag == "soft") else np.nan), dtype=float
            )
        )

    X_raw = (
    np.concatenate(X_list, 0) if X_list else np.zeros((0, len(det_features)), dtype=np.float32)
    )
    y = np.concatenate(y_list, 0) if y_list else np.zeros((0,), dtype=np.float32)
    groups = np.concatenate(g_list, 0) if g_list else np.zeros((0,), dtype=np.int64)
    fp_mask = (
    np.concatenate(fp_mask_list, 0) if fp_mask_list else np.zeros_like(y, dtype=np.float32)
    )
    ep_arr = np.concatenate(ep_list, 0) if ep_list else np.zeros((0,), dtype=np.int64)
    tc_arr = np.concatenate(tc_soft_list, 0) if tc_soft_list else np.zeros((0,), dtype=float)

    # Anchor normalization to healthy geometry: compute q10–q90 bounds on factor=='none' after warm
    none_pool = df_cal_raw[(df_cal_raw["factor"] == "none") & (df_cal_raw["epoch"] >= warm_idx)]
    bounds = {}
    for m in det_features:
        if m in none_pool.columns:
            col = none_pool[m].to_numpy(dtype=float)
            col = col[np.isfinite(col)]
            if col.size >= 8:
                bounds[m] = (float(np.nanpercentile(col, 10)), float(np.nanpercentile(col, 90)))
            else:
                bounds[m] = (float("-inf"), float("inf"))
        else:
            bounds[m] = (float("-inf"), float("inf"))

    # Drop post-warm rows from non-'none' factors whose post-warm median lies outside healthy bounds for any feature
    def _factor_out_of_bounds(df_fac):
        med_ok = True
        post = df_fac[df_fac["epoch"] >= warm_idx]
        if post.empty:
            return False
        for m in det_features:
            if m in post.columns:
                med = np.nanmedian(post[m].to_numpy(dtype=float))
                lo, hi = bounds.get(m, (-np.inf, np.inf))
                if np.isfinite(med) and (med < lo or med > hi):
                    med_ok = False
                    break
        return not med_ok

    drop_idx = []
    for (sd, fc), g in df_cal_raw.groupby(["seed", "factor"]):
        if fc != "none" and _factor_out_of_bounds(g):
            drop_idx.extend(g[g["epoch"] >= warm_idx].index.tolist())
    df_cal_for_norm = df_cal_raw.drop(index=drop_idx) if drop_idx else df_cal_raw

    # (duplicate quantile-based soft-GT threshold block removed)

    stats = _fit_global_robust_norm_precollapse(df_cal_for_norm, det_features)
    Xn = _apply_global_norm_impute(X_raw, stats, det_features)
    X_det = Xn  # det_use_missing=False

    best = _cv_grouped_fit(
        X_det,
        y,
        groups,
        fp_mask,
        steps_grid=CFG["detector_steps_grid"],
        l2_grid=CFG["detector_L2_grid"],
        lr=CFG["detector_lr"],
        folds=CFG["detector_cv_folds"],
        fp_cap=SUCCESS_TARGET["max_early_fp_rate"],
        epochs=ep_arr,
        tc_soft=tc_arr,
        warm_idx=warm_idx,
    )
    if best is None:
        steps, l2, thresh = CFG["detector_steps_grid"][0], CFG["detector_L2_grid"][0], 0.5
    else:
        steps, l2, thresh = best

    w, b = _train_logistic_ridge_balanced(
    X_det, y, groups, steps=steps, lr=CFG["detector_lr"], l2=l2
    )
    model_info = dict(
    weights=w.tolist(),
    bias=float(b),
    thresh=float(thresh),
    features=det_features,
    horizon=int(CFG["detector_horizon"]),
    norm_stats=stats,
    steps=steps,
    l2=l2,
    scheduled_metrics=[],
    det_use_missing=False,
    ph_win=CFG["ph_win"],
    ph_lambda=CFG["ph_lambda"],
    ph_two_sided=int(CFG["ph_two_sided"]),
    )
    save_json(model_info, OUTDIR / f"learned_detector_{tag}.json")

    # --- τ→τ′ mapping under the deployed gate (deterministic; calibration-only) ---
    # Build calibration meta rows and raw feature matrix in deterministic (seed,factor,epoch) order
    meta_rows = []
    X_parts = []
    for (sd, fc), g in df_cal_det.groupby(["seed", "factor"]):
        gg = g.sort_values("epoch").reset_index(drop=True)
        X_raw_g, _ = _metrics_matrix_with_missing(gg, det_features)
        X_parts.append(X_raw_g)
        for _, r in gg.iterrows():
            meta_rows.append(
            {"seed": int(r["seed"]), "factor": str(r["factor"]), "epoch": int(r["epoch"])}
            )
    X_cal_raw = (
    np.concatenate(X_parts, axis=0).astype(np.float32)
    if X_parts
    else np.zeros((0, len(det_features)), dtype=np.float32)
    )
    meta_rows = pd.DataFrame(meta_rows)

    # Normalize with the same robust stats and compute deterministic OOF probabilities for fixed (steps,l2)
    # Build y_cal / groups_cal aligned 1:1 with meta_rows/X_cal_raw
    Xn_cal = _apply_global_norm_impute(X_cal_raw.copy(), stats, det_features)
    y_cal = np.zeros(len(meta_rows), dtype=np.float32)
    groups_cal = meta_rows["seed"].to_numpy(dtype=np.int64)

    horiz = int(CFG["detector_horizon"])

    # Precompute per-(seed,factor) collapse target for labeling positives
    gt_by_sf = {}
    for (sd, fc), g in df_cal_det.groupby(["seed", "factor"]):
        try:
            t_c = g["t_collapse_gt"].iloc[0]
            ctag = str(g["collapse_tag_gt"].iloc[0])
        except Exception:
            t_c, ctag = (np.nan, "none")
        gt_by_sf[(int(sd), str(fc))] = (t_c, ctag)

    # Label positives only for soft collapses within the horizon after warm
    for i, r in meta_rows.iterrows():
        sd = int(r["seed"])
        fc = str(r["factor"])
        ep = int(r["epoch"])
        t_c, ctag = gt_by_sf.get((sd, fc), (np.nan, "none"))
        if pd.notna(t_c) and ctag == "soft" and ep >= warm_idx and (int(t_c) - ep) <= horiz:
            y_cal[i] = 1.0

    p_oof = _oof_probs_for_params(
    Xn_cal,
    y_cal,
    groups_cal,
    steps=steps,
    l2=l2,
    lr=CFG["detector_lr"],
    folds=CFG["detector_cv_folds"],
    )

    # RP adequacy flags on calibration
    rp_flags_cal = rp_adequacy_flags(
        df_cal_raw,
        warm_idx,
        corr_min=float(CFG.get("rp_corr_min", 0.9)),
        min_pts=int(CFG.get("rp_min_pts", 8)),
    )

    # Family-gate parameters
    z_thr = float(CFG.get("family_z_thr", 1.5))
    K = int(CFG.get("family_window", 1))
    warn_consec = int(CFG.get("warn_consec", 3))
    fp_cap = float(SUCCESS_TARGET["max_early_fp_rate"])

    tau_prime, fp_measured = map_threshold_to_gated_fp(
        meta_rows,
        X_cal_raw,
        p_oof,
        det_features,
        stats,
        dir_map,
        rp_flags_cal,
        warm_idx,
        z_thr,
        K,
        warn_consec,
        fp_cap,
    )

    # Persist mapping info and adopt τ′ for evaluation
    save_json(
        {
            "tau_raw": float(thresh),
            "tau_prime": float(tau_prime),
            "fp_measured_cal_gated": float(fp_measured),
            "fp_cap": float(fp_cap),
            "family_z_thr": float(z_thr),
            "warm": int(warm_idx),
            "steps": int(steps),
            "l2": float(l2),
        },
        OUTDIR / f"tau_mapping_{tag}.json",
    )

    # Update learned_detector to mirror deployed threshold (store both raw and deployed τ)
    try:
        # preserve the pre-mapping value as tau_raw if available; fall back to current thresh
        model_info["tau_raw"] = float(model_info.get("thresh", thresh))
        model_info["thresh"] = float(tau_prime)
        save_json(model_info, OUTDIR / f"learned_detector_{tag}.json")
    except Exception as e:
        print(f"[WARN] failed to update learned_detector artifact with tau_prime: {e}")

    thresh = float(tau_prime)

    # ---- Apply learned detector to evaluation runs ----
    rows = []
    for (seed, factor), g in df_eval_det.groupby(["seed", "factor"]):
        gg = g.sort_values("epoch").reset_index(drop=True)
        X_raw, M_ind = _metrics_matrix_with_missing(gg, det_features)
        Xn = _apply_global_norm_impute(X_raw, stats, det_features)
        z = (Xn @ w + b)
        p = _sigmoid_stable(z)
        mask = (gg["epoch"] >= warm_idx) & (p >= thresh)
        hit_idx = np.where(mask.values)[0]

        # Build per-epoch z-scores for detector features using the same robust stats
        Z_eval = X_raw.copy()
        for j, m in enumerate(det_features):
            med, scale = stats.get(m, (0.0, 1.0))
            if scale <= 0:
                scale = 1.0
            col = Z_eval[:, j]
            col = np.where(np.isfinite(col), col, med)
            Z_eval[:, j] = (col - med) / scale

        # Family definitions over det_features only
        col_to_idx = {m: j for j, m in enumerate(det_features)}
        geom_cols = [c for c in ["cos_disp", "var_out_k"] if c in col_to_idx]
        dyn_cols = [c for c in ["ftle", "ftle_lowent"] if c in col_to_idx]
        z_thr = float(CFG.get("family_z_thr", 1.5))
        K = int(CFG.get("family_window", 1))

        def _fam_alarm(epoch_idx: int, cols: List[str]) -> bool:
            if not cols:
                return False
            lo, hi = max(0, epoch_idx - K), min(Z_eval.shape[0] - 1, epoch_idx + K)
            for m in cols:
                d = dir_map.get(m, None)
                if d is None:  # unknown direction -> skip
                    continue
                j = col_to_idx[m]
                zwin = Z_eval[lo : hi + 1, j]
                if d == "up" and np.isfinite(zwin).any() and np.nanmax(zwin) >= z_thr:
                    return True
                if d == "down" and np.isfinite(zwin).any() and np.nanmin(zwin) <= -z_thr:
                    return True
            return False

        # RP adequacy: if JL under-resolved for this (seed,factor), require a non-geometry corroboration
        rp_under = bool(rp_flags.get((int(gg["seed"].iloc[0]), str(gg["factor"].iloc[0])), 0))

        K_warn = int(CFG.get("warn_consec", 3))
        j_end = _first_run_end(hit_idx, K_warn)
        t_warn = None
        if j_end >= 0:
            i = hit_idx[j_end]
            geom_ok = _fam_alarm(i, geom_cols)
            dyn_ok = _fam_alarm(i, dyn_cols)
            gate_ok = (dyn_ok if rp_under else (geom_ok or dyn_ok))
            if gate_ok:
                t_warn = int(gg.iloc[i]["epoch"])

        t_c, ctag = (
        int(gg["t_collapse_gt"].iloc[0]) if pd.notna(gg["t_collapse_gt"].iloc[0]) else None
        ), str(gg["collapse_tag_gt"].iloc[0])

        if t_warn is not None and t_c is not None and not (t_warn < t_c):
            t_warn = None

        lead = (
        float(t_c - t_warn)
        if (ctag in ["soft", "hard"] and t_warn is not None and t_c is not None)
        else float("nan")
        )
        rows.append(
        dict(
        run_id=f"s{int(seed)}-{str(factor)}",
        seed=int(seed),
        factor=str(factor),
        t_warn=t_warn,
        t_collapse=t_c,
        collapse_tag=ctag,
        ph_win=CFG["ph_win"],
        ph_lambda=CFG["ph_lambda"],
        ph_two_sided=int(CFG["ph_two_sided"]),
        warn_vote=CFG["warn_vote"],
        heavy_every=CFG["heavy_every"],
        metric_batches=CFG["metric_batches"],
        var_k_energy=CFG["var_k_energy"],
        var_k_max=CFG["var_k_max"],
        lead_time=lead,
        rp_under_resolved=int(rp_under),
        )
        )
    det_rows = pd.DataFrame(rows)
    det_rows.to_csv(OUTDIR / f"detector_events_{tag}.csv", index=False)

    # ---- Baseline postprocess (PH tables → canonical columns) ----
    def quorum_time(r, metrics, k):
        ts_all = []
        for m in metrics:
            v = r.get(f"t_{m}")
            if pd.notna(v):
                ts_all.append(int(v))
        if not ts_all:
            return None
        ts_sorted = sorted(ts_all)
        for t in ts_sorted:
            if sum(tt <= t for tt in ts_all) >= k:
                return t
        return None

    def postprocess(tr_df, name):
        tcol = _first_t_column(tr_df)
        rows = []
        for _, r in tr_df.iterrows():
            if name == "vote":
                t_warn = quorum_time(r, VOTE_METRICS, CFG["warn_vote"])
            else:
                t_warn = int(r[tcol]) if (tcol and pd.notna(r.get(tcol))) else None
            t_c = int(r["t_collapse"]) if pd.notna(r["t_collapse"]) else None
            if t_warn is not None and t_c is not None and not (t_warn < t_c):
                t_warn = None
            rows.append(
            dict(
            run_id=r["run_id"],
            seed=int(r["seed"]),
            factor=str(r["factor"]),
            t_warn=t_warn,
            t_collapse=t_c,
            collapse_tag=str(r["collapse_tag"]),
            ph_win=r["ph_win"],
            ph_lambda=r["ph_lambda"],
            ph_two_sided=r["ph_two_sided"],
            warn_vote=CFG["warn_vote"],
            heavy_every=CFG["heavy_every"],
            metric_batches=CFG["metric_batches"],
            var_k_energy=CFG["var_k_energy"],
            var_k_max=CFG["var_k_max"],
            lead_time=(
            float(t_c - t_warn)
            if (t_warn is not None and t_c is not None)
            else float("nan")
            ),
            )
            )
        out = pd.DataFrame(rows)
        out.to_csv(OUTDIR / f"baseline_events_{name}_{tag}.csv", index=False)
        return out

    base_ewma = postprocess(tr_ewma, "ewma")
    base_sma = postprocess(tr_sma, "sma")
    base_vote = postprocess(tr_vote_ev, "vote")

    # ---- Sequential PH + rank-drop vote baseline ('seq') ----
    lam = CFG.get("seq_cusum_lambda", CFG["ph_lambda"])
    win_short = CFG.get("ph_win_short", 8)
    rank_win = CFG.get("rank_win", 8)

    seq_rows = []
    for (seed, factor), g in df_eval_raw.groupby(["seed", "factor"]):
        gg = g.sort_values("epoch").reset_index(drop=True)
        # Unified PH preprocessing for pers_H0
        xs_full = _prep_series_for_ph(gg, "pers_H0")
        # Indices eligible for sequential tests: valid finite values at/after warm_idx
        idxs = [
            i
            for i, x in enumerate(xs_full)
            if (np.isfinite(x) and int(gg.iloc[i]["epoch"]) >= warm_idx)
        ]
        ph_seq = [xs_full[i] for i in idxs]

        # Level test over preprocessed series (burn-in respects epoch timeline)
        t_level, _, _ = ph_window_sparse(
            xs_full,
            win=win_short,
            lam=lam,
            direction="down",
            burn_in=warm_idx,
            min_points=CFG["ph_min_points"],
            two_sided=False,
        )
        # CUSUM on the post-warm valid subsequence
        zs_delta = robust_z_series(_delta(ph_seq), win=win_short, burn_in=0)
        t_cusum, _ = cusum_one_sided(zs_delta, lam=lam, direction="down")

        def _map_sub_to_epoch(t_sub):
            return (
                None
                if t_sub is None
                else int(gg.iloc[idxs[t_sub]]["epoch"]) if (0 <= t_sub < len(idxs)) else None
            )

        t_ph = None
        cand = [t for t in [t_level, t_cusum] if t is not None]
        if cand:
            t_ph = _map_sub_to_epoch(min(cand))

        rk_seq = (
            gg["eff_dim_gt"].astype(float).tolist()
            if "eff_dim_gt" in gg.columns
            else gg["eff_dim"].astype(float).tolist()
        )
        t_rank, _, _ = ph_window_sparse(
            rk_seq,
            win=rank_win,
            lam=lam,
            direction="down",
            burn_in=warm_idx,
            min_points=CFG["ph_min_points"],
            two_sided=False,
        )

        t_warn = None
        if (t_ph is not None) and (t_rank is not None):
            t_warn = max(int(t_ph), int(t_rank))

        t_c = int(gg["t_collapse_gt"].iloc[0]) if pd.notna(gg["t_collapse_gt"].iloc[0]) else None
        ctag = str(gg["collapse_tag_gt"].iloc[0])

        if t_warn is not None and t_c is not None and not (t_warn < t_c):
            t_warn = None

        lead = (
            float(t_c - t_warn)
            if (ctag in ["soft", "hard"] and t_warn is not None and t_c is not None)
            else float("nan")
        )
        seq_rows.append(
            dict(
                run_id=f"s{int(seed)}-{str(factor)}",
                seed=int(seed),
                factor=str(factor),
                t_warn=t_warn,
                t_collapse=t_c,
                collapse_tag=ctag,
                ph_win_short=win_short,
                ph_lambda=lam,
                rank_win=rank_win,
                heavy_every=CFG["heavy_every"],
                metric_batches=CFG["metric_batches"],
                var_k_energy=CFG["var_k_energy"],
                var_k_max=CFG["var_k_max"],
                lead_time=lead,
            )
        )
    seq_events = pd.DataFrame(seq_rows)
    seq_events.to_csv(OUTDIR / f"baseline_events_seq_{tag}.csv", index=False)

    # ---- NEWMA baseline on PH + non-scheduled spectral/FTLE channels ----
    newma_rows = []
    for (seed, factor), g in df_eval_raw.groupby(["seed", "factor"]):
        gg = g.sort_values("epoch").reset_index(drop=True)

        def _mask_with_valid(series, valid):
            a = np.asarray(series, dtype=float)
            if valid is None:
                return a
            v = np.asarray(valid, dtype=bool)
            out = a.copy()
            out[~v] = np.nan
            return out

        xs_ph = _prep_series_for_ph(gg, "pers_H0")
        xs_eff = (
            (gg["eff_dim_gt"] if "eff_dim_gt" in gg.columns else gg["eff_dim"])
            .to_numpy()
            .astype(float)
            .tolist()
        )
        xs_var = gg["var_out_k"].to_numpy().astype(float).tolist()
        xs_ftle = _mask_with_valid(
            gg["ftle"], gg["ftle_valid"] if "ftle_valid" in gg.columns else None
        ).tolist()
        t1 = newma_warn_epoch(
            xs_ph,
            fast=CFG["newma_fast"],
            slow=CFG["newma_slow"],
            lam=CFG["ph_lambda"],
            burn_in=warm_idx,
        )
        t2 = newma_warn_epoch(
            xs_eff,
            fast=CFG["newma_fast"],
            slow=CFG["newma_slow"],
            lam=CFG["ph_lambda"],
            burn_in=warm_idx,
        )
        t3 = newma_warn_epoch(
            xs_var,
            fast=CFG["newma_fast"],
            slow=CFG["newma_slow"],
            lam=CFG["ph_lambda"],
            burn_in=warm_idx,
        )
        t4 = newma_warn_epoch(
            xs_ftle,
            fast=CFG["newma_fast"],
            slow=CFG["newma_slow"],
            lam=CFG["ph_lambda"],
            burn_in=warm_idx,
        )
        cand = [t for t in [t1, t2, t3, t4] if t is not None]
        t_warn = int(min(cand)) if cand else None
        t_c = int(gg["t_collapse_gt"].iloc[0]) if pd.notna(gg["t_collapse_gt"].iloc[0]) else None
        ctag = str(gg["collapse_tag_gt"].iloc[0])
        if t_warn is not None and t_c is not None and not (t_warn < t_c):
            t_warn = None
        lead = (
            float(t_c - t_warn)
            if (ctag in ["soft", "hard"] and t_warn is not None and t_c is not None)
            else float("nan")
        )
        newma_rows.append(
            dict(
                run_id=f"s{int(seed)}-{str(factor)}",
                seed=int(seed),
                factor=str(factor),
                t_warn=t_warn,
                t_collapse=t_c,
                collapse_tag=ctag,
                ph_lambda=CFG["ph_lambda"],
                heavy_every=CFG["heavy_every"],
                metric_batches=CFG["metric_batches"],
                var_k_energy=CFG["var_k_energy"],
                var_k_max=CFG["var_k_max"],
                lead_time=lead,
            )
        )
    base_newma = pd.DataFrame(newma_rows)
    base_newma.to_csv(OUTDIR / f"baseline_events_newma_{tag}.csv", index=False)

    # ---- Gate-only baseline (uses per-epoch gate_warn) ----
    gate_rows = []
    for (seed, factor), g in df_eval_raw.groupby(["seed", "factor"]):
        gg = g.sort_values("epoch").reset_index(drop=True)
        warm_idx_local = CFG["warmup"] + CFG["ph_burn"]

        # Determine first post-warm epoch where gate_warn == 1
        gw = None
        if "gate_warn" in gg.columns:
            hits = gg[(gg["epoch"] >= warm_idx_local) & (gg["gate_warn"].astype(float) >= 1.0)]
            if len(hits) > 0:
                gw = int(hits.iloc[0]["epoch"])

        t_warn = gw if gw is not None else None
        t_c = int(gg["t_collapse_gt"].iloc[0]) if pd.notna(gg["t_collapse_gt"].iloc[0]) else None
        ctag = str(gg["collapse_tag_gt"].iloc[0])

        # Enforce causal ordering
        if t_warn is not None and t_c is not None and not (t_warn < t_c):
            t_warn = None

        lead = (
            float(t_c - t_warn)
            if (ctag in ["soft", "hard"] and t_warn is not None and t_c is not None)
            else float("nan")
        )
        gate_rows.append(
            dict(
                run_id=f"s{int(seed)}-{str(factor)}",
                seed=int(seed),
                factor=str(factor),
                t_warn=t_warn,
                t_collapse=t_c,
                collapse_tag=ctag,
                ph_lambda=CFG["ph_lambda"],
                heavy_every=CFG["heavy_every"],
                metric_batches=CFG["metric_batches"],
                var_k_energy=CFG["var_k_energy"],
                var_k_max=CFG["var_k_max"],
                lead_time=lead,
            )
        )
    base_gate = pd.DataFrame(gate_rows)
    base_gate.to_csv(OUTDIR / f"baseline_events_gate_{tag}.csv", index=False)

    # ---- Summaries ----
    summ_det = summarize_detection(det_rows, warm_idx=warm_idx)
    summ_ewma = summarize_detection(base_ewma, warm_idx=warm_idx)
    summ_sma = summarize_detection(base_sma, warm_idx=warm_idx)
    summ_vote = summarize_detection(base_vote, warm_idx=warm_idx)
    summ_seq = summarize_detection(seq_events, warm_idx=warm_idx)
    summ_newma = summarize_detection(base_newma, warm_idx=warm_idx)
    summ_gate = summarize_detection(base_gate, warm_idx=warm_idx)
    summ_gate.to_csv(OUTDIR / f"summary_baseline_gate_{tag}.csv", index=False)

    # ---- Run-level FP (any warn on 'none' after warm) ----
    run_fp_learned = float("nan") if no_eval else summarize_runlevel_fp(det_rows, warm_idx=warm_idx+1)
    run_fp_ewma = summarize_runlevel_fp(base_ewma, warm_idx=warm_idx+1)
    run_fp_sma = summarize_runlevel_fp(base_sma, warm_idx=warm_idx+1)
    run_fp_vote = summarize_runlevel_fp(base_vote, warm_idx=warm_idx+1)
    run_fp_seq = summarize_runlevel_fp(seq_events, warm_idx=warm_idx+1)
    run_fp_newma = summarize_runlevel_fp(base_newma, warm_idx=warm_idx+1)
    run_fp_gate = summarize_runlevel_fp(base_gate, warm_idx=warm_idx+1)

    summ_det.to_csv(OUTDIR / f"summary_learned_{tag}.csv", index=False)
    summ_ewma.to_csv(OUTDIR / f"summary_baseline_ewma_{tag}.csv", index=False)
    summ_sma.to_csv(OUTDIR / f"summary_baseline_sma_{tag}.csv", index=False)
    summ_vote.to_csv(OUTDIR / f"summary_baseline_vote_{tag}.csv", index=False)
    summ_seq.to_csv(OUTDIR / f"summary_baseline_seq_{tag}.csv", index=False)
    summ_newma.to_csv(OUTDIR / f"summary_baseline_newma_{tag}.csv", index=False)

    print("\n[LEARNED DETECTOR] per-run summary:\n", summ_det)
    print("\n[BASELINE EWMA(loss)] per-run summary:\n", summ_ewma)
    print("\n[BASELINE SMA(loss)] per-run summary:\n", summ_sma)
    print("\n[BASELINE vote] per-run summary:\n", summ_vote)
    print("\n[BASELINE seq (ΔPH + rank)] per-run summary:\n", summ_seq)
    print("\n[BASELINE NEWMA (PH + eff_dim/var/ftle)] per-run summary:\n", summ_newma)
    print(f"[LEARNED] run-level FP: {run_fp_learned:.3f}")
    print(f"[EWMA] run-level FP: {run_fp_ewma:.3f}")
    print(f"[SMA] run-level FP: {run_fp_sma:.3f}")
    print(f"[VOTE] run-level FP: {run_fp_vote:.3f}")
    print(f"[SEQ] run-level FP: {run_fp_seq:.3f}")
    print(f"[NEWMA] run-level FP: {run_fp_newma:.3f}")
    print("\n[BASELINE GATE (gate_warn)] per-run summary:\n", summ_gate)
    print(f"[GATE] run-level FP: {run_fp_gate:.3f}")

    # ---- Stamp epoch overlays (learned + baselines) & assert consistency ----
    df_eval_overlay = df_eval_raw.copy()
    df_eval_overlay = mark_events_epochwise(df_eval_overlay, det_rows, prefix="learned")
    df_eval_overlay = mark_events_epochwise(df_eval_overlay, base_ewma, prefix="ewma")
    df_eval_overlay = mark_events_epochwise(df_eval_overlay, base_sma, prefix="sma")
    df_eval_overlay = mark_events_epochwise(df_eval_overlay, base_vote, prefix="vote")
    df_eval_overlay = mark_events_epochwise(df_eval_overlay, seq_events, prefix="seq")
    df_eval_overlay = mark_events_epochwise(df_eval_overlay, base_newma, prefix="newma")
    df_eval_overlay = mark_events_epochwise(df_eval_overlay, base_gate, prefix="gate")

    # ---- P1: don't abort on overlay mismatches; warn and continue ----
    for name, ev in [
        ("learned", det_rows),
        ("ewma", base_ewma),
        ("sma", base_sma),
        ("vote", base_vote),
        ("seq", seq_events),
        ("newma", base_newma),
        ("gate", base_gate),
    ]:
        try:
            assert_overlay_consistency(df_eval_overlay, ev, name)
        except AssertionError as e:
            print(f"[WARN] overlay inconsistency ({name}): {e}")
    # Apply per-method warn persistence (after consistency checks, before writing overlays)
    df_eval_overlay = _apply_warn_persistence(df_eval_overlay, int(CFG.get("warn_consec", 3)))
    safe_write_parquet(df_eval_overlay, OUTDIR / f"bundle_runs_eval_with_overlays_{tag}.parquet")
    _write_both_overlays(df_eval_overlay, OUTDIR)

    # ---- Plots (use unified overlay) ----
    make_plots(df_eval_overlay, det_rows, tag=f"learned_{tag}", overlay_prefix="learned")
    make_plots(df_eval_overlay, base_gate, tag=f"gate_{tag}", overlay_prefix="gate")


# ---------------------------
# Main
# ---------------------------
def main():
    # smoke-mode overrides
    if os.environ.get("SCAR_SMOKE", "0") == "1":
        for k, v in CFG_SMOKE.items():
            CFG[k] = v

    # --- env override for family z-gate (deployment gate, not GT) ---
    v = os.environ.get("SCAR_FAMILY_Z_THR")
    if v is not None:
        try:
            CFG["family_z_thr"] = float(v)
            print(f"[env] family_z_thr={CFG['family_z_thr']:.3f}")
        except Exception as e:
            print(f"[WARN] bad SCAR_FAMILY_Z_THR={v!r}: {e}")

    # --- optional env overrides for GT threshold and quick knobs ---
    v = os.environ.get("SCAR_FIXED_GT_RANK_MIN")
    if v is not None:
        try:
            CFG["gt_rank_min"] = float(v)
            print(f"[env] fixed gt_rank_min={CFG['gt_rank_min']:.3f}")
        except Exception as e:
            print(f"[WARN] bad SCAR_FIXED_GT_RANK_MIN={v!r}: {e}")

    v = os.environ.get("SCAR_PH_BURN")
    if v is not None:
        try:
            CFG["ph_burn"] = int(v)
            print(f"[env] ph_burn={CFG['ph_burn']}")
        except Exception as e:
            print(f"[WARN] bad SCAR_PH_BURN={v!r}: {e}")

    v = os.environ.get("SCAR_DET_H")
    if v is not None:
        try:
            CFG["detector_horizon"] = int(v)
            print(f"[env] detector_horizon={CFG['detector_horizon']}")
        except Exception as e:
            print(f"[WARN] bad SCAR_DET_H={v!r}: {e}")

    # optional env overrides for gating knobs
    v_wc = os.environ.get("SCAR_WARN_CONSEC")
    if v_wc is not None:
        try:
            CFG["warn_consec"] = int(v_wc)
            print(f"[env] warn_consec={CFG['warn_consec']}")
        except Exception as e:
            print(f"[WARN] bad SCAR_WARN_CONSEC={v_wc!r}: {e}")

    v_fw = os.environ.get("SCAR_FAMILY_WINDOW")
    if v_fw is not None:
        try:
            CFG["family_window"] = int(v_fw)
            print(f"[env] family_window={CFG['family_window']}")
        except Exception as e:
            print(f"[WARN] bad SCAR_FAMILY_WINDOW={v_fw!r}: {e}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    env = dict(
    torch=torch.__version__,
    tv=torchvision.__version__,
    cuda=torch.version.cuda,
    cudnn=torch.backends.cudnn.version(),
    device=str(CFG["device"]),
    deterministic=bool(CFG["deterministic"]),
    cuda_hash=_cuda_hash(),
    pip_freeze_md5=_pip_freeze_md5(),
    cfg=CFG,
    )
    save_json(env, OUTDIR / "env.json")
    save_json(CFG, OUTDIR / "cfg_phase4.json")
    # one-time reproducibility capsule (non-fatal)
    try:
        repro_path = OUTDIR / "repro.json"
        if not repro_path.exists():
            repro = {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(CFG["device"]),
            "deterministic": bool(CFG["deterministic"]),
            "cuda_hash": _cuda_hash(),
            "pip_freeze_md5": _pip_freeze_md5(),
            }
            save_json(repro, repro_path)
    except Exception:
        pass
    # pip freeze artifact
    try:
        Path(OUTDIR / "pip_freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        )
    except Exception as e:
        Path(OUTDIR / "pip_freeze.txt").write_text(f"<pip freeze failed: {repr(e)}>")

    tag = "smoke" if os.environ.get("SCAR_SMOKE", "0") == "1" else "full_v2"
    df_all = run_sweep(tag=tag)
    if df_all is None or df_all.empty:
        print("No data; exiting.")
        return
    safe_write_parquet(df_all, OUTDIR / f"bundle_runs_{tag}.parquet")

    evaluate(df_all, tag=tag)
    print("✓ Artifacts in:", OUTDIR.resolve())


if __name__ == "__main__":
    main()


