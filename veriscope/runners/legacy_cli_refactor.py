# Veriscope Bundle (REFACTOR: runners/legacy_cli.py)
#
#
# Env:
#   SCAR_OUTDIR=...      # output dir
#   SCAR_DATA=...        # data root (CIFAR-10 / STL10 will be downloaded if absent)
#   SCAR_SMOKE=1         # optional: tiny sweep for quick end-to-end


from __future__ import annotations

# --- debug faulthandler (early) ---
import os
import faulthandler
import signal
import sys
import math
import numpy as np

# Enable with VS_FAULTHANDLER=1. Send `kill -USR1 <pid>` to dump Python stacks.
if os.environ.get("VS_FAULTHANDLER", "0") == "1":
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    print("[debug] faulthandler SIGUSR1 registered", flush=True)
# --- end debug faulthandler ---
from typing import Any, Optional, Dict, cast
from pathlib import Path

# Central mutable configuration dict (single declaration for type-checkers)
CFG: Dict[str, Any] = {}

# ---- Moved mid-file imports ----
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Union, overload, Callable, Iterable

# Python 3.10+ has TypeAlias in typing; for 3.9 use typing_extensions.
try:  # version compatibility shim
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:  # on 3.9 this will fire
    from typing_extensions import TypeAlias  # type: ignore[misc]

from collections.abc import Iterable as _Iter
from typing import Literal as _Literal


# --- Typing-only handles (no runtime reassignment to type names) ---
if TYPE_CHECKING:
    from veriscope.core.window import FRWindow, WindowDecl
    from veriscope.core.transport import DeclTransport, DeclTransport as Transport
    from veriscope.core.gate import GateEngine

# Optional registrar slot with a precise type; None until wired
RegisterMetrics = Optional[
    Callable[
        ["Transport", Iterable[Callable[..., Any]], str],
        None,
    ]
]
register_metrics: RegisterMetrics = None

import matplotlib

matplotlib.use("Agg")
import traceback
from dataclasses import dataclass  # noqa: F401
from typing import List, Tuple, Iterable


import matplotlib.pyplot as plt
from typing import Any


# --- scalar-safe int helper (NumPy → Python int) ---
def safe_int_from_float(x: Union[float, np.floating]) -> int:
    """Round-trip through float to satisfy typing (NumPy scalars → Python int)."""
    return int(float(x))


def _as_float_list(x: Any) -> List[float]:
    """Return a Python list of floats from x, tolerating scalar np.floating/None."""
    try:
        arr = np.asarray(x if x is not None else [], dtype=float)
    except Exception:
        arr = np.array([], dtype=float)
    return arr.tolist()


import numpy.typing as npt
from typing import Iterable

# --- pandas imports ---
import pandas as pd


# --- typed scalar extractors for pandas/NumPy ---
def series_head_int(s: object, default: int = 0) -> int:
    if isinstance(s, pd.Series) and not s.empty:
        v = s.iloc[0]
        try:
            if isinstance(v, (int, np.integer)):
                return int(v)
            if isinstance(v, (float, np.floating)) and np.isfinite(v):
                return safe_int_from_float(v)
            if isinstance(v, str):
                v2 = v.strip()
                return int(v2) if v2 else default
        except Exception:
            pass
    return default


def series_head_float(s: object, default: float = 0.0) -> float:
    if isinstance(s, pd.Series) and not s.empty:
        v = s.iloc[0]
        try:
            if isinstance(v, (int, np.integer, float, np.floating)):
                return float(v)
            if isinstance(v, str):
                v2 = v.strip()
                return float(v2) if v2 else default
        except Exception:
            pass
    return default


from typing import SupportsFloat  # safe even if similar imports exist


def iter_float(a: npt.ArrayLike | SupportsFloat) -> Iterable[float]:
    """Yield floats from arraylike; treat scalars as length-1 iterable via ndarray coercion."""
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 0:
        yield float(arr)
        return
    for x in arr.ravel():
        yield float(x)


FloatArr: TypeAlias = np.ndarray

from typing import cast as _cast


def qlin_vec(a: Iterable[float] | npt.ArrayLike, qs: npt.ArrayLike) -> np.ndarray:
    """Vector quantiles with linear method; returns float ndarray."""
    arr = np.asarray(a, dtype=float)
    quan = _cast(Any, np.quantile)
    try:
        return np.asarray(quan(arr, qs, method="linear"), dtype=float)
    except TypeError:
        return np.asarray(quan(arr, qs, interpolation="linear"), dtype=float)


# --- Helper: Typed DataFrame.get with Series default to please mypy ---
def df_get_series(df: pd.DataFrame, key: str) -> pd.Series:
    """Typed DataFrame.get with a Series default to please mypy."""
    return cast(pd.Series, df.get(key, pd.Series(dtype=float)))


import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchvision  # type: ignore[import-untyped]
from torch.utils.data import DataLoader

# Gate heavy optional deps for type checking
if TYPE_CHECKING:
    import torchvision  # type: ignore[import-untyped]


# Safe percentile wrapper that matches NumPy's current typing
def pct_linear(a: Iterable[float] | npt.ArrayLike, q: float) -> float:
    return qlin(a, float(q) / 100.0)


# --- central config seed (non-fatal if module is absent) ---
try:
    from veriscope.config import CFG as _CFG_CENTER

    for _k, _v in dict(_CFG_CENTER).items():
        CFG.setdefault(_k, _v)
except Exception:
    # Keep runner robust if central CFG is absent
    pass

# ---- FR spine (core) imports; optional until SCAR_FR=1 ----
try:
    from veriscope.core.gate import GateEngine
    from veriscope.core.transport import DeclTransport
    from veriscope.core.window import FRWindow, WindowDecl, window_decl_identity_hash
    from veriscope.core.calibration import aggregate_epsilon_stat as _core_aggregate_epsilon_stat
except Exception:
    GateEngine = None  # type: ignore[assignment]
    DeclTransport = None  # type: ignore[assignment]
    FRWindow = None  # type: ignore[assignment]
    WindowDecl = None  # type: ignore[assignment]
    _core_aggregate_epsilon_stat = None  # type: ignore[assignment]
    window_decl_identity_hash = None  # type: ignore[assignment]

FRW_CLS: Optional[type] = FRWindow
TRANS_CLS: Optional[type] = DeclTransport
GATE_CLS: Optional[type] = GateEngine
DECL_CLS: Optional[type] = WindowDecl


def agg_eps_adapter(
    window: Any,
    counts: Optional[Dict[str, int]] = None,
    *,
    counts_by_metric: Optional[Dict[str, int]] = None,
    alpha: float = 0.05,
    **_ignored: Any,
) -> float:
    """Signature-stable ε_stat aggregator.

    Callers may pass counts either positionally (`counts`) or as the keyword
    `counts_by_metric`. Returns 0.0 on any failure.

    This is deliberately permissive because the runner may operate in a partially
    wired state (e.g., FR not importable) and we don't want ε aggregation to crash
    the sweep.
    """
    fn = _core_aggregate_epsilon_stat
    if fn is None:
        return 0.0

    c = counts_by_metric if counts_by_metric is not None else (counts or {})

    try:
        out = fn(window, c, float(alpha))
        out_f = float(out)
        return out_f if np.isfinite(out_f) and out_f >= 0.0 else 0.0
    except TypeError:
        # Tolerate kw-only implementations.
        try:
            out = fn(window, counts_by_metric=c, alpha=float(alpha))
            out_f = float(out)
            return out_f if np.isfinite(out_f) and out_f >= 0.0 else 0.0
        except Exception:
            return 0.0
    except Exception:
        return 0.0


# Default ε aggregator is the local adapter; FR wiring (below) may override.
AGGREGATE_EPSILON = agg_eps_adapter
# NOTE: agg_eps_adapter is defined once near the top of this file; do not redefine it here.


# ---- legacy imports (stay as-is) ----
from veriscope.runners.legacy import runtime

from veriscope.runners.legacy.utils import (
    save_json,
    update_json,
    file_md5,
    as_int,
    as_float,
    to_numeric_series,
    qlin,
    quantile2,
    _as_float_array,
)


# --- Helper: append a message to sweep.log with timestamp and pid (audit trail) ---
def _append_sweep_log(msg: str) -> None:
    """Append a message to sweep.log unconditionally (creates if absent).

    Use this for audit trail messages that must survive stdout truncation.
    Safe to call from any process; includes timestamp + pid prefix.
    """
    try:
        import time
        import os
        from pathlib import Path

        outdir = globals().get("OUTDIR", None)
        if outdir is None:
            return
        outdir = Path(outdir)

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        pid = os.getpid()
        log_path = outdir / "sweep.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts} pid={pid}] {msg}\n")
            f.flush()
    except Exception:
        pass  # Never let logging failures break the pipeline


from veriscope.runners.legacy.determinism import (
    seed_all,
    new_gen,
    seed_worker,
)
from veriscope.runners.legacy.budget import WindowBudget, BudgetLedger

from veriscope.runners.legacy.metrics_heavy import (
    sliced_w2_gpu_budget,
    topo_h0_jl_agg,
)

from veriscope.runners.legacy.data import (
    _stl10_monitor_dataset,
    make_loader,
    subset_loader_from_indices,
    load_splits,
    FactorisedTrainDataset,
    DropoutAwareSampler,
    _u01_from_hash,
)

from veriscope.runners.legacy.model import (
    make_model,
    lr_at,
    penult,
    make_opt as _make_opt_model,
)

from veriscope.runners.legacy.features import (
    _features_for_loader,
    _std0,
    variance_outside_k,
    spectral_r2,
    cosine_dispersion,
)

from veriscope.runners.legacy.monitors import (
    monitor_entropy,
    monitor_avg_conf,
    monitor_accuracy,
    monitor_margin_median,
)

from veriscope.runners.legacy.probes import (
    collect_feature_snapshot,
    kappa_sens_probe,
)


# baseline detectors (moved out of CLI)
from veriscope.runners.legacy.detectors.baselines import (
    SCHEDULED_METRICS,
    _prep_series_for_ph,
    robust_z_series,
    ph_window_sparse,
    _delta,
    cusum_one_sided,
    newma_warn_epoch,
    calibrate_ph_directions,
)

# learned detector utilities (moved out of CLI)
from veriscope.runners.legacy.detectors.learned import (
    _first_run_end,
)

from veriscope.core.ipm import dPi_product_tv

# ---- Legacy eval core (moved out of CLI) ----
from veriscope.runners.legacy.eval.core import (
    compute_events,
    mark_events_epochwise,
    assert_overlay_consistency,
    summarize_detection,
    summarize_runlevel_fp,
    rp_adequacy_flags,
    recompute_gate_series_under_decl,
    _first_t_column,
)


def _assert_runner_wired() -> None:
    if FRW_CLS is None or TRANS_CLS is None or GATE_CLS is None or DECL_CLS is None:
        raise RuntimeError("runner not wired: missing class bindings")


# ---- typing shims for dynamic inputs (JSON/env/argparse) ----
NumberLike = Union[int, float, np.integer, np.floating, str]


@overload
def to_float(x: int | float | np.integer | np.floating) -> float: ...
@overload
def to_float(x: str) -> float: ...
def to_float(x: Any) -> float:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise TypeError(f"to_float: expected number-like, got {type(x).__name__}")


@overload
def to_int(x: int) -> int: ...
@overload
def to_int(x: float) -> int: ...
@overload
def to_int(x: np.integer) -> int: ...
@overload
def to_int(x: np.floating) -> int: ...
@overload
def to_int(x: str) -> int: ...
def to_int(x: Any) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s.lower().startswith("0x"):
            return int(s, 16)
        try:
            return int(s, 10)
        except ValueError:
            return safe_int_from_float(float(s))
    raise TypeError(f"to_int: expected int-like, got {type(x).__name__}")


def as_int_cfg(x: Any, default: int = 0) -> int:
    """Alias used inside summary/emitters to avoid local shadowing of `as_int`."""
    return as_int(x, default=default)


# ---- percentile helper with stable semantics across NumPy versions ----
def percentile_linear(a: Iterable[float] | npt.ArrayLike, q: Any) -> float:
    """Return the q-th percentile (q in [0,100]) with linear interpolation semantics."""
    return qlin(a, to_float(q) / 100.0)


# --- Inserted: typing and pandas numeric shims ---
def _to_num(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(s, errors="coerce")


def to_numeric_opt(s: Optional[pd.Series], *, errors: _Literal["raise", "coerce", "ignore"] = "coerce") -> pd.Series:
    """Like pd.to_numeric but accepts Optional[Series] and always returns Series[float]."""
    if s is None:
        return pd.Series(dtype=float)
    eno: _Literal["raise", "coerce"] = "coerce" if errors == "ignore" else errors
    return pd.to_numeric(s, errors=eno).astype(float, copy=False)


def _float_pair(x: object) -> tuple[float, float]:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return (
            as_float(x[0], default=float("nan")),  # type: ignore[arg-type]
            as_float(x[1], default=float("nan")),  # type: ignore[arg-type]
        )
    return (float("nan"), float("nan"))


if TYPE_CHECKING:
    from filelock import SoftFileLock as FileLock
else:
    try:
        from filelock import SoftFileLock as _RuntimeFileLock
    except Exception:  # pragma: no cover

        class _RuntimeFileLock:
            def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    FileLock = _RuntimeFileLock  # runtime alias only


# --- env truthy and centralized run tag ---
def env_truthy(name: str, default: str = "0") -> bool:
    v = str(os.environ.get(name, default)).strip().lower()
    return v in ("1", "true", "yes", "on")


CAL = env_truthy("SCAR_CALIB")
SMOKE = env_truthy("SCAR_SMOKE")
DEBUG_GATE = env_truthy("SCAR_DEBUG_GATE")
DEBUG_SMOKE = env_truthy("SCAR_DEBUG")  # broader debug flag
RUN_TAG = "cal" if CAL else ("smoke" if SMOKE else "full_v2")

# Ensure deterministic cuBLAS workspace is configured before torch/cuBLAS init (CUDA 11.x/12.x)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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

# --- explicit functional imports to satisfy static checkers and provide stable callables ---
try:
    from torch.linalg import eigvals as t_eigvals
    from torch.linalg import eigvalsh as t_eigvalsh
    from torch.linalg import vector_norm as t_vector_norm
except Exception:  # pragma: no cover
    t_eigvalsh = None
    t_eigvals = None
    t_vector_norm = None
try:
    from torch.nn.functional import avg_pool2d as _t_avg_pool2d
except Exception:  # pragma: no cover
    _t_avg_pool2d = None  # type: ignore[assignment]


def t_avg_pool2d(*args: Any, **kwargs: Any) -> Any:
    """Compat wrapper so static checkers see a callable; raises if F.avg_pool2d is missing."""
    if _t_avg_pool2d is None:
        raise RuntimeError("torch.nn.functional.avg_pool2d unavailable")
    return _t_avg_pool2d(*args, **kwargs)


if t_eigvalsh is None:

    def t_eigvalsh(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("torch.linalg.eigvalsh unavailable")


if t_eigvals is None:

    def t_eigvals(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("torch.linalg.eigvals unavailable")


if t_vector_norm is None:

    def t_vector_norm(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("torch.linalg.vector_norm unavailable")


amp_mode: Any = None
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
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        class _NoGradScaler:
            def __init__(self, enabled=False):
                self.enabled = bool(enabled)

            def scale(self, loss):
                return loss

            def unscale_(self, opt):
                pass

            def step(self, opt):
                opt.step()

            def update(self):
                pass

        class _NoAmp:
            autocast = _NoAutocast
            GradScaler = _NoGradScaler

        amp_mode = _NoAmp()
    else:
        import inspect as _inspect

        _sig = _inspect.signature(_amp_mod.autocast)
        if "device_type" in _sig.parameters:
            amp_mode = _amp_mod
        else:
            # Wrap autocast to ignore device_type on older torch.cuda.amp
            class _AutocastWrapper:
                def __init__(self, device_type=None, enabled=True):
                    self._ctx = _amp_mod.autocast(enabled=enabled)

                def __enter__(self):
                    return self._ctx.__enter__()

                def __exit__(self, exc_type, exc, tb):
                    return self._ctx.__exit__(exc_type, exc, tb)

            class _AmpShim:
                autocast = _AutocastWrapper
                GradScaler = _amp_mod.GradScaler

            amp_mode = _AmpShim()
else:
    amp_mode = _amp_mod

# Backward-compat for legacy references; keep a stable alias for older call sites.
amp: Any = amp_mode
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

_SW2_CPU_WARNED = False

os.environ.setdefault("PYTHONHASHSEED", "0")

try:
    if mp.current_process().name == "MainProcess":
        print("Torch:", torch.__version__, "TV:", torchvision.__version__, "CUDA:", torch.version.cuda)
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
except Exception:
    pass

# Numerically-stable sigmoid used across train/OOF/eval


def _sigmoid_stable(z: Any, cap: float = 60.0) -> Any:
    """Return sigmoid(z) with clipping to avoid overflow in exp()."""
    zc = np.clip(z, -float(cap), float(cap))
    return 1.0 / (1.0 + np.exp(-zc))


# Global ledger is instantiated after CFG is defined.
BUDGET: Optional[BudgetLedger] = None


def calibrate_window_from_controls(
    df_control: pd.DataFrame,
    metrics: List[str],
    weights: Dict[str, float],
    bins: int,
    epsilon: float,
    interventions: tuple,
) -> WindowDecl:
    cal_ranges: Dict[str, Tuple[float, float]] = {}

    # Guard empty
    if df_control is None or len(df_control) == 0:
        for m in metrics:
            cal_ranges[m] = (0.0, 1.0)
        return WindowDecl(
            epsilon=epsilon,
            metrics=metrics,
            weights=weights,
            bins=bins,
            interventions=interventions,
            cal_ranges=cal_ranges,
        )

    for m in metrics:
        lo, hi = 0.0, 1.0
        if m in df_control.columns:
            arr = pd.to_numeric(df_control[m], errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size >= 4:
                lo = float(np.quantile(arr, 0.01))
                hi = float(np.quantile(arr, 0.99))
            elif arr.size > 0:
                lo = float(np.min(arr))
                hi = float(np.max(arr))

            if np.isfinite(lo) and np.isfinite(hi):
                if hi <= lo + 1e-12:
                    # width floor: 5% of scale or 1e-2, whichever larger
                    scale = max(abs(lo), abs(hi), 1.0)
                    w = max(0.05 * scale, 1e-2)
                    lo, hi = lo - 0.5 * w, hi + 0.5 * w
                else:
                    # padding
                    pad = 0.05 * (hi - lo)
                    lo, hi = lo - pad, hi + pad

        # Nonnegativity for these metrics in your pipeline
        lo = max(0.0, float(lo))
        hi = max(lo + 1e-9, float(hi)) if np.isfinite(hi) else 1.0

        # Final sanity fallback
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            lo, hi = (0.0, 1.0)

        cal_ranges[m] = (lo, hi)

    try:
        print(f"[cal] calibrate_window_from_controls: cal_ranges={cal_ranges}")
    except Exception:
        pass

    return WindowDecl(
        epsilon=epsilon,
        metrics=metrics,
        weights=weights,
        bins=bins,
        interventions=interventions,
        cal_ranges=cal_ranges,
    )


def _apply_for_window_decl(window_decl: Any):
    """Return DeclTransport.apply if installed on the decl; else identity-as-float."""
    try:
        dt = getattr(window_decl, "_DECL_TRANSPORT", None)
        if dt is not None and hasattr(dt, "apply"):
            return dt.apply  # type: ignore[return-value]
    except Exception:
        pass
    return lambda _name, arr: np.asarray(arr, dtype=float)


# --- Calibrate epsilon from controls ---
def calibrate_epsilon_from_controls(
    df_control: pd.DataFrame, window_decl: WindowDecl, W: int, q: float = 0.995
) -> Tuple[float, int]:
    """
    Returns (epsilon, n_vals). Falls back to CFG['gate_epsilon'] on empty or invalid input.
    Always enforces CFG['gate_epsilon_floor'] as a lower bound.
    """
    eps_floor = as_float(CFG.get("gate_epsilon_floor", 0.02), default=0.02)
    if (not np.isfinite(eps_floor)) or (eps_floor < 0.0):
        eps_floor = 0.02

    vals: List[float] = []
    try:
        if "factor" in df_control.columns:
            df_control = df_control.copy()
            df_control["factor"] = df_control["factor"].astype(str).str.lower().str.strip()
        if df_control is None or len(df_control) == 0:
            raise ValueError("empty_controls")
        if "epoch" in df_control.columns:
            df_control["epoch"] = to_numeric_series(df_control.get("epoch"), errors="coerce")
        mets = [m for m in getattr(window_decl, "metrics", []) if m in df_control.columns]
        apply = _apply_for_window_decl(window_decl)
        if not mets:
            raise ValueError("no_metric_columns")
        for _, g in df_control.groupby(["seed", "factor"], sort=False):
            g = g.sort_values("epoch")
            if len(g) < 2 * max(1, int(W)):
                continue
            for i in range(0, max(0, len(g) - 2 * int(W) + 1)):
                past = {m: _as_float_array(to_numeric_opt(g.get(m)).iloc[i : i + W].to_numpy()) for m in mets}
                recent = {m: _as_float_array(to_numeric_opt(g.get(m)).iloc[i + W : i + 2 * W].to_numpy()) for m in mets}
                ok = True
                for m in mets:
                    pa = past[m]
                    re = recent[m]
                    if pa.size < max(1, W // 2) or re.size < max(1, W // 2):
                        ok = False
                        break
                if not ok:
                    continue
                try:
                    tv = dPi_product_tv(window_decl, past, recent, apply=apply)
                    if np.isfinite(tv):
                        vals.append(float(tv))
                except Exception:
                    continue
    except Exception as _e:
        try:
            print(
                f"[WARN] calibrate_epsilon_from_controls failed early: {_e}; fallback to CFG gate_epsilon={CFG.get('gate_epsilon', 0.08)}"
            )
        except Exception:
            pass
        fallback = float(_effective_gate_epsilon(CFG, OUTDIR))
        return max(fallback, float(eps_floor)), 0

    if not vals:
        try:
            print(
                f"[WARN] calibrate_epsilon_from_controls: no valid TV samples; returning CFG gate_epsilon={CFG.get('gate_epsilon', 0.08)}"
            )
        except Exception:
            pass
        fallback = float(_effective_gate_epsilon(CFG, OUTDIR))
        return max(fallback, float(eps_floor)), 0

    eps_new, n_vals = _robust_eps(vals, q=float(q), CFG=CFG, out_dir=OUTDIR)

    eps_raw = float(eps_new)
    eps_clamped = max(eps_raw, float(eps_floor))
    if int(n_vals) > 0 and eps_clamped == float(eps_floor) and eps_raw < float(eps_floor):
        try:
            print(f"[cal] gate_epsilon clamped to floor={eps_floor:.4f} (raw={eps_raw:.6f})")
        except Exception:
            pass

    return float(eps_clamped), int(n_vals)


def _evidence_scaled_epsilon(
    eps_base: float,
    evidence: int,
    *,
    min_evidence: int,
    max_inflation: float,
) -> float:
    """Inflate ε when evidence is low to avoid small-sample / quantization false alarms.

    Returns eps_base once evidence >= min_evidence.
    Linear inflation up to eps_base * max_inflation at evidence=0.
    """
    try:
        eb = float(eps_base)
    except Exception:
        eb = 0.0
    try:
        ev = int(evidence)
    except Exception:
        ev = 0
    try:
        me = max(1, int(min_evidence))
    except Exception:
        me = 1
    try:
        mi = float(max_inflation)
    except Exception:
        mi = 1.0
    if mi < 1.0:
        mi = 1.0
    if ev >= me:
        return eb
    ratio = max(0.0, min(1.0, float(ev) / float(me)))
    infl = 1.0 + (mi - 1.0) * (1.0 - ratio)
    return eb * float(infl)


def _gain_bits_from_recent_losses(
    train_losses_nats: np.ndarray,
    *,
    baseline: str,
    ewma_losses_nats: Optional[np.ndarray] = None,
    num_classes: int = 10,
) -> float:
    """
    Compute gain in bits/sample from recent loss arrays.
    baseline:
      - "uniform": gain_bits = (log(C) - mean(train_loss))/log(2)
      - "ewma":    gain_bits = mean(ewma_loss - train_loss)/log(2)  (legacy; tends to 0 at convergence)
    """
    ln2 = math.log(2.0)
    x = np.asarray(train_losses_nats, dtype=float)
    xm = x[np.isfinite(x)]
    if xm.size == 0:
        return float("nan")
    b = str(baseline or "").strip().lower()
    if b == "uniform":
        base_nats = float(math.log(max(2, int(num_classes))))
        return float((base_nats - float(np.mean(xm))) / ln2)
    if ewma_losses_nats is None:
        return float("nan")
    y = np.asarray(ewma_losses_nats, dtype=float)
    ym = y[np.isfinite(y)]
    n = min(xm.size, ym.size)
    if n <= 0:
        return float("nan")
    return float((float(np.mean(ym[:n] - xm[:n]))) / ln2)


def gate_check(
    window: WindowDecl,
    past: Dict[str, np.ndarray],
    recent: Dict[str, np.ndarray],
    counts: Dict[str, int],
    gain: float,
    gain_thresh: float,
    kappa_sens: float,
    eps_stat_alpha: float = 0.05,
) -> Tuple[int, Dict[str, Any]]:
    """Joint gate: prequential gain + fixed-partition product–TV with ε_stat and κ_sens."""
    thr = float(gain_thresh)
    # Evidence must be computed from metric counts only (ignore meta keys like "_epoch").
    counts_metric: Dict[str, int] = {}
    try:
        for k, v in (counts or {}).items():
            ks = str(k)
            if ks.startswith("_"):
                continue
            try:
                counts_metric[ks] = int(v)
            except Exception:
                continue
    except Exception:
        counts_metric = {}
    # --- FR branch: license via D_W with common transport; preserves legacy audit keys ---
    global USE_FR, FR_WIN, GE
    if USE_FR and (FR_WIN is not None) and (GE is not None):
        try:
            # Aggregate ε_stat using the existing routine, then delegate to GateEngine
            eps_stat_value = AGGREGATE_EPSILON(window, counts_by_metric=counts_metric, alpha=float(eps_stat_alpha))
            # Clamp NaNs to 0 before passing
            if not np.isfinite(eps_stat_value):
                eps_stat_value = 0.0
            # Missing κ should mean "not checked" under GateEngine semantics.
            # GateEngine uses `np.isfinite(kappa_sens)` to decide whether κ is checked.
            kappa_in = float(kappa_sens) if (kappa_sens is not None) and np.isfinite(kappa_sens) else float("nan")

            gr = GE.check(
                past={m: np.asarray(past.get(m, np.array([], float)), dtype=float) for m in window.weights.keys()},
                recent={m: np.asarray(recent.get(m, np.array([], float)), dtype=float) for m in window.weights.keys()},
                counts_by_metric={str(k): int(v) for k, v in (counts_metric or {}).items()},
                gain_bits=float(gain) if (gain is not None) and np.isfinite(gain) else float("nan"),
                kappa_sens=kappa_in,
                eps_stat_value=float(eps_stat_value),
            )

            aud = getattr(gr, "audit", {}) or {}

            # min_evidence provenance: for FR, prefer GE.min_evidence (env/CFG precedence in init_fr)
            try:
                _min_evidence_fr = int(getattr(GE, "min_evidence", as_int(CFG.get("gate_min_evidence", 0), default=0)))
            except Exception:
                _min_evidence_fr = as_int(CFG.get("gate_min_evidence", 0), default=0)

            # total_evidence provenance: prefer GateEngine audit if present, else fallback to counts sum
            try:
                _total_evidence_fr_fallback = int(sum(int(v) for v in (counts_metric or {}).values()))
            except Exception:
                _total_evidence_fr_fallback = 0
            try:
                _total_evidence_fr = int(
                    aud.get("total_evidence", aud.get("evidence_total", _total_evidence_fr_fallback))
                )
            except Exception:
                _total_evidence_fr = int(_total_evidence_fr_fallback)

            # Evaluability and pending-warning semantics (critical under persistence):
            evaluated_fr = bool(aud.get("evaluated", True))
            # Prefer explicit fields from GateResult and audit (GateEngine commonly emits warn_pending in audit)
            warn_pending_fr = bool(getattr(gr, "warn", False)) or bool(aud.get("warn_pending", False))

            if not evaluated_fr:
                # Prefer GateEngine-provided reason if present; keep it stable
                _reason_fr = str(aud.get("reason", "not_evaluated_fr")) or "not_evaluated_fr"
            else:
                _reason_fr = "evaluated_fr_ok" if bool(getattr(gr, "ok", False)) else "evaluated_fr_fail"

            # Map FR audit keys to legacy ones so downstream code remains unchanged,
            # but ADD missing channels required by persistence policies.
            audit = {
                "reason": str(_reason_fr),
                "evaluated": bool(evaluated_fr),
                "warn_pending": bool(warn_pending_fr),
                # Lane provenance (lets offline recompute mirror correctly)
                "gate_lane": "fr",
                "scar_fr": 1,
                # Evidence provenance
                "total_evidence": int(_total_evidence_fr),
                "min_evidence": int(_min_evidence_fr),
                # Core diagnostics
                "gain_bits": as_float(aud.get("gain_bits", np.nan), default=float("nan")),
                "worst_tv": as_float(aud.get("worst_DW", np.nan), default=float("nan")),
                "eps_stat": as_float(aud.get("eps_stat", np.nan), default=float("nan")),
                "eps_eff": as_float(aud.get("eps_eff", np.nan), default=float("nan")),
                "kappa_sens": as_float(aud.get("kappa_sens", np.nan), default=float("nan")),
                # Persistence / counters (names match what offline/analysis expects)
                "policy": str(aud.get("policy", "")) if aud.get("policy", "") is not None else "",
                "persistence_k": as_int(aud.get("persistence_k", 0), default=0),
                "persistence_fail": bool(aud.get("persistence_fail", False)),
                "consecutive_exceedances": as_int(aud.get("consecutive_exceedances", 0), default=0),
                "dw_exceeds_threshold": bool(aud.get("dw_exceeds_threshold", False)),
                # Optional per-component OK bits (if GateEngine emits them)
                "ok_gain": bool(aud.get("ok_gain", True)),
                "ok_stab": bool(aud.get("ok_stab", True)),
                # Existing metadata
                "counts_by_metric": {k: int(v) for k, v in (counts_metric or {}).items()},
                "eps_aggregation": "cap_to_frac",
                "gain_units": CFG.get("gate_gain_units", "bits/sample"),
                "eps_stat_cap_fraction": float(CFG.get("gate_eps_stat_max_frac", 0.25)),
                "gain_thresh_used": float(thr),
            }

            # Return value semantics: 1 means PASS/neutral.
            # If GateEngine says "not evaluated", force neutral PASS.
            flag = int(bool(getattr(gr, "ok", True))) if evaluated_fr else 1
            return int(flag), audit
        except Exception as _fr_exc:
            # Fall through to legacy path on any FR error, but make it visible (once)
            global _FR_FALLBACK_WARNED
            if not _FR_FALLBACK_WARNED:
                try:
                    if mp.current_process().name == "MainProcess":
                        print(f"[gate][WARN] FR branch failed, falling back to legacy: {_fr_exc!r}")
                        _FR_FALLBACK_WARNED = True
                except Exception:
                    pass
    # Early evidence gate: require a minimum number of finite transported pairs across metrics
    try:
        min_evidence = as_int(CFG.get("gate_min_evidence", 16), default=16)
    except Exception:
        min_evidence = 16
    try:
        total_evidence = int(sum(int(v) for v in (counts_metric or {}).values()))
    except Exception:
        total_evidence = 0

    # Debug: checkpoint evidence (rate-limited)
    _dbg_epoch = -1
    try:
        _dbg_epoch = int(counts.get("_epoch", -1)) if isinstance(counts, dict) else -1
    except Exception:
        _dbg_epoch = -1

    if DEBUG_GATE and mp.current_process().name == "MainProcess":
        # Print every 4 epochs in smoke; otherwise print only on explicit skip/fail blocks.
        _every = 4 if env_truthy("SCAR_SMOKE") else 999999
        if (_dbg_epoch >= 0) and (_dbg_epoch % _every == 0):
            try:
                print(
                    f"[gate][dbg] epoch={_dbg_epoch} total_evidence={total_evidence} min={min_evidence} counts={counts}"
                )
            except Exception:
                pass

    if total_evidence < min_evidence:
        if DEBUG_GATE and mp.current_process().name == "MainProcess":
            try:
                print(f"[gate][dbg] SKIP insufficient_evidence: {total_evidence} < {min_evidence}")
            except Exception:
                pass
        return 1, {
            "reason": "not_evaluated_insufficient_evidence",
            "evaluated": False,
            "warn_pending": False,
            "gate_lane": "legacy",
            "scar_fr": 0,
            "total_evidence": int(total_evidence),
            "min_evidence": int(min_evidence),
            "worst_tv": float("nan"),
            "eps_stat": float("nan"),
            "eps_eff": float("nan"),
            "kappa_sens": float(kappa_sens) if (kappa_sens is not None) and np.isfinite(kappa_sens) else float("nan"),
            "gain_bits": float(gain) if (gain is not None) and np.isfinite(gain) else float("nan"),
            "counts_by_metric": {k: int(v) for k, v in (counts_metric or {}).items()},
            "eps_aggregation": "cap_to_frac",
            "eps_stat_cap_fraction": float(CFG.get("gate_eps_stat_max_frac", 0.25)),
            "gain_units": CFG.get("gate_gain_units", "bits/sample"),
            "gain_thresh_used": float(thr),
        }

    # Evidence-scaled ε to avoid spurious stability failures at tiny N (e.g., W=3 => evidence=6)
    # eps_scaled is used for eps_stat capping and eps_eff in the stability check.
    try:
        min_full = as_int(CFG.get("gate_min_evidence_full_eps", 16), default=16)
    except Exception:
        min_full = 16
    # Ensure inflation still applies on the first evaluated epoch (often total_evidence == min_evidence).
    # If min_full == min_evidence, inflation would be off exactly when we most need it.
    min_full = max(int(min_full), int(min_evidence) + 1)

    try:
        max_infl = as_float(CFG.get("gate_eps_inflation_max", 4.0), default=4.0)
    except Exception:
        max_infl = 4.0
    if not np.isfinite(max_infl) or max_infl < 1.0:
        max_infl = 1.0

    eps_base = float(window.epsilon)
    eps_scaled = float(
        _evidence_scaled_epsilon(
            eps_base,
            int(total_evidence),
            min_evidence=int(min_full),
            max_inflation=float(max_infl),
        )
    )

    worst = 0.0
    intervs = window.interventions or (lambda x: x,)
    _DECL_TRANSPORT = None
    try:
        _DECL_TRANSPORT = getattr(window, "_DECL_TRANSPORT", None)
    except Exception:
        _DECL_TRANSPORT = None
    if _DECL_TRANSPORT is not None:
        _apply = _DECL_TRANSPORT.apply  # type: ignore
    else:
        _apply = lambda name, arr: np.asarray(arr, float)
    for T in intervs:

        def _apply_T(name: str, arr: np.ndarray, _T: Any = T) -> np.ndarray:
            return np.asarray(_T(_apply(name, arr)), dtype=float)

        try:
            tv_sum = float(dPi_product_tv(window, past, recent, apply=_apply_T))
        except Exception:
            tv_sum = float("nan")

        if np.isfinite(tv_sum):
            worst = max(worst, tv_sum)
        else:
            worst = max(worst, float("inf"))

    # Sampling-slack ε_stat: aggregate, guard, and cap to a fraction of ε
    eps_stat = AGGREGATE_EPSILON(window, counts_by_metric=counts_metric, alpha=eps_stat_alpha)
    eps_stat = float(eps_stat if np.isfinite(eps_stat) else 0.0)
    max_frac = float(CFG.get("gate_eps_stat_max_frac", 0.25))
    max_frac = float(min(max(max_frac, 0.0), 1.0))  # clamp knob to [0,1]
    eps_cap = float(eps_scaled) * max_frac
    eps_stat = float(min(max(0.0, eps_stat), eps_cap))

    # Gain check (GateEngine parity): non-finite => not checked => OK
    gain_checked = bool(np.isfinite(gain))
    ok_gain = True if (not gain_checked) else bool(float(gain) >= float(thr))

    # Stability: worst TV must be ≤ (ε_scaled + ε_sens − ε_stat), never negative
    worst = float(worst if np.isfinite(worst) else np.inf)

    # Sensitivity budget ε_sens is also stability slack in GateEngine/offline parity.
    eps_sens = as_float(CFG.get("gate_epsilon_sens", 0.04), default=0.04)
    eps_sens = float(max(0.0, eps_sens))

    eps_eff = max(0.0, float(eps_scaled) + float(eps_sens) - float(eps_stat))
    ok_stability = worst <= eps_eff
    if DEBUG_GATE and mp.current_process().name == "MainProcess":
        # Always print on stability failure (useful signal without flooding logs)
        if not ok_stability:
            try:
                print(
                    f"[gate][dbg] stability_fail worst_tv={worst:.6f} eps_eff={eps_eff:.6f} "
                    f"eps_scaled={float(eps_scaled):.6f} eps_base={float(eps_base):.6f} eps_stat={eps_stat:.6f}"
                )
            except Exception:
                pass

    # Sensitivity budget κ_sens ≤ ε_sens
    kappa_checked = (kappa_sens is not None) and bool(np.isfinite(kappa_sens))
    ok_kappa = True if (not kappa_checked) else (float(kappa_sens) <= eps_sens)
    ok = bool(ok_gain and ok_stability and ok_kappa)
    reason = (
        "evaluated_ok"
        if ok
        else (
            "evaluated_fail_gain"
            if (gain_checked and (not ok_gain))
            else (
                "evaluated_fail_stability"
                if not ok_stability
                else ("evaluated_fail_kappa" if not ok_kappa else "evaluated_fail_unknown")
            )
        )
    )

    return int(ok), {
        "reason": str(reason),
        "evaluated": True,
        "warn_pending": False,
        "gate_lane": "legacy",
        "scar_fr": 0,
        "total_evidence": int(total_evidence),
        "min_evidence": int(min_evidence),
        "gain_bits": float(gain),
        "worst_tv": float(worst),
        "eps_stat": float(eps_stat),
        "eps_eff": float(eps_eff),
        "eps_base": float(eps_base),
        "eps_scaled": float(eps_scaled),
        "eps_inflation_factor": float(float(eps_scaled) / max(float(eps_base), 1e-12)),
        "min_evidence_full_eps": int(min_full),
        "gate_eps_inflation_max": float(max_infl),
        # Missing κ ⇒ unchecked (NaN), NOT 0.0.
        "kappa_sens": float(kappa_sens) if np.isfinite(kappa_sens) else float("nan"),
        "counts_by_metric": {k: int(v) for k, v in (counts_metric or {}).items()},
        "eps_aggregation": "cap_to_frac",
        "gain_units": CFG.get("gate_gain_units", "bits/sample"),
        "eps_stat_cap_fraction": float(CFG.get("gate_eps_stat_max_frac", 0.25)),
        "gain_thresh_used": float(thr),
    }


# --- resolved ε fallback from precedence_summary + robust ε estimator ---
def _effective_gate_epsilon(CFG: dict, out_dir: Path) -> float:
    try:
        p = Path(out_dir) / "precedence_summary.json"
        if p.exists():
            data = json.loads(p.read_text())
            final = data.get("final", {})
            if isinstance(final, dict):
                x = final.get("gate_epsilon", None)
                if isinstance(x, (int, float)):
                    return float(x)
            for k in ("gate_epsilon_final", "final_gate_epsilon", "gate_epsilon"):
                x = data.get(k, None)
                if isinstance(x, (int, float)):
                    return float(x)
    except Exception:
        pass
    x = CFG.get("gate_epsilon", None)
    if isinstance(x, (int, float)):
        return float(x)
    return 0.08  # last resort only


def _robust_eps(vals, q: float, CFG: dict, out_dir: Path) -> Tuple[float, int]:
    q = 0.0 if q < 0.0 else (1.0 if q > 1.0 else float(q))
    try:
        arr = vals.to_numpy() if hasattr(vals, "to_numpy") else np.asarray(vals)
        if np.issubdtype(arr.dtype, np.number):
            a = np.asarray(arr, dtype=float).ravel()
            a = a[np.isfinite(a)]
            if a.size == 0:
                return _effective_gate_epsilon(CFG, out_dir), 0
            eps = float(qlin(a, q))
            return eps, int(a.size)
    except Exception:
        pass
    cleaned = []
    try:
        import torch as _torch
    except Exception:
        _torch = None
    from collections.abc import Iterable as _CollIterable

    _it = vals if isinstance(vals, _CollIterable) and not isinstance(vals, (str, bytes, bytearray)) else [vals]
    for x in _it:
        if isinstance(x, (bool, np.bool_)):
            continue
        if isinstance(x, (float, np.floating)):
            cleaned.append(float(x))
        elif isinstance(x, (int, np.integer)):
            cleaned.append(float(x))
        elif (_torch is not None) and _torch.is_tensor(x) and x.numel() == 1:
            cleaned.append(float(x.item()))
    a = np.asarray(cleaned, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return _effective_gate_epsilon(CFG, out_dir), 0
    eps = float(qlin(a, q))
    return eps, int(a.size)


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
# --- Calibration hard-declare & launch markers (env-driven) ---
try:
    if mp.current_process().name == "MainProcess":
        _calib_env = str(os.environ.get("SCAR_CALIB", "0")).strip()
        _CALIBRATION_ACTIVE = _calib_env in ("1", "true", "TRUE", "yes", "on")
        _meta = {
            "ts": time.time(),
            "calibration": bool(_CALIBRATION_ACTIVE),
            "veriscope_version": "0.1.0",
            "pid": int(os.getpid()),
            "host": os.uname().nodename if hasattr(os, "uname") else "",
            "env": {
                "SCAR_OUTDIR": os.environ.get("SCAR_OUTDIR", ""),
                "SCAR_DATA": os.environ.get("SCAR_DATA", ""),
                "SCAR_SMOKE": os.environ.get("SCAR_SMOKE", ""),
                "SCAR_CALIB": os.environ.get("SCAR_CALIB", ""),
            },
        }
        # Persist launch metadata atomically
        save_json(_meta, OUTDIR / "launch.json")
        if _CALIBRATION_ACTIVE:
            # Touch a simple marker and log an explicit line for early visibility
            try:
                (OUTDIR / "calibration.marker").write_text("calibration=true\n")
            except Exception:
                pass
            try:
                with open(OUTDIR / "sweep.log", "a") as _lg:
                    print(
                        "[calib] hard override active; locked calibration mode declared at launch.",
                        file=_lg,
                        flush=True,
                    )
            except Exception:
                pass
except Exception:
    pass
DATA_ROOT = os.environ.get("SCAR_DATA", "./data")

C = 10  # CIFAR-10 classes


CFG.update(
    dict(
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
        gt_min_epoch=0,  # minimum epoch before soft GT can trigger (0 = use warm_idx)
        # workers & I/O
        num_workers=int(os.environ.get("SCAR_NUM_WORKERS", "0")),
        # ops controls
        max_failures=6,  # abort sweep if this many runs fail
        skip_cos_disp_in_smoke=True,  # short-circuit cosine_dispersion when SCAR_SMOKE=1
        skip_gns_in_smoke=True,  # short-circuit gradient_noise_scale when SCAR_SMOKE=1
        # compatibility / backend flags
        compat_mode=True,  # keep PH baselines on ripser during this calibration cycle
        lock_sw2_backend=True,  # reserved: keep a stable SW2 backend per run
    )
)

# Optional env overrides for regime knobs (apply early so smoke helpers see them).
# NOTE: These env vars only set CFG values; smoke overwrite/bypass is handled elsewhere.
try:
    for _env_name, _cfg_key, _cast in (
        ("SCAR_WARMUP", "warmup", int),
        ("SCAR_PH_BURN", "ph_burn", int),
        ("SCAR_GATE_WINDOW", "gate_window", int),
        ("SCAR_FACTOR_START_EPOCH", "factor_start_epoch", int),
        # GT env overrides (so SMOKE_ENV_BYPASS can act as a true override, not just a bypass)
        ("SCAR_GT_RANK_MIN", "gt_rank_min", float),
        ("SCAR_GT_PATIENCE", "gt_patience", int),
        ("SCAR_GT_MIN_EPOCH", "gt_min_epoch", int),
        ("SCAR_NEWMA_MIN_POINTS", "newma_min_points", int),
        ("SCAR_NEWMA_MIN_AGREEMENT", "newma_min_agreement", int),
        ("SCAR_NEWMA_CALIB_Q", "newma_calib_q", float),
    ):
        _v = os.environ.get(_env_name)
        if _v is None:
            continue
        try:
            CFG[_cfg_key] = _cast(str(_v).strip())
            if mp.current_process().name == "MainProcess":
                print(f"[env] {_cfg_key}={CFG[_cfg_key]} (from {_env_name})")
        except Exception as _e:
            if mp.current_process().name == "MainProcess":
                print(f"[WARN] bad {_env_name}={_v!r}: {_e}")
except Exception:
    pass

# Default family z-gate; overridden by SCAR_FAMILY_Z_THR if set
CFG.setdefault("family_z_thr", 2.903)


def _env_float_in_range(name: str, default: float, lo: float, hi: float) -> float:
    s = os.environ.get(name)
    if s is None:
        return default
    try:
        x = as_float(s, default=default)
        if lo <= x <= hi:
            return x
    except Exception:
        pass
    try:
        print(f"[WARN] ignoring {name}={s!r}; using {default}")
    except Exception:
        pass
    return default


def warm_idx_from_cfg(cfg: Dict[str, Any]) -> int:
    """Return warm index (warmup + ph_burn) using runtime CFG values."""
    return as_int(cfg.get("warmup"), default=0) + as_int(cfg.get("ph_burn"), default=0)


def gate_first_evaluable_epoch(cfg: Dict[str, Any]) -> int:
    """Return the first epoch where the gate can evaluate (needs 2W history)."""
    W = as_int(cfg.get("gate_window", 16), default=16)
    return 2 * W


def validate_gate_evaluability(cfg: Dict[str, Any], context: str = "") -> bool:
    """
    Validate that gate can actually evaluate given the epoch count.

    Uses gate_first_evaluable_epoch() to determine the actual evaluability
    threshold, ensuring this validator stays consistent with the gate logic.

    Args:
        cfg: Configuration dictionary
        context: Optional context string for log messages

    Returns:
        True if at least one epoch will be evaluable, False otherwise
    """
    first_evaluable = gate_first_evaluable_epoch(cfg)
    epochs = as_int(cfg.get("epochs", 72), default=72)
    W = as_int(cfg.get("gate_window", 16), default=16)

    # Gate can evaluate starting at epoch index `first_evaluable`
    # (0-indexed: needs `first_evaluable` prior epochs of history)
    # So we need epochs > first_evaluable for at least one evaluation
    evaluable_count = max(0, epochs - first_evaluable)

    ctx_str = f" ({context})" if context else ""

    if evaluable_count == 0:
        max_safe_W = max(1, (epochs - 1) // 2)
        msg = (
            f"[CRITICAL]{ctx_str} Gate will NEVER evaluate: "
            f"epochs={epochs}, gate_window={W}, first_evaluable_epoch={first_evaluable}. "
            f"Fix: set gate_window<={max_safe_W} OR epochs>={first_evaluable + 1}"
        )

        # MainProcess-only: print + sweep.log
        try:
            if mp.current_process().name == "MainProcess":
                print(msg, flush=True)
                _append_sweep_log(
                    f"CRITICAL{ctx_str}: gate_window={W} incompatible with epochs={epochs}; "
                    f"gate will never evaluate (first_evaluable={first_evaluable})"
                )
        except Exception:
            pass

        return False

    # Warn if very few epochs will be evaluable (< 4)
    if evaluable_count < 4:
        try:
            if mp.current_process().name == "MainProcess":
                print(
                    f"[WARN]{ctx_str} Only {evaluable_count} epochs will be evaluable "
                    f"(epochs={epochs}, gate_window={W}, first_evaluable={first_evaluable})",
                    flush=True,
                )
        except Exception:
            pass

    return True


def require_gate_eval(cfg: Dict[str, Any]) -> bool:
    """
    When True, we fail fast if the gate will never evaluate.

    Default policy:
      - Require in smoke (SCAR_SMOKE=1)
      - Require if explicitly requested (SCAR_REQUIRE_GATE_EVAL=1)
      - Optional heuristic: require when SCAR_CANARY_SEEDS is set
    """
    if env_truthy("SCAR_REQUIRE_GATE_EVAL"):
        return True
    if env_truthy("SCAR_SMOKE"):
        return True
    # Optional heuristic (keep only if you want it):
    if os.environ.get("SCAR_CANARY_SEEDS") is not None:
        return True
    return False


def require_gate_eval_or_die(cfg: Dict[str, Any], *, context: str = "") -> None:
    """
    Fail fast if gate evaluation is required but impossible.
    NOTE: No printing here. Logging is done in validate_gate_evaluability (MainProcess-only).
    """
    if not require_gate_eval(cfg):
        return

    ok = validate_gate_evaluability(cfg, context=context)
    if ok:
        return

    epochs = as_int(cfg.get("epochs", 72), default=72)
    W = as_int(cfg.get("gate_window", 16), default=16)
    first_eval = gate_first_evaluable_epoch(cfg)

    raise RuntimeError(
        f"Gate will never evaluate ({context}): epochs={epochs}, gate_window={W}, "
        f"first_evaluable_epoch={first_eval} (needs epochs >= {first_eval + 1})."
    )


def _warn_precedes_collapse(t_warn: Optional[int], t_c: Optional[int]) -> bool:
    """Causal validity check.
    - If no warning -> invalid (False).
    - If no collapse -> keep warning (True) so FP is measurable.
    - If collapse exists -> require precedence (<= in smoke, < otherwise).
    """

    if t_warn is None:
        return False
    if t_c is None:
        return True

    # Debug / correctness mode: force strict precedence even in smoke.
    if env_truthy("SCAR_STRICT_PRECEDENCE"):
        return int(t_warn) < int(t_c)
    if env_truthy("SCAR_SMOKE"):
        return int(t_warn) <= int(t_c)
    return int(t_warn) < int(t_c)


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

# Explicitly annotated containers (used later in the run)
buckets: dict[str, list[int]] = {}
logs: list[dict[str, Any]] = []
out: list[str] = []

# Vote baseline metrics — gradient/loss removed to avoid GT leakage / cadence bias.
# NOTE: eff_dim (projected/observable) is safe; eff_dim_gt would be target leakage.
VOTE_METRICS = ["cos_disp", "var_out_k", "ftle", "mon_entropy", "eff_dim"]

# TTL for scheduled metrics propagation to avoid stale ffill artifacts
CFG.setdefault("scheduled_ttl", 2 * CFG.get("heavy_every", 6))
CFG.setdefault("penult_dim", 512)
# Family gate neighborhood size (epochs) for local confirmation
CFG.setdefault("gate_early_exit", True)
try:
    if (OUTDIR / "window_decl_calibrated.json").exists() or (OUTDIR / "gate_gain_thresh_calibration.json").exists():
        CFG["gate_early_exit"] = 0
        # Optional during calibration: neuter warn persistence instead of early-exit switch
        # CFG["warn_consec"] = 10**9
except Exception:
    pass

# Finite-window heavy-metric budgets (env-overridable)
CFG.setdefault("total_heavy_budget_ms", 180_000)
CFG.setdefault("ripser_calls_cap", 1_000_000)
CFG.setdefault("sw2_calls_cap", 1_000_000)

# Per-run cumulative budgets for each heavy metric.
# Defaults: same as total_heavy_budget_ms so total still dominates.
CFG.setdefault("sw2_run_budget_ms", CFG.get("total_heavy_budget_ms", 180_000))
CFG.setdefault("ripser_run_budget_ms", CFG.get("total_heavy_budget_ms", 180_000))

# Gate parameters for fixed-partition finite realism (predeclared Φ_W)
CFG.setdefault("gate_window", 16)
CFG.setdefault("gate_bins", 16)
CFG.setdefault("gate_epsilon", 0.08)
CFG.setdefault("gate_epsilon_floor", 0.02)  # prevent degenerate ε=0 on small samples
CFG.setdefault("gate_eps_stat_max_frac", 0.25)
CFG.setdefault("gate_gain_thresh", 0.05)  # bits per sample (calibrated against controls)
CFG.setdefault("gate_gain_q", 0.05)
CFG.setdefault("gate_gain_thresh_cap", 0.15)  # cap for calibrated gain threshold (bits/sample)
CFG.setdefault("gate_epsilon_sens", 0.04)  # dedicated κ_sens budget
CFG.setdefault("gate_min_evidence", 16)
CFG.setdefault("gate_min_evidence_full_eps", 16)  # evidence at which ε inflation fully turns off
CFG.setdefault("gate_eps_inflation_max", 4.0)  # max multiplier for ε when evidence is low
CFG.setdefault("gate_gain_units", "bits/sample")

# NEWMA baseline: calibration + guards
CFG.setdefault("newma_lambda", None)  # None => calibrate from controls
CFG.setdefault("newma_min_points", 3)  # minimum finite points after burn before NEWMA can trigger
CFG.setdefault("newma_min_agreement", 2)  # minimum metrics that must agree for NEWMA warn
CFG.setdefault("newma_calib_q", 0.995)  # quantile for calibration

# Instantiate the global budget ledger using current CFG limits
try:
    _wb = WindowBudget(
        # Per-run cumulative budgets (ms) for each heavy metric
        sw2_ms=as_int(CFG.get("sw2_run_budget_ms", CFG.get("total_heavy_budget_ms", 180_000)), default=180_000),
        ripser_ms=as_int(CFG.get("ripser_run_budget_ms", CFG.get("total_heavy_budget_ms", 180_000)), default=180_000),
        total_heavy_ms=as_int(CFG.get("total_heavy_budget_ms", 180_000), default=180_000),
        sw2_calls=as_int(CFG.get("sw2_calls_cap", 1_000_000), default=1_000_000),
        ripser_calls=as_int(CFG.get("ripser_calls_cap", 1_000_000), default=1_000_000),
    )
    BUDGET = BudgetLedger(_wb)
except Exception:
    BUDGET = BudgetLedger(WindowBudget())

# Install shared runtime state so other legacy modules can see CFG/OUTDIR/BUDGET.
try:
    runtime.install_runtime(cfg=CFG, outdir=OUTDIR, budget=BUDGET)
except Exception:
    # Keep runner robust even if runtime wiring fails in the refactor sandbox.
    pass

# Smoke-mode overrides (fast E2E)
CFG_SMOKE = dict(
    seeds_calib=[401, 402, 403, 512],  # include one SOFT collapse seed for calibration
    seeds_eval=[511, 512],
    epochs=24,
    # run heavy metrics on a predictable cadence
    heavy_every=4,
    rp_repeats=1,
    sw2_n_proj=64,
    # gate: W=1 so 2W history exists by epoch 2 (matches earliest GT collapses in smoke)
    gate_window=1,
    gate_min_evidence=3,  # will be clamped down to a NaN-tolerant bound in reconcile_cfg_inplace()
    ph_min_points=3,  # smoke has ~7 finite heavy points; default 10 blocks PH detection
    gate_min_evidence_full_eps=12,
    # Gate epsilon: smoke calibration is underdetermined; use a realistic preset with a protective floor.
    gate_epsilon=0.50,
    gate_epsilon_floor=0.40,
    gate_eps_inflation_max=1.5,  # With a realistic base epsilon, inflation can be modest.
    gate_eps_stat_max_frac=0.10,  # Smoke: keep eps_stat from eating eps_scaled (reduces first-eval false fails)
    gate_bins=4,  # Smoke: finer TV resolution than bins=2 while staying light
    # warm / PH burn
    warmup=2,
    ph_burn=0,
    # delay pathological factors so the gate can build enough history before collapse onset
    factor_start_epoch=12,  # was 8
    # reduce persistence requirements in short runs
    warn_consec=2,
    # NEWMA: conservative settings for smoke to avoid spurious FPs from tiny calibration pool
    newma_min_agreement=2,
    newma_calib_q=0.99,
    detector_horizon=3,
    # smoke GT knobs (empirically grounded for eff_dim_gt distribution)
    gt_rank_min=1.30,  # must be below min healthy eff_dim_gt (~1.367 observed)
    gt_patience=5,  # was 3
    gt_min_epoch=10,  # NEW: don't allow soft GT before this epoch
    # Smoke: keep full traces; do not halt runs early unless explicitly enabled via SCAR_GATE_HALT=1
    gate_early_exit=False,
    gate_gain_q=0.05,
    gate_gain_thresh=0.02,
)

# Env vars that, if set, should prevent smoke from overwriting that key
SMOKE_ENV_BYPASS: Dict[str, str] = {
    "gate_bins": "SCAR_GATE_BINS",
    "gate_min_evidence": "SCAR_GATE_MIN_EVIDENCE",
    "gate_epsilon": "SCAR_GATE_EPSILON",
    "gate_epsilon_floor": "SCAR_GATE_EPSILON_FLOOR",
    "warn_consec": "SCAR_WARN_CONSEC",
    "family_window": "SCAR_FAMILY_WINDOW",
    "gate_gain_q": "SCAR_GATE_GAIN_Q",
    "gate_gain_thresh": "SCAR_GATE_GAIN_THRESH",
    "gate_min_evidence_full_eps": "SCAR_GATE_MIN_EVIDENCE_FULL_EPS",
    "gate_eps_inflation_max": "SCAR_GATE_EPS_INFLATION_MAX",
    "gate_eps_stat_max_frac": "SCAR_GATE_EPS_STAT_MAX_FRAC",
    "gate_early_exit": "SCAR_GATE_HALT",
    "warmup": "SCAR_WARMUP",
    "ph_burn": "SCAR_PH_BURN",
    "gate_window": "SCAR_GATE_WINDOW",
    "factor_start_epoch": "SCAR_FACTOR_START_EPOCH",
    "gt_rank_min": "SCAR_GT_RANK_MIN",
    "gt_patience": "SCAR_GT_PATIENCE",
    "gt_min_epoch": "SCAR_GT_MIN_EPOCH",
    "newma_min_agreement": "SCAR_NEWMA_MIN_AGREEMENT",
    "newma_calib_q": "SCAR_NEWMA_CALIB_Q",
    "newma_min_points": "SCAR_NEWMA_MIN_POINTS",
}

# Smoke-eligible keys (single source of truth).
# NOTE: “eligible” means smoke reconciliation MAY overwrite these keys *if*:
#   (a) the key is present in CFG_SMOKE (i.e., a smoke value is defined), and
#   (b) there is no active env bypass for that key (SMOKE_ENV_BYPASS), and
#   (c) for `epochs`, no explicit epoch clamp env var is present.
# In other words: this list is a whitelist; CFG_SMOKE is the assignment.
SMOKE_CRITICAL_KEYS: List[str] = [
    "epochs",
    "warmup",
    "ph_burn",
    "heavy_every",
    "metric_batches",
    "gate_window",
    "gate_bins",
    "gate_epsilon",
    "gate_epsilon_floor",
    "gate_min_evidence",
    "gate_min_evidence_full_eps",
    "gate_eps_inflation_max",
    "gate_eps_stat_max_frac",
    "gate_early_exit",
    "warn_consec",
    "family_window",
    "gate_gain_q",
    "gate_gain_thresh",
    "gt_rank_min",
    "gt_patience",
    "gt_min_epoch",
]


def _log_smoke_final_cfg(cfg: Dict[str, Any], stage: str = "") -> None:
    """Log final effective smoke CFG values after reconciliation."""
    if not env_truthy("SCAR_SMOKE"):
        return
    try:
        if mp.current_process().name != "MainProcess":
            return
        stage_tag = f" ({stage})" if stage else ""
        msg = (
            f"[smoke] Final CFG{stage_tag}: "
            f"gate_window={cfg.get('gate_window')} "
            f"gate_bins={cfg.get('gate_bins')} "
            f"gate_epsilon={cfg.get('gate_epsilon')} "
            f"gate_epsilon_floor={cfg.get('gate_epsilon_floor')} "
            f"gate_min_evidence={cfg.get('gate_min_evidence')} "
            f"gate_min_evidence_full_eps={cfg.get('gate_min_evidence_full_eps')} "
            f"gate_eps_inflation_max={cfg.get('gate_eps_inflation_max')} "
            f"gate_eps_stat_max_frac={cfg.get('gate_eps_stat_max_frac')} "
            f"warmup={cfg.get('warmup')} ph_burn={cfg.get('ph_burn')} "
            f"warm_idx={warm_idx_from_cfg(cfg)} epochs={cfg.get('epochs')} "
            f"warn_consec={cfg.get('warn_consec')} "
            f"gate_early_exit={cfg.get('gate_early_exit')}"
        )
        print(msg, flush=True)
        _append_sweep_log(msg)
    except Exception:
        pass


def reconcile_cfg_inplace(cfg: Dict[str, Any], *, stage: str = "") -> None:
    """Make cfg consistent with env + smoke policy.

    Call this after any load that can overwrite cfg (calibration, window_decl load, etc.).
    """
    # 1) Env clamps that reduce work (epochs / seed lists) should win.
    try:
        _apply_env_canary_clamps(cfg)
    except Exception:
        pass

    # 2) Smoke policy for eligible keys (whitelist).
    #    Only keys present in CFG_SMOKE are actually overwritten.
    if env_truthy("SCAR_SMOKE"):
        applied_changes: List[str] = []
        bypassed_keys: List[str] = []

        for k in SMOKE_CRITICAL_KEYS:
            if k not in CFG_SMOKE:
                continue

            # Special-case epochs: skip if any epoch clamp env is present.
            if k == "epochs":
                if (
                    os.environ.get("SCAR_MAX_EPOCHS") is not None
                    or os.environ.get("SCAR_EPOCHS") is not None
                    or os.environ.get("SCAR_N_EPOCHS") is not None
                ):
                    bypassed_keys.append(f"{k} (epoch env clamp)")
                    continue
            else:
                bypass_env = SMOKE_ENV_BYPASS.get(k)
                if bypass_env and os.environ.get(bypass_env) is not None:
                    bypassed_keys.append(f"{k} ({bypass_env}={os.environ.get(bypass_env)})")
                    continue

            old_val = cfg.get(k)
            new_val = CFG_SMOKE[k]
            if old_val != new_val:
                applied_changes.append(f"{k}: {old_val}->{new_val}")
            cfg[k] = new_val

        # Log what was changed (main process only, also to sweep.log)
        try:
            if mp.current_process().name == "MainProcess":
                if applied_changes:
                    preview = applied_changes[:5]
                    ellipsis = "..." if len(applied_changes) > 5 else ""
                    msg = f"[smoke][reconcile:{stage}] Applied {len(applied_changes)} overrides: {preview}{ellipsis}"
                    print(msg, flush=True)
                    _append_sweep_log(msg)
                if bypassed_keys:
                    msg = (
                        f"[smoke][reconcile:{stage}] Bypassed {len(bypassed_keys)} keys: "
                        f"{bypassed_keys[:3]}{'...' if len(bypassed_keys) > 3 else ''}"
                    )
                    print(msg, flush=True)
                    _append_sweep_log(msg)
        except Exception:
            pass

    # 3) Always-on invariants.
    try:
        cfg["var_k_max"] = max(1, int(cfg.get("var_k_max", 32)))
    except Exception:
        pass

    # 4) Smoke: keep gate_min_evidence achievable with NaN tolerance.
    #    With W=3 and 2 metrics, ideal evidence = 6. But one metric may be partially/fully NaN,
    #    so we require only (W * 1) = 3 to tolerate one fully-missing metric.
    #
    # Do not force the gate to refuse evaluation in smoke; if the current value is too high,
    # clamp down to the NaN-tolerant bound unless the user explicitly set an env override.
    if env_truthy("SCAR_SMOKE"):
        W = as_int(cfg.get("gate_window", 3), default=3)
        n_metrics = 2  # assume var_out_k + eff_dim feed the gate in smoke
        ideal_evidence = max(1, int(W) * int(n_metrics))
        nan_tolerant_evidence = max(1, int(W) * max(1, int(n_metrics) - 1))
        if os.environ.get("SCAR_GATE_MIN_EVIDENCE") is None:
            cur = as_int(cfg.get("gate_min_evidence", nan_tolerant_evidence), default=nan_tolerant_evidence)
            if cur > nan_tolerant_evidence:
                cfg["gate_min_evidence"] = int(nan_tolerant_evidence)
                try:
                    print(
                        f"[smoke] gate_min_evidence clamped to {nan_tolerant_evidence} "
                        f"(NaN-tolerant; ideal={ideal_evidence})"
                    )
                except Exception:
                    pass

    # 5) Smoke: ensure epsilon-inflation knobs are set for low-evidence quantization artifacts.
    if env_truthy("SCAR_SMOKE"):
        W = as_int(cfg.get("gate_window", 3), default=3)
        n_metrics = 2
        ideal_evidence = max(1, int(W) * int(n_metrics))

        # Smoke: keep full-eps evidence strictly above the first evaluated evidence so inflation is ON
        # when total_evidence == ideal_evidence (common in short smoke runs).
        # Default target is 2× ideal evidence, but never below min_evidence+1.
        if os.environ.get("SCAR_GATE_MIN_EVIDENCE_FULL_EPS") is None:
            min_ev_now = as_int(cfg.get("gate_min_evidence", 3), default=3)
            full_target = max(int(2 * ideal_evidence), int(min_ev_now) + 1)
            cur_full = as_int(cfg.get("gate_min_evidence_full_eps", full_target), default=full_target)
            if cur_full < full_target:
                cfg["gate_min_evidence_full_eps"] = int(full_target)

        # Validate / sanitize inflation knob in smoke unless explicitly overridden.
        # Do NOT force it upward (that can make the gate vacuously loose);
        # just ensure it is finite and >= 1.0.
        if os.environ.get("SCAR_GATE_EPS_INFLATION_MAX") is None:
            try:
                _default_infl = float(CFG_SMOKE.get("gate_eps_inflation_max", 4.0))
            except Exception:
                _default_infl = 4.0
            cur_infl = as_float(cfg.get("gate_eps_inflation_max", _default_infl), default=_default_infl)
            if (not np.isfinite(cur_infl)) or (float(cur_infl) < 1.0):
                cfg["gate_eps_inflation_max"] = float(_default_infl)

        # In smoke, prevent eps_stat from consuming most of eps_scaled.
        # This reduces first-eval false fails under coarse binning (e.g., bins=2) and tiny evidence.
        if os.environ.get("SCAR_GATE_EPS_STAT_MAX_FRAC") is None:
            cur_frac = as_float(cfg.get("gate_eps_stat_max_frac", 0.10), default=0.10)
            if (not np.isfinite(cur_frac)) or (cur_frac < 0.0) or (cur_frac > 1.0):
                cur_frac = 0.10
            # Never increase an explicit config value; only clamp down if too large.
            if float(cur_frac) > 0.10:
                cfg["gate_eps_stat_max_frac"] = 0.10


def reconcile_window_decl_inplace(window_decl: Any, cfg: Dict[str, Any]) -> None:
    """Apply smoke-mode policy to a WindowDecl in-place (e.g., bins)."""
    if not env_truthy("SCAR_SMOKE"):
        return
    # Respect explicit env override for gate bins.
    if os.environ.get("SCAR_GATE_BINS") is None:
        try:
            bins_smoke = int(cfg.get("gate_bins", CFG_SMOKE.get("gate_bins", 4)))
            cur_bins = int(getattr(window_decl, "bins", bins_smoke))
            if cur_bins != bins_smoke:
                try:
                    if mp.current_process().name == "MainProcess":
                        print(f"[smoke] overriding WindowDecl.bins: {cur_bins} -> {bins_smoke}")
                except Exception:
                    pass
            window_decl.bins = int(bins_smoke)
        except Exception:
            pass


def apply_smoke_overrides_inplace(cfg: Dict[str, Any]) -> None:
    """Legacy smoke-mode helper.

    Only applies non-critical smoke behaviour (e.g. factor_start_epoch); core
    hyperparameters are reconciled via reconcile_cfg_inplace().
    """
    try:
        if env_truthy("SCAR_SMOKE"):
            # Only apply non-critical entries; critical keys are handled in reconcile_cfg_inplace.
            applied_keys: List[str] = []
            for k, v in CFG_SMOKE.items():
                if k in SMOKE_CRITICAL_KEYS:
                    continue
                bypass_env = SMOKE_ENV_BYPASS.get(k)
                if bypass_env and os.environ.get(bypass_env) is not None:
                    continue  # respect explicit env override
                cfg[k] = v
                applied_keys.append(k)

            # Ensure pathological factors start after a healthy prefix in smoke mode
            try:
                factor_start = as_int(cfg.get("factor_start_epoch", 8), default=8)
                facs = cfg.get("factors", [])
                if isinstance(facs, list):
                    for f in facs:
                        if not isinstance(f, dict):
                            continue
                        nm = str(f.get("name", "")).lower().strip()
                        if nm and nm not in ("none", "none_safe"):
                            f.setdefault("factor_start_epoch", factor_start)
            except Exception:
                pass

            try:
                if mp.current_process().name == "MainProcess":
                    # Only log what was actually applied here (non-critical keys). Critical keys
                    # (gate_window, gate_bins, gate_min_evidence, etc.) are applied in reconcile_cfg_inplace().
                    keys_preview = ", ".join(applied_keys[:5])
                    ell = "..." if len(applied_keys) > 5 else ""
                    print(
                        f"[smoke] Applied (legacy, non-critical only): "
                        f"factor_start_epoch={cfg.get('factor_start_epoch')} "
                        f"(keys: {keys_preview}{ell})"
                    )
            except Exception:
                pass
    except Exception:
        pass


# Apply smoke overrides early so downstream calibration/eval uses consistent warm_idx
apply_smoke_overrides_inplace(CFG)

# Re-install shared runtime state so pipeline/modules see the smoke-adjusted CFG too.
try:
    runtime.install_runtime(cfg=CFG, outdir=OUTDIR, budget=BUDGET)
except Exception:
    pass


def seeds_for_eval_from_env(CFG_dict):
    """Return seeds for evaluation based on SCAR_EVAL_SPLIT.

    SCAR_EVAL_SPLIT: "eval" (default) | "calib" | "both".

    Smoke-mode hardening: if SCAR_SMOKE is truthy and SCAR_EVAL_SPLIT is not set,
    default to "both" so FP denominators exist in smoke evaluation.
    """
    raw = os.environ.get("SCAR_EVAL_SPLIT")
    if raw is None and env_truthy("SCAR_SMOKE"):
        mode = "both"
    else:
        mode = (raw or "eval").lower().strip()

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
        s = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL, text=True)[
            :50000
        ]
        return hashlib.md5(s.encode()).hexdigest()
    except Exception:
        return ""


# --- Calibration loader (file → env overrides) + provenance capsule ---
def _read_text_safe(p: Path) -> str:
    try:
        return p.read_text()
    except Exception:
        return ""


def load_calibration(cfg_path: Optional[str]) -> tuple[dict, Optional[str], Optional[str]]:
    """
    Returns: (cal, sha16, cfg_path_str)
    Priority: packaged default file -> user file (if provided) -> env overrides.
    """
    pkg_default = Path(__file__).resolve().parent / "configs" / "phase4_cifar10_v0.1.0.json"
    cal = {
        "gate_min_evidence": 16,
        "gate_gain_thresh": 0.05,
        "gate_gain_units": "bits/sample",
    }
    sha16 = None
    used_path: Optional[Path] = None

    # 1) packaged default
    if pkg_default.exists():
        txt = _read_text_safe(pkg_default)
        try:
            obj = json.loads(txt)
            for k in ("gate_min_evidence", "gate_gain_thresh", "gate_gain_units"):
                if k in obj:
                    cal[k] = obj[k]
            for k in [
                "gate_window",
                "gate_bins",
                "gate_epsilon",
                "gate_eps_stat_max_frac",
                "gate_epsilon_sens",
                "gate_eps_stat_alpha",
            ]:
                if k in obj:
                    CFG[k] = obj[k]
            sha16 = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16]
            used_path = pkg_default
        except Exception:
            pass

    # 2) explicit user path (overrides packaged)
    if cfg_path:
        user_p = Path(cfg_path)
        if user_p.exists():
            txt = _read_text_safe(user_p)
            try:
                obj = json.loads(txt)
                for k in ("gate_min_evidence", "gate_gain_thresh", "gate_gain_units"):
                    if k in obj:
                        cal[k] = obj[k]
                # Allow user JSON to set the same CFG pointers as the packaged default
                for k in [
                    "gate_window",
                    "gate_bins",
                    "gate_epsilon",
                    "gate_eps_stat_max_frac",
                    "gate_epsilon_sens",
                    "gate_eps_stat_alpha",
                ]:
                    if k in obj:
                        CFG[k] = obj[k]
                sha16 = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16]
                used_path = user_p
            except Exception:
                pass

    # 3) env overrides (highest precedence)
    if "SCAR_GATE_MIN_EVIDENCE" in os.environ:
        try:
            cal["gate_min_evidence"] = int(os.environ["SCAR_GATE_MIN_EVIDENCE"])
        except Exception:
            pass
    if "SCAR_GATE_GAIN_THRESH" in os.environ:
        try:
            cal["gate_gain_thresh"] = float(os.environ["SCAR_GATE_GAIN_THRESH"])
        except Exception:
            pass
    if "SCAR_GATE_GAIN_UNITS" in os.environ:
        cal["gate_gain_units"] = os.environ["SCAR_GATE_GAIN_UNITS"]

    # micro-validation (guard types; cal is heterogeneous so mypy treats values as object)
    try:
        gme = as_int(cal.get("gate_min_evidence", 16), default=16)
        cal["gate_min_evidence"] = 16 if gme < 1 else gme
    except Exception:
        cal["gate_min_evidence"] = 16
    try:
        x = as_float(cal.get("gate_gain_thresh", 0.05), default=0.05)
        cal["gate_gain_thresh"] = 0.05 if not (0.0 <= x < 1.0) else x
    except Exception:
        cal["gate_gain_thresh"] = 0.05
    if not isinstance(cal.get("gate_gain_units", None), str):
        cal["gate_gain_units"] = "bits/sample"

    return cal, sha16, (str(used_path) if used_path is not None else None)


def write_calibration_capsule(outdir: Path, cal: dict, src_path: Optional[str], sha16: Optional[str]) -> None:
    """Persist calibration details alongside other provenance files."""
    try:
        payload = {
            "source_path": src_path or "",
            "sha16": sha16 or "",
            "gate_min_evidence": as_int(cal.get("gate_min_evidence"), default=16),
            "gate_gain_thresh": as_float(cal.get("gate_gain_thresh"), default=0.05),
            "gate_gain_units": str(cal.get("gate_gain_units", "bits/sample")),
            "gate_window": as_int(CFG.get("gate_window", 16), default=16),
            "gate_bins": as_int(CFG.get("gate_bins", 16), default=16),
            "gate_epsilon": as_float(CFG.get("gate_epsilon", 0.08), default=0.08),
            "gate_epsilon_floor": as_float(CFG.get("gate_epsilon_floor", 0.02), default=0.02),
            "gate_eps_stat_max_frac": as_float(CFG.get("gate_eps_stat_max_frac", 0.25), default=0.25),
            "gate_epsilon_sens": as_float(CFG.get("gate_epsilon_sens", 0.04), default=0.04),
            "gate_eps_stat_alpha": as_float(CFG.get("gate_eps_stat_alpha", 0.05), default=0.05),
        }
        update_json(outdir / "calibration_provenance.json", payload)
    except Exception:
        pass


# Load packaged/user/env calibration and stamp provenance
try:
    _cal, _cal_sha16, _cal_path = load_calibration(os.environ.get("SCAR_CALIBRATION"))
    CFG["gate_min_evidence"] = as_int(_cal.get("gate_min_evidence", 16), default=16)
    CFG["gate_gain_thresh"] = as_float(_cal.get("gate_gain_thresh", 0.05), default=0.05)
    CFG["gate_gain_units"] = str(_cal.get("gate_gain_units", "bits/sample"))
    try:
        write_calibration_capsule(OUTDIR, _cal, _cal_path, _cal_sha16)
    except Exception:
        pass
    if mp.current_process().name == "MainProcess":
        print(
            f"[cal] min_evidence={CFG['gate_min_evidence']} gain_thresh={CFG['gate_gain_thresh']:.3f} ({CFG['gate_gain_units']})"
        )
    # Reapply environment overrides LAST so they win over any preloads
    try:
        v = os.environ.get("SCAR_GATE_MIN_EVIDENCE")
        if v is not None:
            try:
                CFG["gate_min_evidence"] = int(v)
                if mp.current_process().name == "MainProcess":
                    print(f"[env] gate_min_evidence={CFG['gate_min_evidence']}")
            except Exception as _e:
                if mp.current_process().name == "MainProcess":
                    print(f"[WARN] bad SCAR_GATE_MIN_EVIDENCE={v!r}: {_e}")
        v = os.environ.get("SCAR_GATE_GAIN_THRESH")
        if v is not None:
            try:
                CFG["gate_gain_thresh"] = float(v)
                if mp.current_process().name == "MainProcess":
                    print(f"[env] gate_gain_thresh={CFG['gate_gain_thresh']:.4f}")
            except Exception as _e:
                if mp.current_process().name == "MainProcess":
                    print(f"[WARN] bad SCAR_GATE_GAIN_THRESH={v!r}: {_e}")
        v = os.environ.get("SCAR_GATE_GAIN_UNITS")
        if v is not None:
            CFG["gate_gain_units"] = str(v)
            if mp.current_process().name == "MainProcess":
                print(f"[env] gate_gain_units={CFG['gate_gain_units']!r}")
        v = os.environ.get("SCAR_GATE_EPSILON")
        if v is not None:
            try:
                CFG["gate_epsilon"] = float(v)
                if mp.current_process().name == "MainProcess":
                    print(f"[env] gate_epsilon={CFG['gate_epsilon']:.6f}")
            except Exception as _e:
                if mp.current_process().name == "MainProcess":
                    print(f"[WARN] bad SCAR_GATE_EPSILON={v!r}: {_e}")
        v = os.environ.get("SCAR_GATE_HALT")
        if v is not None:
            try:
                CFG["gate_early_exit"] = 1 if env_truthy("SCAR_GATE_HALT") else 0
                if mp.current_process().name == "MainProcess":
                    print(f"[env] gate_early_exit={int(bool(CFG['gate_early_exit']))} (from SCAR_GATE_HALT)")
            except Exception as _e:
                if mp.current_process().name == "MainProcess":
                    print(f"[WARN] bad SCAR_GATE_HALT={v!r}: {_e}")
    except Exception:
        pass
    try:
        ge = float(CFG.get("gate_epsilon", 0.08))
        if not (0.005 <= ge <= 0.5):
            if mp.current_process().name == "MainProcess":
                print(f"[WARN] gate_epsilon out of range ({ge}); clamping to [0.005, 0.5]")
            CFG["gate_epsilon"] = float(min(max(ge, 0.005), 0.5))
    except Exception:
        CFG["gate_epsilon"] = 0.08
    # Emit a simple precedence summary for auditability
    try:
        _prec = {
            "calibration_loaded_from": _cal_path or "",
            "env": {
                "SCAR_GATE_MIN_EVIDENCE": os.environ.get("SCAR_GATE_MIN_EVIDENCE", ""),
                "SCAR_GATE_GAIN_THRESH": os.environ.get("SCAR_GATE_GAIN_THRESH", ""),
                "SCAR_GATE_GAIN_UNITS": os.environ.get("SCAR_GATE_GAIN_UNITS", ""),
                "SCAR_GATE_HALT": os.environ.get("SCAR_GATE_HALT", ""),
            },
            "final": {
                "gate_min_evidence": as_int_cfg(CFG.get("gate_min_evidence", 16), default=16),
                "gate_gain_thresh": as_float(CFG.get("gate_gain_thresh", 0.05), default=0.05),
                "gate_gain_units": str(CFG.get("gate_gain_units", "bits/sample")),
                "gate_window": as_int_cfg(CFG.get("gate_window", 16), default=16),
                "gate_bins": as_int_cfg(CFG.get("gate_bins", 16), default=16),
                "gate_epsilon": as_float(CFG.get("gate_epsilon", 0.08), default=0.08),
                "gate_epsilon_floor": as_float(CFG.get("gate_epsilon_floor", 0.02), default=0.02),
                "gate_eps_stat_max_frac": as_float(CFG.get("gate_eps_stat_max_frac", 0.25), default=0.25),
                "gate_epsilon_sens": as_float(CFG.get("gate_epsilon_sens", 0.04), default=0.04),
                "gate_eps_stat_alpha": as_float(CFG.get("gate_eps_stat_alpha", 0.05), default=0.05),
                "gate_early_exit": bool(CFG.get("gate_early_exit", True)),
                "run_tag": RUN_TAG,
                "calibration_mode": bool(CAL),
                "smoke_mode": bool(SMOKE),
            },
        }
        save_json(_prec, OUTDIR / "precedence_summary.json")
    except Exception:
        pass
except Exception as e:
    try:
        print(f"[WARN] calibration load failed: {e!r}; using defaults")
    except Exception:
        pass

# Reconcile CFG once more after calibration load (calibration may have overwritten gate_* knobs)
try:
    reconcile_cfg_inplace(CFG, stage="post_calibration")
    _log_smoke_final_cfg(CFG, stage="post_calibration")
    runtime.install_runtime(cfg=CFG, outdir=OUTDIR, budget=BUDGET)
except Exception:
    pass

# --- Smoke-mode sanity: clamp unreachable min_evidence (avoid never-evaluated gates) ---
try:
    if env_truthy("SCAR_SMOKE") and os.environ.get("SCAR_GATE_MIN_EVIDENCE") is None:
        W = as_int(CFG.get("gate_window", 3), default=3)
        min_ev = as_int(CFG.get("gate_min_evidence", 4), default=4)

        # Conservative in-practice upper bound: ~W samples per metric.
        # Smoke default assumes ~2 metrics feed the gate.
        n_metrics = 2
        max_possible = max(1, int(W) * int(n_metrics))

        if min_ev > max_possible:
            try:
                print(
                    f"[smoke][WARN] gate_min_evidence={min_ev} likely unreachable with "
                    f"gate_window={W} and n_metrics~{n_metrics}; clamping to {max_possible}"
                )
            except Exception:
                pass
            CFG["gate_min_evidence"] = int(max_possible)
except Exception:
    pass

# Repro capsule: persist environment/cuda/library hashes once per sweep
try:
    repro_path = OUTDIR / "repro.json"
    if mp.current_process().name == "MainProcess":
        if not repro_path.exists():
            save_json(
                {
                    "cuda_hash_md5": _cuda_hash(),
                    "pip_freeze_md5": _pip_freeze_md5(),
                    "torch": torch.__version__,
                    "torchvision": torchvision.__version__,
                    "cuda": torch.version.cuda,
                    "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
                    "deterministic": bool(CFG.get("deterministic", True)),
                },
                repro_path,
            )
except Exception:
    pass


# Echo calibration state to stdout once
try:
    if mp.current_process().name == "MainProcess":
        if str(os.environ.get("SCAR_CALIB", "0")).strip() in ("1", "true", "TRUE", "yes", "on"):
            print(
                "[calib] launch declared in calibration mode (SCAR_CALIB=1). Locked keys should be respected by drivers."
            )
except Exception:
    pass


def safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to parquet; fall back to CSV on failure.

    This runner is used in long sweeps where robustness matters more than strict
    parquet availability (e.g., missing pyarrow/fastparquet, transient IO issues).
    This helper must never raise.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        df.to_parquet(path, index=False)
        return
    except Exception as e:
        # Fallback to CSV with the same basename.
        try:
            csvp = path.with_suffix(".csv")
            df.to_csv(csvp, index=False)
            try:
                print(f"[WARN] parquet write failed ({path.name}); wrote CSV {csvp.name}: {e!r}")
            except Exception:
                pass
        except Exception:
            # Last resort: swallow to keep sweeps alive.
            try:
                print(f"[WARN] failed to write parquet or CSV for {path}: {e!r}")
            except Exception:
                pass


def _epochwise_collapse_tag_gt(epochs: np.ndarray, t_gt: Optional[int], tag: str, warm_idx: int) -> np.ndarray:
    """Return per-epoch GT tags in {'none','soft','hard'}.

    Guardrails:
    - pre-warm epochs are always 'none'
    - if t_gt is None or tag not in {'soft','hard'}, all epochs are 'none'
      (with normalization: accepts case/whitespace and prefix variants like 'SOFT', 'soft ', 'soft_collapse')
    - otherwise epochs >= t_gt and >= warm_idx take `tag`
    """
    ep = np.asarray(epochs, dtype=int)
    out = np.full(ep.shape, "none", dtype=object)

    if t_gt is None:
        return out

    # Normalize tag: accept case/whitespace/prefix variants from upstream GT logic.
    # This prevents silent all-'none' tagging when gt_collapse_time returns e.g. 'SOFT' or 'soft_*'.
    try:
        tag_s = str(tag).strip().lower()
    except Exception:
        tag_s = ""

    if tag_s.startswith("soft"):
        tag_s = "soft"
    elif tag_s.startswith("hard"):
        tag_s = "hard"
    else:
        return out

    # tag turns on at/after collapse time, but never before warm
    mask = (ep >= int(t_gt)) & (ep >= int(warm_idx))
    out[mask] = tag_s

    # hard guarantee
    out[ep < int(warm_idx)] = "none"
    return out


def _runlevel_collapse_tag(df: pd.DataFrame, warm_idx: int) -> str:
    """Reduce epochwise GT tags to a single run-level tag (hard > soft > none).

    Preference is to consider post-warm epochs only when available.
    """
    if df is None or (not isinstance(df, pd.DataFrame)) or df.empty:
        return "none"
    if "collapse_tag_gt" not in df.columns:
        return "none"

    use = df
    if "epoch" in df.columns:
        ep = pd.to_numeric(df["epoch"], errors="coerce")
        post = df[ep >= int(warm_idx)]
        if not post.empty:
            use = post

    tags = {str(x) for x in use["collapse_tag_gt"].dropna().astype(str).unique().tolist()}
    if "hard" in tags:
        return "hard"
    if "soft" in tags:
        return "soft"
    return "none"


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
        df[col] = df.groupby(keys)[col].transform(lambda s: s.rolling(k, min_periods=k).sum().ge(k))
    return df


# --- Inserted shims for text and numeric conversion ---
def _as_text(x: Any) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="replace")
    return str(x)


def safe_int(x: Any) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, str):
        v = x.strip()
        return int(v) if v and v.lstrip("+-").isdigit() else 0
    if isinstance(x, (float, np.floating)):
        return int(x)
    try:
        return int(float(x))
    except Exception:
        return 0


def safe_float(x: Any) -> float:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except Exception:
            return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


# --- General numeric guards for Any/object-typed inputs ---

# --- FR integration (optional import; runtime holders below) ---
try:
    from veriscope.fr_integration import (
        FRWindow,
        DeclTransport,
        GateEngine,
        WindowDecl,
        init_fr,
        resolve_eps as _FR_RESOLVE_EPS,
        install_window_decl,
        write_window_audit,
        write_window_provenance_from_decl,
    )
except Exception:
    # Keep core/legacy types intact; only null out optional FR wiring hooks
    init_fr = None  # type: ignore[assignment]
    _FR_RESOLVE_EPS = None  # type: ignore[assignment]
    install_window_decl = None  # type: ignore[assignment]
    write_window_audit = None  # type: ignore[assignment]
    write_window_provenance_from_decl = None  # type: ignore[assignment]
    FRWindow = None  # type: ignore[assignment]
    DeclTransport = None  # type: ignore[assignment]
    GateEngine = None  # type: ignore[assignment]
    WindowDecl = None  # type: ignore[assignment]

# If fr_integration exposes a resolver, hook it into AGGREGATE_EPSILON
if _FR_RESOLVE_EPS is not None:
    AGGREGATE_EPSILON = _FR_RESOLVE_EPS  # type: ignore[assignment]

# --- FR runtime holders (do not shadow types) ---
from typing import Callable, Iterable


# ---- tiny shims that mypy loves ----
from typing import Any, Iterable, overload, Union


def as_opt_int(x: Any) -> Optional[int]:
    """Int or None; use when the target really is optional."""
    v = as_int(x, default=-1)
    return v if v >= 0 else None


def s2f(s: Any) -> np.ndarray:
    """Series[Any] | None -> np.ndarray[float], tolerant of junk."""
    if isinstance(s, pd.Series):
        try:
            return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        except Exception:
            pass
    return np.array([], dtype=float)


def _as_float_guard_dup(x: Any, default: float = 0.0) -> float:
    try:
        if isinstance(x, (np.floating,)):
            return float(x)  # type: ignore[arg-type]
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, (str, bytes, bytearray)):
            return float(str(x))
        if hasattr(x, "__float__"):
            return float(x)  # type: ignore[arg-type]
    except Exception:
        pass
    return default


# ---- typed guards useful for mypy-clean comparisons/membership ----


def safe_in(item: Any, container: Any) -> bool:
    """Like `item in container` but returns False for non-iterables."""
    try:
        if isinstance(container, _Iter):
            return item in container  # type: ignore[operator]
        return False
    except Exception:
        return False


# --- Inserted helper: as_json_dict ---
def as_json_dict(x: Any) -> dict[str, Any]:
    """Best-effort normalize bytes/str/dict to a JSON dict; otherwise return {}."""
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", errors="replace")
        except Exception:
            return {}
    if isinstance(x, str):
        try:
            obj = json.loads(x)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    if isinstance(x, dict):
        return x
    return {}


def _is_admissible_control(row_or_df: Union[pd.Series, pd.DataFrame]) -> Union[bool, pd.Series]:
    """Admissibility predicate for calibration control pools.

    A row/run is admissible iff:
      1) factor == 'none' (case/whitespace tolerant)
      2) PRE-GT stamp says collapse_tag == 'none' (if available)

    If the PRE-GT stamp column is missing, this predicate defaults to allowing rows
    (so older bundles remain usable), but callers SHOULD log if they expected the column.
    """
    if isinstance(row_or_df, pd.DataFrame):
        df = row_or_df
        if "factor" not in df.columns:
            return pd.Series(False, index=df.index)
        factor_ok = df["factor"].astype(str).str.lower().str.strip() == "none"
        gt_ok: Union[bool, pd.Series] = True
        if "_run_collapse_tag_gt_precal" in df.columns:
            gt_ok = df["_run_collapse_tag_gt_precal"].astype(str) == "none"
        return factor_ok & gt_ok

    # row path
    row = row_or_df
    factor_ok = str(row.get("factor", "")).lower().strip() == "none"
    gt_ok = str(row.get("_run_collapse_tag_gt_precal", "none")) == "none"
    return bool(factor_ok and gt_ok)


#
# ---------------------------
# Model & schedule
# ---------------------------


def make_opt(model):
    """Compatibility wrapper: delegate to legacy.model.make_opt using global CFG."""
    return _make_opt_model(model, CFG)


#
# ---------------------------
# Metrics (multi-batch mean/std)
# ---------------------------


# --- monitor_margin_median: median logit margin (top-1 minus top-2) over loader (label-free)


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
    mask_bool = per <= thr
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


# --- Helper for calibrating gate_gain_thresh from controls ---
def calibrate_gate_gain_thresh_from_controls(df: pd.DataFrame, W: int, warm: int, q: float = 0.05) -> float:
    """
    Calibrate gate_gain threshold (in bits/sample) from control ('none') runs post-warm.
    Uses sliding windows of length W over (ewma_loss - train_loss), averaged per finite window.
    Returns the q-quantile as the threshold and writes a small capsule to OUTDIR.
    """
    # Defensive clamp: q must be a valid quantile in [0, 1]
    try:
        q = float(q)
    except Exception:
        q = 0.05
    q = float(min(max(q, 0.0), 1.0))
    try:
        ep = pd.to_numeric(df["epoch"], errors="coerce")
        sub = df[_is_admissible_control(df) & (ep >= int(warm))].copy()
    except Exception:
        return float(CFG.get("gate_gain_thresh", 0.05))
    vals = []
    try:
        for _, g in sub.groupby(["seed", "factor"]):
            g = g.sort_values("epoch")
            x = to_numeric_opt(g.get("train_loss"), errors="coerce").to_numpy(dtype=float)
            b = to_numeric_opt(g.get("ewma_loss"), errors="coerce").to_numpy(dtype=float)
            if len(x) < W or len(b) < W:
                continue
            for i in range(len(x) - W + 1):
                xs = x[i : i + W]
                bs = b[i : i + W]
                m = np.isfinite(xs) & np.isfinite(bs)
                n = int(m.sum())
                if n == 0:
                    continue
                baseline = (
                    str(os.environ.get("SCAR_GATE_GAIN_BASELINE", CFG.get("gate_gain_baseline", "uniform")))
                    .strip()
                    .lower()
                )

                ln2 = math.log(2.0)
                if baseline == "uniform":
                    base_nats = float(math.log(max(2, int(CFG.get("num_classes", 10)))))
                    gain_bits = (base_nats - float(np.mean(xs[m]))) / ln2
                else:
                    gain_bits = float(np.mean(bs[m] - xs[m])) / ln2

                vals.append(float(gain_bits))
    except Exception:
        pass
    if not vals:
        return float(CFG.get("gate_gain_thresh", 0.05))
    thr = float(np.quantile(np.array(vals, dtype=float), q))
    cap = as_float(CFG.get("gate_gain_thresh_cap", 0.15), default=0.15)
    thr = float(max(0.0, min(thr, cap)))
    try:
        save_json(
            {"gate_gain_thresh": thr, "q": q, "warm": int(warm), "W": int(W)},
            OUTDIR / "gate_gain_thresh_calibration.json",
        )
    except Exception:
        pass
    return thr


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


def _coh_persist_below(corr_series: np.ndarray, thresh: float = 0.85, min_len: int = 5) -> np.ndarray:
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


# --- flatten model parameters and compute L2 norm (for weight drift)
def _flatten_params_l2(model: nn.Module) -> Tuple[float, torch.Tensor]:
    vec = []
    with torch.no_grad():
        for p in model.parameters():
            vec.append(p.detach().float().view(-1).cpu())
    if not vec:
        return 0.0, torch.zeros(0, dtype=torch.float32)
    v = torch.cat(vec, dim=0)
    return float(t_vector_norm(v).item()), v


# --- gradient noise scale (per-batch, microbatch split)
def gradient_noise_scale(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor, micro: int = 4) -> Tuple[float, int]:
    """Compute a cheap GNS proxy on a single batch by comparing microbatch gradients.
    BN‑safe: does not mutate running_mean/var or num_batches_tracked. Restores modes.
    Returns (gns, n_micro) with gns=nan on failure.
    """
    if env_truthy("SCAR_SMOKE") and CFG.get("skip_gns_in_smoke", False):
        return float("nan"), 0
    was_training = model.training

    # Snapshot BatchNorm buffers and per‑module training flags; set BN to eval so buffers don't update
    bn_state = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nbt = getattr(m, "num_batches_tracked", None)

            # --- BN anchor / device-dtype guard (mypy-safe) ---
            anchor = (
                m.running_mean
                if getattr(m, "running_mean", None) is not None
                else (m.weight if getattr(m, "weight", None) is not None else None)
            )
            if isinstance(anchor, torch.Tensor):
                dev = anchor.device
                dt = anchor.dtype
            else:
                dev = torch.device("cpu")
                dt = torch.float32

            # snapshot BN buffers with Tensor-aware clone guards
            rm = getattr(m, "running_mean", None)
            rv = getattr(m, "running_var", None)
            mu_snap = rm.clone() if isinstance(rm, torch.Tensor) else torch.zeros(m.num_features, device=dev, dtype=dt)
            var_snap = rv.clone() if isinstance(rv, torch.Tensor) else torch.ones(m.num_features, device=dev, dtype=dt)

            bn_state.append(
                (
                    m,
                    mu_snap,
                    var_snap,
                    (nbt.clone() if isinstance(nbt, torch.Tensor) else nbt),
                    m.training,
                )
            )
            m.eval()

    try:
        # Train mode for non‑BN layers so dropout/etc. match training behavior
        model.train()
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

        if len(grads) < 2:
            return float("nan"), len(grads)
        if any((g is None) or (getattr(g, "numel", lambda: 0)() == 0) for g in grads):
            return float("nan"), 0

        G = torch.stack([g if g.numel() > 0 else torch.zeros_like(grads[0]) for g in grads], dim=0)
        gbar = G.mean(dim=0)
        num = torch.sum((G - gbar).pow(2))
        den = torch.sum(gbar.pow(2)) + 1e-12
        gns = float((num / den).item() * (len(grads) / max(1, len(grads) - 1)))
        return gns, len(grads)
    except Exception:
        return float("nan"), 0
    finally:
        # Restore BN buffers and original BN training flags
        try:
            for m, mu, var, nbt, was_m_training in bn_state:
                if m.running_mean is not None:
                    m.running_mean.copy_(mu)
                if m.running_var is not None:
                    m.running_var.copy_(var)
                if nbt is not None and hasattr(m, "num_batches_tracked") and torch.is_tensor(m.num_batches_tracked):
                    m.num_batches_tracked.copy_(nbt)
                m.train() if was_m_training else m.eval()
        except Exception:
            pass
        # Restore overall model mode
        model.train() if was_training else model.eval()


# ---------------------------
# GT (unsupervised): robust hard + rank-only soft
# ---------------------------
def gt_collapse_time(run_df: pd.DataFrame, grad_cutoff: float) -> Tuple[Optional[int], str]:
    patience = int(CFG.get("gt_patience", 2))
    gt_min_epoch = int(CFG.get("gt_min_epoch", 0))
    g = run_df.sort_values("epoch")
    ep_float = to_numeric_series(g["epoch"], errors="coerce").to_numpy(dtype=float)
    # Avoid NaN -> int64 MIN_INT corruption (e.g., -9223372036854775808)
    ep = np.where(np.isfinite(ep_float), ep_float, -1.0).astype(np.int64)
    nan_flag = g["nan_flag"].astype(bool).to_numpy()
    grad_rel = g["grad_norm_rel"].to_numpy()
    warm_idx = warm_idx_from_cfg(CFG)
    # Effective minimum epoch for SOFT GT: max(warm_idx, gt_min_epoch)
    soft_min_epoch = max(int(warm_idx), int(gt_min_epoch))

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
    epochs_expected = as_opt_int(CFG.get("epochs"))
    if epochs_expected is not None and len(ep) > 0:
        e_last = int(ep[-1])  # g is sorted by epoch; last observed epoch
        if (e_last < epochs_expected - 1) and bool(nan_flag[-1]):
            # metadata check removed; keep legacy nan_flag gate
            return e_last, "hard"

    # SOFT: rank-only (native eff_dim below threshold) with patience
    eff = g["eff_dim_gt"].to_numpy() if "eff_dim_gt" in g.columns else g["eff_dim"].to_numpy()
    consec = 0
    t_first: Optional[int] = None
    for t in range(len(ep)):
        if ep[t] < soft_min_epoch:
            consec = 0
            t_first = None
            continue

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
# Learned detector utilities (moved out of CLI)
# ---------------------------
from veriscope.runners.legacy.detectors.learned import (
    _metrics_matrix_with_missing,
    _fit_global_robust_norm_precollapse,
    _apply_global_norm_impute,
    _train_logistic_ridge_balanced,
    _cv_grouped_fit,
    _oof_probs_for_params,
    map_threshold_to_gated_fp,
)


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
    # Ensure any preloaded WindowDecl is installed and FR is wired when run_one is invoked directly
    try:
        if "WINDOW_DECL" in globals() and (WINDOW_DECL is not None):
            if callable(install_window_decl):
                install_window_decl(WINDOW_DECL)
            _wire_fr_from_decl(WINDOW_DECL)
    except Exception:
        # Never let FR wiring failures break the core training loop.
        pass

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
                update_json(
                    OUTDIR / "env_probe.json",
                    {
                        "cuda_hash": _cuda_hash(),
                        "pip_freeze_md5": _pip_freeze_md5(),
                        "torch": torch.__version__,
                        "torchvision": torchvision.__version__,
                        "cuda": torch.version.cuda,
                        "deterministic_flag": bool(CFG.get("deterministic", True)),
                    },
                )
        except Exception:
            pass
        _DET_LOGGED = True
    amp_enabled = CFG["amp"] and (not CFG["deterministic"]) and device.type == "cuda"

    tr_aug, tr_take, mon_val, norm_ref, splits_path = load_splits(
        seed=seed,
        cfg=CFG,
        outdir=OUTDIR,
        data_root=DATA_ROOT,
    )
    run_id = f"s{seed}-{factor['name']}"
    tr_ds = FactorisedTrainDataset(tr_aug, tr_take, factor=factor, seed=seed)
    sampler = DropoutAwareSampler(tr_ds, CFG["batch"], seed=seed) if factor["name"] == "class_dropout_window" else None

    # monitor loaders
    ent_every = as_int(CFG.get("external_monitor", {}).get("ent_every"), default=2)
    pool_loader = None
    ent_loader = None
    monitor_source = str(CFG.get("monitor_source", "clean_val"))

    if monitor_source == "external":
        try:
            if monitor_ds is None:
                raise RuntimeError("external monitor dataset unavailable")

            rng = np.random.default_rng(777000 + seed)
            N = len(monitor_ds)
            pool_size = min(as_int(CFG.get("external_monitor", {}).get("pool_size"), default=N), N)
            if pool_size <= 0:
                raise RuntimeError("external monitor pool_size is 0")

            pool_idxs = rng.choice(np.arange(N, dtype=np.int64), size=pool_size, replace=False)
            pool_loader = subset_loader_from_indices(
                monitor_ds,
                pool_idxs,
                CFG["batch"],
                shuffle=True,
                seed=seed,
                device=device,
                num_workers=int(CFG.get("num_workers", 0)),
            )

            # disjoint entropy set (fallback to pool if diff is empty)
            ent_pool = np.setdiff1d(np.arange(N, dtype=np.int64), pool_idxs, assume_unique=True)
            if ent_pool.size == 0:
                ent_pool = pool_idxs

            ent_n = min(as_int(CFG.get("external_monitor", {}).get("ent_subset"), default=ent_pool.size), ent_pool.size)
            if ent_n <= 0:
                raise RuntimeError("external monitor ent_subset is 0")

            ent_idxs = rng.choice(ent_pool, size=ent_n, replace=False)
            ent_loader = subset_loader_from_indices(
                monitor_ds,
                ent_idxs,
                CFG["batch"],
                shuffle=False,
                seed=seed + 1,
                device=device,
                num_workers=int(CFG.get("num_workers", 0)),
            )

            # Ensure Python ints and give mypy a precise type
            pool_list = cast(List[int], pool_idxs.astype(int).tolist())
            ent_list = cast(List[int], ent_idxs.astype(int).tolist())
            update_json(
                splits_path,
                {
                    "STL10_POOL": pool_list,
                    "STL10_ENT": ent_list,
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
                def __init__(self, base):
                    self.base = base

                def __len__(self):
                    return len(self.base)

                def __getitem__(self, i):
                    x, y = self.base[i]
                    try:
                        dev = x.device
                    except Exception:
                        dev = "cpu"
                    g = torch.Generator(device=dev).manual_seed(123 + seed * 997 + int(i))
                    if isinstance(x, torch.Tensor):
                        noise = torch.randn(x.shape, dtype=x.dtype, device=dev, generator=g) * 0.02
                        x2 = (x + noise).clamp(-3, 3)
                        if (
                            int(_u01_from_hash("flip_probe", seed, int(i)) * 2) == 1
                            and x2.ndim == 3
                            and x2.shape[-1] >= 2
                        ):
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
                print(f"[WARN] norm_ref fetch failed for dim check: {e2}; defaulting penult_dim=512")
                d_pen = 512
            else:
                d_pen = as_int(penult(model, xb_chk.to(device)).shape[1], default=512)
        else:
            d_pen = as_int(penult(model, xb_chk.to(device)).shape[1], default=512)
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
            if cnt >= as_int(CFG.get("metric_total_cap", 10**9), default=10**9):
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
        if WINDOW_DECL is None and ("WindowDecl" in globals()) and (WindowDecl is not None):
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
            if callable(install_window_decl):
                install_window_decl(window_decl)
            try:
                _wire_fr_from_decl(window_decl)
            except Exception:
                pass
            if callable(write_window_provenance_from_decl):
                try:
                    write_window_provenance_from_decl(OUTDIR, window_decl)
                except Exception:
                    pass
            if callable(write_window_audit):
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
    from typing import Optional as _Optional

    ewma_loss: _Optional[float] = None

    # --- gate early-exit state (k-consecutive gate_warn after warm) ---
    gate_hits_consec = 0
    gate_halt_epoch = None
    warn_consec_eff = as_int(os.environ.get("SCAR_WARN_CONSEC", str(CFG.get("warn_consec", 3))), default=3)
    warm_epoch = warm_idx_from_cfg(CFG)
    gate_early_exit = bool(CFG.get("gate_early_exit", True))

    # Effective early-exit policy: default OFF in smoke for full trace capture.
    # Can be forced via env SCAR_GATE_HALT=1 (or disabled via SCAR_GATE_HALT=0).
    gate_early_exit_eff = bool(gate_early_exit)
    if env_truthy("SCAR_SMOKE") and os.environ.get("SCAR_GATE_HALT") is None:
        gate_early_exit_eff = False
    if os.environ.get("SCAR_GATE_HALT") is not None:
        gate_early_exit_eff = env_truthy("SCAR_GATE_HALT")
    gate_early_exit_eff = bool(gate_early_exit_eff)

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
        lr = lr_at(epoch, CFG["epochs"], CFG["base_lr"], CFG["warmup"], CFG["cosine"]) * base_lr_scale
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
        clip_eff: Optional[float] = (
            as_float(override_clip, default=float("nan"))
            if (override_clip is not None)
            else as_float(CFG.get("grad_clip_norm", 0.0), default=0.0)
        )
        if clip_eff is not None and clip_eff <= 0:
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

        s_w = as_int(CFG.get("slope_w"), default=9)
        tol = as_float(CFG.get("slope_tol"), default=1e-3)
        p_min = as_int(CFG.get("slope_persist_min"), default=3)

        s_eff = _rolling_slope(np.array(_eff_dim_series, dtype=float), w=s_w)
        s_var = _rolling_slope(np.array(_var_out_k_series, dtype=float), w=s_w)
        s_r2 = _rolling_slope(np.array(_r2_series, dtype=float), w=s_w)
        slope_r2 = float(s_r2[-1]) if s_r2.size and np.isfinite(s_r2[-1]) else float("nan")
        r_eff = _slope_regime(s_eff, tol=tol)
        r_var = _slope_regime(s_var, tol=tol)
        rf_eff = _regime_persist_flags(r_eff, min_len=p_min)
        rf_var = _regime_persist_flags(r_var, min_len=p_min)

        c_w = as_int(CFG.get("corr_w"), default=21)
        c_th = as_float(CFG.get("coh_thresh"), default=0.85)
        c_min = as_int(CFG.get("coh_minlen"), default=5)
        corr_ev = _rolling_corr(np.array(_eff_dim_series, dtype=float), np.array(_var_out_k_series, dtype=float), w=c_w)
        coh_flags = _coh_persist_below(corr_ev, thresh=c_th, min_len=c_min)

        slope_eff_dim = float(s_eff[-1]) if s_eff.size and np.isfinite(s_eff[-1]) else float("nan")
        slope_var_out_k = float(s_var[-1]) if s_var.size and np.isfinite(s_var[-1]) else float("nan")
        reg_eff_dim = float(r_eff[-1]) if r_eff.size and np.isfinite(r_eff[-1]) else float("nan")  # -1/0/+1
        reg_var_out_k = float(r_var[-1]) if r_var.size and np.isfinite(r_var[-1]) else float("nan")
        reg_eff_neg_persist = bool(rf_eff["neg"][-1]) if rf_eff["neg"].size else False
        reg_var_pos_persist = bool(rf_var["pos"][-1]) if rf_var["pos"].size else False
        corr_effdim_varoutk = float(corr_ev[-1]) if corr_ev.size and np.isfinite(corr_ev[-1]) else float("nan")
        coh_break_persist = bool(coh_flags[-1]) if coh_flags.size else False

        # heavy metrics
        pers_h0, topo_done, topo_ms, topo_n_used = float("nan"), 0, 0.0, 0
        sw, sw_ms, sw_proj_done, sw_valid = float("nan"), 0.0, 0, 0
        sw_nat, sw_nat_ms, sw_nat_proj_done, sw_nat_valid = float("nan"), 0.0, 0, 0
        if (epoch % as_int(CFG.get("heavy_every"), default=1)) == 0:
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
            _sw2_series.append(float("nan"))

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
        s_sw2 = _rolling_slope(np.array([v for v in _sw2_series], dtype=float), w=max(3, CFG["heavy_every"]))
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
            slope_margin = float(s_mar[-1]) if s_mar.size and np.isfinite(s_mar[-1]) else float("nan")
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
        batch_i = 0  # number of successfully processed batches so far
        while True:
            # ---- Fetch with one retry on transient DataLoader issues ----
            try:
                xb, yb = next(it)
            except StopIteration:
                break
            except Exception as e:
                if not did_retry_fetch:
                    print(
                        f"[WARN] batch fetch failed once; retrying by rebuilding iterator… (seed={seed}, epoch={epoch}, batch_i={batch_i}): {repr(e)}"
                    )
                    did_retry_fetch = True
                    # IMPORTANT: when `next(it)` raises, the iterator state is not reliable.
                    # Best-effort: rebuild the iterator and fast-forward to the same batch index.
                    try:
                        it = iter(tr_loader)
                        for _ in range(int(batch_i)):
                            next(it)
                        xb, yb = next(it)
                    except StopIteration:
                        break
                    except Exception as e2:
                        nan_seen = True
                        print(
                            f"[WARN] batch fetch failed on retry (seed={seed}, epoch={epoch}, batch_i={batch_i}): {repr(e2)}"
                        )
                        traceback.print_exc()
                        break
                else:
                    nan_seen = True
                    print(f"[WARN] batch fetch failed (seed={seed}, epoch={epoch}, batch_i={batch_i}): {repr(e)}")
                    traceback.print_exc()
                    break

            xb, yb = xb.to(device), yb.to(device)

            # ---- Train step with one retry (recomputes forward/backward) ----
            step_attempts = 0
            while True:
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
                    break  # success
                except Exception as e:
                    if step_attempts == 0 and (not did_retry_step):
                        print(
                            f"[WARN] train step failed once; retrying same batch… (seed={seed}, epoch={epoch}): {repr(e)}"
                        )
                        did_retry_step = True
                        step_attempts += 1
                        # Clear gradients and retry from a clean state
                        try:
                            opt.zero_grad(set_to_none=True)
                        except Exception:
                            pass
                        continue
                    nan_seen = True
                    print(f"[WARN] train step failed (seed={seed}, epoch={epoch}): {repr(e)}")
                    traceback.print_exc()
                    # Abort epoch training loop on repeated failure
                    step_attempts = 999
                    break

            if step_attempts >= 999:
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
            batch_i += 1
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

        # Robust EWMA: never overwrite with NaN; initialize sensibly.
        alpha = float(CFG.get("ewma_alpha", 0.1))
        if epoch == 0:
            ewma_loss = float(train_loss) if np.isfinite(train_loss) else 0.0
        else:
            if np.isfinite(train_loss):
                prev = ewma_loss if (ewma_loss is not None and np.isfinite(ewma_loss)) else float(train_loss)
                ewma_loss = alpha * float(train_loss) + (1.0 - alpha) * float(prev)
            else:
                # Keep previous EWMA if available; otherwise fall back to 0.
                ewma_loss = float(ewma_loss) if (ewma_loss is not None and np.isfinite(ewma_loss)) else 0.0

        loss_hist.append(train_loss)
        sma_loss = float(np.mean(loss_hist[-CFG["sma_k"] :])) if len(loss_hist) >= CFG["sma_k"] else float("nan")

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
                d = t_vector_norm(cur_vec - _prev_param_vec).item()
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
        # Defaults for logging even when the gate is NOT evaluated.
        # Semantics: gate_warn=1 means PASS/neutral; gate_warn=0 means FAIL.
        # Therefore any non-evaluated epoch must be neutral to avoid false failures.
        gate_warn = 1
        gate_gain = float("nan")
        gate_worst_tv = float("nan")
        gate_eps_stat = float("nan")
        gate_kappa = float("nan")
        gate_total_evidence = 0
        gate_reason = "not_evaluated"
        gate_counts_json = ""
        gate_exc = ""
        # --- Additional gate channels (needed for persistence triage) ---
        gate_ok = 1
        gate_fail = 0
        gate_evaluated = 0
        gate_warn_pending = 0
        gate_warn = 0  # explicit init: 0 = no warning, 1 = warning triggered
        gate_eps_eff = float("nan")
        gate_consecutive_exceedances = 0
        gate_dw_exceeds_threshold = 0
        gate_policy_effective = ""
        # --- Gate lane provenance (needed for robust offline recompute) ---
        # Set even if the gate isn't evaluated on this epoch.
        gate_lane = "fr" if bool(USE_FR) else "legacy"
        scar_fr = 1 if gate_lane == "fr" else 0

        # Default min-evidence used: under FR the source of truth is GE.min_evidence.
        gate_min_evidence_used = int(as_int(CFG.get("gate_min_evidence", 0), default=0))
        try:
            if bool(USE_FR) and (GE is not None):
                gate_min_evidence_used = int(getattr(GE, "min_evidence", gate_min_evidence_used))
        except Exception:
            pass
        gate_epsilon_eff = as_float(
            getattr(WINDOW_DECL, "epsilon", CFG.get("gate_epsilon", 0.08)),
            default=as_float(CFG.get("gate_epsilon", 0.08), default=0.08),
        )
        gate_gain_thresh_eff = as_float(CFG.get("gate_gain_thresh", 0.05), default=0.05)

        W_gate = as_int(CFG.get("gate_window", 16), default=16)

        # Explicit non-eval reasons (avoid silent fallthrough / swallowed exceptions)
        if WINDOW_DECL is None:
            gate_reason = "no_window_decl"
            gate_warn = 1
            gate_total_evidence = 0
            gate_counts_json = ""
        elif len(logs) < 2 * int(W_gate):
            gate_reason = "insufficient_history"
            gate_warn = 1
            gate_total_evidence = 0
            gate_counts_json = ""
        else:
            try:
                recent = logs[-(2 * int(W_gate)) :]

                def _series(seg, key):
                    v = np.array([float(r.get(key, np.nan)) for r in seg], dtype=float)
                    return v[np.isfinite(v)]

                # metric names from the installed declaration (force str keys for consistency)
                _metrics_raw = getattr(WINDOW_DECL, "metrics", tuple(getattr(WINDOW_DECL, "weights", {}).keys()))
                metrics_for_tv = tuple(str(m) for m in _metrics_raw)

                past_dict_all = {m: _series(recent[: int(W_gate)], m) for m in metrics_for_tv}
                recent_dict_all = {m: _series(recent[int(W_gate) :], m) for m in metrics_for_tv}

                # counts under the declaration’s common transport adapter
                _adapter = getattr(WINDOW_DECL, "_DECL_TRANSPORT", None)
                _apply = _adapter.apply if (_adapter is not None) else (lambda name, arr: np.asarray(arr, float))

                counts_all: Dict[str, int] = {}
                for m in metrics_for_tv:
                    a = _apply(m, past_dict_all.get(m, np.array([], dtype=float)))
                    b = _apply(m, recent_dict_all.get(m, np.array([], dtype=float)))
                    na = int(np.isfinite(np.asarray(a, float)).sum())
                    nb = int(np.isfinite(np.asarray(b, float)).sum())
                    counts_all[m] = int(min(na, nb))

                # Only pass metrics with evidence into gate_check (prevents NaN audit via empty series)
                metrics_use = [m for m in metrics_for_tv if int(counts_all.get(m, 0)) > 0]
                past_dict = {m: past_dict_all[m] for m in metrics_use}
                recent_dict = {m: recent_dict_all[m] for m in metrics_use}
                counts = {m: int(counts_all[m]) for m in metrics_use}

                gate_total_evidence = int(sum(int(v) for v in counts.values()))
                try:
                    gate_counts_json = json.dumps(counts, sort_keys=True)
                except Exception:
                    gate_counts_json = ""

                # prequential gain (bits/sample) over the recent half
                ml = _series(recent[int(W_gate) :], "train_loss")
                bl = _series(recent[int(W_gate) :], "ewma_loss")

                baseline = (
                    str(os.environ.get("SCAR_GATE_GAIN_BASELINE", CFG.get("gate_gain_baseline", "uniform")))
                    .strip()
                    .lower()
                )

                gate_gain = _gain_bits_from_recent_losses(
                    ml,
                    baseline=baseline,
                    ewma_losses_nats=bl,  # used only when baseline == "ewma"
                    num_classes=int(CFG.get("num_classes", 10)),
                )

                # κ_sens probe sparsely (augmentation-only by default)
                gate_kappa = 0.0
                try:
                    if (epoch % max(1, as_int(CFG.get("heavy_every", 6), default=6))) == 0:
                        gate_kappa = kappa_sens_probe(
                            model,
                            opt,
                            pool_loader,
                            device,
                            WINDOW_DECL,
                            ref_mu_sig,
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
                    gain_thresh=gate_gain_thresh_eff,
                    # Missing κ should mean "not checked" end-to-end.
                    # Pass NaN so GateEngine treats it as unchecked.
                    kappa_sens=(gate_kappa if np.isfinite(gate_kappa) else float("nan")),
                    eps_stat_alpha=float(CFG.get("gate_eps_stat_alpha", 0.05)),
                )

                gate_warn = int(flag)
                audit = audit or {}
                gate_reason = str(audit.get("reason", "evaluated_unknown")) or "evaluated_unknown"
                # --- Provenance + persistence channels from audit (FR or legacy) ---
                try:
                    gate_lane = str(audit.get("gate_lane", gate_lane))
                except Exception:
                    pass
                try:
                    scar_fr = int(audit.get("scar_fr", scar_fr))
                except Exception:
                    pass

                try:
                    gate_total_evidence = int(audit.get("total_evidence", gate_total_evidence))
                except Exception:
                    pass
                try:
                    gate_min_evidence_used = int(audit.get("min_evidence", gate_min_evidence_used))
                except Exception:
                    pass

                # Robust evaluated flag: handle None explicitly to avoid silent unevaluated
                ev = audit.get("evaluated", None)
                if ev is None:
                    gate_evaluated = int(bool(gate_reason.startswith("evaluated")))
                else:
                    gate_evaluated = int(bool(ev))

                gate_warn_pending = int(bool(audit.get("warn_pending", False)))
                gate_eps_eff = as_float(audit.get("eps_eff", np.nan), default=float("nan"))
                gate_consecutive_exceedances = as_int(audit.get("consecutive_exceedances", 0), default=0)
                gate_dw_exceeds_threshold = int(bool(audit.get("dw_exceeds_threshold", False)))
                gate_policy_effective = str(audit.get("policy", "")) if audit.get("policy", "") is not None else ""

                # gate_ok/gate_fail come from audit["ok"], NOT from gate_warn or flag
                # Default to OK=True if audit doesn't provide it (fail-closed would be annoying)
                gate_ok_val = audit.get("ok", True)
                gate_ok = int(bool(gate_ok_val)) if gate_evaluated else 1
                gate_fail = int((not bool(gate_ok_val)) and gate_evaluated)

                # gate_warn represents a *warning state* (threshold exceeded or persistence pending)
                # Only meaningful once evaluated; 1 = warning triggered, 0 = no warning
                gate_warn = int(gate_evaluated and (gate_warn_pending == 1 or gate_dw_exceeds_threshold == 1))

                # --- Gate diagnostics for offline analysis ---
                gate_epsilon_eff = as_float(
                    getattr(WINDOW_DECL, "epsilon", CFG.get("gate_epsilon", 0.08)),
                    default=as_float(CFG.get("gate_epsilon", 0.08), default=0.08),
                )
                gate_worst_tv = as_float(audit.get("worst_tv", np.nan), default=float("nan"))
                gate_eps_stat = as_float(audit.get("eps_stat", np.nan), default=float("nan"))
                gate_gain = as_float(audit.get("gain_bits", gate_gain), default=float("nan"))
                gate_kappa = as_float(audit.get("kappa_sens", gate_kappa), default=float("nan"))

                # effective parameters actually used for this check
                gate_gain_thresh_eff = as_float(CFG.get("gate_gain_thresh", 0.05), default=0.05)

            except Exception as e:
                gate_exc = repr(e)
                gate_reason = (
                    "not_evaluated_exception"  # not "evaluated" so reason-prefix checks don't treat it as evaluated
                )
                gate_warn = 1  # neutral PASS — won't trigger early-exit
                # Log once per process for visibility
                try:
                    if mp.current_process().name == "MainProcess":
                        print(f"[gate][WARN] exception during gate_check (neutral PASS): {e!r}")
                except Exception:
                    pass

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
                gate_window=as_int(CFG.get("gate_window", 16), default=16),
                gate_bins=as_int(CFG.get("gate_bins", 16), default=16),
                gate_epsilon=float(gate_epsilon_eff),
                gate_gain_thresh=float(gate_gain_thresh_eff),
                # fixed-partition gate diagnostics
                gate_eps_stat=float(gate_eps_stat),
                gate_kappa=float(gate_kappa),
                gate_total_evidence=int(gate_total_evidence),
                gate_lane=str(gate_lane),
                scar_fr=int(scar_fr),
                gate_min_evidence_used=int(gate_min_evidence_used),
                gate_ok=int(gate_ok),
                gate_fail=int(gate_fail),
                gate_evaluated=int(gate_evaluated),
                gate_warn_pending=int(gate_warn_pending),
                gate_eps_eff=float(gate_eps_eff),
                gate_consecutive_exceedances=int(gate_consecutive_exceedances),
                gate_dw_exceeds_threshold=int(gate_dw_exceeds_threshold),
                gate_policy_effective=str(gate_policy_effective),
                gate_reason=str(gate_reason),
                gate_counts=str(gate_counts_json),
                gate_exc=str(gate_exc),
                # WindowDecl identity for offline comparability
                window_decl_hash=(
                    window_decl_identity_hash(WINDOW_DECL)
                    if (WINDOW_DECL is not None and window_decl_identity_hash is not None)
                    else ""
                ),
                window_decl_epsilon=(
                    float(getattr(WINDOW_DECL, "epsilon", float("nan"))) if WINDOW_DECL is not None else float("nan")
                ),
                window_decl_bins=(int(getattr(WINDOW_DECL, "bins", 0)) if WINDOW_DECL is not None else 0),
            )
        )
        # hard-stop training when the finite-window gate FAILS for k consecutive evaluated epochs (after warm)
        if gate_early_exit_eff:
            if epoch >= warm_epoch:
                # gate_check() returns 1 when the gate PASSES (stable) and 0 when it FAILS (instability).
                # Only count consecutive FAILs, and only when the gate was actually evaluated (not "insufficient_history", etc.).
                reason_eval = isinstance(gate_reason, str) and gate_reason.startswith("evaluated")
                if reason_eval and int(gate_warn) == 0:
                    gate_hits_consec += 1
                else:
                    gate_hits_consec = 0
                if gate_hits_consec >= warn_consec_eff and gate_halt_epoch is None:
                    gate_halt_epoch = int(epoch)
                    # persist a terminal marker compatible with resume logic
                    try:
                        (OUTDIR / f"runs_seed{seed}_{tag}.done.json").write_text(
                            json.dumps(
                                {
                                    "terminal": True,
                                    "reason": "gate_halt",
                                    "epoch": int(epoch),
                                    "gate_hits_consec": int(gate_hits_consec),
                                }
                            )
                        )
                    except Exception:
                        pass
                    break

    df = pd.DataFrame(logs)

    # Per-run gate summary: always persist to OUTDIR/sweep.log (auditability) but only print to stdout in smoke/debug.
    if mp.current_process().name == "MainProcess":
        try:

            def _is_eval(r: Dict[str, Any]) -> bool:
                return str(r.get("gate_reason", "")).startswith("evaluated")

            eval_ct = 0
            eval_warn_ok = 0
            not_eval_ct = 0
            insuff_ct = 0
            reason_counts: Dict[str, int] = {}

            for r in logs:
                reason = str(r.get("gate_reason", "unknown"))
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if _is_eval(r):
                    eval_ct += 1
                    # gate_warn==1 means PASS among evaluated epochs
                    if int(r.get("gate_warn", 0) or 0) == 1:
                        eval_warn_ok += 1
                else:
                    not_eval_ct += 1
                    if "insufficient" in reason.lower():
                        insuff_ct += 1

            # Top 3 reasons for immediate diagnosis
            top_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])[:3]
            reason_summary = " ".join(f"{k}:{v}" for k, v in top_reasons)

            line = (
                f"[run] seed={seed} factor={str(factor.get('name', ''))} epochs={len(logs)} "
                f"gate_evaluated={eval_ct} gate_eval_ok={eval_warn_ok} "
                f"gate_not_eval={not_eval_ct} gate_insufficient={insuff_ct} "
                f"reasons=[{reason_summary}]"
            )

            # Always-on: append a single line/run to sweep.log to survive stdout truncation.
            try:
                with open(OUTDIR / "sweep.log", "a", encoding="utf-8") as _lg:
                    print(line, file=_lg, flush=True)
            except Exception:
                pass

            # Stdout only when explicitly debugging or in smoke mode.
            if env_truthy("SCAR_SMOKE") or DEBUG_SMOKE:
                try:
                    print(line, flush=True)
                except Exception:
                    pass
        except Exception:
            pass

    return df


# ---------------------------
# Direction calibration for PH & grad cutoff
# ---------------------------
def calibrate_grad_cutoff_per_factor(df_cal: pd.DataFrame) -> Dict[str, float]:
    """\
    Robust per-factor cutoff for gradient-explosion detection.

    Goal: avoid pathological cutoffs when the calibration distribution has near-zero dispersion
    (e.g., smoke runs) while still being conservative in production.

    Policy:
      - exclude epochs with nan_flag or non-finite grad_norm_rel
      - restrict to post-warm epochs
      - for runs without (preliminary) collapse, restrict to an early stable window (< warm + ph_win)
      - for runs with (preliminary) collapse time t_collapse_gt, restrict to epochs < t_collapse_gt
      - estimate cutoff as max(floor, median + K * 1.4826 * max(MAD, mad_abs_floor), q_upper)
      - if too few samples, fall back to floor

    Notes:
      - This function is intended to be called on `df_cal_raw` after the initial GT pass
        where `t_collapse_gt` is computed with `grad_cutoff=np.inf`, so the preliminary
        collapse time is not itself driven by the grad cutoff being calibrated.
    """
    out: Dict[str, float] = {}

    warm = as_int(CFG.get("warmup"), default=0) + as_int(CFG.get("ph_burn"), default=0)
    guard = warm + as_int(CFG.get("ph_win"), default=0)  # early stable window cap

    # Calibration knobs (conservative defaults)
    floor = as_float(CFG.get("gt_grad_cutoff_floor", 0.10), default=0.10)
    K = as_float(CFG.get("gt_grad_cutoff_k_mad", 6.0), default=6.0)
    q_upper = as_float(CFG.get("gt_grad_cutoff_q", 0.995), default=0.995)
    mad_abs_floor = as_float(CFG.get("gt_grad_cutoff_mad_abs_floor", 0.002), default=0.002)
    min_samples = as_int(CFG.get("gt_grad_cutoff_min_samples", 32), default=32)

    # Sanity clamps
    if not np.isfinite(floor) or floor <= 0:
        floor = 0.10
    if not np.isfinite(K) or K <= 0:
        K = 6.0
    if (not np.isfinite(q_upper)) or (q_upper <= 0.0) or (q_upper >= 1.0):
        q_upper = 0.995
    if (not np.isfinite(mad_abs_floor)) or (mad_abs_floor < 0.0):
        mad_abs_floor = 0.002
    if min_samples <= 0:
        min_samples = 32

    for factor, gfac in df_cal.groupby("factor"):
        vals: List[float] = []
        for _, g in gfac.groupby(["seed", "factor"]):
            g = g.sort_values("epoch")

            ep = pd.to_numeric(g.get("epoch", pd.Series(dtype=float)), errors="coerce")
            gr = pd.to_numeric(g.get("grad_norm_rel", pd.Series(dtype=float)), errors="coerce")
            nf = g.get("nan_flag", pd.Series([0] * len(g))).fillna(0).astype(bool)

            # preliminary collapse time (should be derived from grad_cutoff=np.inf pass)
            t_c = np.nan
            if "t_collapse_gt" in g.columns:
                try:
                    t_c = float(pd.to_numeric(g["t_collapse_gt"], errors="coerce").iloc[0])
                except Exception:
                    t_c = np.nan

            sub = pd.DataFrame({"epoch": ep, "grad": gr, "nan": nf})
            sub = sub[(~sub["nan"]) & np.isfinite(sub["grad"]) & np.isfinite(sub["epoch"])]
            sub = sub[sub["epoch"] >= int(warm)]

            if pd.isna(t_c):
                # no preliminary collapse -> use early stable window
                sub = sub[sub["epoch"] < int(guard)]
            else:
                t_c_i = as_int(t_c, default=-1)
                if t_c_i >= 0:
                    sub = sub[sub["epoch"] < int(t_c_i)]

            if len(sub) > 0:
                vals.extend([float(x) for x in sub["grad"].to_list() if np.isfinite(x)])

        # If insufficient data, fall back to a conservative floor (not inf)
        if len(vals) < int(min_samples):
            out[str(factor)] = float(floor)
            continue

        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < int(min_samples):
            out[str(factor)] = float(floor)
            continue

        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        mad_eff = max(float(mad), float(mad_abs_floor))

        est_mad = float(med + float(K) * 1.4826 * mad_eff)

        # With small n, a fixed "0.999" is effectively "max"; adapt slightly.
        q_eff = min(float(q_upper), 1.0 - (1.0 / float(arr.size)))
        try:
            est_q = float(np.quantile(arr, q_eff))
        except Exception:
            est_q = float(np.max(arr))

        cutoff = max(float(floor), float(est_mad), float(est_q))
        out[str(factor)] = float(cutoff)

    return out


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
    lo, hi = quantile2(boots, alpha / 2.0, 1 - alpha / 2.0)
    return (float(lo), float(hi))


def safe_auc_by_factor(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    min_per_side: int = 2,
    family_map: Optional[Dict[str, str]] = None,
    B: int = 1000,
) -> pd.DataFrame:
    """
    Compute per-factor Δ and AUC with bootstrap CIs. Ensures each factor has at least
    `min_per_side` seeds on both sides of label_col. Optionally pool factors via `family_map`.
    Expects one row per (seed,factor) with last-epoch summary scores.
    Returns rows with: group (factor or family), n_pos, n_neg, delta_med, delta_lo, delta_hi,
    auc, auc_lo, auc_hi.
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["group", "n_pos", "n_neg", "delta_med", "delta_lo", "delta_hi", "auc", "auc_lo", "auc_hi"]
        )

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
                dlo, dhi = quantile2(deltas, 0.025, 0.975)
            else:
                dlo, dhi = (np.nan, np.nan)
            out_rows.append(
                dict(
                    group=str(grp),
                    n_pos=len(pos),
                    n_neg=len(neg),
                    delta_med=float(d),
                    delta_lo=float(dlo),
                    delta_hi=float(dhi),
                    auc=np.nan,
                    auc_lo=np.nan,
                    auc_hi=np.nan,
                )
            )
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
        d_lo, d_hi = quantile2(deltas, 0.025, 0.975) if deltas else (np.nan, np.nan)
        auc = _wilcoxon_auc(pos, neg)
        a_lo, a_hi = quantile2(aucs, 0.025, 0.975) if aucs else (np.nan, np.nan)
        out_rows.append(
            dict(
                group=str(grp),
                n_pos=len(pos),
                n_neg=len(neg),
                delta_med=d_med,
                delta_lo=float(d_lo),
                delta_hi=float(d_hi),
                auc=float(auc),
                auc_lo=float(a_lo),
                auc_hi=float(a_hi),
            )
        )
    return pd.DataFrame(out_rows)


from typing import Iterable, Callable
from typing import Any, Optional

# --- FR integration holders (process-global; typed for mypy) ---
USE_FR: bool = False
FR_WIN: Optional["FRWindow"] = None  # current FRWindow instance (or None)
_DECL_TRANSPORT: Optional["DeclTransport"] = None  # kept for backward compatibility
GE: Optional["GateEngine"] = None  # GateEngine instance/factory
WINDOW_DECL: Optional["WindowDecl"] = None  # WindowDecl from calibration
DECL_INSTALL: Optional[Callable[..., None]] = None
FR_ASSERT_NATURALITY = None  # kept for backward compatibility
_FR_FALLBACK_WARNED: bool = False


def _wire_fr_from_decl(win: "WindowDecl") -> None:
    """
    Wire USE_FR / FR_WIN / GE from a WindowDecl via fr_integration.init_fr().
    Falls back cleanly to legacy gating if FR is unavailable.
    """
    global USE_FR, FR_WIN, GE, WINDOW_DECL
    WINDOW_DECL = win
    # If FR is not importable or no decl, fall back cleanly
    if init_fr is None or win is None:
        USE_FR, FR_WIN, GE = False, None, None
        return
    try:
        # init_fr reads SCAR_FR and other knobs from env
        use_fr, fr_win, ge = init_fr(win, CFG, env=os.environ)
    except Exception:
        # Any failure → legacy path
        USE_FR, FR_WIN, GE = False, None, None
        return
    USE_FR = bool(use_fr)
    FR_WIN = fr_win
    GE = ge


# --- epsilon aggregation helper (legacy default; do not clobber FR resolver if installed) ---
# NOTE: `_core_aggregate_epsilon_stat` is imported once near the top of this file; do not re-import here.
# Guard for safety in case the top-level import was removed/modified.
if "_core_aggregate_epsilon_stat" not in globals():
    _core_aggregate_epsilon_stat = None


# Only set the default adapter if nothing else (e.g., FR resolver) already installed it.
try:
    if globals().get("AGGREGATE_EPSILON") is None:
        AGGREGATE_EPSILON = agg_eps_adapter
except Exception:
    AGGREGATE_EPSILON = agg_eps_adapter


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
                m.append(float((y[j] - y[i]) / (x[j] - x[i])))
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
    from typing import cast

    for key, g in df_runs.groupby(["seed", "factor"]):
        sd, fc = cast(tuple[Any, Any], key)
        gg = g.sort_values("epoch")
        tail_g = gg.tail(tail)
        x = tail_g["epoch"].to_numpy(dtype=float)
        ft = tail_g.get("ftle", pd.Series(dtype=float)).to_numpy(dtype=float)
        dr = tail_g.get("drift_abs", pd.Series(dtype=float)).to_numpy(dtype=float)
        s_ft = _theil_sen_slope(ft, x)
        s_dr = _theil_sen_slope(dr, x)
        post = gg[gg["epoch"] >= int(warm_idx)]
        ft_ser = post.get("ftle", pd.Series(dtype=float))
        ft_arr = np.array([], dtype=float)
        if isinstance(ft_ser, pd.Series):
            ft_arr = pd.to_numeric(ft_ser, errors="coerce").to_numpy(dtype=float)
        ft_ok = bool(np.all(np.nan_to_num(ft_arr) <= 0.0)) if len(post) else False
        rows.append(
            dict(
                seed=as_int(sd, default=0),
                factor=str(fc),
                slope_ftle=s_ft,
                slope_drift_abs=s_dr,
                slope_ftle_pos=bool(np.isfinite(s_ft) and s_ft > 0),
                slope_drift_pos=bool(np.isfinite(s_dr) and s_dr > 0),
                ftle_nonpos_after_warm=ft_ok,
            )
        )
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
    warm = warm_idx_from_cfg(CFG)
    ctl = (
        df_runs[(df_runs["factor"] == "none") & (df_runs["epoch"] >= warm)]
        if isinstance(df_runs, pd.DataFrame)
        else pd.DataFrame()
    )
    ctl_n = 0
    try:
        if not ctl.empty and ("ftle" in ctl.columns):
            arr = ctl["ftle"].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            ctl_n = int(arr.size)
            ctl_q99 = float(qlin(arr, 0.99)) if arr.size > 0 else np.nan
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
    from typing import cast

    for key, g in df_runs.groupby(["seed", "factor"]):
        sd, fc = cast(tuple[Any, Any], key)
        gg = g.sort_values("epoch")
        post = gg[gg["epoch"] >= warm] if "epoch" in gg.columns else gg

        # FTLE boundedness with guarded cap fallback
        q99_run = np.nan
        if "ftle" in post.columns and len(post) > 0:
            vals = post["ftle"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                q99_run = float(qlin(vals, 0.99))

        # Start with the global cap computed above; if NaN, fall back to calibration or CFG
        cap: float = float(ftle_cap)
        if not np.isfinite(cap):
            try:
                import glob
                import json as _json

                paths = sorted(glob.glob(str(OUTDIR / "gt_rank_calibration_*.json")))
                if paths:
                    _obj = _json.loads(Path(paths[-1]).read_text())
                    cap = float(_obj.get("ftle_q99_cap", float("nan")))
            except Exception:
                pass
        if not np.isfinite(cap):
            cap = as_float(CFG.get("ftle_q99_cap", float("nan")), default=float("nan"))

        inv_row = dict(seed=as_int(sd, default=0), factor=str(fc))
        if np.isfinite(cap) and np.isfinite(q99_run):
            inv_row["ftle_bounded"] = int(q99_run <= cap)
            inv_row["ftle_q99_run"] = float(q99_run)
            inv_row["ftle_cap"] = float(cap)
        else:
            inv_row["ftle_q99_run"] = float(q99_run) if np.isfinite(q99_run) else float("nan")
            inv_row["ftle_cap"] = float("nan")
            inv_row["ftle_bounded_skipped"] = 1

        # Other invariants
        neg_ok = bool(np.nanmax(gg.get("neg_eigs", pd.Series([0], dtype=float)).to_numpy(dtype=float)) == 0)
        inv_row["neg_eigs_zero"] = int(neg_ok)

        try:
            k_arr = gg.get("k_used", pd.Series([np.nan], dtype=float)).to_numpy(dtype=float)
            k_used_med = float(np.nanmedian(k_arr))
        except Exception:
            k_used_med = float("nan")
        inv_row["k_used_median"] = float(k_used_med) if np.isfinite(k_used_med) else float("nan")
        ok_range = bool(np.isfinite(k_used_med)) and (1 <= k_used_med <= as_int(CFG.get("var_k_max", 32), default=32))
        inv_row["k_used_in_range"] = 1 if ok_range else 0

        inv.append(inv_row)

    invariants = pd.DataFrame(inv)
    out = {
        "invariants": invariants.to_dict(orient="records"),
        "summary": {
            "ftle_bounded_pass_rate": float(
                np.mean(invariants.get("ftle_bounded", pd.Series([], dtype=float)).astype(bool))
            )
            if not invariants.empty and ("ftle_bounded" in invariants.columns)
            else np.nan,
            "neg_eigs_zero_pass_rate": float(
                np.mean(invariants.get("neg_eigs_zero", pd.Series([], dtype=float)).astype(bool))
            )
            if not invariants.empty and ("neg_eigs_zero" in invariants.columns)
            else np.nan,
            "k_used_in_range_pass_rate": float(
                np.mean(invariants.get("k_used_in_range", pd.Series([], dtype=float)).astype(bool))
            )
            if not invariants.empty and ("k_used_in_range" in invariants.columns)
            else np.nan,
            "ftle_bounded_skipped_count": (
                int(
                    float(
                        to_numeric_series(invariants.get("ftle_bounded_skipped", pd.Series([], dtype=float)))
                        .fillna(0)
                        .sum()
                    )
                )
                if ("ftle_bounded_skipped" in invariants.columns)
                else 0
            ),
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
        import torch as _t

        prov["torch"] = getattr(_t, "__version__", None)
    except Exception:
        pass
    try:
        import torch as _t

        prov["cuda"] = getattr(getattr(_t, "version", None), "cuda", None)
    except Exception:
        pass
    if artifact_csv is not None:
        try:
            prov["artifact_md5"] = file_md5(artifact_csv)
        except Exception:
            pass

    prov["ftle_cap_source"] = "abs_cap" if abs_cap is not None else "control_q99_margin"
    prov["ftle_control_q99"] = float(ctl_q99) if np.isfinite(ctl_q99) else None
    prov["ftle_control_q99_n"] = int(ctl_n)
    prov["ftle_margin"] = float(margin)
    prov["ftle_cap_value"] = float(cap) if np.isfinite(cap) else None
    prov["warm_idx"] = int(warm)

    # Gate parameters for audit consistency
    prov["gate_epsilon"] = as_float(CFG.get("gate_epsilon", 0.08), default=0.08)
    prov["gate_epsilon_floor"] = as_float(CFG.get("gate_epsilon_floor", 0.02), default=0.02)
    prov["gate_window"] = as_int(CFG.get("gate_window", 16), default=16)
    prov["gate_bins"] = as_int(CFG.get("gate_bins", 16), default=16)
    prov["gate_gain_thresh"] = as_float(CFG.get("gate_gain_thresh", 0.05), default=0.05)
    prov["gate_eps_stat_max_frac"] = as_float(CFG.get("gate_eps_stat_max_frac", 0.25), default=0.25)
    prov["gate_min_evidence"] = as_int(CFG.get("gate_min_evidence", 16), default=16)
    out["provenance"] = prov

    try:
        save_json(out, OUTDIR / "artifact_invariants_provenance.json")
    except Exception:
        pass
    return out


# ---------------------------
# Plotting (reads unified epoch overlays)
# ---------------------------
def _spaghetti(ax, df_noise: pd.DataFrame, metric: str):
    for _, g in df_noise.groupby("seed"):
        gg = g[g.epoch >= 0].sort_values("epoch")
        seed = 0
        if "seed" in gg.columns and len(gg) > 0:
            seed = as_int(pd.to_numeric(gg["seed"], errors="coerce").iloc[0], default=0)
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
        "gate_warn_calib",
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
    from typing import cast

    for key, sub in df.groupby(["seed", "factor"]):
        seed, factor = cast(tuple[Any, Any], key)
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
        ax.text(-2, y, f"{factor} s={as_int(seed, default=0)} [{tagc}]", ha="right", va="center", fontsize=8)
        y += 1
    ax.axvspan(shade0, shade1, color="#eee", alpha=0.5)
    warm = warm_idx_from_cfg(CFG)
    he = as_int(CFG.get("heavy_every"), default=1)
    heavy0 = int(((warm + he - 1) // max(he, 1)) * max(he, 1))
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
    from typing import cast

    for key, g in df.groupby(["seed", "factor"]):
        seed, factor = cast(tuple[Any, Any], key)
        idx = g.index
        for m in SCHEDULED_METRICS:
            if m in df.columns:
                df.loc[idx, m] = g[m].ffill()
    return df


def _stationaryize_scheduled(df_in: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df_in.copy()
    from typing import cast

    for key, g in df.groupby(["seed", "factor"]):
        seed, factor = cast(tuple[Any, Any], key)
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


# --- budget reset helper ---
def _reset_budget() -> None:
    """Reset the process-global heavy-metric budget/call counters for a fresh run/seed."""
    global BUDGET
    try:
        if BUDGET is not None and hasattr(BUDGET, "reset"):
            BUDGET.reset()
    except Exception:
        # Keep sweeps robust even if the budget ledger is misconfigured.
        pass


# --- env int + canary clamps (optional) ---
def _env_int(name: str) -> Optional[int]:
    """Parse env var as int; returns None on missing/blank/invalid."""
    v = os.environ.get(name)
    if v is None:
        return None
    v = str(v).strip()
    if not v:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _apply_env_canary_clamps(cfg: Dict[str, Any]) -> None:
    """Optionally clamp epochs and seed lists using env vars.

    - Epoch clamp: SCAR_MAX_EPOCHS (preferred), else SCAR_EPOCHS, else SCAR_N_EPOCHS.
    - Seed clamp (both splits): SCAR_SEED_COUNT.
    - Eval-only seed clamp: SCAR_CANARY_SEEDS.

    No effect unless the corresponding env var is set to a positive int.
    """
    # ---- epoch clamp ----
    epoch_cap = _env_int("SCAR_MAX_EPOCHS") or _env_int("SCAR_EPOCHS") or _env_int("SCAR_N_EPOCHS")
    if epoch_cap is not None and int(epoch_cap) > 0:
        cfg_epochs = as_int(cfg.get("epochs", int(epoch_cap)), default=int(epoch_cap))
        new_epochs = min(int(cfg_epochs), int(epoch_cap))
        if as_int(cfg.get("epochs", new_epochs), default=new_epochs) != new_epochs:
            print(f"[cfg] env clamp: epochs {cfg.get('epochs')} -> {new_epochs} (cap={epoch_cap})")
        cfg["epochs"] = int(new_epochs)

    # ---- seed clamp (both splits) ----
    seed_cap = _env_int("SCAR_SEED_COUNT")
    if seed_cap is not None and int(seed_cap) > 0:
        sc = int(seed_cap)
        if "seeds_calib" in cfg:
            cfg["seeds_calib"] = list(cfg.get("seeds_calib") or [])[:sc]
        if "seeds_eval" in cfg:
            cfg["seeds_eval"] = list(cfg.get("seeds_eval") or [])[:sc]
        print(f"[cfg] env clamp: seeds_calib<= {sc}, seeds_eval<= {sc} (SCAR_SEED_COUNT)")

    # ---- seed clamp (eval only) ----
    eval_cap = _env_int("SCAR_CANARY_SEEDS")
    if eval_cap is not None and int(eval_cap) > 0:
        se = int(eval_cap)
        if "seeds_eval" in cfg:
            cfg["seeds_eval"] = list(cfg.get("seeds_eval") or [])[:se]
        print(f"[cfg] env clamp: seeds_eval<= {se} (SCAR_CANARY_SEEDS)")


# ---------------------------
# Sweep
# ---------------------------


# --- resume completeness helper ---
def _epoch_bounds_csv(path: Path) -> tuple[Optional[int], Optional[int], int]:
    """Return (emin, emax, nrows) for a CSV shard, or (None,None,0) on failure."""
    try:
        df = pd.read_csv(path, usecols=["epoch"])
        if "epoch" not in df.columns:
            return None, None, 0
        df = df.copy()
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        ep = to_numeric_series(df["epoch"], errors="coerce")
        arr = ep.to_numpy(dtype=float)
        # Filter to finite values only to avoid np.nanmin/nanmax returning NaN
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None, None, 0
        mn = int(np.min(arr))
        mx = int(np.max(arr))
        return mn, mx, int(len(df))
    except Exception:
        return None, None, 0


def _epoch_bounds(path: Path) -> tuple[Optional[int], Optional[int], int]:
    """Return (emin, emax, nrows) for a parquet shard, or (None,None,0) on failure."""
    try:
        df = pd.read_parquet(path, columns=["epoch"])
        if "epoch" not in df.columns:
            return None, None, 0
        df = df.copy()
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        ep = to_numeric_series(df["epoch"], errors="coerce")
        arr = ep.to_numpy(dtype=float)
        # Filter to finite values only to avoid np.nanmin/nanmax returning NaN
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None, None, 0
        mn = int(np.min(arr))
        mx = int(np.max(arr))
        return mn, mx, int(len(df))
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

    # Reconcile CFG with env/smoke before the sweep starts
    if require_gate_eval(CFG):
        # In require/fail-fast mode, do not validate against a stale CFG.
        reconcile_cfg_inplace(CFG, stage="run_sweep")
        runtime.install_runtime(cfg=CFG, outdir=OUTDIR, budget=BUDGET)
    else:
        try:
            reconcile_cfg_inplace(CFG, stage="run_sweep")
            runtime.install_runtime(cfg=CFG, outdir=OUTDIR, budget=BUDGET)
        except Exception:
            pass

    # Gate evaluability check (warn always; fail-fast when policy requires)
    if require_gate_eval(CFG):
        require_gate_eval_or_die(CFG, context="run_sweep:post_reconcile")
    else:
        validate_gate_evaluability(CFG, context="run_sweep:post_reconcile")

    # factor mapping across ALL seeds
    all_seeds = sorted(set(list(CFG["seeds_calib"]) + list(CFG["seeds_eval"])))
    mapping_all = assign_factors_evenly(all_seeds)
    split_map = {as_int(s, default=0): ("calib" if s in CFG["seeds_calib"] else "eval") for s in all_seeds}
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
        if (parq.exists() and parq.stat().st_size > 0) or (csvp.exists() and csvp.stat().st_size > 0):
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
                and (emax == as_int(CFG.get("epochs", 1), default=1) - 1)
            )

            # CSV fallback completeness: if parquet probe incomplete but CSV exists and is complete, resume from CSV
            if (not complete) and (probe.suffix != ".csv") and csvp.exists():
                emin_csv, emax_csv, nrows_csv = _epoch_bounds_csv(csvp)
                complete_csv = (
                    (emax_csv is not None)
                    and (nrows_csv is not None)
                    and (nrows_csv >= 2)
                    and (emax_csv == as_int(CFG.get("epochs", 1), default=1) - 1)
                )
                if complete_csv:
                    print(f"[resume] skipping seed={seed} (csv complete: emax={emax_csv}, rows={nrows_csv})")
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
                print(f"[resume] skipping seed={seed} ({probe.suffix[1:]} complete: emax={emax}, rows={nrows})")
                paths.append(parq)  # always append the parquet basename for parquet case
                continue
            else:
                print(
                    f"[resume] redoing seed={seed} ({probe.suffix[1:]} incomplete: emin={emin}, emax={emax}, rows={nrows})"
                )
                # fall through to run_one

        f = mapping_all[seed]
        print(f"=== Run[{tag}]: seed={seed} factor={f['name']} ===", flush=True)
        _reset_budget()  # reset heavy-metric budgets/call counters per run/seed
        try:
            df = run_one(seed, tag=tag, monitor_ds=monitor_ds, factor=f)
            safe_write_parquet(df, parq)
            paths.append(parq)
        except Exception as e:
            fail_count += 1
            errlog = OUTDIR / f"errors_{tag}.log"
            msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] seed={seed} factor={f['name']} tag={tag} -> {repr(e)}\n"
            try:
                with open(errlog, "a", encoding="utf-8") as fh:
                    fh.write(msg)
            finally:
                pass
            print(f"[WARN] Run failed (seed={seed}): {e}")
            if fail_count >= as_int(CFG.get("max_failures", 6), default=6):
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
    Path(str(target) + ".md5").write_text(f"{md5}  {target.name}\n")
    print(f"[checksum] {target.name} md5={md5}")

    # End-of-sweep diagnostic summary (debug-only): catches "gate never evaluated" quickly.
    if DEBUG_SMOKE and (df_all is not None) and (not df_all.empty):
        try:
            print("\n[debug] Gate Diagnostic Summary:")
            for col in ["gate_reason", "gate_total_evidence"]:
                if col in df_all.columns:
                    try:
                        vc = df_all[col].value_counts(dropna=False)
                        # Keep output bounded to avoid log spam on large sweeps.
                        top = vc.head(25).to_dict()
                        print(f"  {col}: {top}")
                    except Exception:
                        pass

            if "gate_reason" in df_all.columns:
                try:
                    reasons = [str(r) for r in df_all["gate_reason"].dropna().unique().tolist()]
                    if not any("evaluated" in r for r in reasons):
                        print("[WARN] Gate was NEVER successfully evaluated in any run!")
                except Exception:
                    pass
        except Exception:
            pass

    return df_all


def _compute_gate_warn_time(
    gg: pd.DataFrame,
    col: str,
    warm_idx: int,
    reason_col: Optional[str] = None,
) -> Optional[int]:
    """Compute first post-warm epoch where gate fails for K consecutive evaluated epochs.

    Semantics:
      - gate_warn == 1: PASS; gate_warn == 0: FAIL
      - Non-evaluated epochs are treated as neutral (PASS) and do not contribute to runs.
      - If an evidence column exists, enforce CFG['gate_min_evidence'] as an additional guard.

    Returns the epoch number (int) or None.
    """
    if gg is None or getattr(gg, "empty", True):
        return None
    if "epoch" not in gg.columns or col not in gg.columns:
        return None

    warm_i = as_int(warm_idx, default=0)

    # ---- Build evaluation mask ----
    eval_mask = np.ones(len(gg), dtype=bool)

    # Prefer explicit evaluated bit when present for calibrated stream
    if col == "gate_warn_calib" and "gate_evaluated_calib" in gg.columns:
        ev = pd.to_numeric(gg["gate_evaluated_calib"], errors="coerce").fillna(0).to_numpy(dtype=float)
        eval_mask = eval_mask & (ev >= 0.5)

    # Infer a reason column if not provided
    if reason_col is None:
        if col == "gate_warn_calib" and "gate_reason_calib" in gg.columns:
            reason_col = "gate_reason_calib"
        elif "gate_reason" in gg.columns:
            reason_col = "gate_reason"

    # Reason prefix guard (aligns with existing conventions)
    # IMPORTANT: fill NaNs BEFORE astype(str). Otherwise NaN becomes the literal string "nan"
    # and silently fails the evaluated-prefix check, suppressing warnings.
    if reason_col is not None and reason_col in gg.columns:
        reason_arr = gg[reason_col].fillna("").astype(str).to_numpy()
        eval_mask = eval_mask & np.array([s.startswith("evaluated") for s in reason_arr], dtype=bool)

    # Evidence guard (if present)
    evidence_col: Optional[str] = None
    if col == "gate_warn_calib" and "gate_total_evidence_calib" in gg.columns:
        evidence_col = "gate_total_evidence_calib"
    elif "gate_total_evidence" in gg.columns:
        evidence_col = "gate_total_evidence"

    if evidence_col is not None and evidence_col in gg.columns:
        te = pd.to_numeric(gg[evidence_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        min_ev = float(as_int(CFG.get("gate_min_evidence", 16), default=16))
        if np.isfinite(min_ev) and min_ev > 0:
            eval_mask = eval_mask & (te >= min_ev)

    # ---- Debug: audit why warnings might be getting suppressed ----
    try:
        if (env_truthy("SCAR_SMOKE") or DEBUG_GATE) and mp.current_process().name == "MainProcess":
            n_eval = int(np.sum(eval_mask)) if eval_mask is not None else 0
            # Reason distribution (top-k) if available
            top_reasons = {}
            if reason_col is not None and reason_col in gg.columns:
                try:
                    top_reasons = gg[reason_col].fillna("").astype(str).value_counts().head(8).to_dict()
                except Exception:
                    top_reasons = {}
            print(
                f"[gate_warn_time] col={col} reason_col={reason_col} evidence_col={evidence_col} "
                f"warm={warm_i} n_rows={len(gg)} n_eval={n_eval} top_reasons={top_reasons}",
                flush=True,
            )
    except Exception:
        pass

    epn = pd.to_numeric(gg["epoch"], errors="coerce").to_numpy(dtype=float)
    gw = pd.to_numeric(gg[col], errors="coerce").to_numpy(dtype=float)

    # Non-evaluated epochs are neutral (PASS)
    gw = np.where(eval_mask, gw, 1.0)

    fail_mask = (gw < 0.5) & eval_mask & np.isfinite(gw)
    hit_idx = np.where((epn >= float(warm_i)) & fail_mask)[0]
    if hit_idx.size == 0:
        return None

    K_warn = as_int(CFG.get("warn_consec", 3), default=3)
    j_end = _first_run_end(hit_idx, K_warn)
    if j_end is None or int(j_end) < 0:
        return None

    try:
        # FIX: Return the FIRST epoch of the K-consecutive run, not the LAST.
        # _first_run_end returns an index (into hit_idx) for the last element of the first length-K run.
        j_start = int(j_end) - int(K_warn) + 1
        # Safety clamp (should be unnecessary if _first_run_end is correct, but harmless).
        if j_start < 0:
            j_start = 0

        ep_out = gg.iloc[int(hit_idx[int(j_start)])]["epoch"]
        ep_i = as_int(ep_out, default=-1)
        return ep_i if ep_i >= 0 else None
    except Exception:
        return None


def evaluate(df_all: pd.DataFrame, tag: str):
    # Early guards for empty/epoch-0 aggregates or missing columns
    if df_all is None or len(df_all) == 0 or not {"epoch", "seed"}.issubset(df_all.columns):
        print("[eval] no usable rows or missing epoch/seed; skipping evaluate()")
        return
    try:
        ep_series = to_numeric_series(df_all["epoch"], errors="coerce")
        ep_max = float(np.nanmax(ep_series.to_numpy(dtype=float)))
        if not np.isfinite(ep_max) or ep_max <= 0.0:
            print("[eval] aggregate has only epoch 0; skipping evaluate()")
            return
    except Exception:
        pass
    warm_idx = warm_idx_from_cfg(CFG)
    cal_mask = df_all["seed"].isin(CFG["seeds_calib"])
    eval_seeds = seeds_for_eval_from_env(CFG)
    ev_mask = df_all["seed"].isin(eval_seeds)
    df_cal_raw = df_all[cal_mask].copy()
    df_eval_raw = df_all[ev_mask].copy()
    no_eval = (df_eval_raw is None) or getattr(df_eval_raw, "empty", True)

    # --- Helper: enforce GT collapse time is within observed epoch range ---
    def _sanitize_t_collapse_in_window(g: pd.DataFrame, t_c: Optional[int], ctag: str) -> tuple[Optional[int], str]:
        """If GT proposes a collapse time after the last recorded epoch, treat as no-collapse."""
        try:
            ep = pd.to_numeric(g["epoch"], errors="coerce").to_numpy(dtype=float)
            ep = ep[np.isfinite(ep)]
            epoch_max = int(ep.max()) if ep.size else -1
        except Exception:
            epoch_max = -1

        if t_c is None:
            return None, "none"
        try:
            tci = int(t_c)
        except Exception:
            return None, "none"

        if epoch_max >= 0 and tci > epoch_max:
            return None, "none"
        return tci, str(ctag)

    def _verify_decl_consistency(df: pd.DataFrame, wd_offline: Any) -> bool:
        if df is None or getattr(df, "empty", True):
            return True
        if "window_decl_hash" not in df.columns:
            return True
        if window_decl_identity_hash is None:
            return True
        offline_hash = window_decl_identity_hash(wd_offline)
        online_hashes = (
            df["window_decl_hash"]
            .astype(str)
            .replace({"nan": "", "None": "", "": ""})
            .loc[lambda s: s.str.len() > 0]
            .unique()
            .tolist()
        )
        if len(online_hashes) == 0:
            return True
        if len(online_hashes) > 1:
            print(f"[WARN] multiple WindowDecl configs detected online: {online_hashes}")
        ok = online_hashes[0] == offline_hash
        if not ok:
            print(f"[WARN] offline WindowDecl hash {offline_hash} != online {online_hashes[0]}")
            print("       Any gate_*_calib series is COUNTERFACTUAL.")
        return ok

    if no_eval:
        print(
            "[eval] no evaluation seeds present; will skip learned/baseline evaluation but still write calibration aggregates."
        )
    # --- PRE-GT STAMP (anti-contamination): tag runs before using any “controls” pools ---
    # Rationale: factor labels are not admissibility predicates. Exclude runs that unified GT
    # already marks as soft/hard collapse from ALL calibration pools.
    try:
        if df_cal_raw is not None and isinstance(df_cal_raw, pd.DataFrame) and len(df_cal_raw) > 0:
            rows = []
            for (seed, factor), g in df_cal_raw.groupby(["seed", "factor"], sort=False):
                g = g.sort_values("epoch").copy()

                # Initial GT: infinite cutoff is enough to flag obvious collapse in "controls".
                t_c0, ctag0 = gt_collapse_time(g, grad_cutoff=np.inf)
                t_c0, ctag0 = _sanitize_t_collapse_in_window(g, t_c0, ctag0)

                # Epochwise tag with pre-warm forced to 'none'
                g["t_collapse_gt"] = np.nan if t_c0 is None else int(t_c0)
                ep = pd.to_numeric(g["epoch"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
                g["collapse_tag_gt"] = _epochwise_collapse_tag_gt(ep, t_c0, str(ctag0), int(warm_idx))

                rows.append(g)

            df_cal_raw = pd.concat(rows, ignore_index=True) if rows else df_cal_raw

            # Run-level reduction: hard > soft > none
            def _reduce_run_tag(s: pd.Series) -> str:
                ss = set(s.dropna().astype(str).tolist())
                if "hard" in ss:
                    return "hard"
                if "soft" in ss:
                    return "soft"
                return "none"

            df_cal_raw["_run_collapse_tag_gt_precal"] = df_cal_raw.groupby(["seed", "factor"], sort=False)[
                "collapse_tag_gt"
            ].transform(_reduce_run_tag)

            try:
                if mp.current_process().name == "MainProcess":
                    n_bad = int((df_cal_raw["_run_collapse_tag_gt_precal"].astype(str) != "none").sum())
                    if n_bad > 0:
                        print(f"[calib][guard] excluding collapsed rows from calibration pools (GT != 'none'): {n_bad}")
            except Exception:
                pass
    except Exception as _e:
        try:
            print(f"[calib][guard][WARN] pre-GT stamp failed; proceeding without GT guard: {_e!r}")
        except Exception:
            pass

    # --- calibrate gate gain threshold (bits/sample) from controls ---
    try:
        # Low-quantile calibration: gate fails only on unusually-low gain.
        try:
            q_gain = as_float(os.environ.get("SCAR_GATE_GAIN_Q", CFG.get("gate_gain_q", 0.05)), default=0.05)
        except Exception:
            q_gain = 0.05
        q_gain = float(min(max(float(q_gain), 0.0), 1.0))

        # Filter to evaluable control epochs only (gate needs 2W history before it can evaluate)
        gate_eval_start = gate_first_evaluable_epoch(CFG)
        df_gain_pool = df_cal_raw[
            (df_cal_raw["factor"].astype(str).str.lower().str.strip() == "none")
            & (to_numeric_series(df_cal_raw["epoch"], errors="coerce") >= gate_eval_start)
        ].copy()

        # Only rows where gate was actually evaluated (not insufficient_history)
        if "gate_reason" in df_gain_pool.columns:
            df_gain_pool = df_gain_pool[df_gain_pool["gate_reason"].fillna("").astype(str).str.startswith("evaluated")]

        # Exclude any calibration runs that unified GT already marks as collapsed (anti-contamination)
        try:
            if "_run_collapse_tag_gt_precal" in df_gain_pool.columns:
                df_gain_pool = df_gain_pool[df_gain_pool["_run_collapse_tag_gt_precal"].astype(str) == "none"].copy()
        except Exception as _e:
            try:
                print(f"[calib][WARN] anti-contamination filter failed (gain_pool): {_e!r}")
            except Exception:
                pass

        # Fallback safety: if filtering produced an empty pool, fall back to the least-contaminated pool available
        if getattr(df_gain_pool, "empty", True):
            df_gain_pool = df_cal_raw
            try:
                if "_run_collapse_tag_gt_precal" in df_gain_pool.columns:
                    df_gain_pool = df_gain_pool[
                        df_gain_pool["_run_collapse_tag_gt_precal"].astype(str) == "none"
                    ].copy()
            except Exception as _e:
                try:
                    print(f"[calib][WARN] anti-contamination filter failed (gain_pool fallback): {_e!r}")
                except Exception:
                    pass

        CFG["gate_gain_thresh"] = calibrate_gate_gain_thresh_from_controls(
            df_gain_pool,
            W=as_int(CFG.get("gate_window", 16), default=16),
            warm=gate_eval_start,
            q=q_gain,
        )

        update_json(
            OUTDIR / "gate_gain_thresh_calibration.json",
            {
                "gate_gain_thresh": float(CFG["gate_gain_thresh"]),
                "W": as_int(CFG.get("gate_window", 16), default=16),
                "warm": int(gate_eval_start),
                "q": float(q_gain),
                "pool_rows": int(len(df_gain_pool)) if df_gain_pool is not None else 0,
            },
        )

        # Keep runtime.CFG in sync for offline recompute paths.
        try:
            runtime.install_runtime(cfg=CFG, outdir=OUTDIR, budget=BUDGET)
        except Exception:
            pass

        try:
            if mp.current_process().name == "MainProcess":
                print(f"[calib] gate_gain_thresh (bits/sample) = {CFG['gate_gain_thresh']:.6f} (q={q_gain:.3f})")
        except Exception:
            print(f"[calib] gate_gain_thresh (bits/sample) = {CFG['gate_gain_thresh']:.6f} (q={q_gain:.3f})")

    except Exception as e:
        print(
            f"[WARN] gate gain calibration failed; keeping CFG['gate_gain_thresh']={CFG.get('gate_gain_thresh')}: {e}"
        )

    # --- Calibrate and persist a fixed-partition WindowDecl from controls (factor=='none' post-warm) ---
    try:
        # Pool for WindowDecl range calibration (can use broader set)
        df_control_ranges = df_cal_raw[
            (df_cal_raw["factor"].astype(str).str.lower().str.strip() == "none")
            & (to_numeric_series(df_cal_raw["epoch"], errors="coerce") >= int(warm_idx))
        ]
        # Anti-contamination: exclude any runs marked collapsed by GT (if available)
        try:
            if "_run_collapse_tag_gt_precal" in df_control_ranges.columns:
                df_control_ranges = df_control_ranges[
                    df_control_ranges["_run_collapse_tag_gt_precal"].astype(str) == "none"
                ].copy()
        except Exception as _e:
            try:
                print(f"[calib][WARN] anti-contamination filter failed (control_ranges): {_e!r}")
            except Exception:
                pass
        # Keep legacy variable name for downstream uses (audits, etc.)
        df_control = df_control_ranges

        decl_match = True
        window_decl = calibrate_window_from_controls(
            df_control=df_control_ranges,  # broader pool for ranges
            metrics=["var_out_k", "eff_dim"],
            weights={"var_out_k": 0.5, "eff_dim": 0.5},
            bins=as_int(CFG.get("gate_bins", 16), default=16),
            epsilon=as_float(CFG.get("gate_epsilon", 0.08), default=0.08),  # placeholder
            interventions=(lambda x: x,),
        )

        # Pool for epsilon calibration: MUST be evaluable epochs only
        gate_eval_start = gate_first_evaluable_epoch(CFG)
        df_control_evaluable = df_cal_raw[
            (df_cal_raw["factor"].astype(str).str.lower().str.strip() == "none")
            & (to_numeric_series(df_cal_raw["epoch"], errors="coerce") >= gate_eval_start)
        ].copy()
        # Anti-contamination: exclude any runs marked collapsed by GT (if available)
        try:
            if "_run_collapse_tag_gt_precal" in df_control_evaluable.columns:
                df_control_evaluable = df_control_evaluable[
                    df_control_evaluable["_run_collapse_tag_gt_precal"].astype(str) == "none"
                ].copy()
        except Exception as _e:
            try:
                print(f"[calib][WARN] anti-contamination filter failed (control_evaluable): {_e!r}")
            except Exception:
                pass

        if "gate_reason" in df_control_evaluable.columns:
            df_control_evaluable = df_control_evaluable[
                df_control_evaluable["gate_reason"].fillna("").astype(str).str.startswith("evaluated")
            ]

        # Calibrate epsilon from worst_tv + eps_stat (matches gate acceptance condition)
        try:
            _W = as_int(CFG.get("gate_window", 16), default=16)

            # Compute epsilon from quantile of (worst_tv + eps_stat) on evaluable controls.
            # This matches the gate condition: worst_tv <= epsilon - eps_stat
            if (
                len(df_control_evaluable) >= 16
                and "gate_worst_tv" in df_control_evaluable.columns
                and "gate_eps_stat" in df_control_evaluable.columns
            ):
                worst_tv = pd.to_numeric(df_control_evaluable["gate_worst_tv"], errors="coerce")
                eps_stat = pd.to_numeric(df_control_evaluable["gate_eps_stat"], errors="coerce").fillna(0.0)
                x = (worst_tv + eps_stat).to_numpy(dtype=float)
                x = x[np.isfinite(x)]

                if len(x) >= 16:
                    eps_new = float(np.quantile(x, 0.995))
                    n_vals = int(len(x))
                    if mp.current_process().name == "MainProcess":
                        print(f"[cal] epsilon from q99.5(worst_tv + eps_stat) = {eps_new:.4f} (n={n_vals})")
                else:
                    # Fallback to legacy calibration
                    eps_new, n_vals = calibrate_epsilon_from_controls(df_control_evaluable, window_decl, W=_W, q=0.995)
            else:
                # Fallback to legacy calibration if columns missing or pool too small
                eps_new, n_vals = calibrate_epsilon_from_controls(df_control_evaluable, window_decl, W=_W, q=0.995)
            decl_match = _verify_decl_consistency(df_eval_raw, window_decl)
            eps_floor = as_float(CFG.get("gate_epsilon_floor", 0.02), default=0.02)
            if (not np.isfinite(eps_floor)) or (eps_floor < 0.0):
                eps_floor = 0.02
            eps_new = max(float(eps_new), float(eps_floor))

            CFG["gate_epsilon"] = float(eps_new)
            window_decl.epsilon = float(eps_new)

            # In smoke mode, align WindowDecl bins with effective smoke policy
            reconcile_window_decl_inplace(window_decl, CFG)

            if mp.current_process().name == "MainProcess":
                print(
                    f"[cal] gate_epsilon set: {CFG['gate_epsilon']:.6f} from {int(n_vals)} TV samples (floor={eps_floor:.4f})"
                )
        except Exception:
            pass
        # Persist for future sweeps and install for any downstream checks in this process
        crng = getattr(window_decl, "cal_ranges", {})
        from typing import Mapping, cast

        wmap = cast(Mapping[str, float], getattr(window_decl, "weights", {}) or {})
        mets = list(getattr(window_decl, "metrics", []))
        bins = as_int(getattr(window_decl, "bins", 0), default=0)
        eps = as_float(getattr(window_decl, "epsilon", float("nan")), default=float("nan"))
        save_json(
            {
                "epsilon": float(eps),
                "metrics": list(mets),
                "weights": {str(k): float(v) for k, v in wmap.items()},
                "bins": int(bins),
                "cal_ranges": {str(k): [a, b] for k, v in crng.items() for (a, b) in [_float_pair(v)]},
                "kappa_transport_mismatch": True,
                "eps_stat_max_frac": as_float(CFG.get("gate_eps_stat_max_frac", 0.25), default=0.25),
            },
            OUTDIR / "window_decl_calibrated.json",
        )
        if callable(install_window_decl):
            install_window_decl(window_decl)
        _wire_fr_from_decl(window_decl)
        try:
            write_window_audit(
                OUTDIR,
                window_decl,
                note="Calibrated WindowDecl from controls (post-warm)",
                controls_used=len(df_control),
            )
            write_window_provenance_from_decl(OUTDIR, window_decl)
        except Exception:
            pass
        # --- Recompute per-epoch gate flag under calibrated settings (transport-consistent, offline) ---
        try:
            _W = as_int(CFG.get("gate_window", 16), default=16)
            df_eval_raw = df_eval_raw.sort_values(["seed", "factor", "epoch"]).reset_index(drop=True)
            df_eval_raw = recompute_gate_series_under_decl(df_eval_raw, window_decl, W=_W)
            if not decl_match:
                df_eval_raw["gate_calib_counterfactual"] = 1
                print("[WARN] gate_*_calib columns are COUNTERFACTUAL (offline decl != online decl)")
        except Exception as e:
            print(f"[WARN] offline calibrated gate recompute failed: {e}")
            if "gate_warn_calib" not in df_eval_raw.columns:
                df_eval_raw["gate_warn_calib"] = df_get_series(df_eval_raw, "gate_warn").fillna(0)
    except Exception as e:
        print(f"[WARN] window calibration/persist failed: {e}")

    # ---- Initial GT (t_c0/ctag0 with infinite cutoff) ----
    def add_gt(df_in):
        rows = []
        from typing import cast

        for key, g in df_in.groupby(["seed", "factor"]):
            seed, factor = cast(tuple[Any, Any], key)
            g = g.sort_values("epoch").copy()
            t_c0, ctag0 = gt_collapse_time(g, grad_cutoff=np.inf)
            t_c0, ctag0 = _sanitize_t_collapse_in_window(g, t_c0, ctag0)

            # Keep t_collapse_gt numeric; NaN denotes “no collapse”
            g["t_collapse_gt"] = np.nan if t_c0 is None else int(t_c0)

            # Epochwise tag: 'none' until collapse (post-warm), then 'soft'/'hard'
            ep = pd.to_numeric(g["epoch"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
            g["collapse_tag_gt"] = _epochwise_collapse_tag_gt(ep, t_c0, str(ctag0), int(warm_idx))
            rows.append(g)
        return pd.concat(rows, ignore_index=True) if rows else df_in

    df_cal_raw = add_gt(df_cal_raw)

    # ---- Calibrate grad explosion cutoff per-factor from healthy epochs (robust) ----
    grad_cut_by_factor = calibrate_grad_cutoff_per_factor(df_cal_raw)
    # --- HARDEN: never allow absurdly low or non-finite grad explosion cutoffs ---
    _floor = as_float(CFG.get("gt_grad_cutoff_floor", 0.10), default=0.10)
    if (not np.isfinite(_floor)) or (_floor <= 0):
        _floor = 0.10
    for _k in list(grad_cut_by_factor.keys()):
        try:
            _v = float(grad_cut_by_factor[_k])
        except Exception:
            grad_cut_by_factor[_k] = float(_floor)
            continue
        if (not np.isfinite(_v)) or (_v < float(_floor)):
            grad_cut_by_factor[_k] = float(_floor)
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
        from typing import cast

        for key, g in df_in.groupby(["seed", "factor"]):
            seed, factor = cast(tuple[Any, Any], key)
            g = g.sort_values("epoch").copy()
            cutoff = grad_cut_by_factor.get(str(factor), np.inf)
            t_c, ctag = gt_collapse_time(g, grad_cutoff=cutoff)
            t_c, ctag = _sanitize_t_collapse_in_window(g, t_c, ctag)

            # Keep t_collapse_gt numeric; NaN denotes “no collapse”
            g["t_collapse_gt"] = np.nan if t_c is None else int(t_c)

            # Epochwise tag: 'none' until collapse (post-warm), then 'soft'/'hard'
            ep = pd.to_numeric(g["epoch"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
            g["collapse_tag_gt"] = _epochwise_collapse_tag_gt(ep, t_c, str(ctag), int(warm_idx))
            rows.append(g)
        return pd.concat(rows, ignore_index=True) if rows else df_in

    # ---- Soft GT calibration (eff_dim_gt): env override priority ----
    # Priority: explicit fixed threshold > quantile from env (SCAR_SOFT_Q) > default calibration
    fixed_thr_env = os.environ.get("SCAR_FIXED_GT_RANK_MIN", "").strip()
    soft_q_env = os.environ.get("SCAR_SOFT_Q", "").strip()
    gtq = float(CFG.get("gt_rank_q", 0.075))

    def _soft_quantile_from_controls(df_src, q, warm_idx):
        """Compute q-quantile of eff_dim_gt on control ('none') after warm_idx.

        Anti-contamination: excludes runs marked as collapsed by PRE-GT stamp
        (`_run_collapse_tag_gt_precal != 'none'`) when that column exists.

        Returns None on failure.
        """
        if df_src is None or (not isinstance(df_src, pd.DataFrame)):
            return None
        if ("factor" not in df_src.columns) or ("epoch" not in df_src.columns):
            return None

        # Defensive clamp: q must be a valid quantile in [0, 1]
        try:
            q = float(q)
        except Exception:
            q = 0.05
        q = float(min(max(q, 0.0), 1.0))

        try:
            ep = to_numeric_series(df_src["epoch"], errors="coerce")
            base = df_src[_is_admissible_control(df_src) & (ep >= int(warm_idx))].copy()
        except Exception:
            return None

        col = "eff_dim_gt" if "eff_dim_gt" in base.columns else ("eff_dim" if "eff_dim" in base.columns else None)
        if col is None:
            return None

        try:
            vals = pd.to_numeric(base[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return None
            return float(np.quantile(vals, q))
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
        print(
            f"[calib] gt_rank_min set from factor=='none' @q={gtq:.3f} after warm={warm_idx}: {CFG['gt_rank_min']:.3f}"
        )

    # Compute GT exactly once using this finalized threshold
    df_cal_raw = add_gt_final(df_cal_raw)
    df_eval_raw = add_gt_final(df_eval_raw)

    # sanity: fail fast if GT didn't stick
    assert "collapse_tag_gt" in df_cal_raw.columns, "add_gt_final failed to stamp GT on df_cal_raw"
    if "collapse_tag_gt" not in df_eval_raw.columns:
        print("[eval] WARN: add_gt_final did not stamp GT on df_eval_raw; continuing with empty eval.")

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
    # Drop pers_H0 if unavailable (e.g., ripser absent) to avoid diluting calibration
    try:
        if "pers_H0" in soft_cal.columns and soft_cal["pers_H0"].isna().all():
            ph_metrics = [m for m in ph_metrics if m != "pers_H0"]
    except Exception:
        pass
    dir_map = calibrate_ph_directions(soft_cal, ph_metrics)
    save_json(dir_map, OUTDIR / f"ph_directions_{tag}.json")

    # ---- Run-level collapse tag map for evaluation (needed by summarize_detection) ----
    collapse_tag_map: dict[tuple[int, str], str] = {}
    for key, g in df_eval_raw.groupby(["seed", "factor"]):
        sd, fc = cast(tuple[Any, Any], key)

        if "collapse_tag_gt" in g.columns:
            ctag_this = _runlevel_collapse_tag(g, warm_idx)
        elif "is_collapse_epoch_gt" in g.columns and bool(g["is_collapse_epoch_gt"].fillna(False).astype(bool).any()):
            # legacy fallback if the epochwise tag column isn't present
            ctag_this = "soft"
        else:
            ctag_this = "none"

        collapse_tag_map[(as_int(sd, default=0), str(fc))] = ctag_this

    # ---- RP adequacy on eval: JL/native agreement pre-warm ----
    rp_flags = rp_adequacy_flags(
        df_eval_raw,
        warm_idx,
        corr_min=as_float(CFG.get("rp_corr_min", 0.9), default=0.9),
        min_pts=as_int(CFG.get("rp_min_pts", 8), default=8),
    )
    save_json(
        {
            "corr_min": as_float(CFG.get("rp_corr_min", 0.9), default=0.9),
            "min_pts": as_int(CFG.get("rp_min_pts", 8), default=8),
            "flags": {f"s{seed}-{factor}": as_int(flag, default=0) for (seed, factor), flag in rp_flags.items()},
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
    from typing import cast

    for key, g in df_cal_det.groupby(["seed", "factor"]):
        seed, factor = cast(tuple[Any, Any], key)
        g = g.sort_values("epoch").copy()
        t_c = g["t_collapse_gt"].iloc[0]
        ctag = _runlevel_collapse_tag(g, warm_idx)
        is_nontrig = ctag == "none"
        if pd.notna(t_c) and ctag in ("soft", "hard"):
            tci = as_int(t_c, default=-1)
            g = g[g["epoch"] < tci].copy()
        X, M = _metrics_matrix_with_missing(g, det_features)
        y = np.zeros(len(g), dtype=np.float32)
        if pd.notna(t_c) and ctag == "soft":
            tci = as_int(t_c, default=-1)
            horizon = as_int(CFG.get("detector_horizon"), default=10)
            for t in range(len(g)):
                ep = as_int(g.iloc[t]["epoch"], default=-(10**9))
                if ep >= warm_idx and ((tci - ep) <= horizon):
                    y[t] = 1.0
        grp = np.full(len(g), as_int(seed, default=0), dtype=np.int64)
        X_list.append(X)
        M_list.append(M)
        y_list.append(y)
        g_list.append(grp)
        fp_mask = ((g["epoch"].to_numpy() >= warm_idx) & is_nontrig).astype(np.float32)
        fp_mask_list.append(fp_mask)
        ep_list.append(g["epoch"].to_numpy().astype(np.int64))
        tc_soft_list.append(
            np.full(
                len(g),
                (float(as_int(t_c, default=-1)) if (pd.notna(t_c) and ctag == "soft") else np.nan),
                dtype=float,
            )
        )

    X_raw = np.concatenate(X_list, 0) if X_list else np.zeros((0, len(det_features)), dtype=np.float32)
    y = np.concatenate(y_list, 0) if y_list else np.zeros((0,), dtype=np.float32)
    groups = np.concatenate(g_list, 0) if g_list else np.zeros((0,), dtype=np.int64)
    fp_mask = np.concatenate(fp_mask_list, 0) if fp_mask_list else np.zeros_like(y, dtype=np.float32)
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
                bounds[m] = (pct_linear(col, 10.0), pct_linear(col, 90.0))
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

    w, b = _train_logistic_ridge_balanced(X_det, y, groups, steps=steps, lr=CFG["detector_lr"], l2=l2)
    model_info = dict(
        weights=w.tolist(),
        bias=float(b),
        thresh=float(thresh),
        features=det_features,
        horizon=as_int(CFG.get("detector_horizon", 10), default=10),
        norm_stats=stats,
        steps=steps,
        l2=l2,
        scheduled_metrics=[],
        det_use_missing=False,
        ph_win=as_int(CFG.get("ph_win", 8), default=8),
        ph_lambda=as_float(CFG.get("ph_lambda", 0.0), default=0.0),
        ph_two_sided=as_int(CFG.get("ph_two_sided", 0), default=0),
    )
    save_json(model_info, OUTDIR / f"learned_detector_{tag}.json")

    # --- τ→τ′ mapping under the deployed gate (deterministic; calibration-only) ---
    # Build calibration meta rows and raw feature matrix in deterministic (seed,factor,epoch) order
    meta_rows = []
    X_parts = []
    from typing import cast

    for key, g in df_cal_det.groupby(["seed", "factor"]):
        sd, fc = cast(tuple[Any, Any], key)
        gg = g.sort_values("epoch").reset_index(drop=True)
        X_raw_g, _ = _metrics_matrix_with_missing(gg, det_features)
        X_parts.append(X_raw_g)
        for _, r in gg.iterrows():
            meta_rows.append(
                {
                    "seed": as_int(r["seed"], default=0),
                    "factor": str(r["factor"]),
                    "epoch": as_int(r["epoch"], default=-(10**9)),
                }
            )
    X_cal_raw = (
        np.concatenate(X_parts, axis=0).astype(np.float32)
        if X_parts
        else np.zeros((0, len(det_features)), dtype=np.float32)
    )
    meta_df = pd.DataFrame(meta_rows)

    # Normalize with the same robust stats and compute deterministic OOF probabilities for fixed (steps,l2)
    # Build y_cal / groups_cal aligned 1:1 with meta_df/X_cal_raw
    Xn_cal = _apply_global_norm_impute(X_cal_raw.copy(), stats, det_features)
    y_cal = np.zeros(len(meta_df), dtype=np.float32)
    groups_cal = meta_df["seed"].to_numpy(dtype=np.int64)

    horiz = as_int(CFG.get("detector_horizon"), default=10)

    # Precompute per-(seed,factor) collapse target for labeling positives
    gt_by_sf = {}
    from typing import cast

    for key, g in df_cal_det.groupby(["seed", "factor"]):
        sd, fc = cast(tuple[Any, Any], key)
        try:
            t_c = g["t_collapse_gt"].iloc[0]
            ctag = _runlevel_collapse_tag(g, warm_idx)
        except Exception:
            t_c, ctag = (np.nan, "none")
        gt_by_sf[(as_int(sd, default=0), str(fc))] = (t_c, ctag)

    # Label positives only for soft collapses within the horizon after warm
    # Horizon for detector labeling (typed; tolerates str/np scalar)
    horiz = as_int(CFG.get("detector_horizon"), default=10)

    for i, r in meta_df.iterrows():
        sd = as_int(r["seed"], default=0)
        fc = str(r["factor"])
        ep = as_int(r["epoch"], default=-(10**9))
        t_c, ctag = gt_by_sf.get((sd, fc), (np.nan, "none"))
        tci = as_int(t_c, default=-1)
        if (tci >= 0) and ctag == "soft" and ep >= warm_idx and ((tci - ep) <= horiz):
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
        corr_min=as_float(CFG.get("rp_corr_min", 0.9), default=0.9),
        min_pts=as_int(CFG.get("rp_min_pts", 8), default=8),
    )

    # Family-gate parameters
    z_thr = as_float(CFG.get("family_z_thr", 1.5), default=1.5)
    K = as_int(CFG.get("family_window", 1), default=1)
    warn_consec = as_int(CFG.get("warn_consec", 3), default=3)
    fp_cap = as_float(SUCCESS_TARGET.get("max_early_fp_rate", 0.1), default=0.1)

    tau_prime, fp_measured = map_threshold_to_gated_fp(
        meta_df,
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

    # If there are no evaluation seeds, stop here after calibration-only artifacts.
    # This avoids downstream empty-frame processing (baselines, overlays, invariants) that can produce
    # index errors or meaningless outputs.
    if no_eval:
        print("[eval] no evaluation seeds present; skipping evaluation-only artifacts.")
        try:
            empty_events = pd.DataFrame(
                columns=[
                    "run_id",
                    "seed",
                    "factor",
                    "t_warn",
                    "t_collapse",
                    "collapse_tag",
                    "lead_time",
                ]
            )
            empty_events.to_csv(OUTDIR / f"detector_events_{tag}.csv", index=False)
            # Baseline event placeholders
            empty_events.to_csv(OUTDIR / f"baseline_events_ewma_{tag}.csv", index=False)
            empty_events.to_csv(OUTDIR / f"baseline_events_sma_{tag}.csv", index=False)
            empty_events.to_csv(OUTDIR / f"baseline_events_vote_{tag}.csv", index=False)
            empty_events.to_csv(OUTDIR / f"baseline_events_seq_{tag}.csv", index=False)
            empty_events.to_csv(OUTDIR / f"baseline_events_newma_{tag}.csv", index=False)
            empty_events.to_csv(OUTDIR / f"baseline_events_gate_{tag}.csv", index=False)
            # Summary placeholders
            pd.DataFrame([]).to_csv(OUTDIR / f"summary_learned_{tag}.csv", index=False)
            pd.DataFrame([]).to_csv(OUTDIR / f"summary_baseline_ewma_{tag}.csv", index=False)
            pd.DataFrame([]).to_csv(OUTDIR / f"summary_baseline_sma_{tag}.csv", index=False)
            pd.DataFrame([]).to_csv(OUTDIR / f"summary_baseline_vote_{tag}.csv", index=False)
            pd.DataFrame([]).to_csv(OUTDIR / f"summary_baseline_seq_{tag}.csv", index=False)
            pd.DataFrame([]).to_csv(OUTDIR / f"summary_baseline_newma_{tag}.csv", index=False)
            pd.DataFrame([]).to_csv(OUTDIR / f"summary_baseline_gate_{tag}.csv", index=False)
        except Exception:
            pass
        return

    # --- Invariant (smoke/debug): emitted event times must be in the run's observed epoch set ---
    def _assert_events_in_epoch_set(df_events: pd.DataFrame, df_source: pd.DataFrame, label: str) -> None:
        # Cheap, high-value regression guard: only run in smoke or debug.
        if not (env_truthy("SCAR_SMOKE") or env_truthy("SCAR_DEBUG") or env_truthy("SCAR_DEBUG_GATE")):
            return
        if df_events is None or (not isinstance(df_events, pd.DataFrame)) or df_events.empty:
            return
        if df_source is None or (not isinstance(df_source, pd.DataFrame)) or df_source.empty:
            return

        try:
            epochs_by_run: dict[tuple[int, str], set[int]] = {}
            for (seed0, factor0), g0 in df_source.groupby(["seed", "factor"], sort=False):
                ep = pd.to_numeric(g0["epoch"], errors="coerce").to_numpy(dtype=float)
                ep = ep[np.isfinite(ep)]
                if ep.size == 0:
                    continue
                key0 = (as_int(seed0, default=0), str(factor0))
                epochs_by_run[key0] = set(int(x) for x in ep.astype(int).tolist())

            bad: list[tuple[tuple[int, str], str, int]] = []
            for _, r in df_events.iterrows():
                key = (as_int(r.get("seed"), default=0), str(r.get("factor")))
                ep_set = epochs_by_run.get(key)
                if not ep_set:
                    continue
                for col in ("t_warn", "t_collapse"):
                    tv = r.get(col, np.nan)
                    if tv is None:
                        continue
                    try:
                        fv = float(tv)
                    except Exception:
                        continue
                    if not np.isfinite(fv):
                        continue
                    ti = int(fv)
                    if ti not in ep_set:
                        bad.append((key, col, ti))

            if bad:
                key, col, ti = bad[0]
                msg = (
                    f"[eval][invariant] {label}: {len(bad)} event time(s) not in epoch set; "
                    f"first key={key} col={col} t={ti}"
                )
                try:
                    print(msg, flush=True)
                except Exception:
                    pass
                try:
                    _append_sweep_log(msg)
                except Exception:
                    pass
                raise AssertionError(msg)
        except Exception:
            # If the checker itself fails (not a detected mismatch), do not break the sweep.
            return

    # ---- Apply learned detector to evaluation runs ----
    rows = []
    from typing import cast

    for key, g in df_eval_det.groupby(["seed", "factor"]):
        seed, factor = cast(tuple[Any, Any], key)
        gg = g.sort_values("epoch").reset_index(drop=True)
        X_raw, M_ind = _metrics_matrix_with_missing(gg, det_features)
        Xn = _apply_global_norm_impute(X_raw, stats, det_features)
        z = Xn @ w + b
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
        z_thr = as_float(CFG.get("family_z_thr", 1.5), default=1.5)
        K = as_int(CFG.get("family_window", 1), default=1)

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
        rp_under = bool(rp_flags.get((as_int(gg["seed"].iloc[0], default=0), str(gg["factor"].iloc[0])), 0))

        K_warn = as_int(CFG.get("warn_consec", 3), default=3)
        j_end = _first_run_end(hit_idx, K_warn)
        t_warn = None
        if j_end >= 0:
            i = hit_idx[j_end]
            geom_ok = _fam_alarm(i, geom_cols)
            dyn_ok = _fam_alarm(i, dyn_cols)
            gate_ok = dyn_ok if rp_under else (geom_ok or dyn_ok)
            if gate_ok:
                t_warn = int(gg.iloc[i]["epoch"])

        t_c_raw = gg["t_collapse_gt"].iloc[0]
        t_c = as_int(t_c_raw, default=-1)
        t_c = t_c if t_c >= 0 else None
        ctag = _runlevel_collapse_tag(gg, warm_idx)

        if not _warn_precedes_collapse(t_warn, t_c):
            t_warn = None

        lead = (
            float(t_c - t_warn)
            if (ctag in ["soft", "hard"] and t_warn is not None and t_c is not None)
            else float("nan")
        )
        rows.append(
            dict(
                run_id=f"s{as_int(seed, default=0)}-{str(factor)}",
                seed=as_int(seed, default=0),
                factor=str(factor),
                t_warn=t_warn,
                t_collapse=t_c,
                collapse_tag=ctag,
                ph_win=as_int(CFG.get("ph_win", 8), default=8),
                ph_lambda=as_float(CFG.get("ph_lambda", 0.0), default=0.0),
                ph_two_sided=as_int(CFG.get("ph_two_sided", 0), default=0),
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
    _assert_events_in_epoch_set(det_rows, df_eval_raw, label=f"detector_events_{tag}")
    det_rows.to_csv(OUTDIR / f"detector_events_{tag}.csv", index=False)

    # ---- Baseline postprocess (PH tables → canonical columns) ----
    def quorum_time(r: pd.Series, metrics: List[str], k: int) -> Optional[int]:
        ts_all: List[int] = []
        for m in metrics:
            v = r.get(f"t_{m}")
            iv = as_int(v, default=-1)
            if iv >= 0:
                ts_all.append(iv)
        if not ts_all:
            return None
        ts_sorted = sorted(ts_all)
        for t in ts_sorted:
            if sum(1 for tt in ts_all if tt <= t) >= k:
                return t
        return None

    def postprocess(tr_df, name):
        tcol = _first_t_column(tr_df)
        rows = []
        for _, r in tr_df.iterrows():
            if name == "vote":
                t_warn = quorum_time(r, VOTE_METRICS, as_int(CFG.get("warn_vote", 2), default=2))
            else:
                t_cand = r.get(tcol) if tcol else None
                tw = as_int(t_cand, default=-1)
                t_warn = tw if tw >= 0 else None

            t_cand2 = r.get("t_collapse")
            tc = as_int(t_cand2, default=-1)
            t_c = tc if tc >= 0 else None

            if not _warn_precedes_collapse(t_warn, t_c):
                t_warn = None

            rows.append(
                dict(
                    run_id=r["run_id"],
                    seed=as_int(r.get("seed"), default=0),
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
                    lead_time=(float(t_c - t_warn) if (t_warn is not None and t_c is not None) else float("nan")),
                )
            )
        out = pd.DataFrame(rows)
        _assert_events_in_epoch_set(out, df_eval_raw, label=f"baseline_events_{name}_{tag}")
        out.to_csv(OUTDIR / f"baseline_events_{name}_{tag}.csv", index=False)
        return out

    base_ewma = postprocess(tr_ewma, "ewma")
    base_sma = postprocess(tr_sma, "sma")
    base_vote = postprocess(tr_vote_ev, "vote")

    # ---- Sequential PH + rank-drop vote baseline ('seq') ----
    lam_cfg = CFG.get("seq_cusum_lambda", CFG["ph_lambda"])
    lam_f = as_float(lam_cfg, default=float(CFG.get("ph_lambda", 0.0)))
    win_short = as_int(CFG.get("ph_win_short"), default=8)
    rank_win = as_int(CFG.get("rank_win"), default=8)

    seq_rows = []
    from typing import cast

    for key, g in df_eval_raw.groupby(["seed", "factor"]):
        seed, factor = cast(tuple[Any, Any], key)
        gg = g.sort_values("epoch").reset_index(drop=True)
        # Unified PH preprocessing for pers_H0
        xs_full = _prep_series_for_ph(gg, "pers_H0")
        # Indices eligible for sequential tests: valid finite values at/after warm_idx
        idxs = [
            i
            for i, x in enumerate(xs_full)
            if (np.isfinite(x) and as_int(gg.iloc[i].get("epoch"), default=-1) >= warm_idx)
        ]
        ph_seq = [xs_full[i] for i in idxs]

        # Level test over preprocessed series (burn-in respects epoch timeline)
        t_level, _, _ = ph_window_sparse(
            xs_full,
            win=win_short,
            lam=lam_f,
            direction="down",
            burn_in=warm_idx,
            min_points=as_int(CFG.get("ph_min_points", 0), default=0),
            two_sided=False,
        )
        # CUSUM on the post-warm valid subsequence
        zs_delta = robust_z_series(_delta(ph_seq), win=win_short, burn_in=0)
        t_cusum, _ = cusum_one_sided(zs_delta, lam=lam_f, direction="down")

        def _map_sub_to_epoch(t_sub):
            return (
                None
                if t_sub is None
                else as_int(gg.iloc[idxs[t_sub]].get("epoch"), default=-1)
                if (0 <= t_sub < len(idxs))
                else None
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
            lam=lam_f,
            direction="down",
            burn_in=warm_idx,
            min_points=as_int(CFG.get("ph_min_points", 0), default=0),
            two_sided=False,
        )

        t_warn = None
        if (t_ph is not None) and (t_rank is not None):
            tw1 = as_int(t_ph, default=-1)
            tw2 = as_int(t_rank, default=-1)
            if tw1 >= 0 and tw2 >= 0:
                t_warn = max(tw1, tw2)

        t_c_raw = gg["t_collapse_gt"].iloc[0]
        t_c = as_int(t_c_raw, default=-1)
        t_c = t_c if t_c >= 0 else None
        ctag = _runlevel_collapse_tag(gg, warm_idx)

        if not _warn_precedes_collapse(t_warn, t_c):
            t_warn = None

        lead = (
            float(t_c - t_warn)
            if (ctag in ["soft", "hard"] and t_warn is not None and t_c is not None)
            else float("nan")
        )
        seq_rows.append(
            dict(
                run_id=f"s{as_int(seed, default=0)}-{str(factor)}",
                seed=as_int(seed, default=0),
                factor=str(factor),
                t_warn=t_warn,
                t_collapse=t_c,
                collapse_tag=ctag,
                ph_win_short=win_short,
                ph_lambda=lam_f,
                rank_win=rank_win,
                heavy_every=CFG["heavy_every"],
                metric_batches=CFG["metric_batches"],
                var_k_energy=CFG["var_k_energy"],
                var_k_max=CFG["var_k_max"],
                lead_time=lead,
            )
        )
    seq_events = pd.DataFrame(seq_rows)
    _assert_events_in_epoch_set(seq_events, df_eval_raw, label=f"baseline_events_seq_{tag}")
    seq_events.to_csv(OUTDIR / f"baseline_events_seq_{tag}.csv", index=False)

    # ---- NEWMA: calibrate lambda from controls (runner-proof) ----
    def _calibrate_newma_lambda(
        df_control: pd.DataFrame,
        warm: int,
        fast: float,
        slow: float,
        q: float,
        min_samples: int = 100,
    ) -> float:
        """Calibrate NEWMA threshold λ from control runs.

        Args:
            df_control: DataFrame containing control runs (factor=='none' or 'none_safe')
            warm: Warmup index - only consider epochs >= warm
            fast: Fast EWMA decay rate
            slow: Slow EWMA decay rate
            q: Quantile for threshold selection (e.g., 0.995)
            min_samples: Minimum samples required; returns conservative default if insufficient

        Returns:
            Calibrated λ threshold, or 3.0 if insufficient data
        """
        vals: list[float] = []
        if df_control is None or df_control.empty:
            return 3.0

        gcols = [c for c in ("seed", "factor") if c in df_control.columns]
        groups = df_control.groupby(gcols, sort=False) if gcols else [(None, df_control)]

        for _, g in groups:
            g = g.sort_values("epoch") if "epoch" in g.columns else g

            # Calibrate on ALL metrics used by the NEWMA evaluation to ensure scale alignment
            # Note: pers_H0 is a scheduled metric (may have NaNs), ftle may also be sparse
            for col in ("pers_H0", "eff_dim_gt", "eff_dim", "var_out_k", "ftle"):
                if col not in g.columns:
                    continue
                xs = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)

                # Apply validity mask if available (e.g., ftle_valid, topo_valid for pers_H0)
                valid_col = f"{col}_valid"
                if valid_col in g.columns:
                    valid_mask = g[valid_col].fillna(0).astype(bool).to_numpy()
                    xs = np.where(valid_mask, xs, np.nan)

                # Special handling for pers_H0: also check topo_done when present
                if col == "pers_H0" and "topo_done" in g.columns:
                    topo_mask = pd.to_numeric(g["topo_done"], errors="coerce").fillna(0).to_numpy(dtype=float) >= 1
                    xs = np.where(topo_mask, xs, np.nan)

                mu_f = None
                mu_s = None
                for t, x in enumerate(xs):
                    if not np.isfinite(x):
                        continue
                    a = float(x)
                    if mu_f is None:
                        mu_f = a
                        mu_s = a
                        continue
                    mu_f = (1.0 - float(fast)) * float(mu_f) + float(fast) * a
                    mu_s = (1.0 - float(slow)) * float(mu_s) + float(slow) * a
                    if int(t) >= int(warm):
                        vals.append(abs(float(mu_f) - float(mu_s)))

        if not vals:
            return 3.0

        # Guard against unstable quantile estimation with too few samples
        if len(vals) < int(min_samples):
            try:
                if mp.current_process().name == "MainProcess":
                    print(
                        f"[WARN] NEWMA calibration has only {len(vals)} samples (< {int(min_samples)}); "
                        f"using conservative default λ=3.0"
                    )
            except Exception:
                pass
            return 3.0

        return float(np.quantile(np.asarray(vals, dtype=float), float(q)))

    q_newma = float(CFG.get("newma_calib_q", 0.995))
    min_points_newma = int(CFG.get("newma_min_points", 0))

    # Prefer the eval-aligned controls pool if it exists; otherwise fall back to post-warm controls.
    # Include both "none" and "none_safe" factors as valid controls for calibration.
    try:
        df_control_for_newma = df_control_evaluable  # type: ignore[name-defined]
    except Exception:
        df_control_for_newma = df_cal_raw[
            (df_cal_raw["factor"].astype(str).str.lower().str.strip().isin(["none", "none_safe"]))
            & (to_numeric_series(df_cal_raw["epoch"], errors="coerce") >= int(warm_idx))
        ].copy()

    # Also filter/expand the pre-fetched df_control_evaluable if it exists but was too restrictive
    try:
        if df_control_for_newma is not None and len(df_control_for_newma) > 0:
            factor_col = df_control_for_newma["factor"].astype(str).str.lower().str.strip()
            if not factor_col.isin(["none_safe"]).any():
                none_safe_rows = df_cal_raw[
                    (df_cal_raw["factor"].astype(str).str.lower().str.strip() == "none_safe")
                    & (to_numeric_series(df_cal_raw["epoch"], errors="coerce") >= int(warm_idx))
                ]
                if len(none_safe_rows) > 0:
                    df_control_for_newma = pd.concat([df_control_for_newma, none_safe_rows], ignore_index=True)
    except Exception:
        pass

    if CFG.get("newma_lambda") is None:
        lam_newma = _calibrate_newma_lambda(
            df_control_for_newma,
            warm_idx,
            float(CFG["newma_fast"]),
            float(CFG["newma_slow"]),
            q=q_newma,
        )
        CFG["newma_lambda"] = float(lam_newma)
        try:
            save_json(
                {"newma_lambda": float(lam_newma), "q": float(q_newma), "warm_idx": int(warm_idx)},
                OUTDIR / f"newma_lambda_calibration_{tag}.json",
            )
        except Exception:
            pass
    else:
        lam_newma = as_float(CFG.get("newma_lambda"), default=3.0)

    def _newma_warn(xs: list[float]) -> Optional[int]:
        """Call NEWMA with optional min_points guard (backward-compatible if signature lacks it)."""
        try:
            return newma_warn_epoch(
                xs,
                fast=CFG["newma_fast"],
                slow=CFG["newma_slow"],
                lam=lam_newma,
                burn_in=warm_idx,
                min_points=min_points_newma,
            )
        except TypeError:
            return newma_warn_epoch(
                xs,
                fast=CFG["newma_fast"],
                slow=CFG["newma_slow"],
                lam=lam_newma,
                burn_in=warm_idx,
            )

    # ---- NEWMA baseline on PH + non-scheduled spectral/FTLE channels ----
    newma_rows = []
    from typing import cast

    for key, g in df_eval_raw.groupby(["seed", "factor"]):
        seed, factor = cast(tuple[Any, Any], key)
        gg = g.sort_values("epoch").reset_index(drop=True)

        def _mask_with_valid(series, valid):
            """Apply optional validity mask and coerce non-finite to NaN."""
            a = np.asarray(series, dtype=float)
            out = a.copy()
            out[~np.isfinite(out)] = np.nan
            if valid is None:
                return out
            v = np.asarray(valid, dtype=bool)
            # Only apply mask if shapes line up
            if v.shape == out.shape:
                out[~v] = np.nan
            return out

        xs_ph = _prep_series_for_ph(gg, "pers_H0")
        eff_col = "eff_dim_gt" if "eff_dim_gt" in gg.columns else "eff_dim"
        xs_eff = _mask_with_valid(gg[eff_col], None).tolist()

        xs_var = _mask_with_valid(
            gg["var_out_k"],
            gg["var_out_k_valid"] if "var_out_k_valid" in gg.columns else None,
        ).tolist()

        xs_ftle = _mask_with_valid(
            gg["ftle"],
            gg["ftle_valid"] if "ftle_valid" in gg.columns else None,
        ).tolist()

        # lam_newma is now precomputed above

        t1 = _newma_warn(xs_ph)
        t2 = _newma_warn(xs_eff)
        t3 = _newma_warn(xs_var)
        t4 = _newma_warn(xs_ftle)

        # Require at least N metrics to trigger to reduce FP from single-metric noise
        cand = [t for t in [t1, t2, t3, t4] if t is not None]
        min_newma_agreement = as_int(CFG.get("newma_min_agreement", 2), default=2)

        if len(cand) >= int(min_newma_agreement):
            cand_sorted = sorted(cand)
            tmin = None
            for candidate_epoch in cand_sorted:
                n_triggered = sum(1 for t in cand if t <= candidate_epoch)
                if n_triggered >= int(min_newma_agreement):
                    tmin = candidate_epoch
                    break
        else:
            tmin = None
        tw_tmp = as_int(tmin, default=-1) if tmin is not None else -1
        t_warn = tw_tmp if tw_tmp >= 0 else None

        t_c_raw = gg["t_collapse_gt"].iloc[0]
        t_c = as_int(t_c_raw, default=-1)
        t_c = t_c if t_c >= 0 else None
        ctag = _runlevel_collapse_tag(gg, warm_idx)

        if not _warn_precedes_collapse(t_warn, t_c):
            t_warn = None

        lead = (
            float(t_c - t_warn)
            if (ctag in ["soft", "hard"] and t_warn is not None and t_c is not None)
            else float("nan")
        )
        newma_rows.append(
            dict(
                run_id=f"s{as_int(seed, default=0)}-{str(factor)}",
                seed=as_int(seed, default=0),
                factor=str(factor),
                t_warn=t_warn,
                t_collapse=t_c,
                collapse_tag=ctag,
                ph_lambda=lam_newma,
                heavy_every=CFG["heavy_every"],
                metric_batches=CFG["metric_batches"],
                var_k_energy=CFG["var_k_energy"],
                var_k_max=CFG["var_k_max"],
                lead_time=lead,
            )
        )
    base_newma = pd.DataFrame(newma_rows)
    _assert_events_in_epoch_set(base_newma, df_eval_raw, label=f"baseline_events_newma_{tag}")
    base_newma.to_csv(OUTDIR / f"baseline_events_newma_{tag}.csv", index=False)

    # ---- Gate-only baseline (AUDIT lane): primary uses ONLINE gate_warn ----
    gate_rows = []
    gate_rows_drift_only = []
    gate_calib_rows = []  # optional analysis lane (forced calibrated stream)

    for key, g in df_eval_raw.groupby(["seed", "factor"]):
        seed, factor = cast(tuple[Any, Any], key)
        gg = g.sort_values("epoch").reset_index(drop=True)
        warm_idx_local = max(warm_idx_from_cfg(CFG), gate_first_evaluable_epoch(CFG))

        # Prefer calibrated gate stream when present AND not marked counterfactual; else fall back to online.
        use_calib = False
        if "gate_warn_calib" in gg.columns:
            if "gate_calib_counterfactual" in gg.columns:
                cc = pd.to_numeric(gg["gate_calib_counterfactual"], errors="coerce").fillna(0).to_numpy()
                use_calib = bool(np.nanmax(cc) <= 0)
            else:
                use_calib = True

        warn_col = "gate_warn_calib" if use_calib else "gate_warn"
        reason_col = None
        if use_calib and "gate_reason_calib" in gg.columns:
            reason_col = "gate_reason_calib"
        elif (not use_calib) and "gate_reason" in gg.columns:
            reason_col = "gate_reason"

        # Compute t_warn from the chosen stream (reason-aware when supported).
        try:
            t_warn = _compute_gate_warn_time(gg, warn_col, warm_idx_local, reason_col=reason_col)
        except TypeError:
            t_warn = _compute_gate_warn_time(gg, warn_col, warm_idx_local)

        # Drift-only lane: ignore gain failures entirely; only count stability/kappa failures.
        t_warn_drift = None
        try:
            if reason_col is not None and reason_col in gg.columns:
                epn = pd.to_numeric(gg["epoch"], errors="coerce").to_numpy(dtype=float)
                gw = pd.to_numeric(gg[warn_col], errors="coerce").to_numpy(dtype=float)
                rs = gg[reason_col].fillna("").astype(str).to_numpy()

                # Only consider evaluated epochs.
                eval_mask = np.array([s.startswith("evaluated") for s in rs], dtype=bool)

                # Allowed drift reasons (exclude evaluated_fail_gain by construction).
                allowed = (rs == "evaluated_fail_stability") | (rs == "evaluated_fail_kappa")

                fail_mask = (gw < 0.5) & eval_mask & allowed & (epn >= float(warm_idx_local))
                hit_idx = np.where(fail_mask)[0]

                K_warn = as_int(CFG.get("warn_consec", 3), default=3)
                j_end = _first_run_end(hit_idx, K_warn)
                if (j_end is not None) and int(j_end) >= 0:
                    j_start = int(j_end) - int(K_warn) + 1
                    if j_start < 0:
                        j_start = 0
                    # use the first epoch of the K-length failure run
                    t_warn_drift = as_int(gg.iloc[int(hit_idx[int(j_start)])]["epoch"], default=-1)
                    if t_warn_drift < 0:
                        t_warn_drift = None
        except Exception:
            t_warn_drift = None

        # Collapse time/tag from unified GT
        t_c_raw = gg["t_collapse_gt"].iloc[0] if ("t_collapse_gt" in gg.columns and len(gg) > 0) else np.nan
        tc_i = as_int(t_c_raw, default=-1)
        t_c = tc_i if tc_i >= 0 else None
        ctag = _runlevel_collapse_tag(gg, warm_idx)

        # Enforce precedence only if a collapse exists; otherwise keep warning for FP accounting.
        if (t_c is not None) and (not _warn_precedes_collapse(t_warn, t_c)):
            t_warn = None

        # Drift-only lane: apply the same precedence rule (only if a collapse exists).
        if (t_c is not None) and (not _warn_precedes_collapse(t_warn_drift, t_c)):
            t_warn_drift = None

        # FIX A: if there is a collapse and no warning was found, DO NOT set warn-at-collapse.
        # Write NaN so summaries correctly count it as a miss.
        t_warn_out = np.nan if t_warn is None else float(t_warn)
        t_c_out = np.nan if t_c is None else float(t_c)

        lead = (
            float(t_c - t_warn)  # type: ignore[operator]
            if (ctag in ("soft", "hard") and (t_warn is not None) and (t_c is not None))
            else float("nan")
        )
        detected = int(ctag in ("soft", "hard") and (t_warn is not None) and (t_c is not None))

        gate_rows.append(
            dict(
                run_id=f"s{as_int(seed, default=0)}-{str(factor)}",
                seed=as_int(seed, default=0),
                factor=str(factor),
                t_warn=t_warn_out,
                t_collapse=t_c_out,
                collapse_tag=ctag,
                ph_win=as_int(CFG.get("ph_win", 8), default=8),
                ph_lambda=as_float(CFG.get("ph_lambda", 0.0), default=0.0),
                ph_two_sided=as_int(CFG.get("ph_two_sided", 0), default=0),
                warn_vote=CFG.get("warn_vote", 2),
                heavy_every=CFG.get("heavy_every", 1),
                metric_batches=CFG.get("metric_batches", 1),
                var_k_energy=CFG.get("var_k_energy", 0.95),
                var_k_max=CFG.get("var_k_max", 32),
                lead_time=lead,
                detected=detected,
                gate_source=("calibrated" if use_calib else "online"),
                gate_warn_col=warn_col,
            )
        )

        lead_drift = (
            float(t_c - t_warn_drift)  # type: ignore[operator]
            if (ctag in ("soft", "hard") and (t_warn_drift is not None) and (t_c is not None))
            else float("nan")
        )
        detected_drift = int(ctag in ("soft", "hard") and (t_warn_drift is not None) and (t_c is not None))

        gate_rows_drift_only.append(
            dict(
                run_id=f"s{as_int(seed, default=0)}-{str(factor)}",
                seed=as_int(seed, default=0),
                factor=str(factor),
                t_warn=(np.nan if t_warn_drift is None else float(t_warn_drift)),
                t_collapse=(np.nan if t_c is None else float(t_c)),
                collapse_tag=ctag,
                lead_time=lead_drift,
                detected=detected_drift,
                gate_source=("calibrated" if use_calib else "online"),
                gate_lane="drift_only",
                gate_warn_col=warn_col,
                gate_reason_col=(reason_col if reason_col is not None else ""),
            )
        )

        # Optional counterfactual lane: force calibrated stream even if marked counterfactual; keep separate file.
        if "gate_warn_calib" in gg.columns:
            try:
                t_warn_calib = _compute_gate_warn_time(
                    gg,
                    "gate_warn_calib",
                    warm_idx_local,
                    reason_col=("gate_reason_calib" if "gate_reason_calib" in gg.columns else None),
                )
            except TypeError:
                t_warn_calib = _compute_gate_warn_time(gg, "gate_warn_calib", warm_idx_local)

            if (t_c is not None) and (not _warn_precedes_collapse(t_warn_calib, t_c)):
                t_warn_calib = None

            lead2 = (
                float(t_c - t_warn_calib)
                if (ctag in ("soft", "hard") and t_warn_calib is not None and t_c is not None)
                else float("nan")
            )

            gate_calib_rows.append(
                dict(
                    run_id=f"s{as_int(seed, default=0)}-{str(factor)}",
                    seed=as_int(seed, default=0),
                    factor=str(factor),
                    t_warn=(np.nan if t_warn_calib is None else float(t_warn_calib)),
                    t_collapse=(np.nan if t_c is None else float(t_c)),
                    collapse_tag=ctag,
                    ph_win=as_int(CFG.get("ph_win", 8), default=8),
                    ph_lambda=as_float(CFG.get("ph_lambda", 0.0), default=0.0),
                    ph_two_sided=as_int(CFG.get("ph_two_sided", 0), default=0),
                    warn_vote=CFG.get("warn_vote", 2),
                    heavy_every=CFG.get("heavy_every", 1),
                    metric_batches=CFG.get("metric_batches", 1),
                    var_k_energy=CFG.get("var_k_energy", 0.95),
                    var_k_max=CFG.get("var_k_max", 32),
                    lead_time=lead2,
                    gate_source="calibrated_counterfactual",
                )
            )
    base_gate = pd.DataFrame(gate_rows)
    _assert_events_in_epoch_set(base_gate, df_eval_raw, label=f"baseline_events_gate_{tag}")
    base_gate.to_csv(OUTDIR / f"baseline_events_gate_{tag}.csv", index=False)
    if gate_rows_drift_only:
        base_gate_drift = pd.DataFrame(gate_rows_drift_only)
        _assert_events_in_epoch_set(base_gate_drift, df_eval_raw, label=f"baseline_events_gate_drift_only_{tag}")
        base_gate_drift.to_csv(OUTDIR / f"baseline_events_gate_drift_only_{tag}.csv", index=False)

    # Write counterfactual separately for analysis (never overwrites primary audit lane)
    if gate_calib_rows:
        base_gate_cf = pd.DataFrame(gate_calib_rows)
        _assert_events_in_epoch_set(base_gate_cf, df_eval_raw, label=f"baseline_events_gate_counterfactual_{tag}")
        base_gate_cf.to_csv(OUTDIR / f"baseline_events_gate_counterfactual_{tag}.csv", index=False)

    # ---- Summaries ----
    summ_det = summarize_detection(det_rows, warm_idx=warm_idx)
    summ_ewma = summarize_detection(base_ewma, warm_idx=warm_idx)
    summ_sma = summarize_detection(base_sma, warm_idx=warm_idx)
    summ_vote = summarize_detection(base_vote, warm_idx=warm_idx)
    summ_seq = summarize_detection(seq_events, warm_idx=warm_idx)
    summ_newma = summarize_detection(base_newma, warm_idx=warm_idx)

    # Gate can only detect collapses that occur after it becomes evaluable
    warm_idx_gate = max(int(warm_idx), gate_first_evaluable_epoch(CFG))
    summ_gate = summarize_detection(base_gate, warm_idx=warm_idx_gate)
    summ_gate.to_csv(OUTDIR / f"summary_baseline_gate_{tag}.csv", index=False)

    # ---- Run-level FP (any warn on 'none' after warm) ----
    run_fp_learned = float("nan") if no_eval else summarize_runlevel_fp(det_rows, warm_idx=warm_idx)
    run_fp_ewma = summarize_runlevel_fp(base_ewma, warm_idx=warm_idx)
    run_fp_sma = summarize_runlevel_fp(base_sma, warm_idx=warm_idx)
    run_fp_vote = summarize_runlevel_fp(base_vote, warm_idx=warm_idx)
    run_fp_seq = summarize_runlevel_fp(seq_events, warm_idx=warm_idx)
    run_fp_newma = summarize_runlevel_fp(base_newma, warm_idx=warm_idx)
    # Gate FP accounting should honor the gate's evaluability threshold (needs 2W history)
    run_fp_gate = summarize_runlevel_fp(base_gate, warm_idx=warm_idx_gate)

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
    df_eval_overlay = _apply_warn_persistence(df_eval_overlay, as_int(CFG.get("warn_consec", 3), default=3))
    safe_write_parquet(df_eval_overlay, OUTDIR / f"bundle_runs_eval_with_overlays_{tag}.parquet")
    _write_both_overlays(df_eval_overlay, OUTDIR)

    try:
        compute_invariants_and_provenance(
            df_eval_overlay, artifact_csv=OUTDIR / f"bundle_runs_eval_with_overlays_{tag}.parquet"
        )
    except Exception as e:
        print(f"[WARN] invariants/provenance generation failed: {e}")

    # ---- Plots (use unified overlay) ----
    make_plots(df_eval_overlay, det_rows, tag=f"learned_{tag}", overlay_prefix="learned")
    make_plots(df_eval_overlay, base_gate, tag=f"gate_{tag}", overlay_prefix="gate")


# Note: _calib_fallback_col was removed as dead code.
def main():
    # ---- FAST PATHS: avoid heavy imports/side effects for help/version ----
    _argv = sys.argv[1:]

    # Apply smoke defaults (non-critical) then reconcile env/smoke policy once at entry.
    apply_smoke_overrides_inplace(CFG)
    try:
        reconcile_cfg_inplace(CFG, stage="main_pre")
        runtime.install_runtime(cfg=CFG, outdir=OUTDIR, budget=BUDGET)
    except Exception:
        pass

    # If user only asked for help/version, print and exit quickly.
    if any(a in ("-h", "--help") for a in _argv):
        # Minimal usage to avoid importing argparse or any heavy deps.
        print(
            "veriscope: early warning of soft collapse (Phase-4)\n"
            "Usage: veriscope [options]\n"
            "Common: --help, --version, (env) SCAR_SMOKE=1"
        )
        return 0

    if any(a in ("-V", "--version") for a in _argv):
        try:
            from . import __version__ as _ver
        except Exception:
            _ver = "unknown"
        print(_ver)
        return 0

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

    # Heavy-path setup (imports & side effects only when not in help/version fast path)
    _heavy_path = True  # set by fast path above; remains True here if we didn't early-return
    if _heavy_path:
        OUTDIR.mkdir(parents=True, exist_ok=True)

        # Import heavy libs lazily inside the heavy path
        try:
            import torch as _torch
            import torchvision as _tv
        except Exception:
            # Defer hard failure until an actual run needs them; keep env fields sparse.
            _torch = None
            _tv = None

        env = dict(
            torch=(getattr(_torch, "__version__", None) or "missing"),
            tv=(getattr(_tv, "__version__", None) or "missing"),
            cuda=(getattr(getattr(_torch, "version", None), "cuda", None) if _torch else None),
            cudnn=(
                getattr(getattr(getattr(_torch, "backends", None), "cudnn", None), "version", lambda: None)()
                if _torch
                else None
            ),
            device=str(CFG.get("device", "cpu")),
            deterministic=bool(CFG.get("deterministic", False)),
            cuda_hash=_cuda_hash(),
            pip_freeze_md5=_pip_freeze_md5(),
            cfg=CFG,
        )
        save_json(env, OUTDIR / "env.json")
        # --- defaults & units notes ---
        # Provide a sensible default for the family neighborhood size (used by family gate)
        CFG.setdefault("family_window", 1)  # set to 2 if you prefer a small neighborhood
        # Evidence floor for the family gate (avoid acting on extremely sparse TV windows)
        CFG.setdefault("gate_min_evidence", 16)
        # Optional env override (SCAR_GATE_MIN_EVIDENCE) to tune evidence floor per run
        try:
            CFG["gate_min_evidence"] = int(os.environ.get("SCAR_GATE_MIN_EVIDENCE", CFG.get("gate_min_evidence", 16)))
            if "SCAR_GATE_MIN_EVIDENCE" in os.environ:
                print(f"[env] gate_min_evidence={CFG['gate_min_evidence']}")
        except Exception as e:
            print(f"[WARN] bad SCAR_GATE_MIN_EVIDENCE={os.environ.get('SCAR_GATE_MIN_EVIDENCE')!r}: {e}")
        # Clarify units for gate gain everywhere we log/save config
        CFG.setdefault("gate_gain_units", "bits/sample")

        # Clamp var_k_max to avoid degenerate k (NEW)
        CFG["var_k_max"] = max(1, int(CFG.get("var_k_max", 32)))

        save_json(CFG, OUTDIR / "cfg_phase4.json")
        try:
            print("[cfg] note: 'gate_gain' is measured in bits/sample.")
            print(
                "[cfg] note: eps_stat aggregation uses weighted sum of per-metric BH–C bounds; see audit 'eps_aggregation' and 'counts_by_metric'."
            )
        except Exception:
            pass
        # one-time reproducibility capsule (non-fatal)
        try:
            repro_path = OUTDIR / "repro.json"
            if not repro_path.exists():
                repro = {
                    "torch": getattr(_torch, "__version__", None),
                    "torchvision": getattr(_tv, "__version__", None),
                    "cuda": getattr(getattr(_torch, "version", None), "cuda", None) if _torch else None,
                    "cudnn": (
                        getattr(getattr(getattr(_torch, "backends", None), "cudnn", None), "version", lambda: None)()
                        if _torch
                        else None
                    ),
                    "device": str(CFG.get("device", "cpu")),
                    "deterministic": bool(CFG.get("deterministic", False)),
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

    # ============================================================================
    # --- Load calibration (file → env overrides) + provenance capsule ---
    # ============================================================================
    _cal_path: Optional[str] = None
    _cal_sha16: Optional[str] = None
    try:
        _cal, _cal_sha16, _cal_path = load_calibration(os.environ.get("SCAR_CALIBRATION"))
        CFG["gate_min_evidence"] = as_int(_cal.get("gate_min_evidence", 16), default=16)
        CFG["gate_gain_thresh"] = as_float(_cal.get("gate_gain_thresh", 0.05), default=0.05)
        CFG["gate_gain_units"] = str(_cal.get("gate_gain_units", "bits/sample"))
        try:
            write_calibration_capsule(OUTDIR, _cal, _cal_path, _cal_sha16)
        except Exception:
            pass
        if mp.current_process().name == "MainProcess":
            print(
                f"[cal] min_evidence={CFG['gate_min_evidence']} gain_thresh={CFG['gate_gain_thresh']:.3f} ({CFG['gate_gain_units']})"
            )

        # Reapply environment overrides LAST so they win over any preloads
        try:
            v = os.environ.get("SCAR_GATE_MIN_EVIDENCE")
            if v is not None:
                try:
                    CFG["gate_min_evidence"] = int(v)
                    if mp.current_process().name == "MainProcess":
                        print(f"[env] gate_min_evidence={CFG['gate_min_evidence']}")
                except Exception as _e:
                    if mp.current_process().name == "MainProcess":
                        print(f"[WARN] bad SCAR_GATE_MIN_EVIDENCE={v!r}: {_e}")

            v = os.environ.get("SCAR_GATE_GAIN_THRESH")
            if v is not None:
                try:
                    CFG["gate_gain_thresh"] = float(v)
                    if mp.current_process().name == "MainProcess":
                        print(f"[env] gate_gain_thresh={CFG['gate_gain_thresh']:.4f}")
                except Exception as _e:
                    if mp.current_process().name == "MainProcess":
                        print(f"[WARN] bad SCAR_GATE_GAIN_THRESH={v!r}: {_e}")

            v = os.environ.get("SCAR_GATE_GAIN_UNITS")
            if v is not None:
                CFG["gate_gain_units"] = str(v)
                if mp.current_process().name == "MainProcess":
                    print(f"[env] gate_gain_units={CFG['gate_gain_units']!r}")

            v = os.environ.get("SCAR_GATE_EPSILON")
            if v is not None:
                try:
                    CFG["gate_epsilon"] = float(v)
                    if mp.current_process().name == "MainProcess":
                        print(f"[env] gate_epsilon={CFG['gate_epsilon']:.6f}")
                except Exception as _e:
                    if mp.current_process().name == "MainProcess":
                        print(f"[WARN] bad SCAR_GATE_EPSILON={v!r}: {_e}")

            v = os.environ.get("SCAR_GATE_HALT")
            if v is not None:
                try:
                    CFG["gate_early_exit"] = 1 if env_truthy("SCAR_GATE_HALT") else 0
                    if mp.current_process().name == "MainProcess":
                        print(f"[env] gate_early_exit={int(bool(CFG['gate_early_exit']))} (from SCAR_GATE_HALT)")
                except Exception as _e:
                    if mp.current_process().name == "MainProcess":
                        print(f"[WARN] bad SCAR_GATE_HALT={v!r}: {_e}")
        except Exception:
            pass

        # Clamp epsilon if user/env supplied something unreasonable
        try:
            ge = float(CFG.get("gate_epsilon", 0.08))
            if not (0.005 <= ge <= 0.5):
                if mp.current_process().name == "MainProcess":
                    print(f"[WARN] gate_epsilon out of range ({ge}); clamping to [0.005, 0.5]")
                CFG["gate_epsilon"] = float(min(max(ge, 0.005), 0.5))
        except Exception:
            CFG["gate_epsilon"] = 0.08

    except Exception as e:
        try:
            print(f"[WARN] calibration load failed: {e!r}; using defaults")
        except Exception:
            pass

    # ============================================================================
    # UNCONDITIONAL: Capture pre-reconcile snapshot, reconcile, then write audit
    # ============================================================================

    # Step 1: Capture pre-reconcile snapshot (BEFORE any smoke overrides)
    _pre_reconcile_snapshot: Dict[str, Any] = {}
    try:
        _pre_reconcile_snapshot = {
            "gate_min_evidence": as_int(CFG.get("gate_min_evidence", 16), default=16),
            "gate_gain_thresh": as_float(CFG.get("gate_gain_thresh", 0.05), default=0.05),
            "gate_gain_units": str(CFG.get("gate_gain_units", "bits/sample")),
            "gate_window": as_int(CFG.get("gate_window", 16), default=16),
            "gate_bins": as_int(CFG.get("gate_bins", 16), default=16),
            "gate_epsilon": as_float(CFG.get("gate_epsilon", 0.08), default=0.08),
            "gate_epsilon_floor": as_float(CFG.get("gate_epsilon_floor", 0.02), default=0.02),
            "gate_eps_stat_max_frac": as_float(CFG.get("gate_eps_stat_max_frac", 0.25), default=0.25),
            "gate_epsilon_sens": as_float(CFG.get("gate_epsilon_sens", 0.04), default=0.04),
            "gate_eps_stat_alpha": as_float(CFG.get("gate_eps_stat_alpha", 0.05), default=0.05),
            "gate_early_exit": bool(CFG.get("gate_early_exit", True)),
            "gate_min_evidence_full_eps": as_int(CFG.get("gate_min_evidence_full_eps", 16), default=16),
            "gate_eps_inflation_max": as_float(CFG.get("gate_eps_inflation_max", 4.0), default=4.0),
            "warmup": as_int(CFG.get("warmup", 0), default=0),
            "ph_burn": as_int(CFG.get("ph_burn", 0), default=0),
            "epochs": as_int(CFG.get("epochs", 72), default=72),
            "warn_consec": as_int(CFG.get("warn_consec", 3), default=3),
        }
    except Exception:
        pass

    # Step 2: Reconcile CFG (applies smoke overrides, env clamps, invariants)
    if require_gate_eval(CFG):
        # In require/fail-fast mode, do not continue with a stale CFG snapshot.
        reconcile_cfg_inplace(CFG, stage="post_calibration")
        _log_smoke_final_cfg(CFG, stage="post_calibration")
        runtime.install_runtime(cfg=CFG, outdir=OUTDIR, budget=BUDGET)
    else:
        try:
            reconcile_cfg_inplace(CFG, stage="post_calibration")
            _log_smoke_final_cfg(CFG, stage="post_calibration")
            runtime.install_runtime(cfg=CFG, outdir=OUTDIR, budget=BUDGET)
        except Exception:
            pass

    # Step 2b: Gate evaluability check (warn always; fail-fast when policy requires)
    if require_gate_eval(CFG):
        require_gate_eval_or_die(CFG, context="main:post_calibration")
    else:
        validate_gate_evaluability(CFG, context="main:post_calibration")

    # Step 3: Write precedence summary AFTER reconciliation (includes both snapshots)
    try:
        _prec = {
            "calibration_loaded_from": _cal_path or "",
            "calibration_sha16": _cal_sha16 or "",
            "env": {
                "SCAR_GATE_MIN_EVIDENCE": os.environ.get("SCAR_GATE_MIN_EVIDENCE", ""),
                "SCAR_GATE_GAIN_THRESH": os.environ.get("SCAR_GATE_GAIN_THRESH", ""),
                "SCAR_GATE_GAIN_UNITS": os.environ.get("SCAR_GATE_GAIN_UNITS", ""),
                "SCAR_GATE_HALT": os.environ.get("SCAR_GATE_HALT", ""),
                "SCAR_SMOKE": os.environ.get("SCAR_SMOKE", ""),
                "SCAR_MAX_EPOCHS": os.environ.get("SCAR_MAX_EPOCHS", ""),
                "SCAR_GATE_WINDOW": os.environ.get("SCAR_GATE_WINDOW", ""),
                "SCAR_GATE_BINS": os.environ.get("SCAR_GATE_BINS", ""),
                "SCAR_GATE_EPSILON": os.environ.get("SCAR_GATE_EPSILON", ""),
                "SCAR_WARN_CONSEC": os.environ.get("SCAR_WARN_CONSEC", ""),
                "SCAR_WARMUP": os.environ.get("SCAR_WARMUP", ""),
                "SCAR_PH_BURN": os.environ.get("SCAR_PH_BURN", ""),
            },
            "pre_reconcile": _pre_reconcile_snapshot,
            "final": {
                "gate_min_evidence": as_int(CFG.get("gate_min_evidence", 16), default=16),
                "gate_gain_thresh": as_float(CFG.get("gate_gain_thresh", 0.05), default=0.05),
                "gate_gain_units": str(CFG.get("gate_gain_units", "bits/sample")),
                "gate_window": as_int(CFG.get("gate_window", 16), default=16),
                "gate_bins": as_int(CFG.get("gate_bins", 16), default=16),
                "gate_epsilon": as_float(CFG.get("gate_epsilon", 0.08), default=0.08),
                "gate_epsilon_floor": as_float(CFG.get("gate_epsilon_floor", 0.02), default=0.02),
                "gate_eps_stat_max_frac": as_float(CFG.get("gate_eps_stat_max_frac", 0.25), default=0.25),
                "gate_epsilon_sens": as_float(CFG.get("gate_epsilon_sens", 0.04), default=0.04),
                "gate_eps_stat_alpha": as_float(CFG.get("gate_eps_stat_alpha", 0.05), default=0.05),
                "gate_early_exit": bool(CFG.get("gate_early_exit", True)),
                "gate_min_evidence_full_eps": as_int(CFG.get("gate_min_evidence_full_eps", 16), default=16),
                "gate_eps_inflation_max": as_float(CFG.get("gate_eps_inflation_max", 4.0), default=4.0),
                "warmup": as_int(CFG.get("warmup", 0), default=0),
                "ph_burn": as_int(CFG.get("ph_burn", 0), default=0),
                "warm_idx": warm_idx_from_cfg(CFG),
                "epochs": as_int(CFG.get("epochs", 72), default=72),
                "warn_consec": as_int(CFG.get("warn_consec", 3), default=3),
                "run_tag": str(globals().get("RUN_TAG", "")),
                "calibration_mode": bool(globals().get("CAL", False)),
                "smoke_mode": bool(globals().get("SMOKE", env_truthy("SCAR_SMOKE"))),
            },
            "smoke_applied": bool(env_truthy("SCAR_SMOKE")),
            "reconcile_stage": "post_calibration",
        }
        save_json(_prec, OUTDIR / "precedence_summary.json")
        _append_sweep_log(
            f"precedence_summary written: smoke={_prec['smoke_applied']} final_gate_window={_prec['final']['gate_window']}"
        )
    except Exception:
        pass

    # Step 4: Write dedicated smoke config file for unambiguous audit
    if env_truthy("SCAR_SMOKE"):
        try:
            smoke_final = {
                "description": "Final CFG values after smoke reconciliation (single source of truth)",
                "smoke_critical_keys": {k: CFG.get(k) for k in SMOKE_CRITICAL_KEYS if k in CFG},
                "derived": {
                    "warm_idx": warm_idx_from_cfg(CFG),
                    "gate_first_evaluable_epoch": gate_first_evaluable_epoch(CFG),
                },
                "env_bypasses_active": {
                    k: os.environ.get(v) for k, v in SMOKE_ENV_BYPASS.items() if os.environ.get(v) is not None
                },
            }
            save_json(smoke_final, OUTDIR / "smoke_final_cfg.json")
            _append_sweep_log(
                f"smoke_final_cfg written: gate_window={smoke_final['smoke_critical_keys'].get('gate_window')}"
            )
        except Exception:
            pass

    # --- hand off to pipeline (import late to avoid cycles / stale runtime) ---
    try:
        from veriscope.pipeline import run_sweep as pipeline_run_sweep
        from veriscope.pipeline import evaluate as pipeline_evaluate
    except Exception as e:
        print(f"[WARN] pipeline import failed ({e!r}); falling back to local run_sweep/evaluate")
        pipeline_run_sweep = run_sweep
        pipeline_evaluate = evaluate

    tag = RUN_TAG
    # or, if you want to respect RUN_TAG overrides:
    # tag = CFG.get("RUN_TAG", "full_v2")

    df_all = pipeline_run_sweep(tag=tag)
    if df_all is None or df_all.empty:
        print("No data; exiting.")
        return
    safe_write_parquet(df_all, OUTDIR / f"bundle_runs_{tag}.parquet")

    pipeline_evaluate(df_all, tag=tag)
    print("✓ Artifacts in:", OUTDIR.resolve())


if __name__ == "__main__":
    main()
