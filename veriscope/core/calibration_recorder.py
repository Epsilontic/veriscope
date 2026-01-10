"""Append-only calibration data recorder for offline epsilon analysis.

Records per-check diagnostics to enable:
- Empirical epsilon calibration: fit ε such that P(D_W > ε) ≈ target α
- Stratified analysis by phase, regime_id, or health status
- No decision changes - purely observational

Thread/multiprocess safety:
- By default, only writes from the main process to avoid CSV corruption
- Implements context manager protocol for proper cleanup
- Registers atexit handler as fallback (does NOT rely on __del__)
"""

from __future__ import annotations

import atexit
import csv
import json
import multiprocessing as _mp
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import numpy as np


def _is_main_process() -> bool:
    """Check if we're in the main process."""
    try:
        return _mp.current_process().name == "MainProcess"
    except Exception:
        return True


def _first_not_none(*vals: Any) -> Any:
    """Return the first value that is not None (does NOT treat 0/0.0/"" as missing)."""
    for v in vals:
        if v is not None:
            return v
    return None


def _safe_float(v: Any) -> Optional[float]:
    """Convert to float, return None if not finite."""
    if v is None:
        return None
    try:
        fv = float(v)
        return fv if np.isfinite(fv) else None
    except Exception:
        return None


def _safe_int(v: Any) -> Optional[int]:
    """Convert to int, return None on failure."""
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _safe_str(v: Any) -> str:
    """Convert to string, empty string for None."""
    if v is None:
        return ""
    return str(v)


def _as_opt_bool(x: Any) -> Optional[bool]:
    """Coerce to Optional[bool] safely (handles np.bool_, ints 0/1)."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        if x == 0:
            return False
        if x == 1:
            return True
        return None
    return None


@dataclass
class CalibrationRecord:
    """Single row of calibration data."""

    # Identification
    iter_num: int
    timestamp_utc: str

    # Change detector
    change_worst_DW: Optional[float]
    change_eps_eff: Optional[float]
    change_eps: Optional[float]
    change_eps_stat: Optional[float]
    change_margin_raw: Optional[float]
    change_margin_eff: Optional[float]
    change_margin_rel_eff: Optional[float]
    change_evaluated: bool
    change_ok: Optional[bool]

    # Regime detector
    regime_worst_DW: Optional[float]
    regime_eps_eff: Optional[float]
    regime_eps_stat: Optional[float]
    regime_margin_raw: Optional[float]
    regime_margin_eff: Optional[float]
    regime_margin_rel_eff: Optional[float]
    regime_check_ran: bool
    regime_ok: Optional[bool]

    # Evidence / counts
    evidence_total: Optional[int]

    # Decision attribution
    decision_source: str
    decision_source_stability: str
    decision_source_gate: str

    # State
    regime_state: str
    regime_id: Optional[str]
    regime_active_effective: bool
    regime_shadow_mode: bool

    # Health indicators
    gain_bits: Optional[float]
    gain_ok: Optional[bool]
    combined_ok: bool

    # Optional: per-metric breakdown (JSON-encoded)
    per_metric_tv_json: str = ""

    # Optional: extra metadata (JSON-encoded) to avoid column explosion
    meta_json: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for CSV/JSON serialization."""
        return {
            "iter_num": self.iter_num,
            "timestamp_utc": self.timestamp_utc,
            "change_worst_DW": self.change_worst_DW,
            "change_eps_eff": self.change_eps_eff,
            "change_eps": self.change_eps,
            "change_eps_stat": self.change_eps_stat,
            "change_margin_raw": self.change_margin_raw,
            "change_margin_eff": self.change_margin_eff,
            "change_margin_rel_eff": self.change_margin_rel_eff,
            "change_evaluated": self.change_evaluated,
            "change_ok": self.change_ok,
            "regime_worst_DW": self.regime_worst_DW,
            "regime_eps_eff": self.regime_eps_eff,
            "regime_eps_stat": self.regime_eps_stat,
            "regime_margin_raw": self.regime_margin_raw,
            "regime_margin_eff": self.regime_margin_eff,
            "regime_margin_rel_eff": self.regime_margin_rel_eff,
            "regime_check_ran": self.regime_check_ran,
            "regime_ok": self.regime_ok,
            "evidence_total": self.evidence_total,
            "decision_source": self.decision_source,
            "decision_source_stability": self.decision_source_stability,
            "decision_source_gate": self.decision_source_gate,
            "regime_state": self.regime_state,
            "regime_id": self.regime_id,
            "regime_active_effective": self.regime_active_effective,
            "regime_shadow_mode": self.regime_shadow_mode,
            "gain_bits": self.gain_bits,
            "gain_ok": self.gain_ok,
            "combined_ok": self.combined_ok,
            "per_metric_tv_json": self.per_metric_tv_json,
            "meta_json": self.meta_json,
        }


@dataclass
class CalibrationRecorderConfig:
    """Configuration for calibration recorder."""

    enabled: bool = False
    output_path: Optional[str] = None
    buffer_size: int = 100
    include_per_metric: bool = False

    # Filtering
    only_evaluated: bool = True
    only_healthy: bool = False

    # Multiprocess safety
    main_process_only: bool = True

    # CSV schema safety: if appending to an existing file, fail if header != COLUMNS.
    fail_on_schema_mismatch: bool = True


class CalibrationRecorder:
    """Append-only recorder for calibration data.

    Usage (context manager - recommended):
        with CalibrationRecorder(config) as recorder:
            record = recorder.extract_record(iter_num, audit, combined_ok)
            if record:
                recorder.write(record)

    Usage (manual):
        recorder = CalibrationRecorder(config)
        try:
            # ... recording ...
        finally:
            recorder.close()
    """

    COLUMNS: List[str] = [
        "iter_num",
        "timestamp_utc",
        "change_worst_DW",
        "change_eps_eff",
        "change_eps",
        "change_eps_stat",
        "change_margin_raw",
        "change_margin_eff",
        "change_margin_rel_eff",
        "change_evaluated",
        "change_ok",
        "regime_worst_DW",
        "regime_eps_eff",
        "regime_eps_stat",
        "regime_margin_raw",
        "regime_margin_eff",
        "regime_margin_rel_eff",
        "regime_check_ran",
        "regime_ok",
        "evidence_total",
        "decision_source",
        "decision_source_stability",
        "decision_source_gate",
        "regime_state",
        "regime_id",
        "regime_active_effective",
        "regime_shadow_mode",
        "gain_bits",
        "gain_ok",
        "combined_ok",
        "per_metric_tv_json",
        "meta_json",
    ]

    def __init__(self, config: Optional[CalibrationRecorderConfig] = None):
        self.config = config or CalibrationRecorderConfig()
        self._buffer: List[CalibrationRecord] = []
        self._file: Optional[TextIO] = None
        self._writer: Optional[csv.DictWriter] = None
        self._records_written: int = 0
        self._closed: bool = False
        self._atexit_registered: bool = False

        self._can_write = self._check_can_write()

        if self._can_write and self.config.enabled and self.config.output_path:
            self._open_file()

    def _check_can_write(self) -> bool:
        """Check if this instance should write to disk."""
        if not self.config.enabled:
            return False
        if self.config.main_process_only and not _is_main_process():
            return False
        return True

    def _open_file(self) -> None:
        """Open output file and write header if new."""
        if not self.config.output_path or not self._can_write:
            return

        path = Path(self.config.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = path.exists() and path.stat().st_size > 0

        # Fail-fast schema/header validation to prevent silent CSV corruption across versions.
        if file_exists and bool(getattr(self.config, "fail_on_schema_mismatch", True)):
            with open(path, "r", newline="", encoding="utf-8") as f_in:
                reader = csv.reader(f_in)
                header = next(reader, [])
            if list(header) != list(self.COLUMNS):
                raise RuntimeError(
                    "Calibration CSV schema mismatch.\n"
                    f"  path={str(path)!r}\n"
                    f"  file_header={header}\n"
                    f"  expected_header={self.COLUMNS}\n"
                    "Refusing to append because this would silently corrupt parsing.\n"
                    "Use a new output_path (e.g. calibration_v2.csv) or disable "
                    "fail_on_schema_mismatch (NOT recommended)."
                )

        self._file = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)

        if not file_exists:
            self._writer.writeheader()

        if not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True

    def _atexit_cleanup(self) -> None:
        """Cleanup handler for interpreter shutdown."""
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass

    def __enter__(self) -> "CalibrationRecorder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def extract_record(
        self,
        iter_num: int,
        audit: Dict[str, Any],
        combined_ok: bool,
    ) -> Optional[CalibrationRecord]:
        """Extract calibration record from audit dict."""
        if not self.config.enabled or not self._can_write:
            return None

        evaluated = bool(audit.get("evaluated", True))
        if self.config.only_evaluated and not evaluated:
            return None

        if self.config.only_healthy and not combined_ok:
            return None

        per_metric_json = ""
        if self.config.include_per_metric:
            pm = audit.get("per_metric_tv") or audit.get("drifts") or {}
            if pm:
                try:
                    pm_clean = {}
                    for k, v in pm.items():
                        if isinstance(v, (int, float, np.number)):
                            pm_clean[k] = float(v) if np.isfinite(float(v)) else None
                        else:
                            pm_clean[k] = v
                    per_metric_json = json.dumps(pm_clean)
                except Exception:
                    per_metric_json = ""

        gain_bits_val = _safe_float(_first_not_none(audit.get("change_gain_bits"), audit.get("gain_bits")))

        evidence_total_val = _safe_int(
            _first_not_none(audit.get("evidence_total"), audit.get("total_evidence"), audit.get("evidence"))
        )

        # Keep consistent keys (even if None) for stable downstream parsing
        meta = {
            "base_policy": audit.get("base_policy"),
            "base_persistence_k": audit.get("base_persistence_k"),
            "base_eps_sens": audit.get("base_eps_sens"),
            "base_min_metrics_exceeding_effective": audit.get("base_min_metrics_exceeding_effective"),
            "regime_epsilon": audit.get("regime_epsilon"),
            "regime_policy": audit.get("regime_policy"),
            "regime_persistence_k": audit.get("regime_persistence_k"),
            "regime_eps_sens_used": audit.get("regime_eps_sens_used"),
        }
        try:
            meta_json = json.dumps(meta)
        except Exception:
            meta_json = ""

        return CalibrationRecord(
            iter_num=iter_num,
            timestamp_utc=datetime.utcnow().isoformat(),
            change_worst_DW=_safe_float(audit.get("change_worst_DW")),
            change_eps_eff=_safe_float(audit.get("change_eps_eff")),
            change_eps=_safe_float(audit.get("change_eps")),
            change_eps_stat=_safe_float(audit.get("change_eps_stat")),
            change_margin_raw=_safe_float(audit.get("change_margin_raw")),
            change_margin_eff=_safe_float(audit.get("change_margin_eff")),
            change_margin_rel_eff=_safe_float(audit.get("change_margin_rel_eff")),
            change_evaluated=bool(audit.get("change_evaluated", evaluated)),
            change_ok=_as_opt_bool(audit.get("change_dw_ok")),
            regime_worst_DW=_safe_float(audit.get("regime_worst_DW")),
            regime_eps_eff=_safe_float(audit.get("regime_eps_eff")),
            regime_eps_stat=_safe_float(audit.get("regime_eps_stat")),
            regime_margin_raw=_safe_float(audit.get("regime_margin_raw")),
            regime_margin_eff=_safe_float(audit.get("regime_margin_eff")),
            regime_margin_rel_eff=_safe_float(audit.get("regime_margin_rel_eff")),
            regime_check_ran=bool(audit.get("regime_check_ran", False)),
            regime_ok=_as_opt_bool(audit.get("regime_ok")),
            evidence_total=evidence_total_val,
            decision_source=_safe_str(audit.get("decision_source")),
            decision_source_stability=_safe_str(audit.get("decision_source_stability")),
            decision_source_gate=_safe_str(audit.get("decision_source_gate")),
            regime_state=_safe_str(audit.get("regime_state")),
            regime_id=audit.get("regime_id"),
            regime_active_effective=bool(audit.get("regime_active_effective", False)),
            regime_shadow_mode=bool(audit.get("regime_shadow_mode", False)),
            gain_bits=gain_bits_val,
            gain_ok=_as_opt_bool(audit.get("change_gain_ok")),
            combined_ok=combined_ok,
            per_metric_tv_json=per_metric_json,
            meta_json=meta_json,
        )

    def write(self, record: CalibrationRecord) -> None:
        """Buffer a record, flush if buffer full."""
        if not self._can_write or self._closed:
            return

        self._buffer.append(record)

        if len(self._buffer) >= self.config.buffer_size:
            self.flush()

    def flush(self) -> int:
        """Write buffered records to disk. Returns count written."""
        if not self._buffer or not self._can_write or self._closed:
            return 0

        if self._writer is None and self.config.output_path:
            self._open_file()

        count = 0
        if self._writer:
            for record in self._buffer:
                self._writer.writerow(record.to_dict())
                count += 1
            if self._file:
                self._file.flush()

        self._records_written += count
        self._buffer = []
        return count

    def close(self) -> None:
        """Flush and close output file."""
        if self._closed:
            return

        # CRITICAL FIX: flush BEFORE setting _closed
        # Otherwise flush() will no-op due to the _closed guard and lose buffered records
        self.flush()
        self._closed = True

        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._writer = None

    @property
    def records_written(self) -> int:
        return self._records_written

    @property
    def records_buffered(self) -> int:
        return len(self._buffer)

    @property
    def is_active(self) -> bool:
        return self._can_write and not self._closed
