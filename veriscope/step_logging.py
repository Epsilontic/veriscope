from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from veriscope.core._coerce import (
    coerce_nonneg_int,
    coerce_optional_float,
    coerce_optional_nonneg_int,
    iso_z,
)
from veriscope.core.jsonutil import atomic_append_jsonl, atomic_write_json

UNIVERSAL_METRIC_KEYS = (
    "loss",
    "lr",
    "step_time",
    "grad_norm",
    "update_norm",
    "overflow",
    "nan_count",
)


def _normalize_sketch(sketch: Optional[Sequence[float]], *, max_len: int) -> Optional[list[float]]:
    if sketch is None:
        return None
    values = list(sketch)
    if len(values) > max_len:
        raise ValueError(f"sketch length must be <= {max_len}")
    normalized: list[float] = []
    for idx, raw in enumerate(values):
        val = coerce_optional_float(raw)
        if val is None:
            raise ValueError(f"sketch[{idx}] is not a finite number")
        normalized.append(val)
    return normalized


def _normalize_metrics(metrics: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {}
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        name = str(key).strip()
        if not name:
            continue
        if isinstance(value, float):
            out[name] = coerce_optional_float(value)
            continue
        out[name] = value
    return out


@dataclass(frozen=True)
class UniversalLogPaths:
    jsonl_path: Path
    manifest_path: Path


class UniversalStepLogger:
    """
    Thin JSONL step logger for framework-agnostic training loops.

    The emitted records are intentionally simple:
    - one JSON object per step in `universal_steps.jsonl`
    - a companion `universal_manifest.json` with run-level metadata
    """

    def __init__(
        self,
        outdir: Path | str,
        *,
        run_id: str,
        gate_preset: str = "universal_v0",
        jsonl_name: str = "universal_steps.jsonl",
        manifest_name: str = "universal_manifest.json",
        sketch_max_len: int = 64,
        fsync: bool = False,
        calibration_hooks: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not str(run_id).strip():
            raise ValueError("run_id must be a non-empty string")
        if sketch_max_len <= 0:
            raise ValueError("sketch_max_len must be > 0")

        root = Path(outdir).expanduser()
        root.mkdir(parents=True, exist_ok=True)

        self._paths = UniversalLogPaths(
            jsonl_path=root / str(jsonl_name),
            manifest_path=root / str(manifest_name),
        )
        self._fsync = bool(fsync)
        self._sketch_max_len = int(sketch_max_len)
        self._run_id = str(run_id).strip()
        self._step_count = 0
        self._last_step: Optional[int] = None

        self._manifest: dict[str, Any] = {
            "schema_version": 1,
            "format": "veriscope.universal_step.v1",
            "run_id": self._run_id,
            "gate_preset": str(gate_preset).strip() or "universal_v0",
            "created_ts_utc": iso_z(),
            "universal_metrics": list(UNIVERSAL_METRIC_KEYS),
            "records": 0,
            "run_status": None,
            "runner_exit_code": None,
            "ended_ts_utc": None,
            "jsonl_path": self._paths.jsonl_path.name,
        }
        if calibration_hooks:
            self._manifest["calibration_hooks"] = dict(calibration_hooks)
        atomic_write_json(self._paths.manifest_path, self._manifest)

    @property
    def jsonl_path(self) -> Path:
        return self._paths.jsonl_path

    @property
    def manifest_path(self) -> Path:
        return self._paths.manifest_path

    def log(
        self,
        *,
        step: int,
        loss: Optional[float] = None,
        lr: Optional[float] = None,
        step_time: Optional[float] = None,
        grad_norm: Optional[float] = None,
        update_norm: Optional[float] = None,
        overflow: Optional[int | bool] = None,
        nan_count: Optional[int] = None,
        sketch: Optional[Sequence[float]] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        ts_utc: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        step_int = coerce_nonneg_int(step, field="step")
        if self._last_step is not None and step_int <= self._last_step:
            raise ValueError(f"step must be strictly increasing (got {step_int} after {self._last_step})")

        overflow_int = None
        if overflow is not None:
            overflow_int = 1 if bool(overflow) else 0

        record: dict[str, Any] = {
            "schema_version": 1,
            "record_type": "step",
            "run_id": self._run_id,
            "step": step_int,
            "ts_utc": str(ts_utc).strip() if ts_utc else iso_z(),
            "loss": coerce_optional_float(loss),
            "lr": coerce_optional_float(lr),
            "step_time": coerce_optional_float(step_time),
            "grad_norm": coerce_optional_float(grad_norm),
            "update_norm": coerce_optional_float(update_norm),
            "overflow": overflow_int,
            "nan_count": coerce_optional_nonneg_int(nan_count),
        }

        sketch_norm = _normalize_sketch(sketch, max_len=self._sketch_max_len)
        if sketch_norm is not None:
            record["sketch"] = sketch_norm

        metrics_norm = _normalize_metrics(metrics)
        if metrics_norm:
            record["metrics"] = metrics_norm

        if extra:
            record["extra"] = dict(extra)

        atomic_append_jsonl(self._paths.jsonl_path, record, fsync=self._fsync)
        self._last_step = step_int
        self._step_count += 1
        self._manifest["records"] = int(self._step_count)
        atomic_write_json(self._paths.manifest_path, self._manifest)

    def finalize(
        self,
        *,
        run_status: str = "success",
        runner_exit_code: Optional[int] = 0,
        ended_ts_utc: Optional[str] = None,
    ) -> None:
        self._manifest["run_status"] = str(run_status)
        self._manifest["runner_exit_code"] = runner_exit_code
        self._manifest["ended_ts_utc"] = str(ended_ts_utc).strip() if ended_ts_utc else iso_z()
        atomic_write_json(self._paths.manifest_path, self._manifest)
