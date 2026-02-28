from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from veriscope.core.evidence import compute_loss_delta_z
from veriscope.core.gate import GateEngine
from veriscope.core.gate_session import GateSession
from veriscope.core.transport import DeclTransport
from veriscope.core.window import FRWindow, WindowDecl


def _pkg_version() -> str:
    try:
        import importlib.metadata as md

        return md.version("veriscope")
    except Exception:
        return "unknown"


class ConvenienceLogger:
    """High-level onboarding API: feed loss values, produce a valid capsule."""

    def __init__(
        self,
        outdir: Path,
        *,
        gate_window: int = 20,
        gate_epsilon: float = 0.12,
        gate_min_evidence: int = 8,
        gate_policy: str = "persistence",
        gate_persistence_k: int = 2,
        cadence: int = 10,
        observer_mode: bool = False,
        run_id: Optional[str] = None,
        gate_preset: str = "auto",
    ) -> None:
        self._cadence = max(1, int(cadence))
        self._gate_window = int(gate_window)
        self._loss_history: List[float] = []

        wd = WindowDecl(
            epsilon=float(gate_epsilon),
            metrics=["loss_delta_z"],
            weights={"loss_delta_z": 1.0},
            bins=16,
            interventions=(lambda x: x,),
            cal_ranges={"loss_delta_z": (-6.0, 6.0)},
        )
        transport = DeclTransport(wd)
        wd.attach_transport(transport)
        fr_win = FRWindow(decl=wd, transport=transport, tests=())
        ge = GateEngine(
            frwin=fr_win,
            gain_thresh=0.0,
            eps_stat_alpha=0.05,
            eps_stat_max_frac=0.25,
            eps_sens=0.04,
            min_evidence=int(gate_min_evidence),
            policy=str(gate_policy),
            persistence_k=int(gate_persistence_k),
            min_metrics_exceeding=1,
        )

        # No timestamps or observer-specific fields in window signature to preserve comparability.
        window_signature = {
            "schema_version": 1,
            "code_identity": {"package_version": _pkg_version()},
            "gate_controls": {
                "gate_window": int(gate_window),
                "gate_epsilon": float(gate_epsilon),
                "min_evidence": int(gate_min_evidence),
                "gate_policy": str(gate_policy),
                "gate_persistence_k": int(gate_persistence_k),
                "gain_thresh": 0.0,
                "eps_stat_alpha": 0.05,
                "eps_stat_max_frac": 0.25,
                "eps_sens": 0.04,
                "min_metrics_exceeding": 1,
            },
            "metric_interval": int(self._cadence),
            "metric_pipeline": {"transport": "loss_delta_z_auto"},
        }

        self._session = GateSession(
            outdir=Path(outdir),
            run_id=run_id or uuid.uuid4().hex[:12],
            window_decl=wd,
            gate_engine=ge,
            gate_window=int(gate_window),
            gate_policy=str(gate_policy),
            gate_min_evidence=int(gate_min_evidence),
            gate_preset=str(gate_preset),
            window_signature=window_signature,
            cadence=int(self._cadence),
            observer_mode=bool(observer_mode),
        )

    def step(self, step: int, *, loss: float) -> Optional[str]:
        """Record one training step. Returns a decision when a gate is evaluated."""
        self._loss_history.append(float(loss))

        if int(step) % self._cadence != 0:
            return None

        loss_delta_z = compute_loss_delta_z(loss, self._loss_history[:-1], self._gate_window)
        self._session.record_step(int(step), loss_delta_z=loss_delta_z)

        if int(step) > 0 and int(step) % (self._cadence * self._gate_window) == 0:
            record = self._session.evaluate(step=int(step))
            return str(record.decision)
        return None

    def close(self, *, run_status: str = "success") -> Path:
        return self._session.close(
            run_status=str(run_status),
            ended_ts_utc=datetime.now(timezone.utc),
        )
