# veriscope/core/lifecycle.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from signal import Signals
from typing import Optional

from veriscope.core.artifacts import RunStatus

LifecycleState = str


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _signal_name(signum: Optional[int]) -> Optional[str]:
    if signum is None:
        return None
    try:
        return Signals(signum).name
    except Exception:
        return f"SIG{signum}"


@dataclass
class RunLifecycle:
    run_id: str
    run_kind: str
    started_ts_utc: datetime = field(default_factory=_now_utc)
    state: LifecycleState = "STARTED"
    runner_exit_code: Optional[int] = None
    runner_signal: Optional[str] = None
    wrapper_exit_code: Optional[int] = None
    run_status: RunStatus = "success"
    ended_ts_utc: Optional[datetime] = None
    error_message: Optional[str] = None

    def mark_running(self) -> None:
        self.state = "RUNNING"

    def mark_interrupted(self, signum: int) -> None:
        self.state = "INTERRUPTED"
        self.runner_signal = _signal_name(signum)

    def mark_runner_exit(self, returncode: Optional[int]) -> None:
        self.runner_exit_code = returncode
        if returncode is None:
            return
        if returncode < 0:
            self.runner_signal = _signal_name(-returncode)

    def mark_internal_failure(self, message: str) -> None:
        self.state = "FAILED_INTERNAL"
        self.error_message = message

    def finalize(self, *, run_status: RunStatus, wrapper_exit_code: int) -> None:
        self.run_status = run_status
        self.wrapper_exit_code = wrapper_exit_code
        self.ended_ts_utc = _now_utc()
        self.state = "FINALIZED"


def map_status_and_exit(
    *,
    runner_exit_code: Optional[int],
    runner_signal: Optional[str],
    internal_error: bool,
) -> tuple[RunStatus, int]:
    if internal_error:
        return "veriscope_failure", 3
    if runner_signal is not None:
        return "user_code_failure", 2
    if runner_exit_code is None:
        return "veriscope_failure", 3
    if runner_exit_code != 0:
        return "user_code_failure", 2
    return "success", 0
