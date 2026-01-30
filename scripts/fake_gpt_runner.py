#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from veriscope.core.governance import append_run_started, build_code_identity
from veriscope.core.artifacts import (
    CountsV1,
    Decision,
    ProfileV1,
    ResultsSummaryV1,
    ResultsV1,
    RunStatus,
    WindowSignatureRefV1,
)
from veriscope.core.jsonutil import atomic_write_json, window_signature_sha256

SUCCESS_STATUS: RunStatus = "success"
USER_CODE_FAILURE: RunStatus = "user_code_failure"
SKIP_DECISION: Decision = "skip"


def _read_json_obj(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise TypeError(f"{path.name} must be a JSON object")
    return obj


def _emit_minimal_artifacts(out_dir: Path, *, run_status: RunStatus, runner_exit_code: int) -> None:
    ws_path = out_dir / "window_signature.json"
    ws_obj = _read_json_obj(ws_path) if ws_path.exists() else {"schema_version": 1, "placeholder": True}
    ws_hash = window_signature_sha256(ws_obj)
    ws_ref = WindowSignatureRefV1(hash=ws_hash, path="window_signature.json")

    run_config_path = out_dir / "run_config_resolved.json"
    run_id = "fake-run-id"
    if run_config_path.exists():
        cfg = _read_json_obj(run_config_path)
        run_id = str(cfg.get("run", {}).get("run_id") or run_id)

    try:
        append_run_started(
            out_dir,
            run_id=run_id,
            outdir_path=out_dir,
            argv=sys.argv,
            code_identity=build_code_identity(),
            window_signature_ref={"hash": ws_hash, "path": "window_signature.json"},
            entrypoint={"kind": "runner", "name": "scripts.fake_gpt_runner"},
        )
    except Exception as exc:
        print(f"fake-runner: failed to append governance run_started: {exc}", file=sys.stderr, flush=True)

    profile = ProfileV1(gate_preset="fake_runner", overrides={})
    counts = CountsV1(evaluated=0, skip=0, pass_=0, warn=0, fail=0)

    summary = ResultsSummaryV1(
        run_id=run_id,
        window_signature_ref=ws_ref,
        profile=profile,
        run_status=run_status,
        runner_exit_code=runner_exit_code,
        runner_signal=None,
        started_ts_utc=datetime.now(timezone.utc),
        ended_ts_utc=datetime.now(timezone.utc),
        counts=counts,
        final_decision=SKIP_DECISION,
    )
    summary_payload = summary.model_dump(mode="json", by_alias=True, exclude_none=True)

    results = ResultsV1(
        run_id=run_id,
        window_signature_ref=ws_ref,
        profile=profile,
        run_status=run_status,
        runner_exit_code=runner_exit_code,
        runner_signal=None,
        started_ts_utc=datetime.now(timezone.utc),
        ended_ts_utc=datetime.now(timezone.utc),
        gates=[],
        metrics=[],
    )
    atomic_write_json(out_dir / "results.json", results.model_dump(mode="json", by_alias=True, exclude_none=True))
    atomic_write_json(out_dir / "results_summary.json", summary_payload)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sleep-seconds", type=float, default=3.0)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--ignore-signals", action="store_true")
    parser.add_argument("--emit-artifacts", action="store_true")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--out_json", type=str, default="")
    args, _ = parser.parse_known_args()

    def _handle(signum: int, _frame: Optional[object]) -> None:
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = f"SIG{signum}"
        print(f"fake-runner: received {signame}", file=sys.stderr, flush=True)
        if not args.ignore_signals:
            sys.exit(args.exit_code)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    print("fake-runner: stdout ready", flush=True)
    print("fake-runner: stderr ready", file=sys.stderr, flush=True)

    if args.emit_artifacts and args.out_dir:
        out_dir = Path(args.out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        status = SUCCESS_STATUS if args.exit_code == 0 else USER_CODE_FAILURE
        _emit_minimal_artifacts(out_dir, run_status=status, runner_exit_code=args.exit_code)

    time.sleep(args.sleep_seconds)
    return int(args.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
