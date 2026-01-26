#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import veriscope
from veriscope.core.artifacts import CountsV1, ProfileV1, ResultsSummaryV1, ResultsV1, WindowSignatureRefV1
from veriscope.core.jsonutil import atomic_write_json, canonical_json_sha256


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _emit_minimal_artifacts(outdir: Path, run_id: str, gate_preset: str) -> None:
    window_signature = {
        "schema_version": 1,
        "created_ts_utc": _iso_utc_now(),
        "description": "Fake HF runner (test harness)",
        "code_identity": {"package_version": veriscope.__version__},
        "transport": {"name": "hf_hidden_state_v1", "cadence": "every_1_steps"},
        "evidence": {"metrics": ["var_out_k", "eff_dim"]},
        "gates": {"preset": gate_preset},
        "gate_controls": {"gate_window": 16, "min_evidence": 16},
        "metric_interval": 16,
    }
    ws_hash = canonical_json_sha256(window_signature)
    ws_ref = WindowSignatureRefV1(hash=ws_hash, path="window_signature.json")

    atomic_write_json(outdir / "window_signature.json", window_signature)

    profile = ProfileV1(gate_preset=gate_preset, overrides={})
    counts = CountsV1(evaluated=0, skip=0, pass_=0, warn=0, fail=0)
    summary = ResultsSummaryV1(
        run_id=run_id,
        window_signature_ref=ws_ref,
        profile=profile,
        run_status="success",
        runner_exit_code=0,
        runner_signal=None,
        started_ts_utc=datetime.now(timezone.utc),
        ended_ts_utc=datetime.now(timezone.utc),
        counts=counts,
        final_decision="skip",
    )
    atomic_write_json(outdir / "results_summary.json", summary.model_dump(mode="json", by_alias=True))

    results = ResultsV1(
        run_id=run_id,
        window_signature_ref=ws_ref,
        profile=profile,
        run_status="success",
        runner_exit_code=0,
        runner_signal=None,
        started_ts_utc=datetime.now(timezone.utc),
        ended_ts_utc=datetime.now(timezone.utc),
        gates=[],
        metrics=[],
    )
    atomic_write_json(outdir / "results.json", results.model_dump(mode="json", by_alias=True))


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--gate_preset", type=str, default="tuned_v0")
    args, _ = parser.parse_known_args()

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or "fake-hf-run"
    gate_preset = args.gate_preset or "tuned_v0"
    _emit_minimal_artifacts(outdir, run_id, gate_preset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
