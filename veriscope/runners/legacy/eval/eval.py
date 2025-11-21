# veriscope/runners/legacy/eval/eval.py
from __future__ import annotations



from veriscope.runners.legacy_cli_refactor import (  # type: ignore
    compute_events,
    mark_events_epochwise,
    summarize_detection,
    summarize_runlevel_fp,
    bootstrap_stratified,
    rp_adequacy_flags,
    assert_overlay_consistency,
    recompute_gate_series_under_decl,
    compute_invariants_and_provenance,
    make_plots,
)

__all__ = [
    "compute_events",
    "mark_events_epochwise",
    "summarize_detection",
    "summarize_runlevel_fp",
    "bootstrap_stratified",
    "rp_adequacy_flags",
    "assert_overlay_consistency",
    "recompute_gate_series_under_decl",
    "compute_invariants_and_provenance",
    "make_plots",
]