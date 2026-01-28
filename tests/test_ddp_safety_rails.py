from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import types

import pytest

from veriscope.core.artifacts import AuditV1, GateRecordV1
from veriscope.core.gate import GateEngine
from veriscope.core.transport import DeclTransport
from veriscope.core.window import FRWindow, WindowDecl
from veriscope.runners.hf.emit_artifacts import emit_hf_artifacts_v1


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    if "torch" in sys.modules:
        return
    torch_mod = types.ModuleType("torch")
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")
    cuda_mod = types.SimpleNamespace()
    torch_mod.__path__ = []
    utils_mod.__path__ = []
    data_mod.__path__ = []

    def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    class _DummyCudnn:  # pragma: no cover - only for import safety
        deterministic = False
        benchmark = False

    class _DummyBackends:  # pragma: no cover - only for import safety
        cudnn = _DummyCudnn()

    class _DummyDataLoader:  # pragma: no cover - only for import safety
        pass

    data_mod.DataLoader = _DummyDataLoader
    utils_mod.data = data_mod
    torch_mod.utils = utils_mod
    cuda_mod.manual_seed_all = _noop
    torch_mod.cuda = cuda_mod
    torch_mod.backends = _DummyBackends()
    torch_mod.manual_seed = _noop

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data_mod)


def test_ddp_non_chief_skips_artifact_emission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import veriscope.core.ddp as ddp

    monkeypatch.setattr(ddp, "_dist_module", lambda: None)
    for key in ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")

    outdir = tmp_path / "hf_emit_non_chief"
    window_signature = {"schema_version": 1, "transport": {"name": "hf_hidden_state_v1", "cadence": "every_1_steps"}}
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    gate_records = [GateRecordV1(iter=0, decision="skip", audit=AuditV1(evaluated=False, per_metric_tv={}))]

    result = emit_hf_artifacts_v1(
        outdir=outdir,
        run_id="run-ddp",
        started_ts_utc=now,
        ended_ts_utc=now,
        gate_preset="tuned_v0",
        window_signature=window_signature,
        gate_records=gate_records,
        run_status="success",
        runner_exit_code=0,
        runner_signal=None,
    )

    assert result.emitted is False
    assert not outdir.exists()


def test_ddp_gate_returns_skip_with_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    import veriscope.core.ddp as ddp

    monkeypatch.setattr(ddp, "_dist_module", lambda: None)
    _install_fake_torch(monkeypatch)
    from veriscope.runners.hf.train_hf import _gate_from_history

    for key in ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")

    window_decl = WindowDecl(
        epsilon=0.12,
        metrics=["var_out_k", "eff_dim"],
        weights={"var_out_k": 0.5, "eff_dim": 0.5},
        bins=16,
    )
    transport = DeclTransport(window_decl)
    window_decl.attach_transport(transport)
    fr_win = FRWindow(decl=window_decl, transport=transport, tests=())
    gate_engine = GateEngine(
        frwin=fr_win,
        gain_thresh=0.0,
        eps_stat_alpha=0.05,
        eps_stat_max_frac=0.25,
        eps_sens=0.04,
        min_evidence=1,
        policy="persistence",
        persistence_k=2,
        min_metrics_exceeding=1,
    )

    metric_history = [
        {"iter": 0, "var_out_k": 0.1, "eff_dim": 0.2},
        {"iter": 1, "var_out_k": 0.12, "eff_dim": 0.22},
        {"iter": 2, "var_out_k": 0.11, "eff_dim": 0.21},
        {"iter": 3, "var_out_k": 0.13, "eff_dim": 0.23},
    ]

    record = _gate_from_history(
        gate_engine,
        window_decl,
        metric_history,
        gate_window=2,
        iter_num=3,
        gate_policy="persistence",
        gate_min_evidence=1,
    )

    assert record.decision == "skip"
    assert record.audit.evaluated is False
    assert record.audit.reason == "ddp_unsupported"
    assert record.audit.policy == "persistence"
    assert record.audit.evidence_total == 4
    assert record.audit.min_evidence == 1
    assert record.audit.per_metric_tv == {}


def test_ddp_rank_uses_initialized_dist(monkeypatch: pytest.MonkeyPatch) -> None:
    torch_mod = types.ModuleType("torch")
    dist_mod = types.ModuleType("torch.distributed")
    torch_mod.__path__ = []

    dist_mod.is_available = lambda: True
    dist_mod.is_initialized = lambda: True
    dist_mod.get_rank = lambda: 1
    dist_mod.get_world_size = lambda: 4

    torch_mod.distributed = dist_mod

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist_mod)

    from veriscope.core.ddp import ddp_is_active, ddp_is_chief, ddp_rank, ddp_world_size

    assert ddp_is_active() is True
    assert ddp_is_chief() is False
    assert ddp_rank() == 1
    assert ddp_world_size() == 4


def test_ddp_is_active_false_for_single_rank_dist(monkeypatch: pytest.MonkeyPatch) -> None:
    torch_mod = types.ModuleType("torch")
    dist_mod = types.ModuleType("torch.distributed")
    torch_mod.__path__ = []

    dist_mod.is_available = lambda: True
    dist_mod.is_initialized = lambda: True
    dist_mod.get_rank = lambda: 0
    dist_mod.get_world_size = lambda: 1

    torch_mod.distributed = dist_mod

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist_mod)

    from veriscope.core.ddp import ddp_is_active

    assert ddp_is_active() is False


def test_ddp_barrier_strictness(monkeypatch: pytest.MonkeyPatch) -> None:
    torch_mod = types.ModuleType("torch")
    dist_mod = types.ModuleType("torch.distributed")
    torch_mod.__path__ = []

    calls = {"barrier": 0}

    def _barrier(*_args: object, **_kwargs: object) -> None:
        if "timeout" in _kwargs:
            raise TypeError("timeout unsupported")
        calls["barrier"] += 1

    dist_mod.is_available = lambda: True
    dist_mod.is_initialized = lambda: True
    dist_mod.get_rank = lambda: 0
    dist_mod.get_world_size = lambda: 2
    dist_mod.barrier = _barrier

    torch_mod.distributed = dist_mod

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist_mod)

    from veriscope.core.ddp import ddp_barrier

    monkeypatch.delenv("VERISCOPE_DDP_STRICT_BARRIER", raising=False)
    status = ddp_barrier()
    assert status == "skipped_no_timeout"
    assert calls["barrier"] == 0

    monkeypatch.setenv("VERISCOPE_DDP_STRICT_BARRIER", "1")
    status = ddp_barrier()
    assert status == "performed"
    assert calls["barrier"] == 1
