from __future__ import annotations

from datetime import datetime, timezone
import importlib
import math
from pathlib import Path
import sys
import types

import pytest

from veriscope.core.artifacts import AuditV1, GateRecordV1
from veriscope.core.gate import GateEngine
from veriscope.core.transport import DeclTransport
from veriscope.core.window import FRWindow, WindowDecl
from veriscope.runners.hf.emit_artifacts import emit_hf_artifacts_v1


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch, *, with_dist: bool = False, other_rank_value: float = 0.0
) -> dict[str, int]:
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
    cuda_mod.is_available = lambda: False
    cuda_mod.device_count = lambda: 0
    torch_mod.cuda = cuda_mod
    torch_mod.backends = _DummyBackends()
    torch_mod.manual_seed = _noop

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data_mod)
    if not with_dist:
        return {}

    dist_mod = types.ModuleType("torch.distributed")

    class _FakeScalar:
        def __init__(self, value: float) -> None:
            self._value = float(value)

        def item(self) -> float:
            return float(self._value)

    class _FakeTensor:
        def __init__(self, value: float | list[float]) -> None:
            if isinstance(value, list):
                self._data = [float(v) for v in value]
            else:
                self._data = float(value)

        def item(self) -> float:
            if isinstance(self._data, list):
                raise TypeError("Use indexing for list tensors")
            return float(self._data)

        def __getitem__(self, index: int) -> _FakeScalar:
            if not isinstance(self._data, list):
                raise TypeError("Indexing only supported for list tensors")
            return _FakeScalar(self._data[index])

    class _ReduceOp:
        SUM = "sum"

    def _tensor(value: float | list[float], *_args: object, **_kwargs: object) -> _FakeTensor:
        return _FakeTensor(value)

    calls = {"barrier": 0}

    def _all_reduce(tensor: _FakeTensor, *_args: object, **_kwargs: object) -> None:
        if isinstance(tensor._data, list):
            for i in range(0, len(tensor._data), 2):
                tensor._data[i] = float(tensor._data[i]) + float(other_rank_value)
                if i + 1 < len(tensor._data):
                    tensor._data[i + 1] = float(tensor._data[i + 1]) + 1.0
            return
        tensor._data = float(tensor._data) * 2.0

    def _barrier(*_args: object, **_kwargs: object) -> None:
        calls["barrier"] += 1

    dist_mod.is_available = lambda: True
    dist_mod.is_initialized = lambda: True
    dist_mod.get_rank = lambda: 0
    dist_mod.get_world_size = lambda: 2
    dist_mod.all_reduce = _all_reduce
    dist_mod.barrier = _barrier
    dist_mod.get_backend = lambda: "gloo"
    dist_mod.ReduceOp = _ReduceOp
    torch_mod.distributed = dist_mod
    torch_mod.tensor = _tensor
    torch_mod.float32 = object()
    torch_mod.int64 = object()
    torch_mod.device = lambda *_args, **_kwargs: "cpu"
    torch_mod.cuda.is_available = lambda: False

    monkeypatch.setitem(sys.modules, "torch.distributed", dist_mod)
    return calls


def _install_fake_torch_ddp_init(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
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
    cuda_mod.is_available = lambda: False
    cuda_mod.device_count = lambda: 0
    cuda_mod.set_device = _noop
    torch_mod.cuda = cuda_mod
    torch_mod.backends = _DummyBackends()
    torch_mod.manual_seed = _noop
    torch_mod.device = lambda *_args, **_kwargs: "cpu"

    dist_mod = types.ModuleType("torch.distributed")
    calls: dict[str, object] = {"init": 0, "kwargs": None}

    def _init_process_group(*_args: object, **kwargs: object) -> None:
        calls["init"] = int(calls["init"]) + 1
        calls["kwargs"] = dict(kwargs)

    dist_mod.is_available = lambda: True
    dist_mod.is_initialized = lambda: False
    dist_mod.init_process_group = _init_process_group

    torch_mod.distributed = dist_mod

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data_mod)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist_mod)
    return calls


def test_ddp_non_chief_skips_artifact_emission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import veriscope.core.ddp as ddp

    monkeypatch.setattr(ddp, "_dist_module", lambda: None)
    for key in (
        "WORLD_SIZE",
        "RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    # New policy: env rank/world is only trusted under torchrun/elastic context.
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")

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
        strict_governance=False,
    )

    assert result.emitted is False
    assert not outdir.exists()


def test_hf_finalize_barrier_failure_blocks_full_emission(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch_ddp_init(monkeypatch)
    sys.modules.pop("veriscope.runners.hf.train_hf", None)
    train_hf = importlib.import_module("veriscope.runners.hf.train_hf")

    with pytest.raises(RuntimeError, match="DDP finalize barrier not performed"):
        train_hf._ensure_full_capsule_emission_allowed(
            world_size=2,
            barrier_status="skipped_error",
            governance_failure=None,
        )


def test_hf_run_started_failure_blocks_full_emission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch_ddp_init(monkeypatch)

    transformers_mod = types.ModuleType("transformers")

    class _FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        eos_token_id = 0
        mask_token_id = 0
        pad_token_id = None
        vocab_size = 8

        def __len__(self) -> int:
            return int(self.vocab_size)

    class _FakeModel:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(tie_word_embeddings=False, use_cache=False, vocab_size=8)

        def to(self, _device: object) -> "_FakeModel":
            return self

        def parameters(self) -> list[object]:
            return []

        def buffers(self) -> list[object]:
            return []

        def train(self) -> None:
            return None

    transformers_mod.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: _FakeTokenizer())
    transformers_mod.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: _FakeModel()
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    sys.modules.pop("veriscope.runners.hf.train_hf", None)
    train_hf = importlib.import_module("veriscope.runners.hf.train_hf")

    class _FakeDevice:
        def __init__(self, spec: object) -> None:
            self.type = str(spec)
            self.index = None

        def __str__(self) -> str:
            return self.type

    optimizer = types.SimpleNamespace(param_groups=[{"lr": 0.01}], step=lambda: None, zero_grad=lambda: None)
    monkeypatch.setattr(train_hf.torch, "device", lambda spec: _FakeDevice(spec))
    monkeypatch.setattr(
        train_hf.torch,
        "optim",
        types.SimpleNamespace(AdamW=lambda *_args, **_kwargs: optimizer),
        raising=False,
    )
    monkeypatch.setattr(train_hf, "_is_chief", lambda: True)
    monkeypatch.setattr(train_hf, "ddp_can_communicate", lambda: False)
    monkeypatch.setattr(train_hf, "_tokenize_dataset", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(train_hf, "HFMetricComputer", lambda *args, **kwargs: object())
    monkeypatch.setattr(train_hf, "_build_window_decl", lambda _cfg: object())
    monkeypatch.setattr(train_hf, "_build_gate_engine", lambda _cfg, _decl: object())

    def _fake_build_window_signature(
        _cfg: object, *, created_ts_utc: object, window_decl: object, gate_engine: object
    ) -> dict[str, object]:
        return {"schema_version": 1}

    monkeypatch.setattr(train_hf, "_build_window_signature", _fake_build_window_signature)
    monkeypatch.setattr(train_hf, "build_code_identity", lambda **_kwargs: {"package_version": "test"})
    monkeypatch.setattr(
        train_hf,
        "build_distributed_context",
        lambda: {"distributed_mode": "single_process", "world_size_observed": 1},
    )
    monkeypatch.setattr(
        train_hf, "append_run_started", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    emitted = {"called": False}

    def _emit_fail(*_args: object, **_kwargs: object) -> None:
        emitted["called"] = True
        raise AssertionError("emit_hf_artifacts_v1 should not be reached after run_started failure")

    monkeypatch.setattr(train_hf, "emit_hf_artifacts_v1", _emit_fail)

    cfg = train_hf.HFRunConfig(
        model="fake-model",
        dataset_name="fake-dataset",
        dataset_config=None,
        dataset_path=None,
        dataset_split="train",
        dataset_text_column="text",
        outdir=tmp_path / "hf_run_started_failure",
        run_id="run-started-fail",
        force=False,
        max_steps=0,
        batch_size=1,
        lr=0.01,
        seed=123,
        cadence=1,
        block_size=1,
        device="cpu",
        grad_clip=0.0,
        gate_preset="tuned_v0",
        gate_window=1,
        gate_epsilon=0.1,
        gate_min_evidence=1,
        gate_gain_thresh=0.0,
        gate_policy="persistence",
        gate_persistence_k=1,
        rp_dim=1,
        lr_spike_at=-1,
        lr_spike_len=0,
        lr_spike_mult=1.0,
        lr_spike_verify=False,
        data_corrupt_at=-1,
        data_corrupt_len=0,
        data_corrupt_frac=0.0,
        data_corrupt_mode="replace",
        data_corrupt_target="input_ids",
        data_corrupt_mask_id=0,
    )

    with pytest.raises(RuntimeError, match="run_started_v1: boom"):
        train_hf._run_body(cfg, argv=["train_hf.py"], rank=0, world_size=1, seed_rank=123)

    assert emitted["called"] is False


def test_ddp_gate_returns_skip_with_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    import veriscope.core.ddp as ddp

    monkeypatch.setattr(ddp, "_dist_module", lambda: None)
    _install_fake_torch(monkeypatch)
    sys.modules.pop("veriscope.runners.hf.train_hf", None)
    train_hf = importlib.import_module("veriscope.runners.hf.train_hf")
    _gate_from_history = train_hf._gate_from_history

    for key in (
        "WORLD_SIZE",
        "RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    # New policy: env rank/world is only trusted under torchrun/elastic context.
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")

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


def test_ddp_gate_skips_when_env_active_but_no_comms(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch_ddp_init(monkeypatch)
    sys.modules.pop("veriscope.runners.hf.train_hf", None)
    train_hf = importlib.import_module("veriscope.runners.hf.train_hf")
    monkeypatch.setattr(train_hf, "ddp_is_active", lambda: True)
    monkeypatch.setattr(train_hf, "ddp_can_communicate", lambda: False)
    _gate_from_history = train_hf._gate_from_history

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
    assert record.audit.ddp_barrier_status == "skipped_inactive"
    assert record.audit.per_metric_tv == {}


def test_hf_metric_snapshot_requires_full_past_and_recent_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch_ddp_init(monkeypatch)
    sys.modules.pop("veriscope.runners.hf.train_hf", None)
    train_hf = importlib.import_module("veriscope.runners.hf.train_hf")

    history = [{"iter": 0}, {"iter": 1}, {"iter": 2}]
    past, recent = train_hf._metric_snapshot(history, gate_window=2)
    assert past == []
    assert recent == history

    history_full = [{"iter": 0}, {"iter": 1}, {"iter": 2}, {"iter": 3}]
    past_full, recent_full = train_hf._metric_snapshot(history_full, gate_window=2)
    assert [row["iter"] for row in past_full] == [0, 1]
    assert [row["iter"] for row in recent_full] == [2, 3]


def test_hf_gate_raises_on_non_coercible_metric_value(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch_ddp_init(monkeypatch)
    sys.modules.pop("veriscope.runners.hf.train_hf", None)
    train_hf = importlib.import_module("veriscope.runners.hf.train_hf")
    _gate_from_history = train_hf._gate_from_history

    for key in (
        "WORLD_SIZE",
        "RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
    ):
        monkeypatch.delenv(key, raising=False)

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
        {"iter": 1, "var_out_k": "not-a-number", "eff_dim": 0.22},
        {"iter": 2, "var_out_k": 0.11, "eff_dim": 0.21},
        {"iter": 3, "var_out_k": 0.13, "eff_dim": 0.23},
    ]
    with pytest.raises(ValueError, match="Non-coercible metric value"):
        _gate_from_history(
            gate_engine,
            window_decl,
            metric_history,
            gate_window=2,
            iter_num=3,
            gate_policy="persistence",
            gate_min_evidence=1,
        )


def test_hf_parse_args_rejects_skip_only_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_torch_ddp_init(monkeypatch)
    sys.modules.pop("veriscope.runners.hf.train_hf", None)
    train_hf = importlib.import_module("veriscope.runners.hf.train_hf")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_hf.py",
            "--outdir",
            str(tmp_path / "hf_skip_only"),
            "--max_steps",
            "100",
            "--cadence",
            "10",
            "--gate_window",
            "6",
        ],
    )

    with pytest.raises(ValueError) as excinfo:
        train_hf._parse_args()

    msg = str(excinfo.value)
    assert "Run will produce only skip decisions." in msg
    assert "Need at least 12 snapshots but settings yield only 10." in msg
    assert "increase max_steps to at least 120" in msg
    assert "reduce gate_window to at most 5" in msg
    assert "or reduce cadence." in msg


def test_hf_gate_fail_dominates_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch_ddp_init(monkeypatch)
    sys.modules.pop("veriscope.runners.hf.train_hf", None)
    train_hf = importlib.import_module("veriscope.runners.hf.train_hf")
    _gate_from_history = train_hf._gate_from_history

    for key in (
        "WORLD_SIZE",
        "RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
    ):
        monkeypatch.delenv(key, raising=False)

    class _FakeGateEngine:
        def check(self, **_kwargs: object) -> object:
            return types.SimpleNamespace(
                ok=False,
                warn=True,
                audit={
                    "evaluated": True,
                    "reason": "change_persistence_fail",
                    "policy": "persistence",
                    "worst_DW": 0.24,
                    "eps_eff": 0.08,
                    "per_metric_tv": {"var_out_k": 0.24},
                },
            )

    window_decl = WindowDecl(
        epsilon=0.12,
        metrics=["var_out_k", "eff_dim"],
        weights={"var_out_k": 0.5, "eff_dim": 0.5},
        bins=16,
    )
    transport = DeclTransport(window_decl)
    window_decl.attach_transport(transport)

    metric_history = [
        {"iter": 0, "var_out_k": 0.1, "eff_dim": 0.2},
        {"iter": 1, "var_out_k": 0.12, "eff_dim": 0.22},
        {"iter": 2, "var_out_k": 0.11, "eff_dim": 0.21},
        {"iter": 3, "var_out_k": 0.13, "eff_dim": 0.23},
    ]
    record = _gate_from_history(
        _FakeGateEngine(),
        window_decl,
        metric_history,
        gate_window=2,
        iter_num=3,
        gate_policy="persistence",
        gate_min_evidence=1,
    )

    assert record.decision == "fail"
    assert record.ok is False
    assert record.warn is False


def test_hf_ddp_init_shim_calls_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_torch_ddp_init(monkeypatch)
    initialized = {"value": False}
    dist_mod = sys.modules["torch.distributed"]

    def _is_initialized() -> bool:
        return bool(initialized["value"])

    def _init_process_group(*_args: object, **kwargs: object) -> None:
        calls["init"] = int(calls["init"]) + 1
        calls["kwargs"] = dict(kwargs)
        initialized["value"] = True

    dist_mod.is_initialized = _is_initialized
    dist_mod.init_process_group = _init_process_group
    for key in ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")

    sys.modules.pop("veriscope.runners.hf.train_hf", None)
    train_hf = importlib.import_module("veriscope.runners.hf.train_hf")
    initialized_here = train_hf._maybe_init_ddp()

    assert initialized_here is True
    assert calls["init"] == 1
    assert calls["kwargs"]["backend"] == "gloo"
    assert calls["kwargs"]["init_method"] == "env://"
    assert "timeout" in calls["kwargs"]


def test_hf_ddp_init_shim_skips_when_already_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_torch_ddp_init(monkeypatch)
    dist_mod = sys.modules["torch.distributed"]
    dist_mod.is_initialized = lambda: True
    dist_mod.init_process_group = lambda *_args, **_kwargs: calls.update({"init": 999})
    for key in ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("LOCAL_RANK", "0")
    sys.modules.pop("veriscope.runners.hf.train_hf", None)
    train_hf = importlib.import_module("veriscope.runners.hf.train_hf")

    initialized_here = train_hf._maybe_init_ddp()

    assert initialized_here is False
    assert calls["init"] == 0
    monkeypatch.delenv("VERISCOPE_DDP_CLEANUP", raising=False)
    assert train_hf._should_cleanup_ddp(initialized_here) is False
    assert train_hf._should_cleanup_ddp(True) is True


def test_ddp_gate_aggregates_when_dist_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERISCOPE_DDP_STRICT_GATE_SYNC", raising=False)
    calls = _install_fake_torch(monkeypatch, with_dist=True, other_rank_value=0.5)
    assert "torch" in sys.modules
    sys.modules.pop("veriscope.runners.hf.train_hf", None)
    train_hf = importlib.import_module("veriscope.runners.hf.train_hf")
    _gate_from_history = train_hf._gate_from_history
    _ddp_aggregate_slice = train_hf._ddp_aggregate_slice

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

    past_slice = metric_history[:2]
    agg_metrics = {"var_out_k": "mean", "eff_dim": "mean"}
    aggregated_past = _ddp_aggregate_slice(past_slice, agg_metrics, ["var_out_k", "eff_dim"])
    assert aggregated_past is not None
    assert aggregated_past[0]["var_out_k"] != metric_history[0]["var_out_k"]
    assert record.audit.reason != "ddp_unsupported"
    assert record.audit.ddp_agg == "mean"
    assert record.audit.ddp_barrier_status == "not_requested"
    assert calls["barrier"] == 0


def test_ddp_gate_strict_sync_uses_barrier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERISCOPE_DDP_STRICT_GATE_SYNC", "1")
    monkeypatch.delenv("VERISCOPE_DDP_STRICT_BARRIER", raising=False)
    calls = _install_fake_torch(monkeypatch, with_dist=True, other_rank_value=0.5)
    sys.modules.pop("veriscope.runners.hf.train_hf", None)
    train_hf = importlib.import_module("veriscope.runners.hf.train_hf")
    _gate_from_history = train_hf._gate_from_history

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

    assert record.audit.reason != "ddp_unsupported"
    assert record.audit.ddp_barrier_status == "performed"
    assert calls["barrier"] >= 1


def test_ddp_masked_mean_ignores_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, with_dist=True, other_rank_value=0.5)
    from veriscope.core.ddp import ddp_reduce_mean_scalars_masked

    result = ddp_reduce_mean_scalars_masked([float("nan")])
    assert result is not None
    assert math.isfinite(result[0])
    assert result[0] == pytest.approx(0.5)


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
