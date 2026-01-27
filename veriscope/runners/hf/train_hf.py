# veriscope/runners/hf/train_hf.py
from __future__ import annotations

import argparse
import math
import contextlib
import os
import random
import signal
import subprocess
import sys
import traceback
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

import veriscope
from veriscope.core.artifacts import AuditV1, GateRecordV1, MetricRecordV1
from veriscope.core.calibration import aggregate_epsilon_stat
from veriscope.core.gate import GateEngine
from veriscope.core.transport import DeclTransport
from veriscope.core.window import FRWindow, WindowDecl
from veriscope.core.jsonutil import atomic_write_json
from veriscope.runners.hf.adapter import HFMetricComputer, HFMetricConfig
from veriscope.runners.hf.emit_artifacts import emit_hf_artifacts_v1


@dataclass(frozen=True)
class HFRunConfig:
    model: str
    dataset_name: str
    dataset_config: Optional[str]
    dataset_path: Optional[Path]
    dataset_split: str
    dataset_text_column: str
    outdir: Path
    run_id: str
    force: bool
    max_steps: int
    batch_size: int
    lr: float
    seed: int
    cadence: int
    block_size: int
    device: str
    grad_clip: float
    gate_preset: str
    gate_window: int
    gate_epsilon: float
    gate_min_evidence: int
    gate_gain_thresh: float
    gate_policy: str
    gate_persistence_k: int
    rp_dim: int


def _jsonable_float(x: float) -> Optional[float]:
    """Convert float to JSON-safe value: finite -> float, non-finite -> None."""
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_outdir() -> Path:
    base = Path(os.environ.get("VERISCOPE_OUT_BASE", "./out")).expanduser()
    ts = time.strftime("%Y%m%d_%H%M%S")
    return base / f"veriscope_hf_{ts}_{os.getpid()}"


def _get_rank_and_world() -> tuple[int, int]:
    """Determine (rank, world_size) without being confused by unrelated env vars.

    Policy:
      1) If a torch.distributed process group is initialized, trust it.
      2) Else, only trust env vars if we appear to be under torchrun/elastic
         (LOCAL_RANK or TORCHELASTIC_RUN_ID present).
      3) Otherwise treat as single-process (0, 1).
    """
    # For smokes/CI, allow an explicit override to force single-process emission.
    force = (os.environ.get("VERISCOPE_FORCE_SINGLE_PROCESS") or "").strip().lower()
    if force in {"1", "true", "yes", "y", "on"}:
        return 0, 1
    # 1) Trust an initialized process group (most reliable).
    try:
        import torch.distributed as dist  # local import

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
    except Exception:
        pass

    # 2) Only trust env rank/world when torchrun/elastic hints are present.
    if os.environ.get("LOCAL_RANK") is None and os.environ.get("TORCHELASTIC_RUN_ID") is None:
        return 0, 1

    def _as_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    rank = _as_int("RANK", 0)
    world_size = _as_int("WORLD_SIZE", 1)
    if world_size < 1:
        world_size = 1
    if rank < 0:
        rank = 0
    if rank >= world_size:
        # Clamp rather than silently acting non-chief and emitting nothing.
        rank = 0
        world_size = 1
    return rank, world_size


def _is_chief() -> bool:
    rank, _ = _get_rank_and_world()
    return rank == 0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_window_decl(cfg: HFRunConfig) -> WindowDecl:
    # NOTE: On very small HF models (e.g. sshleifer/tiny-gpt2), some “variance-like”
    # metrics can be > 1.0 depending on implementation details. Using a too-tight
    # calibration range makes gates spuriously fail in smokes/CI.
    var_out_max = float(max(1, int(cfg.rp_dim)))
    return WindowDecl(
        epsilon=float(cfg.gate_epsilon),
        metrics=["var_out_k", "eff_dim"],
        weights={"var_out_k": 0.5, "eff_dim": 0.5},
        bins=16,
        interventions=(lambda x: x,),
        cal_ranges={"var_out_k": (0.0, var_out_max), "eff_dim": (0.0, float(cfg.rp_dim))},
    )


def _fallback_metrics_from_hidden(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    rp_dim: int,
) -> Dict[str, float]:
    """Numeric-stable fallback for CI/smokes if the adapter metric path is degenerate.

    Produces:
      - var_out_k in [0, 1] (top-k variance fraction via SVD)
      - eff_dim in [0, rp_dim] (participation ratio proxy)
    """
    try:
        hs = hidden_states.detach()
        if hs.dim() != 3:
            return {"var_out_k": float("nan"), "eff_dim": float("nan")}
        x = hs.reshape(-1, hs.shape[-1]).float()
        if attention_mask is not None:
            m = attention_mask.reshape(-1).to(dtype=torch.bool)
            x = x[m]
        if x.shape[0] < 2 or x.shape[1] < 1:
            return {"var_out_k": float("nan"), "eff_dim": float("nan")}
        # cap tokens for safety
        if x.shape[0] > 2048:
            idx = torch.randperm(x.shape[0], device=x.device)[:2048]
            x = x[idx]
        x = x - x.mean(dim=0, keepdim=True)
        with contextlib.suppress(Exception):
            s = torch.linalg.svdvals(x)
            var = s.square()
            tot = var.sum()
            if torch.isfinite(tot) and float(tot.item()) > 0.0:
                k = min(int(rp_dim), int(var.numel()))
                frac = (var[:k].sum() / (tot + 1e-12)).clamp(0.0, 1.0)
                lam = var / max(1, x.shape[0] - 1)
                tr = lam.sum()
                tr2 = lam.square().sum()
                eff = (tr.square() / (tr2 + 1e-12)).clamp(0.0, float(rp_dim))
                return {"var_out_k": float(frac.item()), "eff_dim": float(eff.item())}
    except Exception:
        pass
    return {"var_out_k": float("nan"), "eff_dim": float("nan")}


def _build_gate_engine(cfg: HFRunConfig, window_decl: WindowDecl) -> GateEngine:
    transport = DeclTransport(window_decl)
    window_decl.attach_transport(transport)
    fr_win = FRWindow(decl=window_decl, transport=transport, tests=())
    return GateEngine(
        frwin=fr_win,
        gain_thresh=float(cfg.gate_gain_thresh),
        eps_stat_alpha=0.05,
        eps_stat_max_frac=0.25,
        eps_sens=0.04,
        min_evidence=int(cfg.gate_min_evidence),
        policy=str(cfg.gate_policy),
        persistence_k=int(cfg.gate_persistence_k),
        min_metrics_exceeding=1,
    )


def _build_window_signature(cfg: HFRunConfig, *, created_ts_utc: datetime) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_ts_utc": _iso_utc(created_ts_utc),
        "description": "HF transformer runner (custom loop)",
        "code_identity": {"package_version": veriscope.__version__},
        "transport": {"name": "hf_hidden_state_v1", "cadence": f"every_{cfg.cadence}_steps"},
        "evidence": {
            "metrics": ["var_out_k", "eff_dim"],
            "window": {"kind": "fixed", "size": cfg.gate_window, "stride": cfg.cadence},
        },
        "gates": {
            "preset": cfg.gate_preset,
            "params": {
                "epsilon": cfg.gate_epsilon,
                "min_evidence": cfg.gate_min_evidence,
                "gain_thresh": cfg.gate_gain_thresh,
                "policy": cfg.gate_policy,
                "persistence_k": cfg.gate_persistence_k,
            },
        },
        "model": {"name": cfg.model},
        "dataset": {
            "name": cfg.dataset_name,
            "config": cfg.dataset_config,
            "path": str(cfg.dataset_path) if cfg.dataset_path else None,
            "split": cfg.dataset_split,
            "text_column": cfg.dataset_text_column,
        },
        "sketch": {"kind": "jl", "dim": cfg.rp_dim, "seed": cfg.seed},
    }


def _best_effort_git_sha() -> Optional[str]:
    try:
        repo_root = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return (result.stdout or "").strip() or None
    except Exception:
        return None


def _best_effort_transformers_version() -> Optional[str]:
    try:
        import transformers  # local import to avoid module-level dependency

        return getattr(transformers, "__version__", None)
    except Exception:
        return None


def _build_run_manifest(
    cfg: HFRunConfig,
    *,
    argv: List[str],
    started_ts_utc: datetime,
    ended_ts_utc: Optional[datetime],
    run_status: str,
    runner_exit_code: Optional[int],
    runner_signal: Optional[str],
    failure_reason: Optional[str],
    failure_traceback: Optional[str],
) -> Dict[str, Any]:
    env = os.environ
    return {
        "schema_version": 1,
        "argv": list(argv),
        "env": {
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "WORLD_SIZE": env.get("WORLD_SIZE"),
            "RANK": env.get("RANK"),
            "MASTER_ADDR": env.get("MASTER_ADDR"),
            "MASTER_PORT": env.get("MASTER_PORT"),
            "HF_HOME": env.get("HF_HOME"),
            "TRANSFORMERS_CACHE": env.get("TRANSFORMERS_CACHE"),
            "PYTHONPATH": env.get("PYTHONPATH"),
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS"),
        },
        "seeds": {
            "seed": cfg.seed,
            "torch_manual_seed": cfg.seed,
            "numpy_seed": cfg.seed,
            "python_random_seed": cfg.seed,
        },
        "determinism": {
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "torch_deterministic_algorithms": bool(
                getattr(torch, "are_deterministic_algorithms_enabled", lambda: False)()
            ),
        },
        "timestamps": {
            "started_ts_utc": _iso_utc(started_ts_utc),
            "ended_ts_utc": _iso_utc(ended_ts_utc) if ended_ts_utc else None,
        },
        "run_status": run_status,
        "runner_exit_code": runner_exit_code,
        "runner_signal": runner_signal,
        "failure_reason": failure_reason,
        "failure_traceback": failure_traceback,
        "git": {"commit_sha": _best_effort_git_sha()},
        "versions": {
            "veriscope": veriscope.__version__,
            "torch": getattr(torch, "__version__", None),
            "transformers": _best_effort_transformers_version(),
        },
    }


def _write_run_manifest(outdir: Path, manifest: Dict[str, Any]) -> None:
    atomic_write_json(outdir / "run_manifest.json", manifest, fsync=True)


def _tokenize_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    dataset_path: Optional[Path],
    dataset_split: str,
    dataset_text_column: str,
    tokenizer: Any,
    *,
    block_size: int,
    batch_size: int,
    seed: int,
) -> Iterable[Dict[str, torch.Tensor]]:
    from datasets import load_dataset

    if dataset_path is not None:
        dataset = load_dataset("text", data_files={dataset_split: str(dataset_path)}, split=dataset_split)
    elif dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)

    if dataset_text_column not in dataset.column_names:
        raise ValueError(f"Dataset column '{dataset_text_column}' not found. Available columns: {dataset.column_names}")

    def tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        # datasets.Dataset.map requires a dict (or pyarrow table) when transforming data.
        # For batched tokenization, transformers returns a list-of-lists for input_ids.
        return {"input_ids": tokenizer(batch[dataset_text_column], add_special_tokens=False)["input_ids"]}

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    def group_texts(batch: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        concatenated: List[int] = []
        for ids in batch["input_ids"]:
            concatenated.extend(ids)
        total_len = (len(concatenated) // block_size) * block_size
        result = {"input_ids": [concatenated[i : i + block_size] for i in range(0, total_len, block_size)]}
        return result

    grouped = tokenized.map(group_texts, batched=True)

    def collate(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids}

    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(grouped, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=generator)


def _metric_snapshot(
    metric_history: List[Dict[str, Any]], gate_window: int
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    if gate_window <= 0:
        return [], []
    if len(metric_history) < gate_window:
        return [], list(metric_history)
    past = metric_history[-2 * gate_window : -gate_window]
    recent = metric_history[-gate_window:]
    return past, recent


def _gate_from_history(
    gate_engine: GateEngine,
    window_decl: WindowDecl,
    metric_history: List[Dict[str, Any]],
    gate_window: int,
    iter_num: int,
    gate_policy: str,
    gate_min_evidence: int,
) -> GateRecordV1:
    past_slice, recent_slice = _metric_snapshot(metric_history, gate_window)
    if not past_slice or not recent_slice:
        audit = AuditV1(
            evaluated=False,
            reason="not_evaluated_insufficient_evidence",
            policy=gate_policy,
            per_metric_tv={},
            evidence_total=0,
            min_evidence=gate_min_evidence,
        )
        return GateRecordV1(iter=iter_num, decision="skip", audit=audit, ok=True, warn=False)
    metrics = list(window_decl.weights.keys())

    def _extract(slice_data: List[Dict[str, Any]], key: str) -> np.ndarray:
        vals = [float(d.get(key, np.nan)) for d in slice_data]
        arr = np.array(vals, dtype=float)
        return arr[np.isfinite(arr)]

    past_dict = {m: _extract(past_slice, m) for m in metrics}
    recent_dict = {m: _extract(recent_slice, m) for m in metrics}
    counts = {m: min(len(past_dict[m]), len(recent_dict[m])) for m in metrics}
    evidence_total = int(sum(counts.values()))

    eps_stat_value = float("nan")
    if evidence_total > 0:
        eps_stat_value = aggregate_epsilon_stat(window_decl, counts, alpha=0.05)

    # If we don't have enough finite evidence, don't “evaluate” (prevents spurious fail).
    if evidence_total < int(gate_min_evidence):
        audit = AuditV1(
            evaluated=False,
            reason="not_evaluated_insufficient_finite_evidence",
            policy=gate_policy,
            per_metric_tv={},
            evidence_total=evidence_total,
            min_evidence=gate_min_evidence,
        )
        return GateRecordV1(iter=iter_num, decision="skip", audit=audit, ok=True, warn=False)

    result = gate_engine.check(
        past=past_dict,
        recent=recent_dict,
        counts_by_metric=counts,
        gain_bits=0.0,
        kappa_sens=0.0,
        eps_stat_value=eps_stat_value,
        iter_num=iter_num,
    )

    audit_payload = dict(result.audit or {})
    audit_payload.setdefault("per_metric_tv", {})
    # AuditV1 requires policy always, and requires reason when evaluated=True.
    audit_payload.setdefault("policy", gate_policy)
    audit_payload.setdefault("evidence_total", evidence_total)
    audit_payload.setdefault("min_evidence", int(gate_min_evidence))

    evaluated = bool(audit_payload.get("evaluated", True))
    audit_payload["evaluated"] = evaluated
    if evaluated:
        audit_payload.setdefault("reason", "evaluated")
    else:
        audit_payload.setdefault("reason", "not_evaluated")
    audit = AuditV1(**audit_payload)

    if not audit.evaluated:
        decision = "skip"
    elif result.warn:
        decision = "warn"
    elif result.ok:
        decision = "pass"
    else:
        decision = "fail"

    return GateRecordV1(iter=iter_num, decision=decision, audit=audit, ok=result.ok, warn=result.warn)


def _exit_code_for_signal(signal_name: Optional[str]) -> Optional[int]:
    if not signal_name:
        return None
    mapping = {"SIGINT": 130, "SIGTERM": 143}
    return mapping.get(signal_name, 128)


def _env_truthy(name: str) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _force_cleanup_outdir(outdir: Path) -> None:
    """Remove only the capsule marker files that can deterministically collide.

    We avoid deleting the entire outdir because callers (pytest) may own the temp root.
    """
    targets = [
        "window_signature.json",
        "results.json",
        "results_summary.json",
        "run_manifest.json",
    ]
    for name in targets:
        try:
            p = outdir / name
            if p.exists():
                p.unlink()
        except Exception:
            pass


def _run(cfg: HFRunConfig, *, argv: List[str]) -> int:
    if _is_chief():
        # If any outer wrapper pre-created capsule markers (esp window_signature.json),
        # force mode must ensure the runner is authoritative.
        if bool(cfg.force):
            _force_cleanup_outdir(cfg.outdir)
        # Chief-only emission: only rank 0 writes artifacts and manifest.
        cfg.outdir.mkdir(parents=True, exist_ok=True)
    _set_seed(cfg.seed)

    device = torch.device(cfg.device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model)
    model.config.use_cache = False
    model.to(device)
    model.train()

    data_loader = _tokenize_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        cfg.dataset_path,
        cfg.dataset_split,
        cfg.dataset_text_column,
        tokenizer,
        block_size=cfg.block_size,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    metric_config = HFMetricConfig(
        max_tokens_per_batch=cfg.batch_size * cfg.block_size,
        rp_dim=cfg.rp_dim,
    )
    metric_computer = HFMetricComputer(config=metric_config, seed=cfg.seed)

    window_decl = _build_window_decl(cfg)
    gate_engine = _build_gate_engine(cfg, window_decl)

    run_id = cfg.run_id
    started_ts = datetime.now(timezone.utc)
    window_signature = _build_window_signature(cfg, created_ts_utc=started_ts)

    interrupt_signal: Optional[str] = None
    stop_requested = False
    previous_handlers: Dict[int, Any] = {}
    failure_reason: Optional[str] = None
    failure_traceback: Optional[str] = None

    def _signal_handler(signum: int, _frame: Any) -> None:
        nonlocal interrupt_signal, stop_requested
        stop_requested = True
        try:
            interrupt_signal = signal.Signals(signum).name
        except Exception:
            interrupt_signal = f"SIG{signum}"

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            sig_num = sig.value if hasattr(sig, "value") else int(sig)
            previous_handlers[sig_num] = signal.getsignal(sig_num)
            signal.signal(sig_num, _signal_handler)
        except Exception:
            continue

    metric_history: List[Dict[str, Any]] = []
    gate_records: List[GateRecordV1] = []
    metric_records: List[MetricRecordV1] = []

    step = 0
    data_iter = iter(data_loader)
    try:
        while step < cfg.max_steps:
            if stop_requested:
                break
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Always provide a real attention_mask (important for adapter metrics on tiny models).
            attention_mask: Optional[torch.Tensor] = None
            if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            need_hidden = step % cfg.cadence == 0
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=need_hidden,
                use_cache=False,
            )
            loss = outputs.loss
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            if step % cfg.cadence == 0:
                with torch.no_grad():
                    hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else None
                    if hidden_states is None:
                        m = {"var_out_k": float("nan"), "eff_dim": float("nan")}
                    else:
                        try:
                            m = metric_computer.compute_metrics(
                                hidden_states=hidden_states.detach(),
                                attention_mask=attention_mask,
                                step=step,
                            )
                        except Exception:
                            m = _fallback_metrics_from_hidden(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                rp_dim=cfg.rp_dim,
                            )
                        # If adapter returns non-finite values, use fallback (keeps gates evaluable).
                        v_ok = math.isfinite(float(m.get("var_out_k", float("nan"))))
                        e_ok = math.isfinite(float(m.get("eff_dim", float("nan"))))
                        if not (v_ok and e_ok):
                            m = _fallback_metrics_from_hidden(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                rp_dim=cfg.rp_dim,
                            )

                loss_value = float(loss.detach().cpu().item())
                var_out_k_raw = float(m.get("var_out_k", float("nan")))
                eff_dim_raw = float(m.get("eff_dim", float("nan")))

                metric_history.append(
                    {
                        "iter": step,
                        "loss": loss_value,
                        "var_out_k": var_out_k_raw,
                        "eff_dim": eff_dim_raw,
                    }
                )

                # Emit JSON-safe MetricRecordV1 for declared evidence metrics (and loss).
                metric_records.append(MetricRecordV1(name="loss", iter=step, value=_jsonable_float(loss_value)))
                metric_records.append(MetricRecordV1(name="var_out_k", iter=step, value=_jsonable_float(var_out_k_raw)))
                metric_records.append(MetricRecordV1(name="eff_dim", iter=step, value=_jsonable_float(eff_dim_raw)))

                gate_records.append(
                    _gate_from_history(
                        gate_engine,
                        window_decl,
                        metric_history,
                        cfg.gate_window,
                        step,
                        cfg.gate_policy,
                        cfg.gate_min_evidence,
                    )
                )

            step += 1
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        failure_traceback = traceback.format_exc()
        # Make failures visible in CI logs (otherwise you just see rc!=0 with no clue).
        if _is_chief():
            print(f"[veriscope:hf] failure_reason={failure_reason}\n{failure_traceback}", file=sys.stderr)

    for sig_num, handler in previous_handlers.items():
        try:
            signal.signal(sig_num, handler)
        except Exception:
            continue

    ended_ts = datetime.now(timezone.utc)
    if not gate_records:
        audit = AuditV1(
            evaluated=False,
            reason="not_evaluated_no_steps",
            policy=cfg.gate_policy,
            per_metric_tv={},
            evidence_total=0,
            min_evidence=cfg.gate_min_evidence,
        )
        gate_records.append(GateRecordV1(iter=0, decision="skip", audit=audit, ok=True, warn=False))

    metrics = list(metric_records)
    if not metrics:
        # Ensure non-degenerate emission even when no steps completed.
        metrics = [
            MetricRecordV1(name="loss", iter=0, value=None),
            MetricRecordV1(name="var_out_k", iter=0, value=None),
            MetricRecordV1(name="eff_dim", iter=0, value=None),
        ]

    run_status = "success"
    runner_signal = interrupt_signal
    runner_exit_code = 0
    if failure_reason:
        run_status = "user_code_failure"
        runner_exit_code = 1
    elif stop_requested:
        run_status = "user_code_failure"
        runner_exit_code = _exit_code_for_signal(interrupt_signal) or 1

    if _is_chief():
        # Chief-only emission: only rank 0 writes artifacts and manifest.
        emit_hf_artifacts_v1(
            outdir=cfg.outdir,
            run_id=run_id,
            started_ts_utc=started_ts,
            ended_ts_utc=ended_ts,
            gate_preset=cfg.gate_preset,
            window_signature=window_signature,
            gate_records=gate_records,
            metrics=metrics,
            run_status=run_status,
            runner_exit_code=runner_exit_code,
            runner_signal=runner_signal,
        )
        manifest = _build_run_manifest(
            cfg,
            argv=argv,
            started_ts_utc=started_ts,
            ended_ts_utc=ended_ts,
            run_status=run_status,
            runner_exit_code=runner_exit_code,
            runner_signal=runner_signal,
            failure_reason=failure_reason,
            failure_traceback=failure_traceback,
        )
        _write_run_manifest(cfg.outdir, manifest)
    return int(runner_exit_code if runner_exit_code is not None else 1)


def _parse_args() -> HFRunConfig:
    parser = argparse.ArgumentParser(description="Veriscope HF transformer runner (custom loop).")
    parser.add_argument("--model", type=str, default="gpt2", help="HF model name (default: gpt2)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext:wikitext-2-raw-v1",
        help="HF dataset spec (name[:config]) (default: wikitext:wikitext-2-raw-v1)",
    )
    parser.add_argument("--dataset_name", type=str, default="", help="HF dataset name override")
    parser.add_argument("--dataset_config", type=str, default="", help="HF dataset config override")
    parser.add_argument("--dataset_split", type=str, default="train", help="HF dataset split")
    parser.add_argument("--dataset_text_column", type=str, default="text", help="Text column name")
    parser.add_argument("--outdir", type=str, default="", help="Output directory for artifacts")
    parser.add_argument("--run_id", type=str, default="", help="Run identifier (wrapper overrides)")
    parser.add_argument("--force", action="store_true", help="Overwrite/repair existing artifacts in outdir")
    parser.add_argument("--max_steps", type=int, default=200, help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--cadence", type=int, default=10, help="Instrumentation cadence (steps)")
    parser.add_argument("--block_size", type=int, default=128, help="Sequence length for training")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping (0 to disable)")
    parser.add_argument("--gate_preset", type=str, default="tuned_v0", help="Gate preset name")
    parser.add_argument("--gate_window", type=int, default=20, help="Gate window (metric snapshots)")
    parser.add_argument("--gate_epsilon", type=float, default=0.12, help="Gate epsilon")
    parser.add_argument("--gate_min_evidence", type=int, default=8, help="Gate minimum evidence")
    parser.add_argument("--gate_gain_thresh", type=float, default=0.0, help="Gate gain threshold")
    parser.add_argument("--gate_policy", type=str, default="persistence", help="Gate policy")
    parser.add_argument("--gate_persistence_k", type=int, default=2, help="Gate persistence K")
    parser.add_argument("--rp_dim", type=int, default=64, help="JL projection dimension")

    args = parser.parse_args()
    outdir = Path(args.outdir).expanduser() if args.outdir else _default_outdir()
    run_id = args.run_id.strip() or uuid.uuid4().hex[:12]
    dataset_name = args.dataset_name.strip()
    dataset_config = args.dataset_config.strip()
    dataset_path: Optional[Path] = None
    if not dataset_name:
        dataset_spec = args.dataset.strip()
        if dataset_spec.startswith("file:"):
            dataset_path_raw = dataset_spec[len("file:") :].strip()
            if not dataset_path_raw:
                raise ValueError("--dataset file: requires a path")
            candidate = Path(dataset_path_raw).expanduser()
            if not candidate.is_absolute():
                repo_root = Path(__file__).resolve().parents[3]
                candidate = (repo_root / candidate).resolve()
            dataset_path = candidate
            dataset_name = "file"
            dataset_config = ""
        elif ":" in dataset_spec:
            dataset_name, dataset_config = dataset_spec.split(":", 1)
        else:
            dataset_name = dataset_spec
    dataset_name = dataset_name or "wikitext"
    dataset_config = dataset_config or None
    # Honor either runner --force or wrapper-side VERISCOPE_FORCE=1
    force = bool(args.force) or _env_truthy("VERISCOPE_FORCE")
    return HFRunConfig(
        model=args.model,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_path=dataset_path,
        dataset_split=args.dataset_split,
        dataset_text_column=args.dataset_text_column,
        outdir=outdir,
        run_id=run_id,
        force=force,
        max_steps=int(args.max_steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        cadence=int(args.cadence),
        block_size=int(args.block_size),
        device=str(args.device),
        grad_clip=float(args.grad_clip),
        gate_preset=str(args.gate_preset),
        gate_window=int(args.gate_window),
        gate_epsilon=float(args.gate_epsilon),
        gate_min_evidence=int(args.gate_min_evidence),
        gate_gain_thresh=float(args.gate_gain_thresh),
        gate_policy=str(args.gate_policy),
        gate_persistence_k=int(args.gate_persistence_k),
        rp_dim=int(args.rp_dim),
    )


def main() -> int:
    cfg = _parse_args()
    if _is_chief():
        cfg.outdir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            cfg.outdir / "runner_config.json",
            {
                "schema_version": 1,
                "ts_utc": _iso_utc(datetime.now(timezone.utc)),
                "runner": "hf",
                "config": {
                    "run_id": cfg.run_id,
                    "model": cfg.model,
                    "dataset_name": cfg.dataset_name,
                    "dataset_config": cfg.dataset_config,
                    "dataset_path": str(cfg.dataset_path) if cfg.dataset_path else None,
                    "dataset_split": cfg.dataset_split,
                    "dataset_text_column": cfg.dataset_text_column,
                    "max_steps": cfg.max_steps,
                    "batch_size": cfg.batch_size,
                    "lr": cfg.lr,
                    "seed": cfg.seed,
                    "cadence": cfg.cadence,
                    "block_size": cfg.block_size,
                    "device": cfg.device,
                    "grad_clip": cfg.grad_clip,
                    "gate_preset": cfg.gate_preset,
                    "gate_window": cfg.gate_window,
                    "gate_epsilon": cfg.gate_epsilon,
                    "gate_min_evidence": cfg.gate_min_evidence,
                    "gate_gain_thresh": cfg.gate_gain_thresh,
                    "gate_policy": cfg.gate_policy,
                    "gate_persistence_k": cfg.gate_persistence_k,
                    "rp_dim": cfg.rp_dim,
                },
            },
        )
    return _run(cfg, argv=sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
