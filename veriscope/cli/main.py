# veriscope/cli/main.py
from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from veriscope.core.artifacts import CountsV1, ProfileV1, ResultsSummaryV1, RunStatus, WindowSignatureRefV1
from veriscope.core.jsonutil import atomic_write_json, canonical_json_sha256
from veriscope.core.lifecycle import RunLifecycle, map_status_and_exit


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pkg_version() -> str:
    try:
        import importlib.metadata as md

        return md.version("veriscope")
    except Exception:
        return "unknown"


def _git_sha(cwd: Optional[Path] = None) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            check=False,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return None
        s = (r.stdout or "").strip()
        return s or None
    except Exception:
        return None


def _relevant_env(env: Dict[str, str]) -> Dict[str, str]:
    keep_prefixes = ("SCAR_", "VERISCOPE_", "NANOGPT_", "CUDA_", "CUBLAS_")
    keep_exact = {"CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED"}
    out: Dict[str, str] = {}
    for k, v in env.items():
        if k in keep_exact or k.startswith(keep_prefixes):
            out[k] = v
    return dict(sorted(out.items()))


def _default_outdir(kind: str) -> Path:
    base = Path(os.environ.get("VERISCOPE_OUT_BASE", "./out")).expanduser()
    ts = time.strftime("%Y%m%d_%H%M%S")
    return base / f"veriscope_{kind}_{ts}_{os.getpid()}"


def _legacy_default_cifar_outdir() -> Path:
    # IMPORTANT: matches legacy default in veriscope.runners.legacy_cli_refactor
    # OUTDIR = Path(os.environ.get("SCAR_OUTDIR", "./scar_bundle_phase4"))
    return Path("./scar_bundle_phase4")


def _write_resolved_config(outdir: Path, payload: Dict[str, Any]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "run_config_resolved.json"
    atomic_write_json(p, payload)


def _read_json_obj(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"{path.name} must be a JSON object")
    return obj


def _write_run_config_merge(outdir: Path, update: Dict[str, Any]) -> None:
    path = outdir / "run_config_resolved.json"
    if path.exists():
        try:
            existing = _read_json_obj(path)
        except Exception:
            existing = {}
    else:
        existing = {}
    merged = dict(existing)
    merged.update(update)
    _write_resolved_config(outdir, merged)


def _truncate_message(message: str, *, limit: int = 400) -> str:
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def _ensure_window_signature(outdir: Path, *, reason: str, run_kind: str) -> WindowSignatureRefV1:
    ws_path = outdir / "window_signature.json"
    placeholder = {
        "schema_version": 1,
        "placeholder": True,
        "reason": reason,
        "run_kind": run_kind,
    }
    if not ws_path.exists():
        atomic_write_json(ws_path, placeholder)
    try:
        ws_obj = _read_json_obj(ws_path)
    except Exception as exc:
        placeholder_with_error = dict(placeholder)
        err = _truncate_message(f"Invalid window_signature.json: {exc}")
        placeholder_with_error["error"] = {"message": err}
        atomic_write_json(ws_path, placeholder_with_error)
        ws_obj = placeholder_with_error
    ws_hash = canonical_json_sha256(ws_obj)
    return WindowSignatureRefV1(hash=ws_hash, path="window_signature.json")


def _extract_gate_preset(args: List[str]) -> str:
    for idx, arg in enumerate(args):
        if arg.startswith("--gate_preset="):
            return arg.split("=", 1)[1] or "unknown"
        if arg == "--gate_preset" and idx + 1 < len(args):
            return args[idx + 1]
    return "unknown"


def _write_partial_summary(
    *,
    outdir: Path,
    run_id: str,
    ws_ref: WindowSignatureRefV1,
    gate_preset: str,
    started_ts_utc: datetime,
    ended_ts_utc: datetime,
    run_status: RunStatus,
    runner_exit_code: Optional[int],
    runner_signal: Optional[str],
    note: str,
    wrapper_emitted: bool = True,
) -> None:
    profile = ProfileV1(gate_preset=gate_preset, overrides={})
    safe_exit_code = runner_exit_code if runner_exit_code is None or runner_exit_code >= 0 else None
    counts = CountsV1(evaluated=0, skip=0, pass_=0, warn=0, fail=0)
    summary = ResultsSummaryV1(
        run_id=run_id,
        window_signature_ref=ws_ref,
        profile=profile,
        run_status=run_status,
        runner_exit_code=safe_exit_code,
        runner_signal=runner_signal,
        started_ts_utc=started_ts_utc,
        ended_ts_utc=ended_ts_utc,
        counts=counts,
        final_decision="skip",
    )
    payload = summary.model_dump(mode="json", by_alias=True, exclude_none=True)
    payload["partial"] = True
    payload["note"] = note
    payload["wrapper_emitted"] = wrapper_emitted
    atomic_write_json(outdir / "results_summary.json", payload)


VERISCOPE_FAILURE: RunStatus = "veriscope_failure"


def _summary_wrapper_exit_code(outdir: Path) -> Optional[int]:
    summary_path = outdir / "results_summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = ResultsSummaryV1.model_validate_json(summary_path.read_text("utf-8"))
    except Exception:
        return None
    if summary.run_status == "success":
        return 0
    if summary.run_status == "user_code_failure":
        return 2
    if summary.run_status == "veriscope_failure":
        return 3
    return None


def _summary_is_valid(path: Path) -> bool:
    try:
        ResultsSummaryV1.model_validate_json(path.read_text("utf-8"))
        return True
    except Exception:
        return False


def _signal_child(proc: subprocess.Popen, signum: int) -> None:
    if hasattr(os, "killpg"):
        os.killpg(proc.pid, signum)
    else:
        proc.send_signal(signum)


def _relay_stream(stream: Any, target: Any) -> None:
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            target.write(line)
            target.flush()
    except (ValueError, BrokenPipeError, OSError, IOError):
        pass
    except Exception as exc:
        if os.environ.get("VERISCOPE_DEBUG_STREAMS") == "1":
            try:
                sys.stderr.write(f"[veriscope] stream relay error: {exc}\n")
                sys.stderr.flush()
            except Exception:
                pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _start_stream_threads(proc: subprocess.Popen) -> List[threading.Thread]:
    threads: List[threading.Thread] = []
    if proc.stdout is not None:
        t = threading.Thread(target=_relay_stream, args=(proc.stdout, sys.stdout))
        t.daemon = True
        t.start()
        threads.append(t)
    if proc.stderr is not None:
        t = threading.Thread(target=_relay_stream, args=(proc.stderr, sys.stderr))
        t.daemon = True
        t.start()
        threads.append(t)
    return threads


class _SignalForwarder:
    def __init__(self, proc: subprocess.Popen, lifecycle: RunLifecycle, *, timeout_s: float = 2.0) -> None:
        self._proc = proc
        self._lifecycle = lifecycle
        self._timeout_s = timeout_s
        self._interrupts = 0
        self._deadline: Optional[float] = None
        self._killed = False

    @property
    def interrupted(self) -> bool:
        return self._interrupts > 0

    def handle(self, signum: int, _frame: Optional[Any]) -> None:
        self._interrupts += 1
        self._lifecycle.mark_interrupted(signum)
        try:
            _signal_child(self._proc, signum)
        except Exception:
            pass
        if self._interrupts >= 2:
            self._force_kill()
        else:
            self._deadline = time.monotonic() + self._timeout_s

    def maybe_escalate(self) -> None:
        if self._deadline is None or self._killed:
            return
        if time.monotonic() >= self._deadline:
            self._force_kill()

    def _force_kill(self) -> None:
        if self._killed:
            return
        self._killed = True
        try:
            _signal_child(self._proc, signal.SIGKILL)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass


def _acquire_gpu_lock(force: bool) -> Optional[Any]:
    """Global GPU lock to prevent accidental concurrent runs."""
    if force:
        return None
    try:
        from filelock import FileLock, Timeout

        lock_path = os.environ.get("VERISCOPE_GPU_LOCK", "/tmp/veriscope_gpu.lock")
        lock = FileLock(lock_path)
        try:
            lock.acquire(timeout=0)
        except Timeout:
            raise SystemExit(f"Refusing to start: GPU lock already held ({lock_path}). Use --force to bypass.")
        return lock
    except ImportError:
        # fail-open if filelock missing (should not happen given your deps)
        return None


def _run_legacy_subprocess(argv_passthrough: List[str], env: Dict[str, str]) -> int:
    """
    Delegate to the legacy console entrypoint via subprocess.

    This preserves the prior process model and avoids importing torchvision in the wrapper.
    """
    legacy_exe = "veriscope-legacy"
    if shutil.which(legacy_exe) is None:
        raise SystemExit(
            "veriscope-legacy not found on PATH. "
            "Run `pip install -e .` (editable install) so the console script is generated."
        )

    cmd = [legacy_exe] + list(argv_passthrough)
    r = subprocess.run(cmd, env=env)
    return int(r.returncode)


def _run_legacy(argv_passthrough: List[str]) -> int:
    return _run_legacy_subprocess(list(argv_passthrough), env=os.environ.copy())


def _select_cifar_outdir(args_outdir: str) -> Path:
    outdir_str = (args_outdir or "").strip()
    scar_outdir = (os.environ.get("SCAR_OUTDIR") or "").strip()

    if outdir_str:
        return Path(outdir_str).expanduser()
    if scar_outdir:
        return Path(scar_outdir).expanduser()

    # IMPORTANT: preserve legacy default semantics
    return _legacy_default_cifar_outdir()


def _cmd_run_cifar(args: argparse.Namespace) -> int:
    outdir = _select_cifar_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # Set SCAR_OUTDIR to preserve legacy default and ensure artifact goes to the run dir.
    env["SCAR_OUTDIR"] = str(outdir)

    if args.smoke:
        env["SCAR_SMOKE"] = "1"

    legacy_args = list(args.legacy_args or [])
    if legacy_args and legacy_args[0] == "--":
        legacy_args = legacy_args[1:]

    run_id = uuid.uuid4().hex[:12]
    lifecycle = RunLifecycle(run_id=run_id, run_kind="cifar")
    resolved = {
        "schema_version": 1,
        "ts_utc": _now_utc_iso(),
        "package": {"name": "veriscope", "version": _pkg_version()},
        "git_sha": _git_sha(),
        "run": {"kind": "cifar", "run_id": run_id, "outdir": str(outdir)},
        "argv": {
            "veriscope_argv": ["veriscope", "run", "cifar"]
            + (["--smoke"] if args.smoke else [])
            + (["--outdir", str(outdir)] if (args.outdir or "").strip() else [])
            + (args.legacy_args or []),
            "legacy_argv": legacy_args,
            "legacy_entrypoint": "veriscope-legacy (console script)",
        },
        "env": _relevant_env(env),
        "wrapper_exit_code": None,
        "runner_exit_code": None,
        "runner_signal": None,
        "run_status": "running",
        "lifecycle_state": lifecycle.state,
        "started_ts_utc": lifecycle.started_ts_utc.isoformat(),
        "ended_ts_utc": None,
    }
    _write_resolved_config(outdir, resolved)
    print(f"[veriscope] outdir={outdir}")
    print(f"[veriscope] wrote {outdir / 'run_config_resolved.json'}")

    ws_ref = _ensure_window_signature(outdir, reason="wrapper_start", run_kind="cifar")
    lock = _acquire_gpu_lock(force=bool(args.force))
    try:
        try:
            lifecycle.mark_running()
            _write_run_config_merge(outdir, {"lifecycle_state": lifecycle.state})
            legacy_exe = "veriscope-legacy"
            if shutil.which(legacy_exe) is None:
                raise SystemExit(
                    "veriscope-legacy not found on PATH. "
                    "Run `pip install -e .` (editable install) so the console script is generated."
                )
            cmd = [legacy_exe] + list(legacy_args)
            proc = subprocess.Popen(cmd, env=env, start_new_session=True)

            def _handle_signal(signum: int, _frame: Optional[Any]) -> None:
                lifecycle.mark_interrupted(signum)
                try:
                    _signal_child(proc, signum)
                except Exception:
                    pass

            prev_int = signal.signal(signal.SIGINT, _handle_signal)
            prev_term = signal.signal(signal.SIGTERM, _handle_signal)
            try:
                proc.wait()
            finally:
                signal.signal(signal.SIGINT, prev_int)
                signal.signal(signal.SIGTERM, prev_term)
            lifecycle.mark_runner_exit(proc.returncode)

            summary_exit = _summary_wrapper_exit_code(outdir)
            run_status, wrapper_exit = map_status_and_exit(
                runner_exit_code=proc.returncode,
                runner_signal=lifecycle.runner_signal,
                internal_error=False,
            )
            if summary_exit is not None and run_status == "success":
                wrapper_exit = summary_exit
        except Exception as exc:
            lifecycle.mark_internal_failure(str(exc))
            run_status, wrapper_exit = map_status_and_exit(
                runner_exit_code=lifecycle.runner_exit_code,
                runner_signal=lifecycle.runner_signal,
                internal_error=True,
            )

        lifecycle.finalize(run_status=run_status, wrapper_exit_code=wrapper_exit)
        _write_run_config_merge(
            outdir,
            {
                "wrapper_exit_code": lifecycle.wrapper_exit_code,
                "runner_exit_code": lifecycle.runner_exit_code,
                "runner_signal": lifecycle.runner_signal,
                "run_status": lifecycle.run_status,
                "lifecycle_state": lifecycle.state,
                "ended_ts_utc": lifecycle.ended_ts_utc.isoformat() if lifecycle.ended_ts_utc else None,
            },
        )
        if not (outdir / "results_summary.json").exists():
            _write_partial_summary(
                outdir=outdir,
                run_id=run_id,
                ws_ref=ws_ref,
                gate_preset="unknown",
                started_ts_utc=lifecycle.started_ts_utc,
                ended_ts_utc=lifecycle.ended_ts_utc or datetime.now(timezone.utc),
                run_status=lifecycle.run_status,
                runner_exit_code=lifecycle.runner_exit_code,
                runner_signal=lifecycle.runner_signal,
                note="results_summary.json missing; emitted wrapper-level partial summary",
            )
        return int(lifecycle.wrapper_exit_code or 0)
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass


def _cmd_run_gpt(args: argparse.Namespace) -> int:
    outdir_str = (args.outdir or "").strip()
    outdir = Path(outdir_str).expanduser() if outdir_str else _default_outdir("gpt")
    outdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    run_id = uuid.uuid4().hex[:12]
    lifecycle = RunLifecycle(run_id=run_id, run_kind="gpt")

    gpt_args = list(args.gpt_args or [])
    if gpt_args and gpt_args[0] == "--":
        gpt_args = gpt_args[1:]

    # Ensure output is routed to outdir unless user explicitly set it.
    has_out_dir = any(a.startswith("--out_dir") for a in gpt_args)
    has_out_json = any(a.startswith("--out_json") for a in gpt_args)
    if not has_out_dir:
        gpt_args = ["--out_dir", str(outdir)] + gpt_args
    if not has_out_json:
        gpt_args = ["--out_json", f"veriscope_gpt_{time.strftime('%Y%m%d_%H%M%S')}.json"] + gpt_args

    override_cmd = (os.environ.get("VERISCOPE_GPT_RUNNER_CMD") or "").strip()
    if override_cmd:
        cmd = shlex.split(override_cmd) + gpt_args
    else:
        cmd = [sys.executable, "-m", "veriscope.runners.gpt.train_nanogpt"] + gpt_args

    gate_preset = _extract_gate_preset(gpt_args)
    resolved = {
        "schema_version": 1,
        "ts_utc": _now_utc_iso(),
        "package": {"name": "veriscope", "version": _pkg_version()},
        "git_sha": _git_sha(),
        "run": {"kind": "gpt", "run_id": run_id, "outdir": str(outdir)},
        "argv": {
            "veriscope_argv": ["veriscope", "run", "gpt"]
            + (["--outdir", str(outdir)] if outdir_str else [])
            + (args.gpt_args or []),
            "runner_cmd": cmd,
        },
        "env": _relevant_env(env),
        "wrapper_exit_code": None,
        "runner_exit_code": None,
        "runner_signal": None,
        "run_status": "running",
        "lifecycle_state": lifecycle.state,
        "started_ts_utc": lifecycle.started_ts_utc.isoformat(),
        "ended_ts_utc": None,
    }
    _write_resolved_config(outdir, resolved)
    print(f"[veriscope] outdir={outdir}")
    print(f"[veriscope] cmd={' '.join(cmd)}")
    print(f"[veriscope] wrote {outdir / 'run_config_resolved.json'}")

    ws_ref = _ensure_window_signature(outdir, reason="wrapper_start", run_kind="gpt")
    lock = _acquire_gpu_lock(force=bool(args.force))
    try:
        proc: Optional[subprocess.Popen] = None
        threads: List[threading.Thread] = []
        prev_int = None
        prev_term = None
        runner_launch_error: Optional[Exception] = None
        try:
            lifecycle.mark_running()
            _write_run_config_merge(outdir, {"lifecycle_state": lifecycle.state})
            try:
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    start_new_session=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except OSError as exc:
                runner_launch_error = exc

            if runner_launch_error is None and proc is not None:
                threads = _start_stream_threads(proc)
                forwarder = _SignalForwarder(proc, lifecycle)

                prev_int = signal.signal(signal.SIGINT, forwarder.handle)
                prev_term = signal.signal(signal.SIGTERM, forwarder.handle)
                try:
                    while proc.poll() is None:
                        if forwarder.interrupted:
                            forwarder.maybe_escalate()
                            time.sleep(0.1)
                        else:
                            try:
                                proc.wait(timeout=0.5)
                            except subprocess.TimeoutExpired:
                                pass
                finally:
                    signal.signal(signal.SIGINT, prev_int)
                    signal.signal(signal.SIGTERM, prev_term)

                lifecycle.mark_runner_exit(proc.returncode)

                summary_path = outdir / "results_summary.json"
                summary_ok = summary_path.exists() and _summary_is_valid(summary_path)
                run_status, wrapper_exit = map_status_and_exit(
                    runner_exit_code=proc.returncode,
                    runner_signal=lifecycle.runner_signal,
                    internal_error=False,
                )
                if not summary_ok:
                    run_status = VERISCOPE_FAILURE
                    wrapper_exit = 2
                else:
                    summary_exit = _summary_wrapper_exit_code(outdir)
                    if summary_exit is not None and run_status == "success":
                        wrapper_exit = summary_exit
            else:
                lifecycle.mark_internal_failure(str(runner_launch_error))
                run_status = VERISCOPE_FAILURE
                wrapper_exit = 2
        except Exception as exc:
            lifecycle.mark_internal_failure(str(exc))
            run_status, wrapper_exit = map_status_and_exit(
                runner_exit_code=lifecycle.runner_exit_code,
                runner_signal=lifecycle.runner_signal,
                internal_error=True,
            )
        finally:
            for thread in threads:
                thread.join(timeout=1.0)

        lifecycle.finalize(run_status=run_status, wrapper_exit_code=wrapper_exit)
        _write_run_config_merge(
            outdir,
            {
                "wrapper_exit_code": lifecycle.wrapper_exit_code,
                "runner_exit_code": lifecycle.runner_exit_code,
                "runner_signal": lifecycle.runner_signal,
                "run_status": lifecycle.run_status,
                "lifecycle_state": lifecycle.state,
                "ended_ts_utc": lifecycle.ended_ts_utc.isoformat() if lifecycle.ended_ts_utc else None,
            },
        )
        summary_path = outdir / "results_summary.json"
        if not (summary_path.exists() and _summary_is_valid(summary_path)):
            try:
                _write_partial_summary(
                    outdir=outdir,
                    run_id=run_id,
                    ws_ref=ws_ref,
                    gate_preset=gate_preset,
                    started_ts_utc=lifecycle.started_ts_utc,
                    ended_ts_utc=lifecycle.ended_ts_utc or datetime.now(timezone.utc),
                    run_status=lifecycle.run_status,
                    runner_exit_code=lifecycle.runner_exit_code,
                    runner_signal=lifecycle.runner_signal,
                    note="wrapper-emitted partial summary: results_summary.json missing",
                )
            except Exception:
                lifecycle.wrapper_exit_code = 2
                try:
                    _write_run_config_merge(
                        outdir,
                        {
                            "wrapper_exit_code": lifecycle.wrapper_exit_code,
                            "run_status": lifecycle.run_status,
                            "lifecycle_state": lifecycle.state,
                            "ended_ts_utc": lifecycle.ended_ts_utc.isoformat() if lifecycle.ended_ts_utc else None,
                        },
                    )
                except Exception:
                    pass
        return int(lifecycle.wrapper_exit_code or 0)
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _cmd_validate(args: argparse.Namespace) -> int:
    outdir = Path(str(args.outdir)).expanduser()
    from veriscope.cli.governance import validate_governance_log
    from veriscope.cli.validate import validate_outdir

    v = validate_outdir(outdir)
    if not v.ok:
        _eprint(f"INVALID: {v.message}")
        return 2

    strict_governance = bool(getattr(args, "strict_governance", False))
    allow_legacy = bool(getattr(args, "allow_legacy_governance", False))
    require_governance = bool(getattr(args, "require_governance", False))

    gov_path = outdir / "governance_log.jsonl"
    if not gov_path.exists():
        if require_governance:
            _eprint("INVALID: governance_log.jsonl is required but missing")
            return 2
    else:
        validation = validate_governance_log(gov_path, allow_legacy_governance=allow_legacy)
        for warning in validation.warnings:
            _eprint(warning)
        if strict_governance and not validation.ok:
            _eprint("INVALID: governance_log.jsonl failed validation:")
            for err in validation.errors:
                _eprint(f"  {err}")
            return 2
    if strict_governance and not gov_path.exists() and not require_governance:
        _eprint(
            "WARNING:GOVERNANCE_LOG_MISSING governance_log.jsonl not present (strict_governance set, but not required)"
        )

    if v.window_signature_hash:
        print(f"OK window_signature_hash={v.window_signature_hash}")
    else:
        print("OK")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from veriscope.cli.report import render_report_compare, render_report_md

    fmt = str(getattr(args, "format", "text")).strip().lower()
    if bool(getattr(args, "json", False)):
        fmt = "json"
    compare = bool(getattr(args, "compare", False))
    outdirs = [Path(str(p)).expanduser() for p in getattr(args, "outdirs", [])]

    if compare:
        result = render_report_compare(
            outdirs,
            fmt=fmt,
            allow_incompatible=bool(getattr(args, "allow_incompatible", False)),
            allow_gate_preset_mismatch=bool(getattr(args, "allow_gate_preset_mismatch", False)),
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            _eprint(result.stderr)
        return int(result.exit_code)

    if len(outdirs) != 1:
        _eprint("report requires exactly one OUTDIR (or use --compare for multiple)")
        return 2
    if fmt not in ("md", "text"):
        _eprint("report format must be 'md' or 'text' (use --compare for json output)")
        return 2

    try:
        text = render_report_md(outdirs[0], fmt=fmt)
    except Exception as e:
        _eprint(str(e))
        return 2
    print(text)
    return 0


def _cmd_override(args: argparse.Namespace) -> int:
    from veriscope.cli.override import write_manual_judgement

    outdir = Path(str(args.outdir)).expanduser()
    try:
        path, warnings = write_manual_judgement(
            outdir,
            status=str(args.status).strip(),
            reason=str(args.reason),
            reviewer=(str(args.reviewer).strip() if args.reviewer else None),
            ts_utc=(str(args.ts_utc).strip() if args.ts_utc else None),
            force=bool(args.force),
        )
    except (FileExistsError, FileNotFoundError) as exc:
        _eprint(str(exc))
        return 2
    except ValueError as exc:
        _eprint(str(exc))
        return 2
    for warning in warnings:
        _eprint(warning)
    print(f"[veriscope] wrote {path}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """
    Inspect = validate + (optionally) render report.

    Exit codes:
      0 = valid
      2 = invalid artifacts / cannot render
    """
    outdir = Path(str(args.outdir)).expanduser()
    from veriscope.cli.report import render_report_md
    from veriscope.cli.governance import validate_governance_log
    from veriscope.cli.validate import validate_outdir

    v = validate_outdir(outdir, allow_partial=True)
    if not v.ok:
        _eprint(f"INVALID: {v.message}")
        _eprint("")
        _eprint("Expected canonical artifacts under OUTDIR:")
        _eprint("  - window_signature.json")
        _eprint("  - results.json")
        _eprint("  - results_summary.json")
        _eprint("Optional:")
        _eprint("  - run_config_resolved.json")
        _eprint("  - manual_judgement.json")
        _eprint("  - manual_judgement.jsonl")
        _eprint("  - governance_log.jsonl")
        return 2

    strict_governance = bool(getattr(args, "strict_governance", False))
    allow_legacy = bool(getattr(args, "allow_legacy_governance", False))
    require_governance = bool(getattr(args, "require_governance", False))

    gov_path = outdir / "governance_log.jsonl"
    if not gov_path.exists():
        if require_governance:
            _eprint("INVALID: governance_log.jsonl is required but missing")
            return 2
    else:
        validation = validate_governance_log(gov_path, allow_legacy_governance=allow_legacy)
        for warning in validation.warnings:
            _eprint(warning)
        if strict_governance and not validation.ok:
            _eprint("INVALID: governance_log.jsonl failed validation:")
            for err in validation.errors:
                _eprint(f"  {err}")
            return 2
    if strict_governance and not gov_path.exists() and not require_governance:
        _eprint(
            "WARNING:GOVERNANCE_LOG_MISSING governance_log.jsonl not present (strict_governance set, but not required)"
        )

    no_report = bool(getattr(args, "no_report", False))
    if no_report:
        if v.window_signature_hash:
            print(f"OK window_signature_hash={v.window_signature_hash}")
        else:
            print("OK")
        return 0

    fmt = str(getattr(args, "format", "text")).strip().lower()
    try:
        text = render_report_md(outdir, fmt=fmt)
    except Exception as e:
        _eprint(f"OK (validation passed), but report failed: {e}")
        return 2

    print(text)
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    from veriscope.cli.diff import diff_outdirs

    outdir_a = Path(str(args.outdir_a)).expanduser()
    outdir_b = Path(str(args.outdir_b)).expanduser()
    fmt = "json" if bool(getattr(args, "json", False)) else str(getattr(args, "format", "text"))
    result = diff_outdirs(
        outdir_a,
        outdir_b,
        allow_gate_preset_mismatch=bool(getattr(args, "allow_gate_preset_mismatch", False)),
        fmt=fmt,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        _eprint(result.stderr)
    return int(result.exit_code)


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Back-compat: `veriscope` with no args routes to legacy unchanged.
    if not argv:
        return _run_legacy([])

    parser = argparse.ArgumentParser(prog="veriscope")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run an experiment")
    run_sub = p_run.add_subparsers(dest="run_kind")

    p_gpt = run_sub.add_parser("gpt", help="Run the nanoGPT-based runner")
    p_gpt.add_argument("--outdir", type=str, default="", help="Output directory (default: ./out/...)")
    p_gpt.add_argument("--force", action="store_true", help="Bypass GPU lock")
    p_gpt.add_argument("gpt_args", nargs=argparse.REMAINDER, help="Args forwarded to train_nanogpt.py")
    p_gpt.set_defaults(_handler=_cmd_run_gpt)

    p_cifar = run_sub.add_parser("cifar", help="Run the legacy CIFAR runner (unchanged semantics)")
    p_cifar.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Output directory (default: SCAR_OUTDIR or ./scar_bundle_phase4)",
    )
    p_cifar.add_argument("--smoke", action="store_true", help="Set SCAR_SMOKE=1 for this run")
    p_cifar.add_argument("--force", action="store_true", help="Bypass GPU lock")
    p_cifar.add_argument("legacy_args", nargs=argparse.REMAINDER, help="Args forwarded to legacy runner")
    p_cifar.set_defaults(_handler=_cmd_run_cifar)

    p_validate = sub.add_parser("validate", help="Validate canonical artifacts in an OUTDIR")
    p_validate.add_argument("outdir", type=str, help="Artifact directory (OUTDIR)")
    p_validate.add_argument(
        "--strict-governance",
        action="store_true",
        help="Fail validation if governance_log.jsonl is invalid",
    )
    p_validate.add_argument(
        "--allow-legacy-governance",
        action="store_true",
        help="Allow governance_log.jsonl entries missing entry_hash (warn only)",
    )
    p_validate.add_argument(
        "--require-governance",
        action="store_true",
        help="Fail validation if governance_log.jsonl is missing",
    )
    p_validate.set_defaults(_handler=_cmd_validate)

    p_report = sub.add_parser("report", help="Render a human report from an OUTDIR")
    p_report.add_argument("outdirs", nargs="+", type=str, help="Artifact directory (OUTDIR)")
    p_report.add_argument(
        "--format",
        choices=["md", "text", "json"],
        default="text",
        help="Output format (json compare-only)",
    )
    p_report.add_argument("--json", action="store_true", help="Shorthand for --format json (compare only)")
    p_report.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple OUTDIRs and emit a compact table",
    )
    p_report.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="Include incompatible runs in the compare table (marks them as comparable=no)",
    )
    p_report.add_argument(
        "--allow-gate-preset-mismatch",
        action="store_true",
        help="Allow comparisons across different gate_preset values",
    )
    p_report.set_defaults(_handler=_cmd_report)

    p_inspect = sub.add_parser("inspect", help="Validate and summarize an OUTDIR (validate + report)")
    p_inspect.add_argument("outdir", type=str, help="Artifact directory (OUTDIR)")
    p_inspect.add_argument("--format", choices=["md", "text"], default="text", help="Output format")
    p_inspect.add_argument(
        "--no-report",
        action="store_true",
        help="Only validate; do not render a report.",
    )
    p_inspect.add_argument(
        "--strict-governance",
        action="store_true",
        help="Fail validation if governance_log.jsonl is invalid",
    )
    p_inspect.add_argument(
        "--allow-legacy-governance",
        action="store_true",
        help="Allow governance_log.jsonl entries missing entry_hash (warn only)",
    )
    p_inspect.add_argument(
        "--require-governance",
        action="store_true",
        help="Fail validation if governance_log.jsonl is missing",
    )
    p_inspect.set_defaults(_handler=_cmd_inspect)

    p_diff = sub.add_parser("diff", help="Compare two OUTDIRs with contract-aware checks")
    p_diff.add_argument("outdir_a", type=str, help="First OUTDIR")
    p_diff.add_argument("outdir_b", type=str, help="Second OUTDIR")
    p_diff.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    p_diff.add_argument("--json", action="store_true", help="Shorthand for --format json")
    p_diff.add_argument(
        "--allow-gate-preset-mismatch",
        action="store_true",
        help="Allow comparisons across different gate_preset values",
    )
    p_diff.set_defaults(_handler=_cmd_diff)

    p_override = sub.add_parser("override", help="Write a manual judgement artifact to an OUTDIR")
    p_override.add_argument("outdir", type=str, help="Artifact directory (OUTDIR)")
    p_override.add_argument("--status", choices=["pass", "fail"], required=True, help="Manual status")
    p_override.add_argument("--reason", required=True, help="Human-readable reason for the judgement")
    p_override.add_argument("--reviewer", default=None, help="Reviewer name/handle")
    p_override.add_argument("--ts-utc", default=None, help="ISO-8601 timestamp (default: now UTC)")
    p_override.add_argument("--force", action="store_true", help="Allow overwriting manual_judgement.json")
    p_override.set_defaults(_handler=_cmd_override)

    # IMPORTANT: let argparse manage help exit codes
    ns = parser.parse_args(argv)

    if not hasattr(ns, "_handler"):
        # Unknown command shape => legacy forward (muscle-memory safe).
        return _run_legacy(argv)

    return int(ns._handler(ns))  # type: ignore[misc]
