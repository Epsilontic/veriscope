# veriscope/cli/main.py
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    }
    _write_resolved_config(outdir, resolved)
    print(f"[veriscope] outdir={outdir}")
    print(f"[veriscope] wrote {outdir / 'run_config_resolved.json'}")

    lock = _acquire_gpu_lock(force=bool(args.force))
    try:
        return _run_legacy_subprocess(legacy_args, env=env)
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

    cmd = [sys.executable, "-m", "veriscope.runners.gpt.train_nanogpt"] + gpt_args

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
    }
    _write_resolved_config(outdir, resolved)
    print(f"[veriscope] outdir={outdir}")
    print(f"[veriscope] cmd={' '.join(cmd)}")
    print(f"[veriscope] wrote {outdir / 'run_config_resolved.json'}")

    lock = _acquire_gpu_lock(force=bool(args.force))
    try:
        r = subprocess.run(cmd, env=env)
        return int(r.returncode)
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
    from veriscope.cli.validate import validate_outdir

    v = validate_outdir(outdir)
    if v.ok:
        if v.window_signature_hash:
            print(f"OK window_signature_hash={v.window_signature_hash}")
        else:
            print("OK")
        return 0

    _eprint(f"INVALID: {v.message}")
    return 2


def _cmd_report(args: argparse.Namespace) -> int:
    outdir = Path(str(args.outdir)).expanduser()
    from veriscope.cli.report import render_report_md

    fmt = str(getattr(args, "format", "text")).strip().lower()
    try:
        text = render_report_md(outdir, fmt=fmt)
    except Exception as e:
        _eprint(str(e))
        return 2
    print(text)
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
    from veriscope.cli.validate import validate_outdir

    v = validate_outdir(outdir)
    if not v.ok:
        _eprint(f"INVALID: {v.message}")
        _eprint("")
        _eprint("Expected canonical artifacts under OUTDIR:")
        _eprint("  - window_signature.json")
        _eprint("  - results.json")
        _eprint("  - results_summary.json")
        _eprint("Optional:")
        _eprint("  - run_config_resolved.json")
        return 2

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
    p_validate.set_defaults(_handler=_cmd_validate)

    p_report = sub.add_parser("report", help="Render a human report from an OUTDIR")
    p_report.add_argument("outdir", type=str, help="Artifact directory (OUTDIR)")
    p_report.add_argument("--format", choices=["md", "text"], default="text", help="Output format")
    p_report.set_defaults(_handler=_cmd_report)

    p_inspect = sub.add_parser("inspect", help="Validate and summarize an OUTDIR (validate + report)")
    p_inspect.add_argument("outdir", type=str, help="Artifact directory (OUTDIR)")
    p_inspect.add_argument("--format", choices=["md", "text"], default="text", help="Output format")
    p_inspect.add_argument(
        "--no-report",
        action="store_true",
        help="Only validate; do not render a report.",
    )
    p_inspect.set_defaults(_handler=_cmd_inspect)

    # IMPORTANT: let argparse manage help exit codes
    ns = parser.parse_args(argv)

    if not hasattr(ns, "_handler"):
        # Unknown command shape => legacy forward (muscle-memory safe).
        return _run_legacy(argv)

    return int(ns._handler(ns))  # type: ignore[misc]
