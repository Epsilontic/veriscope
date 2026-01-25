"""Shared redaction policy for provenance capture (deny wins).

This module defines the stable, write-time redaction rules used when emitting
run_config_resolved.json and related artifacts. The policy is intentionally
conservative, explicit, and stable across CLI and runner contexts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Tuple

POLICY_REV = "v1"

# Allowlist rules (explicit, stable).
ENV_ALLOWLIST_PREFIXES = ("SCAR_", "VERISCOPE_", "NANOGPT_", "CUDA_", "CUBLAS_")
ENV_ALLOWLIST_KEYS = {"CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED"}

# Denylist regexes for secret-like environment keys (deny wins).
ENV_DENY_REGEXES = [
    r"(?:^|_)(?:API_KEY|ACCESS_KEY|SECRET_KEY|PRIVATE_KEY|SIGNING_KEY|TOKEN|PASSWORD|AUTH|BEARER|SESSION|COOKIE)(?:$|_)",
]

# Denylist regexes for argv flags (matches the flag key, without leading dashes).
ARGV_DENY_REGEXES = [
    r"(?:^|_)(?:API_KEY|ACCESS_KEY|SECRET_KEY|PRIVATE_KEY|SIGNING_KEY|TOKEN|PASSWORD|AUTH|BEARER|SESSION|COOKIE)(?:$|_)",
]

_ENV_DENY_PATTERNS = [re.compile(pat, re.IGNORECASE) for pat in ENV_DENY_REGEXES]
_ARGV_DENY_PATTERNS = [re.compile(pat, re.IGNORECASE) for pat in ARGV_DENY_REGEXES]


def _matches_any(key: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(p.search(key) for p in patterns)


def _allowlisted_env_key(key: str) -> bool:
    if key in ENV_ALLOWLIST_KEYS:
        return True
    return any(key.startswith(prefix) for prefix in ENV_ALLOWLIST_PREFIXES)


def default_env_capture() -> Dict[str, Any]:
    """Return default env_capture metadata (deny wins, stable policy)."""
    allowlist = [f"prefix:{prefix}" for prefix in ENV_ALLOWLIST_PREFIXES] + [
        f"key:{key}" for key in sorted(ENV_ALLOWLIST_KEYS)
    ]
    return {
        "policy_rev": POLICY_REV,
        "allowlist": allowlist,
        "denylist": list(ENV_DENY_REGEXES),
        "redactions_applied": False,
    }


def prepare_env_capture(original_env: Mapping[str, str]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Return a sanitized env dict and capture metadata (deny wins)."""
    env_safe: Dict[str, str] = {}
    redactions_applied = False

    for key, value in original_env.items():
        if _matches_any(key, _ENV_DENY_PATTERNS):
            redactions_applied = True
            continue
        if _allowlisted_env_key(key):
            env_safe[str(key)] = str(value)
        else:
            redactions_applied = True

    env_safe = dict(sorted(env_safe.items()))
    env_capture = default_env_capture()
    env_capture["redactions_applied"] = redactions_applied
    return env_safe, env_capture


def redact_argv(argv_list: list[str]) -> Tuple[list[str], bool]:
    """Redact secret-like argv values while preserving token shape."""
    safe_args: list[str] = []
    redactions_applied = False
    idx = 0

    while idx < len(argv_list):
        arg = argv_list[idx]
        if arg.startswith("--") and "=" in arg:
            key, _value = arg[2:].split("=", 1)
            key_norm = key.replace("-", "_")
            if _matches_any(key_norm, _ARGV_DENY_PATTERNS):
                safe_args.append(f"--{key}=[REDACTED]")
                redactions_applied = True
            else:
                safe_args.append(arg)
            idx += 1
            continue

        if arg.startswith("--"):
            key = arg[2:]
            key_norm = key.replace("-", "_")
            if _matches_any(key_norm, _ARGV_DENY_PATTERNS) and idx + 1 < len(argv_list):
                safe_args.append(arg)
                safe_args.append("[REDACTED]")
                redactions_applied = True
                idx += 2
                continue

        safe_args.append(arg)
        idx += 1

    return safe_args, redactions_applied
