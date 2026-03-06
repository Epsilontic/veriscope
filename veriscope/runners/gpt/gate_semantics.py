from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Tuple


def canonicalize_runner_gate_flags(*, ok: Any, warn: Any, audit: Mapping[str, Any] | None) -> Tuple[bool, bool]:
    """
    Keep runner-emitted gate flags aligned with core GateEngine semantics.

    The runner may normalize audit payloads and reasons, but it must not rewrite
    the underlying pass/warn/fail decision policy.
    """
    ok_b = bool(ok)
    warn_b = bool(warn)
    evaluated = bool((audit or {}).get("evaluated", True))
    if evaluated and not ok_b:
        warn_b = False
    return ok_b, warn_b
