# tests/test_redaction_policy.py
from __future__ import annotations

from veriscope.core.redaction_policy import prepare_env_capture, redact_argv


def test_prepare_env_capture_deny_wins() -> None:
    env = {
        "VERISCOPE_API_KEY": "secret",
        "VERISCOPE_OK": "1",
    }
    env_safe, env_capture = prepare_env_capture(env)
    assert "VERISCOPE_API_KEY" not in env_safe
    assert env_safe["VERISCOPE_OK"] == "1"
    assert env_capture["redactions_applied"] is True


def test_redact_argv_redacts_both_forms() -> None:
    argv = ["--api_key=secret", "--api-key", "abc123", "--token", "abc123", "--other", "value"]
    safe_argv, redacted = redact_argv(argv)
    assert safe_argv == [
        "--api_key=[REDACTED]",
        "--api-key",
        "[REDACTED]",
        "--token",
        "[REDACTED]",
        "--other",
        "value",
    ]
    assert len(safe_argv) == len(argv)
    assert redacted is True


def test_prepare_env_capture_redactions_toggle() -> None:
    env = {"VERISCOPE_SAFE": "1"}
    _, env_capture = prepare_env_capture(env)
    assert env_capture["redactions_applied"] is False
