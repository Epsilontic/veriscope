# tests/test_cli_sanitizer.py
from __future__ import annotations

from veriscope.cli import main as cli_main


def test_sanitize_run_config_payload_redacts_env_and_argv() -> None:
    payload = {
        "env": {"VERISCOPE_API_KEY": "secret", "VERISCOPE_OK": "1"},
        "argv": ["--api_key=secret", "--other", "value"],
    }
    sanitized = cli_main._sanitize_run_config_payload(payload)
    assert "VERISCOPE_API_KEY" not in sanitized.get("env", {})
    assert sanitized["env"]["VERISCOPE_OK"] == "1"
    assert sanitized["argv"][0] == "--api_key=[REDACTED]"
    env_capture = sanitized.get("env_capture", {})
    assert env_capture.get("env_present") is True
    assert env_capture.get("redactions_applied") is True


def test_sanitize_run_config_payload_redacts_dict_argv() -> None:
    payload = {
        "env": {"VERISCOPE_OK": "1"},
        "argv": {
            "veriscope_argv": ["--token", "secret"],
            "runner_cmd": ["--api_key=secret", "--other", "value"],
        },
    }
    sanitized = cli_main._sanitize_run_config_payload(payload)
    argv = sanitized["argv"]
    assert argv["veriscope_argv"][1] == "[REDACTED]"
    assert argv["runner_cmd"][0] == "--api_key=[REDACTED]"
    assert sanitized["env"]["VERISCOPE_OK"] == "1"
    env_capture = sanitized.get("env_capture", {})
    assert env_capture.get("env_present") is True
    assert env_capture.get("redactions_applied") is True
