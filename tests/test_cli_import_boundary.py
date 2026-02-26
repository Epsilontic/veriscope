from __future__ import annotations

import json
import pkgutil
import subprocess
import sys
from typing import Any, Iterable

import pytest
import veriscope.core

pytestmark = pytest.mark.unit


_RUNNER_HEAVY_PREFIXES = (
    "torch",
    "torchvision",
    "transformers",
    "datasets",
    "accelerate",
    "veriscope.runners",
)


def _probe_import_side_effects(imports: Iterable[str]) -> dict[str, Any]:
    code = f"""
import importlib, json, sys
imports = {list(imports)!r}
blocked_prefixes = {_RUNNER_HEAVY_PREFIXES!r}

for module_name in imports:
    importlib.import_module(module_name)

loaded_blocked = sorted(
    module_name
    for module_name in sys.modules
    if any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in blocked_prefixes
    )
)

print(json.dumps({{"loaded_blocked": loaded_blocked, "numpy_loaded": "numpy" in sys.modules}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_cli_main_import_stays_thin() -> None:
    payload = _probe_import_side_effects(["veriscope.cli.main"])
    assert payload["loaded_blocked"] == []
    assert payload["numpy_loaded"] is False


def test_core_imports_do_not_pull_runner_heavy_dependencies() -> None:
    discovered_submodules = sorted(module_info.name for module_info in pkgutil.iter_modules(veriscope.core.__path__))
    imports = ["veriscope.core"] + [f"veriscope.core.{name}" for name in discovered_submodules]
    payload = _probe_import_side_effects(imports)
    assert payload["loaded_blocked"] == []
