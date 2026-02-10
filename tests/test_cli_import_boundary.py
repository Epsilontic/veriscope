from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_cli_main_import_does_not_load_numpy_or_torch() -> None:
    code = (
        "import json, sys; "
        "import veriscope.cli.main; "
        "print(json.dumps({'numpy': 'numpy' in sys.modules, 'torch': 'torch' in sys.modules}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload == {"numpy": False, "torch": False}
