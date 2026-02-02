# veriscope/cli/calibrate.py
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from veriscope.core.pilot_calibration import CalibrationError, calibrate_pilot, render_calibration_md


def run_calibrate(args: Any) -> int:
    control_dir = Path(str(args.control_dir)).expanduser()
    injected_dir = Path(str(args.injected_dir)).expanduser()
    out_path = Path(str(args.out)).expanduser()
    out_md_path = Path(str(args.out_md)).expanduser()

    try:
        output: Dict[str, Any] = calibrate_pilot(control_dir, injected_dir)
        out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
        out_md_path.write_text(render_calibration_md(output), encoding="utf-8")
    except CalibrationError as exc:
        sys.stderr.write(f"{exc.token}: {exc.message}\n")
        return 2
    except Exception as exc:
        sys.stderr.write(f"ERROR:CALIBRATE_INTERNAL {exc.__class__.__name__}: {exc}\n")
        return 3

    print(f"WROTE: {out_path} {out_md_path}")
    return 0
