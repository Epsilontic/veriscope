#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from veriscope.core.pilot_calibration import CalibrationError, calibrate_pilot, render_calibration_md
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score pilot control/injected runs.")
    parser.add_argument("--control-dir", required=True, help="Control capsule OUTDIR")
    parser.add_argument("--injected-dir", required=True, help="Injected capsule OUTDIR")
    parser.add_argument("--out", default="calibration.json", help="Output JSON path")
    parser.add_argument("--out-md", default="calibration.md", help="Output Markdown path")
    args = parser.parse_args(argv)

    try:
        output = calibrate_pilot(Path(args.control_dir), Path(args.injected_dir))
        out_path = Path(args.out)
        out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
        Path(args.out_md).write_text(render_calibration_md(output), encoding="utf-8")
    except CalibrationError as exc:
        sys.stderr.write(f"{exc.token}: {exc.message}\n")
        return 2
    except Exception as exc:
        sys.stderr.write(f"ERROR:PILOT_SCORE_INTERNAL {exc.__class__.__name__}: {exc}\n")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
