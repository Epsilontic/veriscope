from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Optional

import pytest

pytestmark = pytest.mark.unit


def _has_fence(line: str) -> bool:
    return line.lstrip().startswith("```")


def _tracked_markdown_files() -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "--", "*.md"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []

    tracked = [Path(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
    return sorted(path for path in tracked if path.exists())


def test_markdown_fences_are_balanced() -> None:
    md_files = _tracked_markdown_files()
    if not md_files:
        md_files = sorted(Path(".").rglob("*.md"))
    assert md_files, "Expected markdown files in repository."

    failures: list[str] = []
    for md_file in md_files:
        open_fence_line: Optional[int] = None
        for lineno, line in enumerate(md_file.read_text(encoding="utf-8").splitlines(), start=1):
            if not _has_fence(line):
                continue
            if open_fence_line is None:
                open_fence_line = lineno
            else:
                open_fence_line = None

        if open_fence_line is not None:
            failures.append(f"{md_file}:{open_fence_line}")

    assert not failures, "Unmatched ``` fence(s) in repo markdown:\n" + "\n".join(failures)
