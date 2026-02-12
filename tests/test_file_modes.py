from __future__ import annotations

import subprocess

import pytest

pytestmark = pytest.mark.unit


def _tracked_paths_with_modes() -> list[tuple[int, str]]:
    proc = subprocess.run(
        ["git", "ls-files", "--stage"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"git ls-files failed: {proc.stderr.strip()}")

    entries: list[tuple[int, str]] = []
    for line in proc.stdout.splitlines():
        if "\t" not in line:
            continue
        left, path = line.split("\t", 1)
        parts = left.split()
        if len(parts) != 3:
            continue
        mode = int(parts[0], 8)
        entries.append((mode, path))
    return entries


def _is_guarded(path: str) -> bool:
    if path.startswith("scripts/"):
        return False
    if path.startswith("docs/"):
        return True
    return path.endswith(".json") or path.endswith(".jsonl")


def test_docs_and_artifact_json_files_are_not_executable() -> None:
    violations: list[str] = []
    for mode, path in _tracked_paths_with_modes():
        if not _is_guarded(path):
            continue
        if mode & 0o111:
            violations.append(f"{path} ({mode:06o})")

    assert not violations, (
        "Executable mode bits are not allowed for docs/ and tracked *.json/*.jsonl files "
        "outside scripts/.\n"
        "Fix with git update-index --chmod=-x <path>.\n"
        "Violations:\n" + "\n".join(violations)
    )
