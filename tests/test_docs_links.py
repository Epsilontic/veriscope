from __future__ import annotations

from pathlib import Path
import re

import pytest

pytestmark = pytest.mark.unit


DOCS_ROOT = Path("docs")
_LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _normalize_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    if " " in target:
        target = target.split(" ", 1)[0]
    return target


def test_docs_relative_links_exist() -> None:
    md_files = sorted(DOCS_ROOT.rglob("*.md"))
    assert md_files, "Expected markdown files under docs/."

    violations: list[str] = []

    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        for match in _LINK_PATTERN.finditer(content):
            raw_target = match.group(1)
            target = _normalize_target(raw_target)

            if "://" in target or target.startswith("mailto:"):
                continue
            if not (target.startswith("./") or target.startswith("../")):
                continue

            path_part = target.split("#", 1)[0]
            if not path_part:
                continue

            resolved = (md_file.parent / path_part).resolve()
            if resolved.exists():
                continue

            violations.append(
                f"{md_file}: link target '{target}' not found (resolved: {resolved})"
            )

    assert not violations, "Broken relative docs links:\n" + "\n".join(violations)
