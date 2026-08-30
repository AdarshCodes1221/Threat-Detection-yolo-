#!/usr/bin/env python3
"""Validate required files, local Markdown references, and SVG text safety."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "README.md",
    "LICENSE",
    "NOTICE.md",
    "CITATION.cff",
    "config/data.yaml",
    "config/experiments.yaml",
    "docs/DATASET_CARD.md",
    "docs/MODEL_CARD.md",
    "docs/REPRODUCIBILITY.md",
    "docs/ETHICS_AND_LIMITATIONS.md",
}
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_REFERENCE_PATTERN = re.compile(r'(?:src|href)="([^"]+)"')


def local_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().split("#", 1)[0]
    if not target or target.startswith(("http://", "https://", "mailto:", "tel:")):
        return None
    return (source.parent / unquote(target)).resolve()


def main() -> int:
    errors: list[str] = []

    for relative in sorted(REQUIRED):
        if not (ROOT / relative).is_file():
            errors.append(f"Missing required file: {relative}")

    for markdown in ROOT.rglob("*.md"):
        text = markdown.read_text(encoding="utf-8")
        references = LINK_PATTERN.findall(text) + HTML_REFERENCE_PATTERN.findall(text)
        for raw_target in references:
            target = local_target(markdown, raw_target)
            if target is not None and not target.exists():
                errors.append(
                    f"Broken local reference in {markdown.relative_to(ROOT)}: {raw_target}"
                )

    for svg in ROOT.rglob("*.svg"):
        data = svg.read_bytes()
        try:
            data.decode("ascii")
        except UnicodeDecodeError:
            errors.append(f"SVG contains non-ASCII text: {svg.relative_to(ROOT)}")
        if b"<script" in data.lower():
            errors.append(f"SVG contains a script element: {svg.relative_to(ROOT)}")

    if errors:
        print("Repository validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Repository validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
