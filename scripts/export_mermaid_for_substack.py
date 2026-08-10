#!/usr/bin/env python3
"""Export Mermaid fences from Markdown as Substack-ready PNG images.

Examples:
    python3 scripts/export_mermaid_for_substack.py --dry-run
    python3 scripts/export_mermaid_for_substack.py
    python3 scripts/export_mermaid_for_substack.py docs/site/jax/policy.md

The exporter prefers an installed ``mmdc`` executable. If one is unavailable,
it runs a pinned ``@mermaid-js/mermaid-cli`` release through ``npx``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "docs" / "site"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "substack-mermaid"
DEFAULT_NPX_PACKAGE = "@mermaid-js/mermaid-cli@10.9.1"
FENCE_START = re.compile(r"^[ \t]*(`{3,}|~{3,})mermaid[ \t]*$", re.IGNORECASE)


@dataclass(frozen=True)
class Diagram:
    source: Path
    line: int
    index: int
    text: str


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def extract_mermaid(source: Path) -> list[Diagram]:
    """Extract closed Mermaid fences from one Markdown file."""
    lines = source.read_text(encoding="utf-8").splitlines()
    diagrams: list[Diagram] = []
    start_line: int | None = None
    closing_fence = ""
    body: list[str] = []

    for line_number, line in enumerate(lines, start=1):
        if start_line is None:
            match = FENCE_START.match(line)
            if match:
                start_line = line_number
                closing_fence = match.group(1)[0] * len(match.group(1))
                body = []
            continue

        if line.strip() == closing_fence:
            diagrams.append(
                Diagram(
                    source=source,
                    line=start_line,
                    index=len(diagrams) + 1,
                    text="\n".join(body).strip() + "\n",
                )
            )
            start_line = None
            closing_fence = ""
            body = []
        else:
            body.append(line)

    if start_line is not None:
        raise ValueError(
            f"{_display_path(source)}:{start_line}: unclosed Mermaid fence"
        )
    return diagrams


def markdown_files(inputs: Sequence[Path]) -> list[Path]:
    """Resolve files and directories into a sorted, de-duplicated file list."""
    resolved: set[Path] = set()
    for item in inputs:
        path = item if item.is_absolute() else REPO_ROOT / item
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Markdown source does not exist: {path}")
        if path.is_dir():
            resolved.update(candidate.resolve() for candidate in path.rglob("*.md"))
        elif path.suffix.lower() == ".md":
            resolved.add(path)
        else:
            raise ValueError(f"Expected a Markdown file or directory: {path}")
    return sorted(resolved, key=_display_path)


def output_stem(diagram: Diagram) -> str:
    """Create a stable name from source path and fence position."""
    try:
        relative = diagram.source.relative_to(DEFAULT_SOURCE)
    except ValueError:
        try:
            relative = diagram.source.relative_to(REPO_ROOT)
        except ValueError:
            relative = Path(diagram.source.name)
    path_part = "__".join(relative.with_suffix("").parts)
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", path_part).strip("-").lower()
    return f"{slug}--diagram-{diagram.index:02d}"


def renderer_command(explicit_mmdc: str | None, npx_package: str) -> list[str]:
    if explicit_mmdc:
        executable = shutil.which(explicit_mmdc) or explicit_mmdc
        return [executable]

    installed = shutil.which("mmdc")
    if installed:
        return [installed]

    npx = shutil.which("npx")
    if not npx:
        raise RuntimeError(
            "No Mermaid renderer found. Install @mermaid-js/mermaid-cli or install Node.js/npx."
        )
    return [npx, "--yes", "--package", npx_package, "mmdc"]


def render_diagram(
    diagram: Diagram,
    destination: Path,
    renderer: Sequence[str],
    *,
    scale: float,
    width: int,
    theme: str,
    background: str,
    no_browser_sandbox: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="basketworld-mermaid-") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "diagram.mmd"
        output_path = temp_root / "diagram.png"
        input_path.write_text(diagram.text, encoding="utf-8")
        command = [
            *renderer,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--scale",
            str(scale),
            "--width",
            str(width),
            "--theme",
            theme,
            "--backgroundColor",
            background,
        ]
        if no_browser_sandbox:
            puppeteer_config = temp_root / "puppeteer-config.json"
            puppeteer_config.write_text(
                json.dumps({"args": ["--no-sandbox"]}) + "\n",
                encoding="utf-8",
            )
            command.extend(["--puppeteerConfigFile", str(puppeteer_config)])
        subprocess.run(command, check=True)
        if not output_path.is_file():
            raise RuntimeError(f"Mermaid renderer did not create {output_path}")
        output_path.replace(destination)


def collect_diagrams(files: Iterable[Path]) -> list[Diagram]:
    diagrams: list[Diagram] = []
    for source in files:
        diagrams.extend(extract_mermaid(source))
    return diagrams


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Mermaid fences from MkDocs Markdown as high-resolution PNGs."
    )
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        default=[DEFAULT_SOURCE],
        help="Markdown files or directories (default: docs/site)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG destination (default: artifacts/substack-mermaid)",
    )
    parser.add_argument(
        "--scale", type=float, default=2.0, help="PNG scale factor (default: 2)"
    )
    parser.add_argument(
        "--width", type=int, default=1200, help="Render viewport width (default: 1200)"
    )
    parser.add_argument(
        "--theme",
        choices=("default", "neutral", "dark", "forest"),
        default="neutral",
        help="Mermaid theme (default: neutral)",
    )
    parser.add_argument(
        "--background",
        default="white",
        help="PNG background color (default: white; use transparent if desired)",
    )
    parser.add_argument("--mmdc", help="Explicit mmdc executable path")
    parser.add_argument(
        "--no-browser-sandbox",
        action="store_true",
        help="Pass --no-sandbox to Chromium when the host disables browser sandboxing",
    )
    parser.add_argument(
        "--npx-package",
        default=DEFAULT_NPX_PACKAGE,
        help=f"Fallback npx package (default: {DEFAULT_NPX_PACKAGE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered diagrams and output names without invoking Mermaid CLI",
    )
    args = parser.parse_args(argv)
    if args.scale <= 0:
        parser.error("--scale must be greater than zero")
    if args.width <= 0:
        parser.error("--width must be greater than zero")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        files = markdown_files(args.sources)
        diagrams = collect_diagrams(files)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not diagrams:
        print("No Mermaid diagrams found.")
        return 0

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    outputs = [
        (diagram, output_dir / f"{output_stem(diagram)}.png") for diagram in diagrams
    ]

    if args.dry_run:
        for diagram, destination in outputs:
            print(
                f"{_display_path(diagram.source)}:{diagram.line} -> {_display_path(destination)}"
            )
        print(f"Found {len(outputs)} Mermaid diagram(s).")
        return 0

    try:
        renderer = renderer_command(args.mmdc, args.npx_package)
        for diagram, destination in outputs:
            print(
                f"Rendering {_display_path(diagram.source)}:{diagram.line} -> {_display_path(destination)}"
            )
            render_diagram(
                diagram,
                destination,
                renderer,
                scale=args.scale,
                width=args.width,
                theme=args.theme,
                background=args.background,
                no_browser_sandbox=args.no_browser_sandbox,
            )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Exported {len(outputs)} PNG(s) to {_display_path(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
