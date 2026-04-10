"""Convert a PDF file to a single Markdown file.

Usage
-----
    python scripts/pdf2markdown.py --input document.pdf
    python scripts/pdf2markdown.py --input document.pdf --output-dir Output/
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from pymupdf4llm import to_markdown
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def resolve_path(path_value: str, base_dir: Path) -> Path:
    """Resolve *path_value* relative to *base_dir* (absolute paths pass through)."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a PDF file to chunked Markdown.")
    p.add_argument("--input", "-i", required=True, help="Path to the input PDF file")
    p.add_argument("--output-dir", "-o", default=None, help="Output directory (default: same dir as input)")
    p.add_argument("--chunk-size", type=int, default=1000, help="Words per chunk (default: 1000)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parent.parent
    input_path = resolve_path(args.input, workspace_root)

    if not input_path.is_file():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info("Converting PDF to markdown: %s", input_path)
    md_text = to_markdown(str(input_path))

    chunks = chunk_text(md_text, args.chunk_size)

    out_dir = resolve_path(args.output_dir, workspace_root) if args.output_dir else input_path.parent
    os.makedirs(out_dir, exist_ok=True)

    final_path = out_dir / f"{input_path.stem}.md"
    with open(final_path, "w", encoding="utf-8") as f:
        for chunk in tqdm(chunks, desc="Writing chunks"):
            f.write(chunk + "\n\n---\n\n")

    logger.info("Wrote %d chunks merged into %s", len(chunks), final_path)


if __name__ == "__main__":
    main()
