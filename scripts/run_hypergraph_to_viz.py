"""Build hypergraph JSON and HTML visualization from text/markdown files.

Usage
-----
Single file:
    python scripts/run_hypergraph_to_viz.py --input doc.md

Batch (all .md in a folder):
    python scripts/run_hypergraph_to_viz.py --doc-data-dir Data/

From module:
    python -m scripts.run_hypergraph_to_viz --input doc.md
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def resolve_path(path_value: str, base_dir: Path) -> Path:
    """Resolve *path_value* relative to *base_dir* (absolute paths pass through)."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def collect_markdown_files(input_dir: Path) -> list[Path]:
    """Collect .md files from *input_dir* (flat or one-deep subdirectories)."""
    docs = sorted(input_dir.glob("*.md"))
    if docs:
        return docs
    for folder in sorted(d for d in input_dir.iterdir() if d.is_dir()):
        candidate = folder / f"{folder.name}.md"
        if candidate.exists():
            docs.append(candidate)
    return sorted(docs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build hypergraph JSON and HTML visualization from text/markdown files."
    )
    # Input  (one of --input or --doc-data-dir is required)
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input", "-i", default=None, help="Path to a single .md file")
    inp.add_argument("--doc-data-dir", default=None, help="Folder containing .md files (batch mode)")

    # Output
    p.add_argument("--output-dir", "-o", default="Output/HyperGraph", help="Output directory")
    p.add_argument("--overwrite", action="store_true", help="Rebuild even if output already exists")

    # Chunking
    p.add_argument("--chunk-size", type=int, default=2000, help="Chunk size in characters")
    p.add_argument("--chunk-overlap", type=int, default=0, help="Overlap in characters")

    # Parallelism
    p.add_argument("--max-workers", type=int, default=4, help="Parallel LLM workers per chunk batch")

    # Prompt config
    p.add_argument("--prompt-config", default=None, help="Path to custom prompt_config.json")

    # Override .env (optional)
    p.add_argument("--llm-url", default=None, help="LLM server URL (overrides URL env var)")
    p.add_argument("--llm-model", default=None, help="LLM model name (overrides MODEL_NAME env var)")
    p.add_argument("--llm-temperature", type=float, default=None, help="LLM temperature")
    p.add_argument("--embed-url", default=None, help="Embedding server URL (overrides EMBED_URL env var)")
    p.add_argument("--embed-model", default=None, help="Embedding model name (overrides EMBED_MODEL env var)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.prompt_config:
        os.environ["GRAPH_REASONING_PROMPT_CONFIG"] = str(
            resolve_path(args.prompt_config, _REPO_ROOT)
        )

    from GraphReasoning.llm_client import create_llm, generate_structured
    from GraphReasoning.graph_generation import (
        make_hypergraph_from_text, cleanup_cache_dir, HypergraphJSON,
    )
    from GraphReasoning.hypergraph_store import HypergraphBuilder
    from GraphReasoning.hypergraph_viz import visualize_hypergraph
    from GraphReasoning.prompt_config import get_prompt

    output_dir = resolve_path(args.output_dir, _REPO_ROOT)
    os.makedirs(output_dir, exist_ok=True)

    # Resolve input(s)
    if args.input:
        docs = [resolve_path(args.input, _REPO_ROOT)]
    else:
        docs = collect_markdown_files(resolve_path(args.doc_data_dir, _REPO_ROOT))

    if not docs:
        logger.error("No input documents found.")
        sys.exit(1)

    # Initialize LLM client (with optional CLI overrides)
    llm_overrides = {}
    if args.llm_url:
        llm_overrides["base_url"] = args.llm_url
    if args.llm_model:
        llm_overrides["model"] = args.llm_model
    if args.llm_temperature is not None:
        llm_overrides["temperature"] = args.llm_temperature
    client = create_llm(**llm_overrides)

    def generate(
        system_prompt: str | None = None,
        prompt: str = "",
        response_model=HypergraphJSON,
        **_: dict,
    ):
        return generate_structured(
            client,
            system_prompt or get_prompt("runtime", "viz_system_prompt"),
            prompt,
            response_model,
        )

    logger.info("Processing %d document(s) -> %s", len(docs), output_dir)

    for i, doc_path in enumerate(docs):
        if not doc_path.exists():
            logger.error("Input file not found: %s", doc_path)
            continue

        title = doc_path.stem
        graph_root = f"{i}_{title[:100]}"
        json_path = output_dir / f"{graph_root}.json"
        html_path = output_dir / f"{graph_root}.html"

        if not args.overwrite and json_path.exists() and html_path.exists():
            logger.info("[skip] %s (output exists, use --overwrite to rebuild)", title)
            continue

        txt = doc_path.read_text(encoding="utf-8")
        logger.info("[build] %s (%d chars)", title, len(txt))

        out_json, builder, _, _ = make_hypergraph_from_text(
            txt,
            generate,
            generate_figure=None,
            image_list=None,
            graph_root=graph_root,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            do_distill=False,
            do_relabel=False,
            repeat_refine=0,
            verbatim=False,
            data_dir=str(output_dir),
            force_rebuild=args.overwrite,
            max_workers=args.max_workers,
        )

        if not isinstance(builder, HypergraphBuilder):
            builder = HypergraphBuilder.load(out_json)

        visualize_hypergraph(builder, output_html=html_path)
        logger.info("[ok] json=%s | html=%s", out_json, html_path)

    cleanup_cache_dir()
    logger.info("Done!")


if __name__ == "__main__":
    main()
