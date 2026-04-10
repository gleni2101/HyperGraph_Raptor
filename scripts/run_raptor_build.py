"""Build a RAPTOR hierarchical RAG index from text/markdown files.

Usage
-----
Single file:
    python scripts/run_raptor_build.py --input doc.md

Batch (all .md in a folder):
    python scripts/run_raptor_build.py --doc-data-dir Data/

From module:
    python -m scripts.run_raptor_build --input doc.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from GraphReasoning.llm_client import create_llm, create_embed_client
from GraphReasoning.raptor_tree import build_raptor_index
from GraphReasoning.raptor_export import export_all, raptor_to_hypergraph
from GraphReasoning.raptor_retrieval import build_faiss_index, query_raptor
from GraphReasoning.raptor_viz import visualize_raptor
from GraphReasoning.hypergraph_viz import visualize_hypergraph

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
# LLM summarizer wrapper
# ---------------------------------------------------------------------------

def make_llm_call(**overrides) -> callable:
    """Return a ``llm_call(prompt) -> str`` function using the shared LLM."""
    llm = create_llm(**overrides)

    def call(prompt: str) -> str:
        response = llm.invoke(prompt)
        return response.content

    return call


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a RAPTOR hierarchical RAG index from text/markdown files."
    )
    # Input  (one of --input or --doc-data-dir is required)
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input", "-i", default=None, help="Path to a single .md file")
    inp.add_argument("--doc-data-dir", default=None, help="Folder containing .md files (batch mode)")
    p.add_argument("--doc-id", default="", help="Document identifier for metadata (single-file mode)")

    # Output
    p.add_argument("--output-dir", "-o", default="Output/Raptor", help="Output directory")
    p.add_argument("--overwrite", action="store_true", help="Rebuild even if output already exists")

    # Chunking
    p.add_argument("--chunk-size", type=int, default=100, help="Target chunk size in tokens (per RAPTOR paper)")
    p.add_argument("--chunk-overlap", type=int, default=0, help="Overlap in tokens")

    # Tree building
    p.add_argument("--max-depth", type=int, default=5, help="Max summarization levels")
    p.add_argument("--min-cluster", type=int, default=3, help="Min nodes to attempt clustering")
    p.add_argument("--max-k", type=int, default=20, help="Max GMM clusters per level")
    p.add_argument("--membership-threshold", type=float, default=0.1, help="Soft clustering threshold")
    p.add_argument("--max-context-tokens", type=int, default=4096, help="Max tokens per summarization call")
    p.add_argument("--max-workers", type=int, default=4, help="Parallel LLM workers per level")

    # Prompt config
    p.add_argument("--prompt-config", default=None, help="Path to custom prompt_config.json")

    # Override .env (optional)
    p.add_argument("--llm-url", default=None, help="LLM server URL (overrides URL env var)")
    p.add_argument("--llm-model", default=None, help="LLM model name (overrides MODEL_NAME env var)")
    p.add_argument("--llm-temperature", type=float, default=None, help="LLM temperature")
    p.add_argument("--embed-url", default=None, help="Embedding server URL (overrides EMBED_URL env var)")
    p.add_argument("--embed-model", default=None, help="Embedding model name (overrides EMBED_MODEL env var)")

    # Query (optional demo)
    p.add_argument("--query", default=None, help="Optional query to run after building")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Build one document
# ---------------------------------------------------------------------------

def build_one(
    input_path: Path,
    args: argparse.Namespace,
    embed_client,
    llm_call,
    output_root: Path,
) -> None:
    """Build a RAPTOR index for a single document."""
    text = input_path.read_text(encoding="utf-8")
    doc_id = args.doc_id or input_path.stem
    output_dir = output_root / input_path.stem

    if not args.overwrite and (output_dir / "raptor_nodes.json").exists():
        logger.info("[skip] %s (output exists, use --overwrite to rebuild)", input_path.stem)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Read %d characters from %s", len(text), input_path)

    # Build RAPTOR index
    logger.info("Building RAPTOR index for %s ...", input_path.stem)
    index = build_raptor_index(
        text=text,
        embed_client=embed_client,
        llm_call=llm_call,
        doc_id=doc_id,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_depth=args.max_depth,
        min_cluster_input=args.min_cluster,
        max_k=args.max_k,
        membership_threshold=args.membership_threshold,
        max_context_tokens=args.max_context_tokens,
        max_workers=args.max_workers,
    )

    logger.info(
        "Index built: %d nodes, %d edges, %d levels",
        index.node_count, index.edge_count, index.max_level,
    )

    # Export everything
    paths = export_all(index, output_dir)
    for name, p in paths.items():
        logger.info("  %s -> %s", name, p)

    # Optional demo query
    overlay = None
    if args.query:
        logger.info("Running demo query: %s", args.query)
        try:
            faiss_idx = build_faiss_index(index)
        except ImportError:
            logger.warning("FAISS not installed — using brute-force search")
            faiss_idx = None

        results = query_raptor(
            args.query, index, embed_client,
            method="collapsed",
            max_tokens=args.max_context_tokens,
            faiss_index=faiss_idx,
        )

        logger.info("Retrieved %d nodes:", len(results))
        for node, score in results:
            logger.info("  [%.4f] %s: %s", score, node.id, node.text[:80])

        overlay = {
            "retrieved_node_ids": [n.id for n, _ in results],
            "scores": [s for _, s in results],
        }

    # Visualization
    viz_path = visualize_raptor(
        index,
        output_dir / "raptor_viz.html",
        retrieval_overlay=overlay,
    )
    logger.info("RAPTOR tree viz -> %s", viz_path)

    hg_builder = raptor_to_hypergraph(index)
    hg_viz_path = visualize_hypergraph(
        hg_builder,
        output_dir / "raptor_as_hypergraph_viz.html",
    )
    logger.info("RAPTOR-as-hypergraph viz -> %s", hg_viz_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import os
    args = parse_args()

    if args.prompt_config:
        os.environ["GRAPH_REASONING_PROMPT_CONFIG"] = str(
            resolve_path(args.prompt_config, _REPO_ROOT)
        )

    output_root = resolve_path(args.output_dir, _REPO_ROOT)

    # Initialize clients
    embed_client = create_embed_client(base_url=args.embed_url, model=args.embed_model)

    llm_overrides = {}
    if args.llm_url:
        llm_overrides["base_url"] = args.llm_url
    if args.llm_model:
        llm_overrides["model"] = args.llm_model
    if args.llm_temperature is not None:
        llm_overrides["temperature"] = args.llm_temperature
    llm_call = make_llm_call(**llm_overrides)

    # Resolve input(s)
    if args.input:
        docs = [resolve_path(args.input, _REPO_ROOT)]
    else:
        docs = collect_markdown_files(resolve_path(args.doc_data_dir, _REPO_ROOT))

    if not docs:
        logger.error("No input documents found.")
        sys.exit(1)

    logger.info("Processing %d document(s) -> %s", len(docs), output_root)

    for doc_path in docs:
        if not doc_path.exists():
            logger.error("Input file not found: %s", doc_path)
            continue
        build_one(doc_path, args, embed_client, llm_call, output_root)

    logger.info("Done!")


if __name__ == "__main__":
    main()
