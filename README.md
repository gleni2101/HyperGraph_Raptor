# HyperGraph Raptor

An LLM-powered knowledge extraction system that builds hierarchical reasoning structures from unstructured text. Combines hypergraph modeling with RAPTOR (Recursive Abstractive Processing for Tree-organized Retrieval) for multi-level document understanding, semantic search, and interactive visualization.

Designed by Markus J. Buehler and Isabella Stewart at MIT.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [CLI Scripts](#cli-scripts)
  - [run_raptor_build.py](#run_raptor_buildpy)
  - [run_hypergraph_to_viz.py](#run_hypergraph_to_vizpy)
  - [pdf2markdown.py](#pdf2markdownpy)
- [GraphReasoning Package](#graphreasoning-package)
  - [llm_client.py](#llm_clientpy)
  - [prompt_config.py](#prompt_configpy)
  - [graph_generation.py](#graph_generationpy)
  - [graph_tools.py](#graph_toolspy)
  - [graph_analysis.py](#graph_analysispy)
  - [hypergraph_store.py](#hypergraph_storepy)
  - [hypergraph_viz.py](#hypergraph_vizpy)
  - [raptor_tree.py](#raptor_treepy)
  - [raptor_export.py](#raptor_exportpy)
  - [raptor_retrieval.py](#raptor_retrievalpy)
  - [raptor_viz.py](#raptor_vizpy)
  - [utils.py](#utilspy)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Prompt Configuration](#prompt-configuration)
- [Installation](#installation)
- [Sample Data](#sample-data)
- [Troubleshooting](#troubleshooting)

---

## Overview

HyperGraph Raptor extracts n-ary relationships (hyperedges) from text documents using LLMs, building knowledge structures where a single relationship can connect multiple entities simultaneously. Unlike traditional knowledge graphs limited to binary (subject-predicate-object) triples, hypergraphs capture the full complexity of real-world relationships.

The system provides two complementary pipelines:

```
Documents (MD/PDF)
      |
      v
 Text Splitting (chunking)
      |
      +-----------------------------+
      |                             |
      v                             v
  RAPTOR Pipeline              Hypergraph Pipeline
      |                             |
  Embed chunks (BGE-M3)        LLM structured extraction
      |                             |
  Two-step clustering           Parse events into
  (UMAP + GMM/BIC)             HypergraphBuilder
      |                             |
  LLM summarization             Merge across chunks
  per cluster                    (label dedup)
      |                             |
  Recursive levels               JSON persistence
  (DAG with soft edges)          (v2.0 schema)
      |                             |
      v                             v
  Output/Raptor/               Output/HyperGraph/
  (tree, DAG, nodes JSON,      (hypergraph JSON,
   embeddings, HTML viz)        HTML viz)
```

---

## Project Structure

```
HyperGraph_Raptor/
|
|-- GraphReasoning/                 # Core library package
|   |-- __init__.py                 # Public API exports
|   |-- llm_client.py              # LLM + embedding client factory, structured generation
|   |-- prompt_config.py           # Prompt template loader (reads prompt_config.json)
|   |-- graph_generation.py        # Hypergraph extraction pipeline + shared Pydantic models
|   |-- graph_tools.py             # Graph utilities: embeddings, simplification, search, stats
|   |-- graph_analysis.py          # Community detection, path finding, scale-free analysis
|   |-- hypergraph_store.py        # JSON-based hypergraph persistence (HypergraphBuilder)
|   |-- hypergraph_viz.py          # D3.js convex-hull hypergraph visualization
|   |-- raptor_tree.py             # RAPTOR index builder (chunk, embed, cluster, summarize)
|   |-- raptor_export.py           # RAPTOR serialization (tree JSON, DAG JSON, embeddings NPZ)
|   |-- raptor_retrieval.py        # RAPTOR query: collapsed-tree + tree-traversal retrieval
|   |-- raptor_viz.py              # D3.js RAPTOR tree/DAG visualization
|   |-- utils.py                   # Text cleaning and string helpers
|
|-- scripts/                       # Standalone CLI scripts (standardized interface)
|   |-- run_raptor_build.py        # Build RAPTOR hierarchical index
|   |-- run_hypergraph_to_viz.py   # Build hypergraph JSON + HTML visualization
|   |-- pdf2markdown.py            # Convert PDF to Markdown
|
|-- prompt_config.json             # All LLM prompt templates (single source of truth)
|-- .env                           # Environment variables (API keys, URLs, model config)
|-- Data/                          # Sample input documents
|-- Output/                        # All generated outputs (gitignored)
|   |-- Raptor/                    #   RAPTOR outputs per document
|   |-- HyperGraph/                #   Hypergraph outputs per document
|
|-- pyproject.toml                 # Python project metadata
|-- requirements.txt               # Python dependencies
|-- uv.lock                        # Locked dependency tree (UV)
```

---

## CLI Scripts

All scripts share a standardized CLI interface:

| Flag | Description | Available in |
|------|-------------|--------------|
| `--input, -i` | Single input file | all |
| `--doc-data-dir` | Folder of .md files (batch mode) | raptor, hypergraph |
| `--output-dir, -o` | Output directory | all |
| `--overwrite` | Force rebuild even if output exists | raptor, hypergraph |
| `--chunk-size` | Chunk size (tokens or chars) | all |
| `--chunk-overlap` | Chunk overlap | raptor, hypergraph |
| `--max-workers` | Parallel LLM workers | raptor, hypergraph |
| `--prompt-config` | Path to custom prompt_config.json | raptor, hypergraph |
| `--llm-url` | Override LLM server URL | raptor, hypergraph |
| `--llm-model` | Override LLM model name | raptor, hypergraph |
| `--llm-temperature` | Override LLM temperature | raptor, hypergraph |
| `--embed-url` | Override embedding server URL | raptor, hypergraph |
| `--embed-model` | Override embedding model name | raptor, hypergraph |

Input is always `--input -i` (single file) or `--doc-data-dir` (batch), mutually exclusive.

---

### run_raptor_build.py

Builds a RAPTOR hierarchical RAG index from one or more Markdown documents. For each document, it creates a subfolder under the output directory containing the full tree structure, DAG, node catalog, embeddings, and HTML visualizations.

```bash
# Single document
python scripts/run_raptor_build.py --input Data/Cars_Movie.md

# Batch: all .md files in a folder
python scripts/run_raptor_build.py --doc-data-dir Data/

# Custom output + chunk size
python scripts/run_raptor_build.py --input doc.md -o Output/Raptor --chunk-size 200
```

**RAPTOR-specific flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--max-depth` | 5 | Maximum number of summarization levels |
| `--min-cluster` | 3 | Minimum nodes to attempt clustering at a level |
| `--max-k` | 20 | Upper bound on GMM cluster count |
| `--membership-threshold` | 0.1 | Minimum GMM probability to assign a node to a cluster |
| `--max-context-tokens` | 4096 | Maximum tokens allowed in a single summarization call |
| `--query` | None | Run a demo retrieval query after building |

**What it produces** (per document, in `Output/Raptor/<doc_name>/`):

| File | Description |
|------|-------------|
| `raptor_nodes.json` | Full node catalog (text, metadata, token counts) |
| `raptor_tree.json` | Nested tree JSON for `d3.hierarchy` |
| `raptor_dag.json` | Flat `{nodes, links}` DAG with soft-clustering edge weights |
| `raptor_as_hypergraph.json` | RAPTOR clusters converted to hypergraph format |
| `raptor_embeddings.npz` | All node embeddings (compressed NumPy) |
| `raptor_viz.html` | Interactive D3.js tree/DAG visualization |
| `raptor_as_hypergraph_viz.html` | RAPTOR clusters rendered as hypergraph blobs |

---

### run_hypergraph_to_viz.py

Extracts n-ary relationships from Markdown documents using an LLM and produces both a structured hypergraph JSON file and an interactive HTML visualization for each document.

```bash
# Single document
python scripts/run_hypergraph_to_viz.py --input Data/Cars_Movie.md

# Batch
python scripts/run_hypergraph_to_viz.py --doc-data-dir Data/

# Force rebuild
python scripts/run_hypergraph_to_viz.py --doc-data-dir Data/ --overwrite
```

**What it produces** (in `Output/HyperGraph/`):

| File | Description |
|------|-------------|
| `<idx>_<title>.json` | Hypergraph JSON (v2.0 schema: nodes, hyperedges, metadata) |
| `<idx>_<title>.html` | Interactive D3.js visualization with convex-hull hyperedge blobs |

After processing completes, the intermediate chunk cache (`temp/`) is automatically cleaned up.

---

### pdf2markdown.py

Converts a PDF file into a single Markdown file, splitting the content into word-level chunks and merging them back with `---` separators. Used as a preprocessing step before running the main pipelines.

```bash
python scripts/pdf2markdown.py --input document.pdf
python scripts/pdf2markdown.py --input document.pdf -o Data/
```

---

## GraphReasoning Package

### llm_client.py

Centralized factory for LLM and embedding clients. All scripts and modules import from here instead of instantiating their own clients.

**Key exports:**

| Symbol | Description |
|--------|-------------|
| `create_llm(**overrides)` | Creates a LangChain `ChatOpenAI` instance configured from `.env` variables. Handles SSL, timeouts, and httpx client setup. Any keyword argument overrides the corresponding env var. |
| `create_embed_client(**overrides)` | Creates a `LocalBGEClient` for embedding text via a local BGE-M3 server. |
| `generate_structured(client, system_prompt, user_prompt, response_model, ...)` | Invokes the LLM with structured output (Pydantic model) and exponential-backoff retry. Shared by all scripts so retry logic is not duplicated. |
| `LocalBGEClient` | Embedding client that calls a local BGE-M3 server via HTTP. Supports token-limit detection with automatic input shrinking, exponential backoff on server errors, and configurable max input size (default: 19,000 chars / ~8k tokens). |

**Environment variables:** `URL`, `MODEL_NAME`, `OPENAI_API_KEY`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TIMEOUT`, `EMBED_URL`, `EMBED_MODEL`, `EMBED_MAX_CHARS`.

---

### prompt_config.py

Loads and serves prompt templates from the single `prompt_config.json` file at the repository root.

**Key exports:**

| Symbol | Description |
|--------|-------------|
| `load_prompt_config(config_path=None)` | Load the full prompt config dict. Resolves path via: explicit arg > `GRAPH_REASONING_PROMPT_CONFIG` env var > repo root default. |
| `get_prompt(section, key, **kwargs)` | Retrieve a single prompt string from `section.key`, with optional `str.format()` interpolation of `**kwargs` into the template. |

---

### graph_generation.py

The core hypergraph extraction pipeline. Takes raw text, splits it into chunks, sends each chunk to an LLM for structured n-ary relationship extraction, and merges the results into a single `HypergraphBuilder`.

**Key exports:**

| Symbol | Description |
|--------|-------------|
| `Event` / `HypergraphJSON` | Pydantic models for LLM structured output. `HypergraphJSON` has an `events` list of `Event` objects, each with `source: list[str]`, `relation: str`, `target: list[str]`. Shared by all scripts. |
| `make_hypergraph_from_text(txt, generate, ...)` | Top-level function: chunks text via `RecursiveCharacterTextSplitter`, extracts hypergraph per chunk in parallel, merges all chunks, persists as JSON. Returns `(json_path, HypergraphBuilder, None, None)`. |
| `add_new_hypersubgraph_from_text(...)` | Merges a new hypergraph into an existing integrated graph. Handles loading from JSON/pkl, label-based node deduplication, optional embedding updates, and atomic file writes with retry. |
| `df2hypergraph(df, generate, ...)` | Build a merged `HypergraphBuilder` from a DataFrame of text chunks. Uses `ThreadPoolExecutor` for parallel LLM extraction. |
| `hypergraphPrompt(input, generate, ...)` | Extract a hypergraph from a single chunk of text. Checks JSON/pkl cache first, then calls the LLM with the `hypergraph.graphmaker_system/user` prompts, parses the structured response, and caches the result. |
| `make_graph_from_text(txt, generate, ...)` | Legacy binary graph extraction (NetworkX `DiGraph`). Uses `graph.graphmaker_system/user` prompts. |
| `cleanup_cache_dir()` | Removes the intermediate chunk-extraction cache directory (`temp/` by default) after a build completes. |

**Chunk caching:** Each chunk's extracted hypergraph is cached to `{GRAPH_REASONING_CACHE_DIR}/{md5_hash}_hg.json` so re-runs skip already-processed chunks. The cache is cleaned up after the full build.

---

### graph_tools.py

The largest module (~2,600 lines). Provides graph processing utilities used by both the hypergraph and binary graph pipelines.

**Embedding functions:**

| Function | Description |
|----------|-------------|
| `generate_node_embeddings(nodes, tokenizer, model)` | Embed graph nodes using a tokenizer+model pair (HuggingFace-style) or a sentence-transformers-style `model.encode()`. |
| `generate_hypernode_embeddings(nodes, tokenizer, model)` | Same as above, but accepts HyperNetX hypergraphs or `HypergraphBuilder` node lists. |
| `update_node_embeddings(embeddings, G, tokenizer, model)` | Incrementally update embeddings dict: add new nodes, remove stale ones. |
| `update_hypernode_embeddings(embeddings, H, tokenizer, model)` | Same for hypergraph builders. |
| `save_embeddings(embeddings, path)` / `load_embeddings(path)` | Pickle-based embedding persistence. |
| `find_best_fitting_node(keyword, embeddings, tokenizer, model)` | Find the single most similar node to a keyword by cosine similarity. |
| `find_best_fitting_node_list(keyword, embeddings, tokenizer, model, N)` | Return top-N most similar nodes. |

**Visualization functions:**

| Function | Description |
|----------|-------------|
| `visualize_embeddings_2d(embeddings, ...)` | PCA projection to 2D scatter plot with labeled nodes. |
| `visualize_embeddings_2d_pretty(embeddings, ...)` | Enhanced 2D plot with KMeans coloring and sampled labels. |
| `visualize_embeddings_with_gmm_density_voronoi_and_print_top_samples(...)` | GMM density estimation with Voronoi diagram overlay and top representative nodes per cluster. |
| `make_HTML(G, graph_HTML)` | Render a NetworkX graph as a PyVis HTML file. |

**Graph simplification and cleanup:**

| Function | Description |
|----------|-------------|
| `simplify_graph(G, embeddings, tokenizer, model, similarity_threshold, ...)` | Merge semantically similar nodes using cosine similarity. Optionally uses LLM for renaming. |
| `simplify_hypergraph(H, embeddings, tokenizer, model, similarity_threshold, ...)` | Same for hypergraph builders: merges nodes with similar labels, rewrites edge references. |
| `remove_small_fragents(G, size_threshold)` | Remove connected components smaller than the threshold. |
| `remove_small_hyperfragments(H, size_threshold)` | Same for hypergraphs. |
| `simplify_node_name_with_llm(node_name, generate)` | Use LLM to suggest a clearer name for a graph node. |

**Network statistics:**

| Function | Description |
|----------|-------------|
| `graph_statistics_and_plots(G, ...)` | Comprehensive stats: degree distribution, centrality measures, density, diameter, with matplotlib plots. |
| `graph_statistics_and_plots_for_large_graphs(G, ...)` | Optimized version for large graphs, skipping expensive metrics. |
| `analyze_network(G)` | Returns a dict of basic network metrics (nodes, edges, density, diameter, avg clustering). |
| `graph_Louvain(G, ...)` | Apply Louvain community detection and annotate nodes with community IDs. |

**Search and retrieval:**

| Function | Description |
|----------|-------------|
| `extract_keywords_to_nodes(question, embeddings, tokenizer, model, generate)` | Use LLM to extract keywords from a question, then map each keyword to the closest graph node. |
| `local_search(question, G, embeddings, tokenizer, model, generate)` | Answer a question by finding shortest paths between keyword-matched nodes and synthesizing the path report with an LLM. |
| `global_search(question, G, embeddings, tokenizer, model, generate)` | Answer a question by iterating over graph communities, accumulating context, and synthesizing with an LLM. |
| `find_shortest_path_hypersubgraph_between_nodes_local(...)` | Find shortest s-paths in a hypergraph between two keywords, extract the connecting subgraph. |

**Hypergraph-specific s-centrality:**

| Function | Description |
|----------|-------------|
| `s_betweenness_centrality_GLOBAL(H, s)` | Compute s-betweenness centrality for all nodes in a HyperNetX hypergraph. |
| `s_betweenness_centrality_LOCAL(H, node_list, s)` | Compute s-betweenness centrality for a subset of nodes. |
| `s_closeness_centrality_GLOBAL(H, s)` | Compute s-closeness centrality for all nodes. |
| `detect_communities(H, s)` | Detect communities in the s-line graph of a hypergraph. |
| `summarize_communities(communities, H, generate)` | Use LLM to generate natural-language summaries of detected communities. |

---

### graph_analysis.py

Higher-level graph analysis functions built on NetworkX.

| Function | Description |
|----------|-------------|
| `find_shortest_path(G, source, target)` | Find and visualize the shortest path between two nodes. Saves HTML (PyVis) and GraphML. |
| `find_N_paths(G, source, target, N)` | Sample up to N simple paths and visualize each. |
| `find_path(G, embeddings, tokenizer, model, keyword_1, keyword_2)` | End-to-end: map keywords to nearest nodes, find embedding-guided heuristic path, visualize. |
| `heuristic_path_with_embeddings(G, ..., source, target, embeddings)` | Greedy path finder that uses embedding cosine distance as the heuristic. Selects among top-k nearest neighbors at each step. |
| `describe_communities(G, N)` | Detect top-N Louvain communities and print key nodes by degree. |
| `describe_communities_with_plots(G, N)` | Same with bar charts of community sizes and top-degree nodes. |
| `describe_communities_with_plots_complex(G, N)` | Extended version adding average degree, clustering coefficient, and betweenness centrality plots per community. |
| `is_scale_free(G)` / `is_scale_free_simple(G)` | Test if the network follows a power-law degree distribution using the `powerlaw` package. Compares power-law vs exponential fit. |
| `find_all_triplets(G)` | Find all 3-node cliques with exactly 3 edges. |

---

### hypergraph_store.py

JSON-based persistence layer for hypergraphs. Defines the on-disk schema (version 2.0) and the `HypergraphBuilder` class for incremental construction.

**Schema models (Pydantic):**

| Class | Fields | Description |
|-------|--------|-------------|
| `HyperNode` | `id`, `label`, `type` | A single entity node (default type: "concept"). |
| `HyperEdge` | `id`, `label`, `source[]`, `target[]`, `weight`, `chunk_id` | One n-ary relationship. `source` and `target` reference entity labels (not IDs). |
| `Hypergraph` | `metadata`, `nodes`, `hyperedges` | Top-level container with version, timestamp, and source document info. |

**`HypergraphBuilder` class:**

| Method | Description |
|--------|-------------|
| `add_event(relation, source, target, chunk_id, weight)` | Add one hyperedge. Nodes are created or reused automatically by exact label match (deduplication). Returns the new edge ID. |
| `merge(other)` | Merge another builder into this one. Nodes are deduplicated by label; all edges from `other` are copied. |
| `save(path)` | Serialize to JSON (v2.0 schema). Parent directories are created automatically. |
| `HypergraphBuilder.load(path)` | Class method: deserialize from JSON, rebuilding the internal label-to-ID index. |
| `node_count` / `edge_count` | Property accessors for quick stats. |
| `node_labels()` | Return all unique entity labels. |

---

### hypergraph_viz.py

Generates self-contained interactive HTML visualizations of hypergraphs using D3.js.

Each hyperedge is rendered as a **colored convex-hull blob** that encloses all its member nodes simultaneously. A dashed arrow from the source-group centroid to the target-group centroid shows the directionality of the relationship. This is the canonical hypergraph representation: a hyperedge is a *set* of nodes, drawn as an enclosing region rather than a regular edge.

**Features:**
- Force-directed node layout with drag interaction
- Convex-hull blobs colored by hyperedge (12-color cyclic palette)
- Directional arrows (source centroid -> target centroid) inside each blob
- Hover inspection panel showing edge label, sources, targets, and chunk ID
- Node search filter (dims non-matching nodes and their hyperedges)
- Zoom and pan

**Usage:** `visualize_hypergraph(source, output_html="hypergraph.html")` where `source` can be a `HypergraphBuilder`, `Hypergraph`, or path to a JSON file.

---

### raptor_tree.py

Implements the RAPTOR paper's hierarchical index builder. Constructs a tree/DAG bottom-up by repeatedly chunking, embedding, clustering, and summarizing.

**Data structures:**

| Class | Description |
|-------|-------------|
| `RaptorNode` | A node at any tree level. Fields: `id`, `level` (0=leaf), `type` ("leaf"/"summary"), `text`, `token_count`, `embedding`, `metadata`. |
| `RaptorEdge` | A parent-child edge with `source` (parent), `target` (child), and `weight` (GMM membership probability). |
| `RaptorIndex` | The full index containing `nodes` dict, `edges` list, and `max_level`. Provides accessors like `nodes_at_level()`, `children_of()`, `parents_of()`. |

**Build pipeline (`build_raptor_index`):**

1. **Chunking** (`chunk_text`): Sentence-aware, token-based splitting. Keeps sentences intact and supports configurable overlap. Default chunk size: 100 tokens (per paper).

2. **Embedding** (`embed_nodes`): Embeds each node's text using the provided `embed_client.encode()`.

3. **Two-step clustering** (`_two_step_cluster`, per RAPTOR paper Section 3):
   - Step 1 — Global UMAP (large `n_neighbors`) + GMM with BIC model selection -> coarse clusters.
   - Step 2 — For each coarse cluster, local UMAP (small `n_neighbors`) + GMM -> fine-grained clusters.
   - Soft clustering: nodes can belong to multiple clusters (probability >= `membership_threshold`), producing a DAG rather than a strict tree.

4. **Summarization** (`summarize_cluster`): Concatenates cluster member texts and sends to the LLM. Parallelized with `ThreadPoolExecutor`. If combined tokens exceed the budget, `_recluster_if_needed` recursively sub-clusters before summarizing.

5. **Recursion**: Summary nodes are re-embedded and fed back into steps 3-4 for the next level. Stops when: fewer than `min_cluster_input` nodes, single cluster produced, or `max_depth` reached.

---

### raptor_export.py

Serializes a `RaptorIndex` to multiple formats for visualization and persistence.

| Function | Output | Description |
|----------|--------|-------------|
| `export_tree_json(index, path)` | `raptor_tree.json` | Nested JSON for `d3.hierarchy`. Each child assigned to its single strongest parent (max GMM weight). Orphan leaves grouped under a synthetic root. |
| `export_dag_json(index, path)` | `raptor_dag.json` | Flat `{nodes, links}` JSON preserving all soft-clustering edges with weights. For force-directed or layered DAG layouts. |
| `export_nodes_json(index, path)` | `raptor_nodes.json` | Full node catalog with text, metadata, and token counts (no embeddings). |
| `save_embeddings_npz(index, path)` | `raptor_embeddings.npz` | All node embeddings in compressed NumPy format. |
| `raptor_to_hypergraph(index)` | `HypergraphBuilder` | Converts RAPTOR clusters to hypergraph format. Each cluster becomes a hyperedge where children are sources and the summary is the target. Enables 1-1 visual comparison between the two pipelines. |
| `export_all(index, output_dir)` | All of the above | Convenience function that exports everything into one directory. |

---

### raptor_retrieval.py

Query interface for RAPTOR indices. Implements two retrieval strategies from the RAPTOR paper.

**Strategy 1 — Collapsed Tree** (`collapsed_tree_retrieve`):
Flattens all nodes across all levels into a single pool, ranks by cosine similarity to the query, and greedily fills a token budget. This is the stronger method from the paper.

**Strategy 2 — Tree Traversal** (`tree_traverse_retrieve`):
Starts at the highest level, picks top-k nodes by similarity, expands their children, and repeats downward. The `depth` parameter controls specificity vs breadth.

Both methods support optional FAISS acceleration via the `FaissIndex` wrapper class (L2-normalized vectors with inner-product index for cosine similarity).

**High-level API:** `query_raptor(query, index, embed_client, method="collapsed")` — embeds the query, dispatches to the chosen strategy, and returns `[(RaptorNode, score)]`.

---

### raptor_viz.py

Generates self-contained interactive HTML visualizations of RAPTOR trees/DAGs using D3.js.

**Two switchable views:**
1. **Tree view** — Tidy dendrogram from the strict-tree JSON (`d3.tree()`).
2. **DAG view** — Force-directed layout with soft-clustering edges, nodes stratified by level.

**Features:**
- Node color by level (ordinal palette)
- Node radius proportional to token count
- Link thickness/opacity by membership weight
- Retrieval overlay highlighting (if a query was run)
- Search filter and weight threshold slider
- Click-to-expand subtree

---

### utils.py

Small utility module with text-processing helpers.

| Function | Description |
|----------|-------------|
| `extract(string, start, end)` | Extract substring between delimiter characters. |
| `contains_phrase(main_string, phrase)` | Check if a phrase exists in a string. |
| `make_dir_if_needed(dir_path)` | Create directory if it doesn't exist. |
| `remove_markdown_symbols(text)` | Strip Markdown formatting (headers, bold, italic, links, code blocks, lists) leaving plain text. |

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# LLM
URL=https://api.openai.com/v1          # OpenAI-compatible API base URL
MODEL_NAME=gpt-4o                      # Model identifier
OPENAI_API_KEY=sk-...                  # API key
LLM_TEMPERATURE=0                      # Sampling temperature (0 = deterministic)
LLM_MAX_TOKENS=20000                   # Max generation length
LLM_TIMEOUT=120.0                      # HTTP timeout in seconds

# Embeddings
EMBED_URL=http://127.0.0.1:8080        # Embedding server URL
EMBED_MODEL=BAAI/bge-m3                # Embedding model name
EMBED_MAX_CHARS=19000                  # Max input chars (~8k tokens for BGE-M3)
```

Optional overrides:

| Variable | Description |
|----------|-------------|
| `GRAPH_REASONING_CACHE_DIR` | Cache directory for chunk extraction intermediates (default: `temp`). Cleaned up automatically after builds. |
| `GRAPH_REASONING_PROMPT_CONFIG` | Path to a custom `prompt_config.json` file. |

### Prompt Configuration

All LLM prompts are defined in `prompt_config.json`, organized by domain:

| Domain | Keys | Purpose |
|--------|------|---------|
| `hypergraph` | `distill_system/user`, `figure_system/user`, `graphmaker_system/user` | N-ary hyperedge extraction. The `graphmaker` prompts define entity types, relation formats, extraction rules, and JSON output schema. |
| `graph` | `distill_system/user`, `figure_system/user`, `graphmaker_system/user` | Binary knowledge graph extraction (nodes + edges). |
| `raptor` | `summarize_user` | Cluster summarization prompt for RAPTOR tree levels. Rules: preserve all facts, no preamble, merge overlapping info. |
| `graph_tools` | `node_rename_*`, `community_summary_*`, `extract_keywords_*`, `local_search_*`, `global_search_*`, `query_validation_*` | Prompts for graph operations: node renaming, community summarization, keyword extraction, local/global search, and answer validation. |
| `runtime` | `default_system_prompt`, `figure_system/user_prompt`, `viz_system_prompt` | Fallback and utility prompts. |

---

## Installation

### Using UV (recommended)

```bash
git clone <repository-url>
cd HyperGraph_Raptor
uv venv
uv pip install -r requirements.txt
uv pip install -e .
```

### Using pip

```bash
git clone <repository-url>
cd HyperGraph_Raptor
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Prerequisites

- **Python 3.10+**
- **OpenAI-compatible LLM endpoint** (OpenAI API, local vLLM/Ollama, etc.)
- **BGE embedding server** running locally on port 8080 (or any compatible endpoint)
  - Recommended model: `BAAI/bge-m3` (multilingual, 8192-token context)

---

## Sample Data

The `Data/` directory includes sample documents for testing:

| File | Type |
|------|------|
| `Automate the Boring Stuff with Python.md` | Technical (programming book) |
| `Cars_Movie.md` | Narrative (movie plot) |
| `Finding_Nemo.md` | Narrative (movie plot) |

---

## Troubleshooting

**Embedding server not reachable:**
Ensure your BGE embedding server is running at the URL specified in `EMBED_URL`.

**LLM timeout errors:**
Increase `LLM_TIMEOUT` in `.env` or pass `--llm-url` with a faster endpoint.

**Missing API key:**
Set `OPENAI_API_KEY` in `.env` or as an environment variable.

**PDF conversion requires `pymupdf4llm`:**
```bash
uv pip install pymupdf4llm
```
