# Parallelism Overview

This repository now parallelizes LLM-heavy steps in both pipelines while keeping shared state updates deterministic and thread-safe.

## What is parallelized

### RAPTOR pipeline (`GraphReasoning/raptor_tree.py`)
- Parallel section: per-cluster summarization inside `build_raptor_index(...)`.
- Implementation: `ThreadPoolExecutor(max_workers=...)` + `as_completed(...)`.
- Each worker does cluster-local work:
  1. token budget check
  2. optional `_recluster_if_needed(...)`
  3. `summarize_cluster(...)`
  4. parent node + parent→child edge construction
- Thread safety: workers return results; `index.nodes` and `index.edges` are updated only in the main thread after futures complete.

### Hypergraph pipeline (`GraphReasoning/graph_generation.py`)
- `df2hypergraph(...)`: one `hypergraphPrompt(...)` call per chunk is submitted to a thread pool.
- `df2Graph(...)`: one `graphPrompt(...)` call per chunk is submitted to a thread pool.
- Thread safety:
  - `df2hypergraph`: collect `HypergraphBuilder` chunk results first, then merge sequentially.
  - `df2Graph`: collect subgraphs first, then compose sequentially with `nx.compose`.
- Existing per-chunk `try/except` behavior is preserved (failed chunks are skipped, pipeline continues).

## What remains intentionally sequential

- RAPTOR leaf/level embedding via `embed_nodes(...)` at level entry.
- Clustering steps (`_two_step_cluster`, `cluster_nodes`, `_reduce_umap`).
- Final merge/compose phases after threaded extraction/summarization.

## Progress bars

`tqdm` bars were added for:
- RAPTOR depth loop (`RAPTOR levels`)
- RAPTOR per-level summarization (`Summarizing level X`)
- RAPTOR embedding loop (`Embedding nodes`)
- Hypergraph extraction (`Extracting hypergraph chunks`)
- Graph extraction (`Extracting graph chunks`)

## CLI controls

### RAPTOR
- `scripts/run_raptor_build.py`
- New flag: `--max-workers` (default: `4`)

### Hypergraph visualization run
- `scripts/run_hypergraph_to_viz.py`
- New flag: `--max-workers` (default: `4`)
- Default chunk size updated to `--chunk-size 2000`

### New/merge hypergraph run
- `scripts/run_make_new_hypergraph.py`
- New flag: `--max-workers` (default: `4`)

## Tuning guidance

- Start with `--max-workers 4`.
- Increase gradually (`6`, `8`) only if your LLM/embedding endpoints and machine can handle more concurrency.
- If you see more 5xx/timeouts, reduce workers and/or chunk size.
- Cache behavior is unchanged: chunk-level caches are still read/written per chunk ID, so repeated runs can reuse prior outputs.
