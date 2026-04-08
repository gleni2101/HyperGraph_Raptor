"""Clean JSON-based hypergraph storage.

On-disk schema (version 2.0)
-----------------------------
{
  "metadata": {
    "created_at": "ISO-8601",
    "source_document": "...",
    "version": "2.0"
  },
  "nodes": {
    "<id>": {"id": "<id>", "label": "...", "type": "concept"}
  },
  "hyperedges": [
    {
      "id": "<id>",
      "label": "<relation>",
      "source": ["entity label", ...],
      "target": ["entity label", ...],
      "weight": 1.0,
      "chunk_id": "..."
    }
  ]
}
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Schema models
# ---------------------------------------------------------------------------

class HyperNode(BaseModel):
    id: str
    label: str
    type: str = "concept"


class HyperEdge(BaseModel):
    id: str
    label: str
    source: list[str]
    target: list[str]
    weight: float = 1.0
    chunk_id: str = ""


class Hypergraph(BaseModel):
    metadata: dict
    nodes: dict[str, HyperNode]
    hyperedges: list[HyperEdge]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class HypergraphBuilder:
    """Incrementally build and persist a hypergraph as portable JSON.

    Nodes are deduplicated by exact label match.  Source/target lists in
    each edge reference entity labels directly (not node IDs), so the JSON
    file remains human-readable and label changes do not corrupt edges.
    """

    def __init__(self, source_document: str = ""):
        self.graph = Hypergraph(
            metadata={
                "created_at": datetime.now().isoformat(),
                "source_document": source_document,
                "version": "2.0",
            },
            nodes={},
            hyperedges=[],
        )
        self._label_to_id: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_node(self, label: str) -> str:
        label = label.strip()
        if not label:
            return ""
        if label in self._label_to_id:
            return self._label_to_id[label]
        nid = f"n_{uuid.uuid4().hex[:8]}"
        self.graph.nodes[nid] = HyperNode(id=nid, label=label)
        self._label_to_id[label] = nid
        return nid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_event(
        self,
        relation: str,
        source: list[str],
        target: list[str],
        chunk_id: str = "",
        weight: float = 1.0,
    ) -> str:
        """Add one hyperedge from an extracted LLM event.

        Nodes are created or reused automatically by label identity.
        Returns the new edge id.
        """
        clean_source = [s.strip() for s in source if s.strip()]
        clean_target = [t.strip() for t in target if t.strip()]

        if not clean_source or not clean_target:
            raise ValueError("Both source and target must contain at least one non-empty label.")

        for label in clean_source + clean_target:
            self._get_or_create_node(label)

        eid = f"e_{uuid.uuid4().hex[:8]}"
        self.graph.hyperedges.append(
            HyperEdge(
                id=eid,
                label=relation.strip(),
                source=clean_source,
                target=clean_target,
                chunk_id=chunk_id,
                weight=weight,
            )
        )
        return eid

    def merge(self, other: "HypergraphBuilder") -> None:
        """Merge *other* into this builder, deduplicating nodes by label."""
        for node in other.graph.nodes.values():
            self._get_or_create_node(node.label)

        for edge in other.graph.hyperedges:
            eid = f"e_{uuid.uuid4().hex[:8]}"
            self.graph.hyperedges.append(
                HyperEdge(
                    id=eid,
                    label=edge.label,
                    source=edge.source,
                    target=edge.target,
                    chunk_id=edge.chunk_id,
                    weight=edge.weight,
                )
            )

    def save(self, path: str | Path) -> Path:
        """Serialize to JSON. Parent directories are created automatically."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.graph.model_dump_json(indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "HypergraphBuilder":
        """Deserialize from JSON, rebuilding the label→id index."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        graph = Hypergraph(**data)
        builder = cls(source_document=graph.metadata.get("source_document", ""))
        builder.graph = graph
        # Rebuild fast lookup from the loaded nodes
        builder._label_to_id = {n.label: n.id for n in graph.nodes.values()}
        return builder

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self.graph.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.graph.hyperedges)

    def node_labels(self) -> list[str]:
        return list(self._label_to_id.keys())

    def all_members(self, edge: HyperEdge) -> list[str]:
        """Return all unique entity labels that participate in an edge."""
        return list(dict.fromkeys(edge.source + edge.target))

    def __repr__(self) -> str:
        return (
            f"HypergraphBuilder("
            f"nodes={self.node_count}, edges={self.edge_count}, "
            f"source={self.graph.metadata.get('source_document', '')})"
        )
