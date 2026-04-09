"""True hypergraph visualisation using D3.js.

Each hyperedge is rendered as a coloured convex-hull blob that encloses ALL
its member nodes simultaneously (sources AND targets).  A directed arrow from
the source-group centroid to the target-group centroid inside each blob shows
the direction of the relationship.

This is the canonical representation: a hyperedge is a *set* of nodes, not
a pair, and must be drawn as an enclosing region rather than a regular edge.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .hypergraph_store import Hypergraph, HypergraphBuilder


def visualize_hypergraph(
    source: Union["Hypergraph", "HypergraphBuilder", str, Path],
    output_html: Union[str, Path] = "hypergraph.html",
    height: str = "100vh",
    bgcolor: str = "#0f172a",
    node_color: str = "#60a5fa",
    edge_hub_color: str = "#f59e0b",  # unused – kept for compat
    arrow_color: str = "#94a3b8",
    physics: bool = True,
    show_buttons: bool = False,
) -> Path:
    """Render a true hypergraph to an interactive HTML file using D3.js.

    Each hyperedge is drawn as a coloured convex-hull blob that encloses all
    its member nodes.  A directed arrow from the source-group centroid to the
    target-group centroid shows the directionality of the relationship.
    """
    del physics, show_buttons  # kept for backward-compatible signature

    if isinstance(source, (str, Path)):
        source = HypergraphBuilder.load(source).graph
    elif isinstance(source, HypergraphBuilder):
        source = source.graph
    graph: Hypergraph = source  # type: ignore[assignment]

    label_to_id: dict[str, str] = {node.label: node.id for node in graph.nodes.values()}

    # -- Pure concept nodes (NO hub nodes) -----------------------------------
    nodes_data = [
        {"id": node.id, "label": node.label, "type": node.type}
        for node in graph.nodes.values()
    ]

    # -- Hyperedges with member sets -----------------------------------------
    hyperedges_data: list[dict] = []
    for edge in graph.hyperedges:
        source_ids = [label_to_id[l] for l in edge.source if l in label_to_id]
        target_ids = [label_to_id[l] for l in edge.target if l in label_to_id]
        # deduplicate while preserving order
        seen: set[str] = set()
        member_ids: list[str] = []
        for nid in source_ids + target_ids:
            if nid not in seen:
                seen.add(nid)
                member_ids.append(nid)
        if not member_ids:
            continue
        hyperedges_data.append({
            "id": edge.id,
            "label": edge.label,
            "source_ids": source_ids,
            "target_ids": target_ids,
            "member_ids": member_ids,
            "chunk_id": edge.chunk_id,
            "weight": edge.weight,
        })

    graph_data = {"nodes": nodes_data, "hyperedges": hyperedges_data}

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Hypergraph Visualization</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    html,body{{height:100%;background:{bgcolor};color:#e2e8f0;font-family:Segoe UI,Arial,sans-serif;overflow:hidden}}
    #canvas{{position:absolute;inset:0;width:100%;height:{height}}}
    svg{{width:100%;height:100%}}
    .panel{{
      position:absolute;z-index:20;background:rgba(15,23,42,0.88);
      border:1px solid rgba(148,163,184,0.18);border-radius:12px;
      padding:12px 14px;backdrop-filter:blur(8px);
      box-shadow:0 12px 30px rgba(0,0,0,0.3);
    }}
    #controls{{top:16px;left:16px;width:310px}}
    #details{{left:16px;bottom:16px;width:420px;max-height:220px;overflow:auto}}
    #stats{{top:16px;right:16px;min-width:200px}}
    h2{{font-size:17px;margin-bottom:8px;color:#e2e8f0}}
    .muted{{color:#94a3b8;font-size:11px}}
    .legend{{display:flex;gap:14px;margin:8px 0;font-size:11px;flex-wrap:wrap}}
    .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}}
    input{{width:100%;padding:8px 10px;border:1px solid rgba(148,163,184,0.24);border-radius:8px;background:rgba(30,41,59,0.9);color:#fff;font-size:12px;margin:6px 0}}
    .row{{margin:5px 0;font-size:12px}}
    .mono{{font-family:Consolas,monospace;word-break:break-word;font-size:11px}}
    .he-label{{fill:white;font-size:11px;pointer-events:none;font-weight:600;text-shadow:0 1px 3px rgba(0,0,0,0.8)}}
    .node-label{{fill:#e2e8f0;font-size:10px;pointer-events:none}}
    .arrow-marker{{}}
  </style>
</head>
<body>
  <div id="canvas"><svg id="svg"></svg></div>
  <div id="controls" class="panel">
    <h2>Hypergraph Viewer</h2>
    <div class="muted">Each coloured blob is a hyperedge — a set of nodes connected by one relation.<br>Arrow shows source → target direction inside the blob.</div>
    <div class="legend">
      <div><span class="dot" style="background:{node_color}"></span>Entity node</div>
      <div><span class="dot" style="background:rgba(200,150,80,0.7);border:2px solid #f59e0b"></span>Hyperedge blob</div>
    </div>
    <input id="searchInput" type="text" placeholder="Search node labels..."/>
    <div class="muted">Hover blob/node for details · Drag nodes · Scroll to zoom</div>
  </div>
  <div id="stats" class="panel"></div>
  <div id="details" class="panel">Hover over a node or hyperedge blob to inspect.</div>

  <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
  <script>
  (function(){{
    const RAW = {json.dumps(graph_data, ensure_ascii=False)};
    const NODE_R = 14;
    const PALETTE = [
      "#7c3aed","#0ea5e9","#10b981","#f59e0b","#ef4444","#ec4899",
      "#14b8a6","#f97316","#6366f1","#84cc16","#06b6d4","#a855f7"
    ];

    // ── build lookup ──────────────────────────────────────────────────────────
    const nodeById = {{}};
    RAW.nodes.forEach(n => {{ nodeById[n.id] = n; }});

    // ── D3 simulation nodes (mutable x,y) ─────────────────────────────────────
    const simNodes = RAW.nodes.map(n => ({{...n}}));
    const simNodeMap = {{}};
    simNodes.forEach(n => {{ simNodeMap[n.id] = n; }});

    // ── edges for force layout (binary links within each hyperedge member set) ─
    const simLinks = [];
    RAW.hyperedges.forEach(he => {{
      const members = he.member_ids;
      for (let i = 0; i < members.length; i++) {{
        for (let j = i+1; j < members.length; j++) {{
          simLinks.push({{ source: members[i], target: members[j], he: he.id }});
        }}
      }}
    }});

    const W = window.innerWidth, H = window.innerHeight;
    const svg = d3.select("#svg")
      .attr("viewBox", `0 0 ${{W}} ${{H}}`)
      .attr("preserveAspectRatio","xMidYMid meet");

    // marker
    svg.append("defs").append("marker")
      .attr("id","arrowhead")
      .attr("viewBox","0 -5 10 10")
      .attr("refX",18).attr("refY",0)
      .attr("markerWidth",6).attr("markerHeight",6)
      .attr("orient","auto")
      .append("path").attr("d","M0,-5L10,0L0,5").attr("fill","rgba(255,255,255,0.7)");

    const gBlobs  = svg.append("g").attr("class","blobs");
    const gArrows = svg.append("g").attr("class","arrows");
    const gLabels = svg.append("g").attr("class","he-labels");
    const gNodes  = svg.append("g").attr("class","nodes");
    const gNLabels= svg.append("g").attr("class","node-labels");

    // ── zoom / pan ────────────────────────────────────────────────────────────
    const zoom = d3.zoom().scaleExtent([0.05,5]).on("zoom", e => {{
      [gBlobs,gArrows,gLabels,gNodes,gNLabels].forEach(g => g.attr("transform", e.transform));
    }});
    svg.call(zoom);

    // ── force simulation ──────────────────────────────────────────────────────
    const sim = d3.forceSimulation(simNodes)
      .force("link", d3.forceLink(simLinks).id(d=>d.id).distance(90).strength(0.25))
      .force("charge", d3.forceManyBody().strength(-220))
      .force("center", d3.forceCenter(W/2, H/2))
      .force("collision", d3.forceCollide(NODE_R + 6))
      .on("tick", ticked)
      .on("end", () => {{
        // Pin every node once the layout has settled so the graph stops drifting
        simNodes.forEach(n => {{ n.fx = n.x; n.fy = n.y; }});
      }});

    // ── convex-hull blob ──────────────────────────────────────────────────────
    function hullPoints(ids, padding) {{
      const pts = [];
      ids.forEach(id => {{
        const n = simNodeMap[id];
        if (!n) return;
        for (let a = 0; a < 8; a++) {{
          pts.push([n.x + Math.cos(a*Math.PI/4)*padding,
                    n.y + Math.sin(a*Math.PI/4)*padding]);
        }}
      }});
      if (pts.length < 3) {{
        // single node – draw a circle
        const n = simNodeMap[ids[0]];
        if (!n) return null;
        for (let a = 0; a < 16; a++)
          pts.push([n.x + Math.cos(a*Math.PI/8)*(padding+4),
                    n.y + Math.sin(a*Math.PI/8)*(padding+4)]);
      }}
      const hull = d3.polygonHull(pts);
      return hull ? d3.line().x(d=>d[0]).y(d=>d[1]).curve(d3.curveCatmullRomClosed.alpha(0.5))(hull) : null;
    }}

    function groupCentroid(ids) {{
      let sx=0,sy=0,c=0;
      ids.forEach(id => {{
        const n = simNodeMap[id];
        if (n){{ sx+=n.x; sy+=n.y; c++; }}
      }});
      return c ? [sx/c, sy/c] : [W/2,H/2];
    }}

    // ── render elements ───────────────────────────────────────────────────────
    const highlightSet = new Set();  // ids that should be highlighted
    let searchQuery = "";

    const blobPaths = gBlobs.selectAll("path.blob")
      .data(RAW.hyperedges).enter().append("path")
      .attr("class","blob")
      .attr("fill", (d,i) => PALETTE[i % PALETTE.length] + "28")
      .attr("stroke", (d,i) => PALETTE[i % PALETTE.length])
      .attr("stroke-width", 2)
      .attr("stroke-linejoin","round")
      .style("cursor","pointer")
      .on("mouseover", showHEDetail)
      .on("mouseout", clearDetail)
      .on("click", (e,d) => e.stopPropagation());

    const arrowLines = gArrows.selectAll("line.arrow")
      .data(RAW.hyperedges).enter().append("line")
      .attr("class","arrow")
      .attr("stroke","rgba(255,255,255,0.55)")
      .attr("stroke-width",1.5)
      .attr("stroke-dasharray","5,3")
      .attr("marker-end","url(#arrowhead)");

    const heLabelTexts = gLabels.selectAll("text.he-label")
      .data(RAW.hyperedges).enter().append("text")
      .attr("class","he-label")
      .attr("text-anchor","middle")
      .attr("dy","0.35em")
      .text(d => d.label)
      .style("cursor","pointer")
      .on("mouseover", showHEDetail)
      .on("mouseout", clearDetail);

    const nodeSel = gNodes.selectAll("circle.node")
      .data(simNodes).enter().append("circle")
      .attr("class","node")
      .attr("r", NODE_R)
      .attr("fill", "{node_color}")
      .attr("stroke","#0f172a")
      .attr("stroke-width",2)
      .style("cursor","grab")
      .call(
        d3.drag()
          .on("start", (e,d) => {{ d.fx=d.x; d.fy=d.y; }})
          .on("drag",  (e,d) => {{ d.fx=e.x; d.fy=e.y; d.x=e.x; d.y=e.y; ticked(); }})
          .on("end",   (e,d) => {{ /* node stays pinned at new position */ }})
      )
      .on("mouseover", showNodeDetail)
      .on("mouseout", clearDetail);

    const nodeLabelSel = gNLabels.selectAll("text.node-label")
      .data(simNodes).enter().append("text")
      .attr("class","node-label")
      .attr("text-anchor","middle")
      .attr("dy","0.35em")
      .text(d => d.label)
      .style("pointer-events","none");

    function ticked() {{
      blobPaths.each(function(d) {{
        const path = hullPoints(d.member_ids, NODE_R + 20);
        d3.select(this).attr("d", path || "");
      }});

      arrowLines.each(function(d) {{
        const sc = groupCentroid(d.source_ids.length ? d.source_ids : d.member_ids);
        const tc = groupCentroid(d.target_ids.length ? d.target_ids : d.member_ids);
        d3.select(this)
          .attr("x1",sc[0]).attr("y1",sc[1])
          .attr("x2",tc[0]).attr("y2",tc[1]);
      }});

      heLabelTexts.each(function(d) {{
        const c = groupCentroid(d.member_ids);
        d3.select(this).attr("x",c[0]).attr("y",c[1]-NODE_R-24);
      }});

      nodeSel.attr("cx",d=>d.x).attr("cy",d=>d.y);
      nodeLabelSel.attr("x",d=>d.x).attr("y",d=>d.y);

      applySearch();
      updateStats();
    }}

    // ── detail panel ─────────────────────────────────────────────────────────
    const det = document.getElementById("details");

    function showHEDetail(e,d) {{
      det.innerHTML = `
        <div class="row"><strong style="font-size:13px">${{d.label}}</strong> <span class="muted">(hyperedge)</span></div>
        <div class="row"><strong>Sources:</strong> <span class="mono">${{d.source_ids.map(id=>nodeById[id]?.label||id).join(", ")||"—"}}</span></div>
        <div class="row"><strong>Targets:</strong> <span class="mono">${{d.target_ids.map(id=>nodeById[id]?.label||id).join(", ")||"—"}}</span></div>
        <div class="row"><strong>All members:</strong> <span class="mono">${{d.member_ids.map(id=>nodeById[id]?.label||id).join(", ")}}</span></div>
        <div class="row"><strong>Chunk:</strong> <span class="mono">${{d.chunk_id}}</span></div>
      `;
    }}

    function showNodeDetail(e,d) {{
      det.innerHTML = `
        <div class="row"><strong style="font-size:13px">${{d.label}}</strong></div>
        <div class="row"><strong>Type:</strong> <span class="mono">${{d.type||"concept"}}</span></div>
        <div class="row"><strong>ID:</strong> <span class="mono">${{d.id}}</span></div>
      `;
    }}

    function clearDetail() {{
      det.innerHTML = "Hover over a node or hyperedge blob to inspect.";
    }}

    // ── stats panel ───────────────────────────────────────────────────────────
    const statEl = document.getElementById("stats");
    function updateStats() {{
      const vis = simNodes.filter(n=>!n._hidden).length;
      statEl.innerHTML = `
        <div class="row"><strong>Nodes:</strong> ${{RAW.nodes.length}}</div>
        <div class="row"><strong>Hyperedges:</strong> ${{RAW.hyperedges.length}}</div>
        <div class="row"><strong>Visible:</strong> ${{vis}} nodes</div>
      `;
    }}

    // ── search ────────────────────────────────────────────────────────────────
    document.getElementById("searchInput").addEventListener("input", e => {{
      searchQuery = e.target.value.trim().toLowerCase();
      applySearch();
    }});

    function applySearch() {{
      if (!searchQuery) {{
        nodeSel.attr("opacity",1);
        nodeLabelSel.attr("opacity",1);
        blobPaths.attr("opacity",1);
        arrowLines.attr("opacity",1);
        heLabelTexts.attr("opacity",1);
        simNodes.forEach(n=>n._hidden=false);
        return;
      }}
      simNodes.forEach(n => {{ n._hidden = !n.label.toLowerCase().includes(searchQuery); }});
      nodeSel.attr("opacity",d=>d._hidden?0.12:1);
      nodeLabelSel.attr("opacity",d=>d._hidden?0.05:1);

      // dim hyperedges whose members are all hidden
      blobPaths.attr("opacity",(d) => {{
        const anyVis = d.member_ids.some(id => !simNodeMap[id]?._hidden);
        return anyVis ? 0.85 : 0.08;
      }});
      arrowLines.attr("opacity",(d) => {{
        const anyVis = d.member_ids.some(id => !simNodeMap[id]?._hidden);
        return anyVis ? 0.7 : 0.05;
      }});
      heLabelTexts.attr("opacity",(d) => {{
        const anyVis = d.member_ids.some(id => !simNodeMap[id]?._hidden);
        return anyVis ? 1 : 0.05;
      }});
    }}
  }})();
  </script>
</body>
</html>
"""

    out = Path(output_html).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_content, encoding="utf-8")
    print(f"Hypergraph visualization saved -> {out}")
    return out