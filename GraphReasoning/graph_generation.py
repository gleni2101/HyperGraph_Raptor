from GraphReasoning.graph_tools import *
from GraphReasoning.utils import *
from GraphReasoning.graph_analysis import *
from GraphReasoning.prompt_config import get_prompt
from concurrent.futures import ThreadPoolExecutor, as_completed

from IPython.display import display, Markdown
import pandas as pd
import numpy as np
import networkx as nx
import os
import re
import importlib

def _get_misc_properties(self):
    if "misc_properties" in self.columns:
        return self["misc_properties"]
    return pd.Series([{} for _ in range(len(self))], index=self.index, dtype=object)


def _set_misc_properties(self, value):
    self["misc_properties"] = value


pd.DataFrame.misc_properties = property(_get_misc_properties, _set_misc_properties)

try:
    RecursiveCharacterTextSplitter = importlib.import_module(
        "langchain.text_splitters"
    ).RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    RecursiveCharacterTextSplitter = importlib.import_module(
        "langchain_text_splitters"
    ).RecursiveCharacterTextSplitter

try:
    import pdfkit
except ImportError:
    pdfkit = None
from pathlib import Path
import random
from pyvis.network import Network
from tqdm.auto import tqdm

import seaborn as sns

from hashlib import md5


#hypergraph add ons
import json
import pickle  # kept for backward-compat with old binary graphs
from GraphReasoning.hypergraph_store import HypergraphBuilder

try:
    import hypernetx as hnx  # optional – only needed for legacy HNX objects
except ImportError:
    hnx = None  # type: ignore[assignment]



palette = "hls"
# Code based on: https://github.com/rahulnyk/knowledge_graph


def _cache_dir() -> Path:
    cache_root = os.getenv("GRAPH_REASONING_CACHE_DIR", "temp")
    cache_path = Path(cache_root)
    if not cache_path.is_absolute():
        cache_path = (Path.cwd() / cache_path).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _parse_json_object_from_text(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Expected non-empty JSON text.")

    cleaned = text.strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()

    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict):
            return payload

    raise ValueError("Could not parse a JSON object from model output.")


def _coerce_structured_payload(value) -> dict:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        return _parse_json_object_from_text(value)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped

    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if isinstance(dumped, dict):
            return dumped

    fields = {}
    for field_name in ("nodes", "edges", "events"):
        if hasattr(value, field_name):
            fields[field_name] = getattr(value, field_name)
    if fields:
        return fields

    raise ValueError("Unsupported structured output type from generate(...).")


def _item_get(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _to_string_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _to_text(value) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk,
            "chunk_id": md5(chunk.encode()).hexdigest(),
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)

    return df


def df2Graph(df: pd.DataFrame, generate, generate_figure=None, image_list=None, repeat_refine=0, do_distill=True, do_relabel = False, verbatim=False,
            max_workers: int = 4,
          
            ) -> nx.DiGraph:
    
    subgraph_list = []
    rows = list(df.itertuples(index=False))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(
                graphPrompt,
                row.text,
                generate,
                generate_figure,
                image_list,
                {"chunk_id": row.chunk_id},
                do_distill=do_distill,
                do_relabel=do_relabel,
                repeat_refine=repeat_refine,
                verbatim=verbatim,
            ): row.chunk_id
            for row in rows
        }

        for future in tqdm(
            as_completed(future_to_chunk),
            total=len(future_to_chunk),
            desc="Extracting graph chunks",
            unit="chunk",
            leave=True,
        ):
            chunk_id = future_to_chunk[future]
            try:
                subgraph = future.result()
                print(subgraph, type(subgraph))
                subgraph_list.append(subgraph)
            except Exception as exc:
                print(f"Exception while processing chunk {chunk_id}: {exc}")

        
    G = nx.DiGraph()

    for g in subgraph_list:
        G = nx.compose(G, g)
    
    return G

def df2hypergraph(
    df: pd.DataFrame,
    generate,
    generate_figure=None,
    image_list=None,
    repeat_refine: int = 0,
    do_distill: bool = True,
    do_relabel: bool = False,
    verbatim: bool = False,
    max_workers: int = 4,
) -> HypergraphBuilder:
    """Build a merged HypergraphBuilder from all chunks in *df*."""
    merged: HypergraphBuilder | None = None
    chunk_builders: list[HypergraphBuilder] = []

    rows = list(df.itertuples(index=False))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(
                hypergraphPrompt,
                row.text,
                generate,
                generate_figure,
                image_list,
                {"chunk_id": row.chunk_id},
                do_distill=do_distill,
                do_relabel=do_relabel,
                repeat_refine=repeat_refine,
                verbatim=verbatim,
            ): row.chunk_id
            for row in rows
        }

        for future in tqdm(
            as_completed(future_to_chunk),
            total=len(future_to_chunk),
            desc="Extracting hypergraph chunks",
            unit="chunk",
            leave=True,
        ):
            chunk_id = future_to_chunk[future]
            try:
                chunk_builder = future.result()
                if chunk_builder is None:
                    print(f"Skipping chunk {chunk_id} – no events extracted.")
                    continue
                chunk_builders.append(chunk_builder)
            except Exception as exc:
                print(f"Exception while processing chunk {chunk_id}: {exc}")

    for chunk_builder in chunk_builders:
        if merged is None:
            merged = chunk_builder
        else:
            merged.merge(chunk_builder)

    if merged is None:
        print("No valid subgraphs found. Returning empty HypergraphBuilder.")
        return HypergraphBuilder()

    print(f"df2hypergraph complete: {merged.node_count} nodes, {merged.edge_count} edges.")
    return merged

import sys
sys.path.append("..")

import json

def graphPrompt(input: str, generate, generate_figure=None, image_list=None, metadata={}, #model="mistral-openorca:latest",
                do_distill=True, repeat_refine=0,verbatim=False,
               ) -> nx.DiGraph:
    cache_graphml = _cache_dir() / f"{metadata['chunk_id']}.graphml"
    
    try:
        return nx.read_graphml(cache_graphml)
    except:
        pass

    SYS_PROMPT_DISTILL = get_prompt("graph", "distill_system")

    USER_PROMPT_DISTILL = get_prompt("graph", "distill_user", input=input)

    SYS_PROMPT_FIGURE = get_prompt("graph", "figure_system")

    USER_PROMPT_FIGURE = get_prompt("graph", "figure_user", input=input)
    input_fig = ''
    if generate_figure: # if image in the chunk
        
        for image_name in image_list:
            _image_name = image_name.split('/')[-1]
            if _image_name.lower() in input.lower():  
                input_fig =  f'Here is the information in the image: {image_name}' + \
                generate_figure( image = image_name, system_prompt=SYS_PROMPT_FIGURE, prompt=USER_PROMPT_FIGURE)
    
    if do_distill:
        #Do not include names, figures, plots or citations in your response, only facts."
        input = _to_text(generate( system_prompt=SYS_PROMPT_DISTILL, prompt=USER_PROMPT_DISTILL))

    if input_fig:
        input += input_fig
    
    SYS_PROMPT_GRAPHMAKER = get_prompt("graph", "graphmaker_system")
     
    USER_PROMPT = get_prompt("graph", "graphmaker_user", input=input)
    # result = [dict(item, **metadata) for item in result]
    
    print ('Generating triples...')
    result_raw  =  generate( system_prompt=SYS_PROMPT_GRAPHMAKER, prompt=USER_PROMPT)

    try:
        result = _coerce_structured_payload(result_raw)
    except Exception as e:
        print(f"Failed to parse graph JSON for chunk {metadata.get('chunk_id')}: {e}")
        return nx.DiGraph()

    nodes = result.get("nodes", [])
    edges = result.get("edges", [])

    G = nx.DiGraph()
    for node in nodes:
        node_id = _item_get(node, "id")
        if node_id is None:
            continue
        node_type = _item_get(node, "type", "entity")
        G.add_node(str(node_id), type=str(node_type))
    for edge in edges:
        source = _item_get(edge, "source")
        target = _item_get(edge, "target")
        relation = _item_get(edge, "relation", "related_to")
        if source is None or target is None:
            continue
        G.add_edge(str(source), str(target), relation=str(relation), chunk_id=metadata['chunk_id'])

    nx.write_graphml(G, cache_graphml)
    print(f'Generated graph: {G}')

    return G


def hypergraphPrompt(
    input: str,
    generate,
    generate_figure=None,
    image_list=None,
    metadata=None,
    do_distill=True,
    do_relabel=False,
    repeat_refine=0,
    verbatim=False,
) -> HypergraphBuilder | None:
    if metadata is None:
        metadata = {}

    chunk_id = metadata.get("chunk_id", md5(input.encode()).hexdigest())
    cache_json = _cache_dir() / f"{chunk_id}_hg.json"
    cache_pkl = _cache_dir() / f"{chunk_id}.pkl"

    def _builder_from_legacy(value) -> HypergraphBuilder | None:
        if value is None:
            return None
        # Legacy cache was often (hypergraph, dataframe)
        if isinstance(value, tuple) and value:
            value = value[0]
        # Already new format
        if isinstance(value, HypergraphBuilder):
            return value

        builder = HypergraphBuilder(source_document=metadata.get("source_document", ""))

        if hasattr(value, "incidence_dict"):
            for eid, nodes in value.incidence_dict.items():
                members = [str(n).strip() for n in nodes if str(n).strip()]
                if len(members) >= 2:
                    builder.add_event(str(eid), members[:1], members[1:], chunk_id=chunk_id)
            return builder

        return None

    # Preferred cache path: JSON builder
    if cache_json.exists():
        try:
            cached_builder = HypergraphBuilder.load(cache_json)
            if verbatim:
                print(f"Loaded hypergraph cache: {cache_json}")
            return cached_builder
        except Exception as exc:
            print(f"Failed to read JSON cache {cache_json}: {exc}")

    # Backward-compatible cache path: PKL tuple/hypergraph
    if cache_pkl.exists():
        try:
            with open(cache_pkl, "rb") as fin:
                cached = pickle.load(fin)
            cached_builder = _builder_from_legacy(cached)
            if cached_builder is not None:
                try:
                    cached_builder.save(cache_json)
                except Exception:
                    pass
                if verbatim:
                    print(f"Loaded legacy hypergraph cache: {cache_pkl}")
                return cached_builder
        except Exception as exc:
            print(f"Failed to read PKL cache {cache_pkl}: {exc}")
    SYS_PROMPT_DISTILL = get_prompt("hypergraph", "distill_system")

    USER_PROMPT_DISTILL = get_prompt("hypergraph", "distill_user", input=input)

    SYS_PROMPT_FIGURE = get_prompt("hypergraph", "figure_system")

    USER_PROMPT_FIGURE = get_prompt("hypergraph", "figure_user", input=input)
    input_fig = ''
    if generate_figure: # if image in the chunk
        
        for image_name in image_list:
            _image_name = image_name.split('/')[-1]
            if _image_name.lower() in input.lower():  
                input_fig =  f'Here is the information in the image: {image_name}' + \
                generate_figure( image = image_name, system_prompt=SYS_PROMPT_FIGURE, prompt=USER_PROMPT_FIGURE)
    
    if do_distill:
        #Do not include names, figures, plots or citations in your response, only facts."
        input = _to_text(generate( system_prompt=SYS_PROMPT_DISTILL, prompt=USER_PROMPT_DISTILL))

    if input_fig:
        input += input_fig

    SYS_PROMPT_GRAPHMAKER = get_prompt("hypergraph", "graphmaker_system")
 
    #USER_PROMPT = f'Context: ```{input}``` \n\ Extract the hypergraph knowledge graph in structured JSON format: '
    USER_PROMPT = get_prompt("hypergraph", "graphmaker_user", input=input)

    print('Generating hypergraph...')
    validated_raw = generate(system_prompt=SYS_PROMPT_GRAPHMAKER, prompt=USER_PROMPT)

    try:
        validated_result = _coerce_structured_payload(validated_raw)
    except Exception as e:
        print(f"Failed to parse hypergraph JSON for chunk {chunk_id}: {e}")
        return None

    raw_events = validated_result.get("events", [])
    events = []
    for event in raw_events:
        source = _to_string_list(_item_get(event, "source"))
        target = _to_string_list(_item_get(event, "target"))
        relation = _item_get(event, "relation", "related_to")
        if not source or not target:
            continue
        events.append({
            "source": source,
            "relation": str(relation),
            "target": target,
        })

    if not events:
        print(f"No valid events found for chunk {chunk_id}.")
        return None

    builder = HypergraphBuilder(source_document=metadata.get("source_document", ""))
    for event in events:
        try:
            builder.add_event(
                relation=event["relation"],
                source=event["source"],
                target=event["target"],
                chunk_id=chunk_id,
            )
        except ValueError:
            continue

    if builder.edge_count == 0:
        return None

    print(f"Generated hypergraph with {builder.node_count} nodes, {builder.edge_count} edges.")

    try:
        builder.save(cache_json)
    except Exception:
        pass

    # Keep writing legacy pkl cache for backward compatibility with old flows.
    try:
        with open(cache_pkl, "wb") as fout:
            pickle.dump(builder, fout)
    except Exception:
        pass

    return builder


def colors2Community(communities) -> pd.DataFrame:
    
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors

def make_graph_from_text (txt,generate, generate_figure=None, image_list=None,
                          graph_root='graph_root',
                          chunk_size=2500,chunk_overlap=0,do_distill=True, do_relabel=False,
                          repeat_refine=0,verbatim=False,
                          data_dir='./data_output_KG/',
                          save_HTML=False,
                          save_PDF=False,#TO DO
                         ):    
    
    ## data directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)     
    graph_GraphML=  f'{data_dir}/{graph_root}.graphml'  #  f'{data_dir}/result.graphml',

    try:
        G = nx.read_graphml(graph_GraphML)
    except:

        outputdirectory = Path(f"./{data_dir}/") #where graphs are stored from graph2df function
        
    
        splitter = RecursiveCharacterTextSplitter(
            #chunk_size=5000, #1500,
            chunk_size=chunk_size, #1500,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        pages = splitter.split_text(txt)
        print("Number of chunks = ", len(pages))
        if verbatim:
            display(Markdown (pages[0]) )
        
        df = documents2Dataframe(pages)
        df.to_csv(f'{data_dir}/{graph_root}_chunks_clean.csv')

        G = df2Graph(df,generate, generate_figure, image_list, do_distill=do_distill, do_relabel=do_relabel, repeat_refine=repeat_refine,verbatim=verbatim) #model='zephyr:latest' )

        nx.write_graphml(G, graph_GraphML)
        

    graph_HTML = None
    net= None
    output_pdf = None
    if save_HTML:
        net = Network(
                notebook=True,
                cdn_resources="remote",
                height="900px",
                width="100%",
                select_menu=True,
                filter_menu=False,
            )

        net.from_nx(G)
        net.force_atlas_2based(central_gravity=0.015, gravity=-31)

        net.show_buttons()

        graph_HTML= f'{data_dir}/{graph_root}.html'
        net.save_graph(graph_HTML,
                )
        if verbatim:
            net.show(graph_HTML,
                )


        if save_PDF:
            output_pdf=f'{data_dir}/{graph_root}.pdf'
            if pdfkit is None:
                raise ImportError("pdfkit is required for save_PDF=True. Install it with `uv pip install pdfkit` and ensure wkhtmltopdf is available on PATH.")
            pdfkit.from_file(graph_HTML,  output_pdf)
        
    
    return graph_HTML, graph_GraphML, G, net, output_pdf

def make_hypergraph_from_text(
    txt,
    generate,
    generate_figure=None,
    image_list=None,
    graph_root='graph_root',
    chunk_size=2500,
    chunk_overlap=0,
    do_distill=True,
    do_relabel=False,
    repeat_refine=0,
    verbatim=False,
    data_dir='./data_output_KG/',
    force_rebuild=False,
    max_workers: int = 4,
):
    """
    Builds or loads a graph stored in a .pkl file.

    - If `{graph_root}.pkl` exists in `data_dir`, loads and returns it.
    - Otherwise, splits `txt` into chunks, generates a graph `G`, 
      pickles `G` to `{graph_root}.pkl`, and returns it.

    Returns:
    pkl_path (str), G 
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    pkl_path = os.path.join(data_dir, f"{graph_root}.pkl")  # kept for legacy migration check
    sub_dfs_pkl_path = os.path.join(data_dir, f"{graph_root}_sub_dfs.pkl")

    # Prefer JSON store; fall back to legacy pkl for backward-compatibility
    json_path = os.path.join(data_dir, f"{graph_root}.json")

    if (not force_rebuild) and os.path.isfile(json_path):
        G = HypergraphBuilder.load(json_path)
        print(f"Loaded existing hypergraph from {json_path}")
        return json_path, G, None, None
    elif (not force_rebuild) and os.path.isfile(pkl_path):
        print(f"Migrating legacy pickle {pkl_path} -> JSON...")
        with open(pkl_path, "rb") as fh:
            legacy = pickle.load(fh)
        # Wrap legacy hnx.Hypergraph into a minimal HypergraphBuilder
        G = HypergraphBuilder(source_document=graph_root)
        if hasattr(legacy, "incidence_dict"):
            for eid, nodes in legacy.incidence_dict.items():
                members = list(nodes)
                if len(members) >= 2:
                    G.add_event(str(eid), members[:1], members[1:], chunk_id="legacy")
        G.save(json_path)
        print(f"Migration complete -> {json_path}")
        return json_path, G, None, None

    # -- Build from text ------------------------------------------------------
    if force_rebuild:
        print(f"Force rebuild enabled for {graph_root}; ignoring cached hypergraph files.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    pages = splitter.split_text(txt)
    print("Number of chunks =", len(pages))
    if verbatim:
        from IPython.display import Markdown, display
        display(Markdown(pages[0]))

    df = documents2Dataframe(pages)
    df.to_csv(os.path.join(data_dir, f"{graph_root}_chunks_clean.csv"), index=False)

    G = df2hypergraph(
        df,
        generate,
        generate_figure,
        image_list,
        do_distill=do_distill,
        do_relabel=do_relabel,
        repeat_refine=repeat_refine,
        verbatim=verbatim,
        max_workers=max_workers,
    )
    G.graph.metadata["source_document"] = graph_root

    G.save(json_path)
    print(f"Saved hypergraph to {json_path}")

    return json_path, G, None, None


import time
from copy import deepcopy

def add_new_subgraph_from_text(txt=None,generate=None,generate_figure=None, image_list=None, 
                               node_embeddings=None,tokenizer=None, model=None, original_graph=None,
                               data_dir_output='./data_temp/',graph_root='graph_root',
                               chunk_size=10000,chunk_overlap=2000,
                               do_update_node_embeddings=True, do_distill=True, do_relabel = False, 
                               do_simplify_graph=True,size_threshold=10,
                               repeat_refine=0,similarity_threshold=0.95,
                               do_Louvain_on_new_graph=True, 
                               #whether or not to simplify, uses similiraty_threshold defined above
                               return_only_giant_component=False,
                               save_common_graph=False,G_to_add=None,
                               graph_GraphML_to_add=None,
                               verbatim=True,):

    display (Markdown(txt[:32]+"..."))
    graph_GraphML=None
    G_new=None
    
    res=None
    # try:
    start_time = time.time() 

    if verbatim:
        print ("Now create or load new graph...")

    if (G_to_add is not None and graph_GraphML_to_add is not None):
        print("G_to_add and graph_GraphML_to_add cannot be used together. Pick one or the other to provide a graph to be added.")
        return
    elif graph_GraphML_to_add==None and G_to_add==None: #make new if no existing one provided
        print ("Make new graph from text...")
        _, graph_GraphML_to_add, G_to_add, _, _ =make_graph_from_text (txt,generate,
                                 data_dir=data_dir_output,
                                 graph_root=f'graph_root',
                                 chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                 repeat_refine=repeat_refine, 
                                 verbatim=verbatim,
                                 )
        if verbatim:
            print ("New graph from text provided is generated and saved as: ", graph_GraphML_to_add)
    elif G_to_add is None:
        if verbatim:
            print ("Loading or using provided graph... Any txt data provided will be ignored...:", G_to_add, graph_GraphML_to_add)
            G_to_add = nx.read_graphml(graph_GraphML_to_add)
    # res_newgraph=graph_statistics_and_plots_for_large_graphs(G_to_add, data_dir=data_dir_output,                                      include_centrality=False,make_graph_plot=False,                               root='new_graph')
    print("--- %s seconds ---" % (time.time() - start_time))
    # except:
        # print ("ALERT: Graph generation failed...")
        
    print ("Now grow the existing graph...")
    
    # try:
    #Load original graph
    if type(original_graph) == str:
        G = nx.read_graphml(original_graph)
    else:
        G = deepcopy(original_graph)
    print(G, G_to_add)
    G_new = nx.compose(G, G_to_add)

    if do_update_node_embeddings:
        if verbatim:
            print ("Now update node embeddings")
        node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)

    if do_simplify_graph:
        if verbatim:
            print ("Now simplify graph.")
        G_new, node_embeddings = simplify_graph (G_new, node_embeddings, tokenizer, model , 
                                                similarity_threshold=similarity_threshold, use_llm=False, data_dir_output=data_dir_output,
                                verbatim=verbatim,)
    if size_threshold >0:
        if verbatim:
            print ("Remove small fragments")            
        G_new=remove_small_fragents (G_new, size_threshold=size_threshold)
        node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)

    if return_only_giant_component:
        if verbatim:
            print ("Select only giant component...")   
        connected_components = sorted(nx.connected_components(G_new), key=len, reverse=True)
        G_new = G_new.subgraph(connected_components[0]).copy()
        node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)

    if do_Louvain_on_new_graph:
        G_new=graph_Louvain (G_new, graph_GraphML=None)
        if verbatim:
            print ("Done Louvain...")

    if verbatim:
        print ("Done update graph")

    graph_GraphML= f'{data_dir_output}/{graph_root}_integrated.graphml'
    if verbatim:
        print ("Save new graph as: ", graph_GraphML)

    nx.write_graphml(G_new, graph_GraphML)
    if verbatim:
        print ("Done saving new graph")
    
    # res=graph_statistics_and_plots_for_large_graphs(G_new, data_dir=data_dir_output,include_centrality=False,make_graph_plot=False,root='assembled')
    # print ("Graph statistics: ", res)

    # except:
        # print ("Error adding new graph.")
    print(G_new, graph_GraphML)
        # print (end="")

    return graph_GraphML, G_new, G_to_add, node_embeddings, res


# ---------------------------------------------------------------------------
# Hypergraph merging — uses HypergraphBuilder, no HyperNetX dependency
# ---------------------------------------------------------------------------

def add_new_hypersubgraph_from_text(
    txt=None,
    generate=None,
    generate_figure=None,
    image_list=None,
    node_embeddings=None,
    tokenizer=None,
    model=None,
    original_graph: "HypergraphBuilder | str | None" = None,
    data_dir_output: str = "./data_temp/",
    graph_root: str = "graph_root",
    chunk_size: int = 10000,
    chunk_overlap: int = 2000,
    do_update_node_embeddings: bool = True,
    do_distill: bool = True,
    do_relabel: bool = False,
    do_simplify_graph: bool = True,
    size_threshold: int = 10,
    repeat_refine: int = 0,
    similarity_threshold: float = 0.95,
    do_Louvain_on_new_graph: bool = False,  # not meaningful for label-based graphs
    return_only_giant_component: bool = False,
    save_common_graph: bool = False,
    G_to_add: "HypergraphBuilder | None" = None,
    graph_pkl_to_add=None,  # kept for signature compat, ignored
    sub_dfs=None,           # kept for signature compat, ignored
    verbatim: bool = True,
) -> tuple:
    """Merge a new HypergraphBuilder into the integrated graph and persist as JSON.

    Returns
    -------
    (integrated_json_path, merged_builder, G_to_add, node_embeddings, None)
    """
    import time as _time

    t0 = _time.time()

    os.makedirs(data_dir_output, exist_ok=True)
    integrated_json = os.path.join(data_dir_output, f"{graph_root}_integrated.json")

    # -- Resolve the graph-to-add ------------------------------------------
    if G_to_add is not None and graph_pkl_to_add is not None:
        raise ValueError("Provide only one of G_to_add or graph_pkl_to_add.")

    if G_to_add is None and graph_pkl_to_add is None:
        if not txt:
            raise ValueError("Either G_to_add or txt must be provided.")
        if verbatim:
            print("Building new hypergraph from text…")
        _, G_to_add, _, _ = make_hypergraph_from_text(
            txt,
            generate,
            generate_figure=generate_figure,
            image_list=image_list,
            graph_root=graph_root,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            do_distill=do_distill,
            do_relabel=do_relabel,
            repeat_refine=repeat_refine,
            verbatim=verbatim,
            data_dir=data_dir_output,
        )
    elif G_to_add is None:
        # Legacy: load from old pkl path
        if verbatim:
            print(f"Loading legacy pkl: {graph_pkl_to_add}")
        with open(graph_pkl_to_add, "rb") as fh:
            legacy = pickle.load(fh)
        G_to_add = HypergraphBuilder(source_document=graph_root)
        if hasattr(legacy, "incidence_dict"):
            for eid, nodes in legacy.incidence_dict.items():
                members = list(nodes)
                if len(members) >= 2:
                    G_to_add.add_event(str(eid), members[:1], members[1:], chunk_id="legacy_pkl")

    if verbatim:
        print(f"Loaded sub-graph in {_time.time() - t0:.2f}s — merging…")

    # -- Resolve the base (original) graph ----------------------------------
    if original_graph is None:
        H = HypergraphBuilder(source_document="integrated")
    elif isinstance(original_graph, str):
        # Could be a JSON path or old pkl path
        if original_graph.endswith(".json"):
            H = HypergraphBuilder.load(original_graph)
        else:
            with open(original_graph, "rb") as fh:
                leg = pickle.load(fh)
            H = HypergraphBuilder(source_document="integrated")
            if hasattr(leg, "incidence_dict"):
                for eid, nodes in leg.incidence_dict.items():
                    members = list(nodes)
                    if len(members) >= 2:
                        H.add_event(str(eid), members[:1], members[1:], chunk_id="legacy_pkl")
    elif isinstance(original_graph, HypergraphBuilder):
        H = deepcopy(original_graph)
    else:
        raise TypeError(f"Unsupported type for original_graph: {type(original_graph)}")

    # -- Merge --------------------------------------------------------------
    H.merge(G_to_add)
    if verbatim:
        print(f"Merged — {H.node_count} nodes, {H.edge_count} edges.")

    # -- Optional embedding update ------------------------------------------
    if do_update_node_embeddings and node_embeddings is not None:
        if verbatim:
            print("Updating node embeddings…")
        node_embeddings = update_hypernode_embeddings(
            node_embeddings, H, tokenizer, model, verbatim=verbatim
        )

    # -- Persist ------------------------------------------------------------
    def _safe_json_write(builder: HypergraphBuilder, path: str, retries: int = 6) -> None:
        tmp = path + ".tmp"
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                builder.save(tmp)
                os.replace(tmp, path)
                return
            except PermissionError as exc:
                last_exc = exc
                wait = attempt * 1.0
                if verbatim:
                    print(f"Permission error writing {path} (attempt {attempt}/{retries}), retry in {wait}s")
                _time.sleep(wait)
            finally:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
        raise PermissionError(f"Could not write {path} after {retries} attempts") from last_exc

    _safe_json_write(H, integrated_json)
    if verbatim:
        print(f"Integrated hypergraph saved -> {integrated_json}")

    return integrated_json, H, G_to_add, node_embeddings, None


