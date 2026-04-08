import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS: dict[str, Any] = {
    "graph": {
        "distill_system": "You are provided with a context chunk (delimited by ```). Respond with a concise heading, summary, and a bulleted list of key facts. Omit author names, references, and citations.",
        "distill_user": "Rewrite this text so it stands alone with all necessary context. Extract and organize any table data. Focus on factual content.\n\n```{input}```",
        "figure_system": "You analyze figures and diagrams. Report the factual content in detail. If the image is not informational, return an empty string. Include the full image path.",
        "figure_user": "Describe this figure factually. Extract data, labels, relationships, and structure.\n\n```{input}```",
        "graphmaker_system": "You are a knowledge-graph extractor for logistics and supply chain documents.\n\nGiven a text chunk delimited by ```, extract entities and binary relationships.\n\nEntity types to look for: shipment, order, SKU, package, pallet, container, lane, route, stop, facility, warehouse, dock, carrier, vehicle, driver, customer, supplier, region, country, event, exception, KPI, SLA, cost, time_window, inventory_state, system (TMS, WMS, ERP), process, rule, constraint.\n\nRelation labels should be specific and operational: ships_to, departs_from, arrives_at, stored_in, handled_by, delayed_by, violates, satisfies, depends_on, constrained_by, triggers, updates, measured_by, contains, routes_through, assigned_to, scheduled_for, caused_by, mitigates, replaces.\n\nRules:\n- Keep technical terms and abbreviations exactly as written (ETA, OTIF, ASN, POD)\n- Each node needs an id and a type\n- Each edge needs source, target, and relation\n- Omit author names, citations, and generic filler\n- When an entity appears in multiple relationships, reuse the same node id\n\nReturn a JSON object with keys \"nodes\" and \"edges\".",
        "graphmaker_user": "Context: ```{input}```\n\nExtract the knowledge graph. Return only a valid JSON object with keys \"nodes\" and \"edges\".",
    },
    "hypergraph": {
        "distill_system": "You are provided with a context chunk (delimited by ```). Respond with a concise heading, summary, and a bulleted list of key facts. Omit author names, references, and citations.",
        "distill_user": "Rewrite this text so it stands alone with all necessary context. Extract and organize any table data. Focus on factual content.\n\n```{input}```",
        "figure_system": "You analyze figures and diagrams. Report the factual content in detail. If the image is not informational, return an empty string. Include the full image path.",
        "figure_user": "Describe this figure factually. Extract data, labels, relationships, and structure.\n\n```{input}```",
        "graphmaker_system": (
            "You are a domain-agnostic hypergraph relationship extractor. Read a text chunk and extract "
            "n-ary relationships (hyperedges) that capture how multiple entities participate in one fact, "
            "event, process, role assignment, property assignment, or causal chain.\n\n"
            "A hyperedge connects multiple source entities to multiple target entities through one specific relation. "
            "Use this for technical manuals, scientific text, policies, narratives, movie summaries, tables, and bullet lists.\n\n"
            "Extraction strategy:\n"
            "1. Read the full chunk before extracting. Identify canonical entities such as documents, products, films, "
            "characters, organizations, people, places, dates, themes, systems, constraints, and outcomes.\n"
            "2. Convert each meaningful statement into one hyperedge:\n"
            "   - source: the main subject(s), actor(s), input(s), cause(s), or owner(s)\n"
            "   - relation: a specific normalized verb phrase in snake_case\n"
            "   - target: the object(s), value(s), recipient(s), effect(s), or result(s)\n"
            "3. For tables, treat each row as facts. Example: a movie title can be linked to release date, runtime, studio, rating, and box office.\n"
            "4. For cast lists or role lists, connect the character/entity to the actor or role with a specific relation such as voiced_by, played_by, works_as, owns, rules, helps, opposes.\n"
            "5. For plot summaries, resolve pronouns to named entities whenever possible and extract concrete story events.\n"
            "6. When a section heading is only organizational (for example: Main Characters, Act I, Key Themes), do NOT extract it as an entity unless the text explicitly treats it as a real concept.\n"
            "7. Do NOT create vague nodes like \"details\", \"overview\", \"section\", \"movie information\", or \"plot summary\" unless they are actual semantic entities in the text.\n\n"
            "Rules:\n"
            "- Be concrete, not generic. Prefer directed_by over related_to. Prefer has_release_date over has_property.\n"
            "- Preserve proper nouns and literal values exactly as written when they are meaningful targets.\n"
            "- Reuse the exact same entity string across events for the same entity.\n"
            "- Include metadata facts, role assignments, world-building facts, and causal story events when present.\n"
            "- Each event must have at least one source and one target.\n"
            "- Return only JSON.\n\n"
            "Return a JSON object with one key \"events\". Each event has:\n"
            "- \"source\": list[str]\n"
            "- \"relation\": str\n"
            "- \"target\": list[str]\n\n"
            "Examples:\n"
            "1. {\"source\": [\"The SpongeBob SquarePants Movie\"], \"relation\": \"has_release_date\", \"target\": [\"November 19, 2004\"]}\n"
            "2. {\"source\": [\"SpongeBob SquarePants\"], \"relation\": \"voiced_by\", \"target\": [\"Tom Kenny\"]}\n"
            "3. {\"source\": [\"Plankton\"], \"relation\": \"frames\", \"target\": [\"Mr. Krabs\"]}\n"
            "4. {\"source\": [\"TMS\", \"Munich warehouse\", \"DHL\"], \"relation\": \"routes_shipments_to\", \"target\": [\"customers in Austria\", \"48-hour SLA\"]}"
        ),
        "graphmaker_user": (
            "Context: ```{input}```\n\n"
            "Extract all meaningful hyperedges. Include factual metadata, table rows, list items, role assignments, "
            "narrative events, causal links, constraints, and dependencies. Ignore formatting-only headings and generic "
            "document labels. Return only a valid JSON object with key \"events\"."
        ),
    },
    "runtime": {
        "default_system_prompt": "You extract structured relationships from a text chunk. Return a JSON object with one key \"events\". Each event has: source (list[str]), relation (str), target (list[str]). Be thorough and specific.",
        "figure_system_prompt": "You are an assistant who describes figures and diagrams in factual detail.",
        "figure_user_prompt": "Describe this figure in detail. Include all data, labels, axes, legends, and relationships shown.",
    },
}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_config_path(config_path: str | None = None) -> Path:
    if config_path:
        return Path(config_path).expanduser().resolve()
    env_path = os.getenv("GRAPH_REASONING_PROMPT_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "prompt_config.json").resolve()


def load_prompt_config(config_path: str | None = None) -> dict[str, Any]:
    resolved = _resolve_config_path(config_path)
    if not resolved.exists():
        return DEFAULT_PROMPTS

    try:
        with resolved.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            return DEFAULT_PROMPTS
        return _deep_merge(DEFAULT_PROMPTS, loaded)
    except Exception:
        return DEFAULT_PROMPTS


def get_prompt(section: str, key: str, config_path: str | None = None, **kwargs) -> str:
    prompts = load_prompt_config(config_path=config_path)
    section_data = prompts.get(section, {}) if isinstance(prompts, dict) else {}
    template = section_data.get(key, "") if isinstance(section_data, dict) else ""
    if not isinstance(template, str):
        return ""
    if kwargs:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    return template
