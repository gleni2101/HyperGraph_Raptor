import json
import os
from pathlib import Path
from typing import Any


def _resolve_config_path(config_path: str | None = None) -> Path:
    """Resolve the path to prompt_config.json.

    Priority:
      1. Explicit *config_path* argument
      2. GRAPH_REASONING_PROMPT_CONFIG environment variable
      3. prompt_config.json at the repository root (one level above this file)
    """
    if config_path:
        return Path(config_path).expanduser().resolve()
    env_path = os.getenv("GRAPH_REASONING_PROMPT_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "prompt_config.json").resolve()


def load_prompt_config(config_path: str | None = None) -> dict[str, Any]:
    """Load the prompt configuration from *prompt_config.json*.

    Raises ``FileNotFoundError`` if the resolved JSON file does not exist,
    since the JSON file is now the single source of truth.
    """
    resolved = _resolve_config_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Prompt configuration file not found: {resolved}\n"
            "Please ensure prompt_config.json exists at the repository root "
            "or set the GRAPH_REASONING_PROMPT_CONFIG environment variable."
        )

    with resolved.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object in {resolved}, got {type(loaded).__name__}")

    return loaded


def get_prompt(section: str, key: str, config_path: str | None = None, **kwargs) -> str:
    """Return a single prompt string from *section* → *key*.

    Any extra ``**kwargs`` are interpolated into the template via ``str.format``.
    """
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