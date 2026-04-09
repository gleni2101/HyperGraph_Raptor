"""Centralized LLM and embedding client factory.

All scripts and modules should import from here instead of
instantiating their own clients. Configuration is read from
environment variables (typically loaded from ``.env``).

Environment variables
---------------------
LLM:
    URL              – OpenAI-compatible base URL  (required)
    MODEL_NAME       – Model identifier            (required)
    OPENAI_API_KEY   – API key                     (required)
    LLM_TEMPERATURE  – Sampling temperature        (default: 0)
    LLM_MAX_TOKENS   – Max generation tokens       (default: 20000)
    LLM_TIMEOUT      – HTTP timeout in seconds     (default: 120)

Embeddings:
    EMBED_URL        – Embedding server base URL   (default: http://127.0.0.1:8080)
    EMBED_MODEL      – Embedding model name        (default: BAAI/bge-m3)
    EMBED_MAX_CHARS  – Max input characters        (default: 19000)
"""
from __future__ import annotations

import logging
import os
import random
import ssl
import time
import re
from pathlib import Path
from typing import Any, Type

import httpx
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Auto-load .env from repo root
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_REPO_ROOT / ".env")


# ---------------------------------------------------------------------------
# Embedding client
# ---------------------------------------------------------------------------

class LocalBGEClient:
    """Embedding client for a local BGE-M3 (or compatible) server.

    BGE-M3 supports up to 8192 tokens (~19k chars). The default
    EMBED_MAX_CHARS of 19000 chars (~8k tokens) uses the full capacity.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        max_input_chars: int | None = None,
    ):
        self.base_url = (base_url or os.getenv("EMBED_URL", "http://127.0.0.1:8080")).rstrip("/")
        self.model = model or os.getenv("EMBED_MODEL", "BAAI/bge-m3")
        self.client = httpx.Client(timeout=timeout)
        # BGE-M3 max context = 8192 tokens ≈ ~19k chars.
        # Default to 19000 chars to use the full 8k token capacity.
        self.max_input_chars = max_input_chars if max_input_chars is not None else int(os.getenv("EMBED_MAX_CHARS", "19000"))

    _TOKEN_LIMIT_RE = re.compile(
        r"too\s+large\s+to\s+process|too\s+many\s+tokens|token\s+limit|maximum\s+context",
        re.IGNORECASE,
    )

    def _is_token_limit_error(self, status_code: int, body: str) -> bool:
        return status_code in (400, 413, 422, 500) and bool(self._TOKEN_LIMIT_RE.search(body or ""))

    def encode(
        self,
        text: str,
        *,
        max_retries: int = 3,
        max_shrinks: int = 8,
        min_chars: int = 128,
    ) -> np.ndarray:
        log = logging.getLogger(__name__)
        if len(text) > self.max_input_chars:
            log.warning("Truncating embedding input from %d to %d chars",
                        len(text), self.max_input_chars)
            text = text[: self.max_input_chars]
        log.debug("Embedding request: %d chars", len(text))

        resp = None
        transport_attempt = 0
        shrink_attempt = 0

        while True:
            resp = self.client.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": self.model, "input": text},
            )

            if resp.status_code == 200:
                return np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)

            # Token-limit style failures should retry with smaller text without
            # consuming transport retry budget.
            if self._is_token_limit_error(resp.status_code, resp.text):
                if shrink_attempt >= max_shrinks or len(text) <= min_chars:
                    break

                new_len = max(min_chars, len(text) // 2)
                if new_len >= len(text):
                    break

                shrink_attempt += 1
                text = text[:new_len]
                log.warning(
                    "Input too many tokens, shrinking to %d chars (shrink %d/%d)",
                    len(text), shrink_attempt, max_shrinks,
                )
                continue

            # Non-token-limit server failures use exponential-backoff retries.
            if resp.status_code >= 500 and transport_attempt < max_retries:
                transport_attempt += 1
                wait = 2 ** transport_attempt
                log.warning(
                    "Embedding server returned %s, retrying in %ds (attempt %d/%d)",
                    resp.status_code, wait, transport_attempt, max_retries,
                )
                time.sleep(wait)
                continue

            break

        log.error("Embedding failed (HTTP %d), text length: %d chars, "
                  "response: %s", resp.status_code, len(text),
                  resp.text[:500])
        try:
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Embedding failed after retries/shrinks "
                f"(status: {resp.status_code}, text length: {len(text)} chars)"
            ) from exc

        raise RuntimeError(
            f"Embedding failed after retries/shrinks "
            f"(status: {resp.status_code}, text length: {len(text)} chars)"
        )

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _resolve_ssl(cert_path: str | None = None) -> Any:
    """Return an SSL context or True/False for httpx verify."""
    if cert_path and os.path.exists(cert_path):
        return ssl.create_default_context(cafile=cert_path)
    default_cert = _REPO_ROOT / "certs" / "knapp.pem"
    if default_cert.exists():
        return ssl.create_default_context(cafile=str(default_cert))
    return True


def create_llm(
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    verify_ssl: bool = True,
    trust_env: bool = True,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance using env vars as defaults.

    Any explicit keyword argument overrides the corresponding env var.
    """
    _base_url = base_url or os.getenv("URL")
    _model = model or os.getenv("MODEL_NAME")
    _api_key = api_key or os.getenv("OPENAI_API_KEY")

    if not _base_url or not _model or not _api_key:
        raise ValueError(
            "Missing LLM configuration. Set URL, MODEL_NAME, and "
            "OPENAI_API_KEY in your .env file or pass them explicitly."
        )

    _temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0"))
    _max_tokens = max_tokens if max_tokens is not None else int(os.getenv("LLM_MAX_TOKENS", "20000"))
    _timeout = timeout if timeout is not None else float(os.getenv("LLM_TIMEOUT", "120"))

    verify_value = _resolve_ssl() if verify_ssl else False

    http_client = httpx.Client(
        verify=verify_value,
        timeout=_timeout,
        trust_env=trust_env,
    )

    return ChatOpenAI(
        base_url=_base_url,
        model=_model,
        api_key=_api_key,
        http_client=http_client,
        max_tokens=_max_tokens,
        temperature=_temperature,
    )


def create_embed_client(
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = 120.0,
    max_input_chars: int | None = None,
) -> LocalBGEClient:
    """Create a LocalBGEClient using env vars as defaults."""
    return LocalBGEClient(base_url=base_url, model=model, timeout=timeout, max_input_chars=max_input_chars)


# ---------------------------------------------------------------------------
# Structured generation with retry
# ---------------------------------------------------------------------------

def generate_structured(
    client: ChatOpenAI,
    system_prompt: str,
    user_prompt: str,
    response_model: Type[BaseModel],
    *,
    retries: int = 6,
    retry_delay: float = 2.0,
    retry_backoff: float = 2.0,
    max_delay: float = 30.0,
) -> BaseModel:
    """Invoke *client* with structured output and exponential-backoff retry.

    Parameters
    ----------
    client : ChatOpenAI
        The LangChain ChatOpenAI instance.
    system_prompt / user_prompt : str
        Messages to send.
    response_model : Type[BaseModel]
        Pydantic model for ``with_structured_output``.
    retries / retry_delay / retry_backoff / max_delay
        Retry configuration.

    Returns the parsed Pydantic object.
    """
    log = logging.getLogger(__name__)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    structured_llm = client.with_structured_output(response_model)
    delay = max(0.0, retry_delay)
    total_attempts = max(1, retries + 1)
    last_exc: Exception | None = None

    for attempt in range(1, total_attempts + 1):
        t0 = time.time()
        try:
            resp = structured_llm.invoke(messages)
            elapsed = time.time() - t0
            if resp is None:
                raise ValueError("with_structured_output returned None")
            log.info(
                "generate_structured OK (attempt %d/%d, %.1fs)",
                attempt, total_attempts, elapsed,
            )
            return resp
        except Exception as exc:
            elapsed = time.time() - t0
            last_exc = exc
            if attempt >= total_attempts:
                log.error(
                    "generate_structured FAILED after %d attempts (%.1fs): %r",
                    total_attempts, elapsed, exc,
                )
                break
            log.warning(
                "generate_structured attempt %d/%d failed (%.1fs): %r — retrying in %.1fs",
                attempt, total_attempts, elapsed, exc, delay,
            )
            time.sleep(delay + random.uniform(0, 0.5))
            delay = min(max_delay, max(0.1, delay * retry_backoff))

    raise last_exc  # type: ignore[misc]