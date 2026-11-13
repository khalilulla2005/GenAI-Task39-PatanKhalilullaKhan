from __future__ import annotations

import json
import urllib.error
from dataclasses import dataclass
from urllib.request import Request, urlopen

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from src.config import Settings
from src.prompts import TaskType, build_chat_prompt

MAX_REQUEST_CHARS = 20000


@dataclass(frozen=True)
class OllamaStatus:
    available: bool
    model_available: bool
    installed_models: tuple[str, ...]


def create_chat_model(settings: Settings) -> Runnable:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError(
            "The Ollama integration is missing. Install it with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    client_kwargs: dict[str, object] = {
        "timeout": settings.request_timeout,
    }
    if settings.api_key:
        client_kwargs["headers"] = {
            "Authorization": f"Bearer {settings.api_key}",
        }

    return ChatOllama(
        model=settings.model,
        base_url=settings.base_url,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        num_predict=settings.num_predict,
        client_kwargs=client_kwargs,
    )


def check_ollama(settings: Settings) -> OllamaStatus:
    headers = {"Accept": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    request = Request(
        f"{settings.base_url}/api/tags",
        headers=headers,
        method="GET",
    )

    try:
        with urlopen(request, timeout=settings.request_timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(
            f"Could not reach Ollama at {settings.base_url}: {reason}. "
            "Start Ollama or configure a reachable endpoint."
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("Ollama returned an invalid model-list response.") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise RuntimeError("Ollama returned an unexpected model-list response.")

    model_names = tuple(
        item["name"]
        for item in payload["models"]
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    )
    return OllamaStatus(
        available=True,
        model_available=settings.model in model_names,
        installed_models=model_names,
    )


def run_assistant(
    task: TaskType | str,
    request: str,
    settings: Settings,
    model: Runnable | None = None,
) -> str:
    normalized_request = request.strip()
    if not normalized_request:
        raise ValueError("The request must not be empty.")
    if len(normalized_request) > MAX_REQUEST_CHARS:
        raise ValueError(
            f"The request exceeds the {MAX_REQUEST_CHARS}-character limit."
        )

    chain = (
        build_chat_prompt(task)
        | (model or create_chat_model(settings))
        | StrOutputParser()
    )
    response = chain.invoke({"request": normalized_request})
    if not response.strip():
        raise RuntimeError("Ollama returned an empty response.")
    return response.strip()
