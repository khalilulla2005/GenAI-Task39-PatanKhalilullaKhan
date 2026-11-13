from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from typing import Mapping
from urllib.parse import urlparse

from dotenv import load_dotenv

_MODEL_PATTERN = re.compile(r"[A-Za-z0-9._/-]+(?::[A-Za-z0-9._-]+)?")


@dataclass(frozen=True)
class Settings:
    base_url: str
    model: str
    temperature: float
    num_ctx: int
    num_predict: int
    request_timeout: int
    api_key: str = field(repr=False)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        secrets: Mapping[str, str] | None = None,
    ) -> Settings:
        if environ is None:
            load_dotenv()
            source = os.environ
        else:
            source = environ

        base_url = _setting(
            source,
            secrets,
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434",
        ).rstrip("/")
        parsed_url = urlparse(base_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or parsed_url.username
            or parsed_url.password
            or parsed_url.path not in {"", "/"}
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError(
                "OLLAMA_BASE_URL must be an HTTP(S) origin without embedded "
                "credentials, a path, query, or fragment."
            )

        model = _setting(source, secrets, "OLLAMA_MODEL", "codellama:7b")
        if not _MODEL_PATTERN.fullmatch(model):
            raise ValueError("OLLAMA_MODEL must be a valid Ollama model tag.")

        temperature = _float_setting(
            source,
            secrets,
            "OLLAMA_TEMPERATURE",
            0.2,
            minimum=0,
            maximum=2,
        )
        num_ctx = _integer_setting(
            source,
            secrets,
            "OLLAMA_NUM_CTX",
            8192,
            minimum=256,
            maximum=131072,
        )
        num_predict = _integer_setting(
            source,
            secrets,
            "OLLAMA_NUM_PREDICT",
            2048,
            minimum=1,
            maximum=32768,
        )
        request_timeout = _integer_setting(
            source,
            secrets,
            "OLLAMA_REQUEST_TIMEOUT",
            180,
            minimum=1,
            maximum=900,
        )
        api_key = _setting(source, secrets, "OLLAMA_API_KEY", "")

        return cls(
            base_url=base_url,
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            request_timeout=request_timeout,
            api_key=api_key,
        )

    def with_overrides(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
    ) -> Settings:
        values: dict[str, str] = {}
        if base_url is not None:
            values["OLLAMA_BASE_URL"] = base_url
        if model is not None:
            values["OLLAMA_MODEL"] = model
        if not values:
            return self
        current = {
            "OLLAMA_BASE_URL": self.base_url,
            "OLLAMA_MODEL": self.model,
            "OLLAMA_TEMPERATURE": str(self.temperature),
            "OLLAMA_NUM_CTX": str(self.num_ctx),
            "OLLAMA_NUM_PREDICT": str(self.num_predict),
            "OLLAMA_REQUEST_TIMEOUT": str(self.request_timeout),
            "OLLAMA_API_KEY": self.api_key,
        }
        current.update(values)
        return Settings.from_env(current)


def _setting(
    environ: Mapping[str, str],
    secrets: Mapping[str, str] | None,
    name: str,
    default: str,
) -> str:
    for source in (environ, secrets or {}):
        value = source.get(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _integer_setting(
    environ: Mapping[str, str],
    secrets: Mapping[str, str] | None,
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = _setting(environ, secrets, name, str(default))
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not minimum <= parsed_value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return parsed_value


def _float_setting(
    environ: Mapping[str, str],
    secrets: Mapping[str, str] | None,
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = _setting(environ, secrets, name, str(default))
    try:
        parsed_value = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not minimum <= parsed_value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return parsed_value
