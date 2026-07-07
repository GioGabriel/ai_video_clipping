from __future__ import annotations

from dataclasses import dataclass

import httpx

from src.core.config import AppSettings

RECOMMENDED_OLLAMA_MODELS = (
    "llama3",
    "deepseek-r1",
    "deepseek-r1:8b",
    "qwen2.5",
    "qwen2.5:7b",
)


@dataclass(slots=True)
class OllamaModelCatalog:
    models: list[str]
    default_model: str
    available: bool


class OllamaRuntimeService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.settings.ollama_base_url,
                timeout=min(float(self.settings.ollama_request_timeout_seconds), 15.0),
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def list_models(self) -> OllamaModelCatalog:
        default_model = self.settings.ollama_model.strip() or "llama3"

        try:
            response = self._get_client().get("/api/tags")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError, RuntimeError):
            return OllamaModelCatalog(
                models=self._merge_models([], default_model),
                default_model=default_model,
                available=False,
            )

        raw_models = payload.get("models", []) if isinstance(payload, dict) else []
        names = [
            str(item.get("name", "")).strip()
            for item in raw_models
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        ]

        return OllamaModelCatalog(
            models=self._merge_models(names, default_model),
            default_model=default_model,
            available=True,
        )

    def _merge_models(self, discovered_models: list[str], default_model: str) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()

        for name in [*discovered_models, default_model, *RECOMMENDED_OLLAMA_MODELS]:
            normalized_name = self._normalize_alias(name)
            if not name or normalized_name in seen:
                continue
            normalized.append(name)
            seen.add(normalized_name)

        return normalized or [default_model]

    @staticmethod
    def _normalize_alias(model_name: str) -> str:
        normalized = model_name.strip()
        if normalized.endswith(":latest"):
            return normalized[: -len(":latest")]
        return normalized
