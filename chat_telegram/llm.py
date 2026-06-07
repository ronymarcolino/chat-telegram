from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from ollama import Client, chat

from chat_telegram.config import Settings
from chat_telegram.conversation import ChatMessage


OllamaChat = Callable[..., Any]


class ChatService(Protocol):
    def reply(self, messages: Sequence[ChatMessage]) -> str:
        pass


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float


class OllamaChatService:
    def __init__(
        self,
        model: str,
        *,
        think: bool = True,
        host: str | None = None,
        chat_client: OllamaChat = chat,
    ) -> None:
        self._model = model
        self._think = think
        self._chat_client = Client(host=host).chat if host else chat_client

    def reply(self, messages: Sequence[ChatMessage]) -> str:
        response = self._chat_client(
            model=self._model,
            messages=list(messages),
            think=self._think,
        )

        return response["message"]["content"]


class OpenAICompatibleChatService:
    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._http_client = http_client or httpx.Client(timeout=config.timeout_seconds)

    def reply(self, messages: Sequence[ChatMessage]) -> str:
        response = self._http_client.post(
            self._chat_completions_url,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._config.model,
                "messages": list(messages),
            },
        )
        response.raise_for_status()

        payload = response.json()
        return payload["choices"][0]["message"]["content"]

    @property
    def _chat_completions_url(self) -> str:
        return f"{self._config.base_url.rstrip('/')}/chat/completions"


def build_chat_service(settings: Settings) -> ChatService:
    if not settings.llm_provider:
        raise RuntimeError("LLM_PROVIDER is required to specify which LLM provider to use.")

    if settings.llm_provider == "ollama":
        return OllamaChatService(
            model=settings.llm_model,
            think=settings.model_think,
            host=settings.llm_host_url,
        )

    if settings.llm_provider == "nvidia":
        return OpenAICompatibleChatService(
            OpenAICompatibleConfig(
                base_url=settings.llm_host_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                timeout_seconds=settings.request_timeout_seconds,
            )
        )
    return OpenAICompatibleChatService(
        OpenAICompatibleConfig(
            base_url=settings.llm_host_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_seconds=settings.request_timeout_seconds,
        )
    )
