from collections.abc import Callable, Sequence
from typing import Any

from ollama import chat

from chat_telegram.conversation import ChatMessage


OllamaChat = Callable[..., Any]


class OllamaChatService:
    def __init__(
        self,
        model: str,
        *,
        think: bool = True,
        chat_client: OllamaChat = chat,
    ) -> None:
        self._model = model
        self._think = think
        self._chat_client = chat_client

    def reply(self, messages: Sequence[ChatMessage]) -> str:
        response = self._chat_client(
            model=self._model,
            messages=list(messages),
            think=self._think,
        )

        return response["message"]["content"]

