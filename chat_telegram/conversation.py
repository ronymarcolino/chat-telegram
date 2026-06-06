from collections import defaultdict
from collections.abc import MutableMapping
from typing import Literal, TypedDict


Role = Literal["user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._conversations: MutableMapping[int, list[ChatMessage]] = defaultdict(list)

    def add_user_message(self, chat_id: int, content: str) -> list[ChatMessage]:
        return self._add_message(chat_id, "user", content)

    def add_assistant_message(self, chat_id: int, content: str) -> list[ChatMessage]:
        return self._add_message(chat_id, "assistant", content)

    def messages_for(self, chat_id: int) -> list[ChatMessage]:
        return list(self._conversations[chat_id])

    def _add_message(self, chat_id: int, role: Role, content: str) -> list[ChatMessage]:
        self._conversations[chat_id].append({"role": role, "content": content})
        return self.messages_for(chat_id)

