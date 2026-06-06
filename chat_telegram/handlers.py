import logging

from telegram import Update
from telegram.ext import ContextTypes

from chat_telegram.conversation import InMemoryConversationStore
from chat_telegram.llm import OllamaChatService


LOGGER = logging.getLogger(__name__)


class TelegramHandlers:
    def __init__(
        self,
        conversations: InMemoryConversationStore,
        chat_service: OllamaChatService,
    ) -> None:
        self._conversations = conversations
        self._chat_service = chat_service

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None:
            return

        await update.message.reply_text(
            "Hello! Send me a message and I will reply using AI!"
        )

    async def handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.message is None or update.effective_chat is None:
            return

        user_message = update.message.text
        if not user_message:
            return

        chat_id = update.effective_chat.id
        await update.message.reply_text("Thinking...")

        messages = self._conversations.add_user_message(chat_id, user_message)

        try:
            response_text = self._chat_service.reply(messages)
        except Exception:
            LOGGER.exception("Failed to generate Ollama response")
            await update.message.reply_text("Something went wrong.")
            return

        self._conversations.add_assistant_message(chat_id, response_text)
        await update.message.reply_text(response_text)

