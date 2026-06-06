from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler
from telegram.ext import filters

from chat_telegram.config import Settings
from chat_telegram.conversation import InMemoryConversationStore
from chat_telegram.handlers import TelegramHandlers
from chat_telegram.llm import OllamaChatService


def build_application(settings: Settings | None = None) -> Application:
    settings = settings or Settings.from_env()

    conversations = InMemoryConversationStore()
    chat_service = OllamaChatService(
        model=settings.ollama_model,
        think=settings.ollama_think,
    )
    handlers = TelegramHandlers(conversations, chat_service)

    app = ApplicationBuilder().token(settings.telegram_token).build()
    app.add_handler(CommandHandler("start", handlers.start))
    app.add_handler(
        MessageHandler(filters.TEXT & (~filters.COMMAND), handlers.handle_message)
    )

    return app

