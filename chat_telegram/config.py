from dataclasses import dataclass
import os

from dotenv import load_dotenv


DEFAULT_OLLAMA_MODEL = "qwen3:4b"


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_think: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        telegram_token = os.getenv("TELEGRAM_TOKEN")
        if not telegram_token:
            raise RuntimeError("TELEGRAM_TOKEN is required.")

        return cls(
            telegram_token=telegram_token,
            ollama_model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            ollama_think=_parse_bool(os.getenv("OLLAMA_THINK"), default=True),
        )


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise RuntimeError(
        "OLLAMA_THINK must be one of: 1, true, yes, on, 0, false, no, off."
    )

