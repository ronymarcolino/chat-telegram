from dataclasses import dataclass
import os
from typing import Literal

from dotenv import load_dotenv


DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0

LlmProvider = Literal["ollama", "openai-compatible", "nvidia"]


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    llm_provider: LlmProvider
    llm_model: str | None = None
    model_think: bool = True
    llm_host_url: str | None = None
    llm_api_key: str | None = None
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        telegram_token = os.getenv("TELEGRAM_TOKEN")
        if not telegram_token:
            raise RuntimeError("TELEGRAM_TOKEN is required.")

        provider = _parse_provider(os.getenv("LLM_PROVIDER", "ollama"))
        return cls(
            telegram_token=telegram_token,
            llm_provider=provider,
            llm_model=os.getenv("LLM_MODEL", _default_model_for(provider)),
            llm_host_url=os.getenv("LLM_HOST_URL"),
            model_think=_parse_bool(os.getenv("LLM_THINK"), default=True),
            llm_api_key=_openai_api_key_for(provider),
            request_timeout_seconds=_parse_float(
                os.getenv("LLM_REQUEST_TIMEOUT_SECONDS"),
                default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
                name="LLM_REQUEST_TIMEOUT_SECONDS",
            ),
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


def _parse_provider(value: str) -> LlmProvider:
    normalized = value.strip().lower()
    if normalized in {"ollama", "openai-compatible", "nvidia"}:
        return normalized

    raise RuntimeError(
        "LLM_PROVIDER must be one of: ollama, openai-compatible, nvidia."
    )


def _default_model_for(provider: LlmProvider) -> str:
    if os.getenv("OPENAI_BASE_URL"):
        return os.getenv("OPENAI_BASE_URL")
    if os.getenv("NVIDIA_BASE_URL"):
        return os.getenv("NVIDIA_BASE_URL")

    return os.getenv("LLM_MODEL")


def _openai_base_url_for(provider: LlmProvider) -> str | None:
    if os.getenv("OPENAI_BASE_URL"):
        return os.getenv("OPENAI_BASE_URL")
    if os.getenv("NVIDIA_BASE_URL"):
        return os.getenv("NVIDIA_BASE_URL")

    return os.getenv("LLM_API_BASE_URL")


def _openai_api_key_for(provider: LlmProvider) -> str | None:
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    if os.getenv("NVIDIA_API_KEY"):
        return os.getenv("NVIDIA_API_KEY")
    return os.getenv("LLM_API_KEY")


def _parse_float(value: str | None, *, default: float, name: str) -> float:
    if value is None:
        return default

    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number.") from exc

    if parsed <= 0:
        raise RuntimeError(f"{name} must be greater than zero.")

    return parsed
